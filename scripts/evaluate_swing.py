#!/usr/bin/env python
"""Paired ItemCF/Swing test: single-channel recall and four-channel RRF pipeline.

Shared channels run once per user. Only the collaborative channel changes;
this is NOT a five-channel comparison. No training or test-set tuning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_pipeline import build, user_subset_hash
from src.data.dataset import ProcessedData
from src.recall import CandidateMerger, load_swing_recall
from src.serving.app import configure_threads
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.provenance import dataset_fingerprint, git_state


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_difference(a, b):
    """Swing minus ItemCF; descriptive paired normal-approximation 95% CI."""
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    se = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else 0.0
    mean = float(delta.mean())
    return {"delta": mean, "ci95_low": mean-1.96*se, "ci95_high": mean+1.96*se}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--itemcf-index", default="artifacts/itemcf_neighbors.npz")
    ap.add_argument("--swing-index", default="artifacts/swing_neighbors.npz")
    ap.add_argument("--content-embeddings", default="artifacts/content_embeddings.npy")
    ap.add_argument("--two-tower-run", default="results/runs/two_tower_base")
    ap.add_argument("--mm-run", default="results/runs/mm_concat_20260917-075552_b84b95")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-users", type=int, default=10000)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--out-prefix", default="results/tables/swing")
    ap.add_argument("--log-file", default="results/logs/swing_evaluation.log")
    ap.add_argument("--resume", action="store_true", help="resume a matching 1000-user progress snapshot")
    args = ap.parse_args()
    if args.max_users < 1:
        ap.error("max-users must be positive")
    log = get_logger("mmrec.swing_evaluation", ROOT / args.log_file)
    configure_threads()
    data = ProcessedData.load(ROOT / args.processed_dir)
    provenance = {**git_state(), "dataset_hash": dataset_fingerprint(data),
                  "mapping_hash": sha256(ROOT / args.processed_dir / "mappings.json")}
    source_paths = [args.itemcf_index, args.swing_index, args.content_embeddings,
                    f"{args.two_tower_run}/best.pt", f"{args.mm_run}/best.pt"]
    provenance["artifact_sha256"] = {p: sha256(ROOT / p) for p in source_paths}
    with np.load(ROOT / args.itemcf_index) as old, np.load(ROOT / args.swing_index) as new:
        if old["neighbors"].shape != new["neighbors"].shape:
            raise ValueError("ItemCF/Swing neighbor budgets differ")
        if not np.array_equal(old["freq"], new["freq"]):
            raise ValueError("ItemCF/Swing training frequencies differ")
        provenance["top_m"] = int(old["neighbors"].shape[1])
        provenance["swing_parameters"] = {k: float(new[k]) for k in ("alpha1", "alpha2", "beta")}
    build_args = SimpleNamespace(**vars(args), two_tower_device=args.device,
                                 merge_method="rrf", mm_concat_run=args.mm_run,
                                 collaborative_channel="itemcf", sasrec_run=None)
    baseline, registry = build(build_args, data)
    if baseline.source_names != ["popular", "itemcf", "semantic", "two_tower"]:
        raise ValueError("all four baseline channels are required")
    swing = load_swing_recall(ROOT / args.swing_index, data)
    combined = CandidateMerger([*baseline.strategies, swing])
    mergers = {"itemcf": baseline,
               "swing": CandidateMerger([baseline.strategies[0], swing, *baseline.strategies[2:]])}
    ranker = registry.get("mm_concat")
    users = np.sort(np.random.default_rng(args.eval_seed).choice(
        data.num_users, min(args.max_users, data.num_users), replace=False)).astype(np.int64)
    n = len(users)
    ks = [100, 200, 500, 1000, 2000]
    channel_ks = ks[:-1]
    channel_hits = {c: np.zeros((n, 4), bool) for c in ("itemcf", "swing")}
    channel_counts = {c: np.zeros(n, dtype=int) for c in channel_hits}
    metrics = {c: {m: np.zeros((n, len(ks))) for m in ("candidate_recall", "Recall@20", "NDCG@20")}
               for c in mergers}
    full = np.zeros((n, 2))
    all_items = ranker.valid_item_ids()
    prefix = ROOT / args.out_prefix
    progress_path = ROOT / "artifacts" / f"{prefix.name}_progress.npz"
    code_paths = ["scripts/evaluate_swing.py", "scripts/evaluate_pipeline.py",
                  "src/recall/itemcf.py", "src/recall/swing.py", "src/recall/pipeline.py",
                  "src/recall/semantic.py", "src/recall/two_tower.py", "src/pipeline/rankers.py"]
    provenance["code_sha256"] = {p: sha256(ROOT / p) for p in code_paths}
    contract = json.dumps({"args": {k:v for k,v in vars(args).items() if k != "resume"},
                           "provenance": provenance}, sort_keys=True)
    state = {"users": users, "full": full,
             **{f"{c}_hits": v for c,v in channel_hits.items()},
             **{f"{c}_counts": v for c,v in channel_counts.items()},
             **{f"{c}_{m}": v for c,group in metrics.items() for m,v in group.items()}}
    completed, previous_elapsed = 0, 0.0
    if args.resume:
        with np.load(progress_path, allow_pickle=False) as saved:
            if str(saved["contract"]) != contract or not np.array_equal(saved["users"], users):
                raise ValueError("progress snapshot protocol/code/artifact mismatch")
            completed = int(saved["completed"])
            previous_elapsed = float(saved["elapsed_s"])
            if not 0 <= completed <= n:
                raise ValueError("invalid progress snapshot length")
            for key, array in state.items():
                if saved[key].shape != array.shape:
                    raise ValueError(f"invalid progress array: {key}")
                array[...] = saved[key]
    start = time.perf_counter()
    log.info(f"paired test: {n} users, seed={args.eval_seed}, hash={user_subset_hash(users)}")
    for row, user in enumerate(users.tolist()):
        if row < completed:
            continue
        history = data.test_history(user).tolist()
        target = int(data.test_target[user])
        results = combined.recall(user, history, per_source_k=1000)
        if results.get("_errors"):
            raise RuntimeError(results["_errors"])
        for c in channel_hits:
            ids = [x.item_id for x in results[c]]
            channel_counts[c][row] = len(ids)
            for j, k in enumerate(channel_ks):
                channel_hits[c][row, j] = target in ids[:k]
        vector = ranker.encode_user(history)
        scores = ranker.seen_mask_overhead(history, ranker.score_with_user_vector(vector, all_items))
        rank = int((scores > scores[target-1]).sum()) + 1
        if np.isfinite(scores[target-1]) and rank <= 20:
            full[row] = (1, 1/np.log2(rank+1))
        for c, merger in mergers.items():
            merged = merger.merge({name: results[name] for name in merger.source_names})
            ids = merged.item_ids()
            for j, k in enumerate(ks):
                candidates = ids[:k]
                if target not in candidates:
                    continue
                metrics[c]["candidate_recall"][row, j] = 1
                candidate_scores = ranker.score_with_user_vector(vector, candidates)
                order = np.argsort(-candidate_scores)[:20]
                top = [candidates[pos] for pos in order]
                if target in top:
                    metrics[c]["Recall@20"][row, j] = 1
                    metrics[c]["NDCG@20"][row, j] = 1/np.log2(top.index(target)+2)
        if (row+1) % 1000 == 0:
            log.info(f"{row+1}/{n} users ({time.perf_counter()-start:.0f}s)")
            tmp = progress_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **state, completed=row+1, contract=contract,
                                elapsed_s=previous_elapsed+time.perf_counter()-start)
            tmp.replace(progress_path)
    retrieval = [{"channel": c, "num_users": n, "mean_candidates": float(channel_counts[c].mean()),
                  **{f"Recall@{k}": float(channel_hits[c][:,j].mean()) for j,k in enumerate(channel_ks)}}
                 for c in channel_hits]
    pipeline, differences = [], []
    for j, k in enumerate(ks):
        for c in mergers:
            pipeline.append({"collaborative_channel": c, "candidate_k": k, "num_users": n,
                "merge_method": "rrf", "per_source_k": 1000,
                **{m: float(v[:,j].mean()) for m,v in metrics[c].items()},
                "recall_retention": float(metrics[c]["Recall@20"][:,j].mean()/full[:,0].mean())
                    if full[:,0].mean() else None})
        for m in metrics["itemcf"]:
            differences.append({"candidate_k": k, "metric": m,
                **paired_difference(metrics["itemcf"][m][:,j], metrics["swing"][m][:,j])})
    buckets = data.popularity_bucket[data.test_target[users]]
    zero = data.train_freq[data.test_target[users]] == 0
    groups = {"head": buckets == 2, "middle": buckets == 1,
              "tail_warm": (buckets == 0) & ~zero, "zero_train": zero}
    breakdown = []
    for name, mask in groups.items():
        for c in channel_hits:
            breakdown.append({"bucket": name, "channel": c, "num_users": int(mask.sum()),
                **{f"Recall@{k}": float(channel_hits[c][mask,j].mean()) if mask.any() else None
                   for j,k in enumerate(channel_ks)}})
    a, b = (channel_hits[c][:,-1] for c in ("itemcf", "swing"))
    overlap = {"k": 1000, "both_hits": int((a & b).sum()),
               "itemcf_only_hits": int((a & ~b).sum()), "swing_only_hits": int((b & ~a).sum())}
    for suffix, rows in (("recall", retrieval), ("pipeline", pipeline),
                         ("paired_deltas", differences), ("popularity", breakdown)):
        write_csv(Path(f"{prefix}_{suffix}.csv"), rows)
    artifact = {"protocol": {**vars(args), "num_users": n, "user_subset_hash": user_subset_hash(users),
                    "split": "test", "per_source_k": 1000, "candidate_sizes": ks,
                    "merge_method": "rrf", "rerank": "off", "channels": {c:m.source_names for c,m in mergers.items()}},
                "provenance": provenance, "retrieval": retrieval, "pipeline": pipeline,
                "paired_differences": differences, "popularity": breakdown, "overlap": overlap,
                "full_catalogue": {"Recall@20": float(full[:,0].mean()), "NDCG@20": float(full[:,1].mean())},
                "elapsed_s": previous_elapsed+time.perf_counter()-start}
    save_json(artifact, Path(f"{prefix}_comparison.json"))
    raw = {"users": users, "full": full, **{f"{c}_hits": h for c,h in channel_hits.items()}}
    raw.update({f"{c}_{m}": v for c, group in metrics.items() for m,v in group.items()})
    raw_path = ROOT / "artifacts" / f"{prefix.name}_paired_users.npz"
    np.savez_compressed(raw_path, **raw)
    log.info(f"completed: {prefix}_comparison.json; retrieval={retrieval}; overlap={overlap}")


if __name__ == "__main__":
    main()
