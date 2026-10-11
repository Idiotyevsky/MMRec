#!/usr/bin/env python
"""Cache a fixed four-channel ranking, then evaluate MMR quality/diversity.

No model is trained. Test lambda variants are descriptive, not used to select
the serving default. The default opt-in lambda (.8) is fixed before evaluation.
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
from src.rerank import RerankConfig, content_diversity, rerank
from src.serving.app import configure_threads
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.provenance import dataset_fingerprint, git_state


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--itemcf-index", default="artifacts/itemcf_neighbors.npz")
    ap.add_argument("--content-embeddings", default="artifacts/content_embeddings.npy")
    ap.add_argument("--two-tower-run", default="results/runs/two_tower_base")
    ap.add_argument("--mm-run", default="results/runs/mm_concat_20260917-075552_b84b95")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-users", type=int, default=10000)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--candidate-k", type=int, default=1000)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--final-k", type=int, default=20)
    ap.add_argument("--lambdas", nargs="+", type=float, default=[.7, .8, .9, .95])
    ap.add_argument("--cache", default="artifacts/mmr_ranked_candidates.npz")
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--out", default="results/tables/mmr_tradeoff.csv")
    ap.add_argument("--json-out", default="results/tables/mmr_evaluation.json")
    args = ap.parse_args()
    if args.max_users < 1 or not 1 <= args.final_k <= args.window <= args.candidate_k:
        ap.error("require 1 <= final-k <= window <= candidate-k and positive max-users")
    for value in args.lambdas:
        RerankConfig(mmr=True, mmr_lambda=value, mmr_candidate_k=args.window)
    configure_threads()
    log = get_logger("mmrec.evaluate_mmr", ROOT / "results/logs/mmr_evaluation.log")
    data = ProcessedData.load(ROOT / args.processed_dir)
    users = np.sort(np.random.default_rng(args.eval_seed).choice(
        data.num_users, min(args.max_users, data.num_users), replace=False)).astype(np.int64)
    n = len(users)
    embedding = np.load(ROOT / args.content_embeddings)
    if embedding.ndim != 2 or embedding.shape[0] != data.num_items+1:
        raise ValueError("MMR content catalogue mismatch")
    paths = [args.itemcf_index, args.content_embeddings,
             f"{args.two_tower_run}/best.pt", f"{args.two_tower_run}/two_tower_item_embeddings.npy",
             f"{args.mm_run}/best.pt", f"{args.mm_run}/config.yaml"]
    provenance = {**git_state(), "dataset_hash": dataset_fingerprint(data),
                  "mapping_hash": digest(ROOT / args.processed_dir / "mappings.json"),
                  "artifact_sha256": {p:digest(ROOT / p) for p in paths}}
    contract = {"dataset_hash": provenance["dataset_hash"], "mapping_hash": provenance["mapping_hash"],
                "artifact_sha256": provenance["artifact_sha256"], "user_hash": user_subset_hash(users),
                "candidate_k": args.candidate_k, "window": args.window,
                "per_source_k": 1000, "merge_method": "rrf",
                "sources": ["popular", "itemcf", "semantic", "two_tower"]}
    stamp = json.dumps(contract, sort_keys=True)
    cache_path = ROOT / args.cache
    ids = np.zeros((n, args.window), dtype=np.int64)
    scores = np.full((n, args.window), -np.inf, dtype=np.float32)
    candidate_hit = np.zeros(n, dtype=bool)
    completed = 0
    if cache_path.exists() and args.reuse_cache:
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache["contract"]) != stamp or not np.array_equal(cache["users"], users):
                raise ValueError("ranking cache data/checkpoint/index/protocol mismatch")
            if cache["ids"].shape != ids.shape or cache["scores"].shape != scores.shape:
                raise ValueError("ranking cache shape mismatch")
            ids[:] = cache["ids"]
            scores[:] = cache["scores"]
            candidate_hit[:] = cache["candidate_hit"]
            completed = int(cache["completed"])
            if not 0 <= completed <= n:
                raise ValueError("invalid ranking cache progress")
        log.info(f"loaded ranked cache: {completed}/{n} users")
    start = time.perf_counter()
    if completed < n:
        build_args = SimpleNamespace(**vars(args), two_tower_device=args.device,
            merge_method="rrf", mm_concat_run=args.mm_run, sasrec_run=None,
            collaborative_channel="itemcf")
        merger, registry = build(build_args, data)
        if merger.source_names != contract["sources"]:
            raise ValueError("all four recall channels required")
        ranker = registry.get("mm_concat")
        for row in range(completed, n):
            user = int(users[row])
            history = data.test_history(user).tolist()
            merged = merger.recall_and_merge(user, history, per_source_k=1000, total_k=args.candidate_k)
            if merged.stats.get("errors"):
                raise RuntimeError(merged.stats["errors"])
            candidates = np.asarray(merged.item_ids(), dtype=np.int64)
            candidate_hit[row] = int(data.test_target[user]) in candidates
            ranking = ranker.score_candidates(history, candidates)
            order = np.argsort(-ranking, kind="stable")[:args.window]
            ids[row,:len(order)] = candidates[order]
            scores[row,:len(order)] = ranking[order]
            if (row+1) % 1000 == 0 or row+1 == n:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, users=users, ids=ids, scores=scores,
                                    candidate_hit=candidate_hit, completed=row+1, contract=stamp)
                tmp.replace(cache_path)
                log.info(f"ranked cache: {row+1}/{n} users ({time.perf_counter()-start:.0f}s)")
    configs = [("score_only", RerankConfig())] + [(f"mmr_{value:g}",
        RerankConfig(mmr=True, mmr_lambda=value, mmr_candidate_k=args.window)) for value in args.lambdas]
    accum = {name:{"hit": [], "ndcg": [], "ild": [], "near_dup": [], "valid_pairs": [],
                   "latency_ms": [], "head_share": [], "tail_share": [], "zero_share": [],
                   "unique_items": set(), "changed": [], "fallbacks": 0} for name,_ in configs}
    eval_start = time.perf_counter()
    for row, user in enumerate(users.tolist()):
        entries = [{"item_id": int(item), "ranking_score": float(score),
                    "exploration_candidate": bool(data.train_freq[item] == 0)}
                   for item,score in zip(ids[row],scores[row]) if item > 0]
        history = set(data.test_history(user).tolist())
        target = int(data.test_target[user])
        base_ids = ids[row,:args.final_k].tolist()
        for name, cfg in configs:
            t0 = time.perf_counter()
            result = rerank(entries, cfg, args.final_k, history, content_embeddings=embedding)
            latency = (time.perf_counter()-t0)*1000
            selected = [e["item_id"] for e in result.items]
            hit = target in selected
            diversity = content_diversity(embedding[selected])
            a = accum[name]
            a["hit"].append(hit)
            a["ndcg"].append(1/np.log2(selected.index(target)+2) if hit else 0)
            a["ild"].append(diversity["ild"])
            a["near_dup"].append(diversity["near_duplicate_pair_share"])
            a["valid_pairs"].append(diversity["valid_pair_fraction"])
            a["latency_ms"].append(latency)
            buckets = data.popularity_bucket[selected]
            a["head_share"].append(float((buckets == 2).mean()))
            a["tail_share"].append(float((buckets == 0).mean()))
            a["zero_share"].append(float((data.train_freq[selected] == 0).mean()))
            a["unique_items"].update(selected)
            a["changed"].append(sum(x != y for x,y in zip(base_ids,selected)))
            a["fallbacks"] += bool(cfg.mmr and not result.applied.get("mmr",{}).get("applied"))
        if (row+1) % 2000 == 0:
            log.info(f"MMR replay: {row+1}/{n} users ({time.perf_counter()-eval_start:.0f}s)")
    def mean_valid(values):
        valid = [v for v in values if v is not None]
        return float(np.mean(valid)) if valid else None
    rows, paired = [], []
    for name,cfg in configs:
        a = accum[name]
        rows.append({"method": name, "num_users": n, "user_subset_hash": user_subset_hash(users),
            "candidate_k": args.candidate_k, "mmr_window": args.window, "lambda": cfg.mmr_lambda if cfg.mmr else 1,
            "candidate_recall": float(candidate_hit.mean()),
            f"Recall@{args.final_k}": float(np.mean(a["hit"])),
            f"NDCG@{args.final_k}": float(np.mean(a["ndcg"])),
            "ILD": mean_valid(a["ild"]), "near_duplicate_pair_share": mean_valid(a["near_dup"]),
            "valid_pair_fraction": float(np.mean(a["valid_pairs"])),
            "catalogue_coverage": len(a["unique_items"])/data.num_items,
            "unique_items": len(a["unique_items"]), "head_share": float(np.mean(a["head_share"])),
            "tail_share": float(np.mean(a["tail_share"])), "zero_train_share": float(np.mean(a["zero_share"])),
            "rerank_cpu_mean_ms": float(np.mean(a["latency_ms"])),
            "rerank_cpu_p50_ms": float(np.median(a["latency_ms"])),
            "rerank_cpu_p95_ms": float(np.percentile(a["latency_ms"],95)),
            "mean_changed_positions": float(np.mean(a["changed"])), "fallback_users": a["fallbacks"]})
        if cfg.mmr:
            for metric in ("hit", "ndcg", "ild"):
                delta = np.asarray(a[metric],dtype=float) - np.asarray(accum["score_only"][metric],dtype=float)
                delta = delta[np.isfinite(delta)]
                se = float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else 0
                paired.append({"method": name, "metric": metric, "delta": float(delta.mean()),
                               "ci95_low": float(delta.mean()-1.96*se), "ci95_high": float(delta.mean()+1.96*se)})
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    save_json({"protocol": {**vars(args), "num_users": n, "user_subset_hash": user_subset_hash(users),
                   "split": "test", "exploration": False, "lambda_policy": "fixed grid, no test-set selection"},
               "cache_contract": contract, "provenance": provenance, "results": rows,
               "paired_differences": paired,
               "diversity_note": "Content cosine proxy, not measured category diversity; unavailable pairs excluded.",
               "latency_note": "CPU rerank only, includes MMR trace diagnostics; excludes recall/ranking."},
              ROOT / args.json_out)
    log.info(f"complete: {out}; results={rows}")


if __name__ == "__main__":
    main()
