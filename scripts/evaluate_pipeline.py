#!/usr/bin/env python
"""Two-stage serving evaluation with a same-checkpoint full-ranking baseline.

    python scripts/evaluate_pipeline.py
    python scripts/evaluate_pipeline.py --candidate-sizes 100 500 1000 --max-users 20000

For one ranker checkpoint and one **fixed random sample of users**, this measures
both:

* the strict full-catalogue ranking (every item, history masked), and
* the two-stage pipeline at several candidate budgets (recall → merge → rank).

and reports how much of the full-ranking quality the pipeline retains::

    recall_retention = pipeline Recall@20 / same-checkpoint full Recall@20

This is the only apples-to-apples way to state a retention number.  Comparing a
single-seed pipeline run on 10 000 users against a 3-seed full-ranking mean on
100 000 users mixes two different protocols and is not a measurement.

The user sample is drawn with a fixed seed and its hash is written to the JSON
artifact, so the comparison is reproducible and free of id-order bias.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.data.popularity import BUCKET_NAMES  # noqa: E402
from src.pipeline import RankerRegistry, RankerSpec  # noqa: E402
from src.pipeline.rankers import discover_run  # noqa: E402
from src.recall import (  # noqa: E402
    CandidateMerger,
    ItemCFRecall,
    PopularRecall,
    SemanticRecall,
    load_itemcf_index,
)
from src.serving.app import configure_threads  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.evaluate_pipeline")


def user_subset_hash(users: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(users, dtype=np.int64).tobytes()).hexdigest()[:16]


def build(args, data: ProcessedData):
    strategies = [PopularRecall(data.train_freq, data.num_items)]
    ic_path = ROOT / args.itemcf_index
    if ic_path.exists():
        ic = load_itemcf_index(ic_path)
        strategies.append(ItemCFRecall(ic["neighbors"], ic["sims"], data.num_items))
    else:
        LOG.warning(f"ItemCF index missing at {ic_path}")
    emb_path = ROOT / args.content_embeddings
    if emb_path.exists():
        strategies.append(SemanticRecall(np.load(emb_path), data.num_items))
    else:
        LOG.warning(f"content embeddings missing at {emb_path}")
    merger = CandidateMerger(strategies)

    specs = {}
    for name in ("sasrec", "mm_concat"):
        override = getattr(args, f"{name}_run", None)
        run = Path(override) if override else discover_run(name)
        if run is None:
            raise SystemExit(f"no finished run for ranker {name!r}")
        specs[name] = RankerSpec(name, run)
    registry = RankerRegistry(data, specs, device=args.device)
    return merger, registry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--itemcf-index", default="artifacts/itemcf_neighbors.npz")
    ap.add_argument("--content-embeddings", default="artifacts/content_embeddings.npy")
    ap.add_argument("--sasrec-run", default=None, help="default: newest finished 'sasrec' run")
    ap.add_argument("--mm-run", default=None, help="default: newest finished 'mm_concat' run")
    ap.add_argument("--ranker", default="mm_concat")
    ap.add_argument("--candidate-sizes", nargs="*", type=int, default=[100, 200, 500, 1000, 2000])
    ap.add_argument("--recall-k", type=int, default=1000, help="per-channel budget")
    ap.add_argument("--final-k", type=int, default=20)
    ap.add_argument("--max-users", type=int, default=10000)
    ap.add_argument("--eval-seed", type=int, default=42, help="seed for the user sample")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/tables/pipeline_tradeoff.csv")
    ap.add_argument("--source-out", default="results/tables/recall_source_contribution.csv")
    args = ap.parse_args()

    configure_threads()
    data = ProcessedData.load(ROOT / args.processed_dir)
    merger, registry = build(args, data)
    model = registry.get(args.ranker)

    # ---- fixed random user sample (not users[:N], which is id-order biased) ----
    rng = np.random.default_rng(args.eval_seed)
    n = min(args.max_users, data.num_users)
    users = np.sort(rng.choice(data.num_users, size=n, replace=False)).astype(np.int64)
    uhash = user_subset_hash(users)
    targets = data.test_target[users]
    LOG.info(f"dataset: {data.summary()} | ranker={args.ranker}")
    LOG.info(f"user sample: {n} users, seed={args.eval_seed}, hash={uhash}")

    all_items = model.valid_item_ids()
    n_users = users.shape[0]
    max_k = max(args.candidate_sizes)

    per_size = {
        k: {"cand_hit": np.zeros(n_users, dtype=bool),
            "final_hit": np.zeros(n_users, dtype=bool),
            "ndcg": np.zeros(n_users, dtype=np.float64),
            "pool": np.zeros(n_users, dtype=np.int64),
            "contrib": {**{name: 0 for name in merger.source_names}, "multiple": 0},
            "contrib_by_bucket": {b: {**{name: 0 for name in merger.source_names}, "multiple": 0}
                                  for b in BUCKET_NAMES.values()}}
        for k in args.candidate_sizes
    }
    # same-checkpoint full-catalogue baseline
    full_hit = np.zeros(n_users, dtype=bool)
    full_ndcg = np.zeros(n_users, dtype=np.float64)
    bucket_n = {b: 0 for b in BUCKET_NAMES.values()}
    t0 = time.perf_counter()

    for i, u in enumerate(users.tolist()):
        hist = [int(x) for x in data.test_history(u)]
        tgt = int(targets[i])
        bname = BUCKET_NAMES.get(int(data.popularity_bucket[tgt]), "unknown")
        if bname in bucket_n:
            bucket_n[bname] += 1

        vec = model.encode_user(hist)

        # ---- full-catalogue ranking with the SAME checkpoint and history ----
        full_scores = model.seen_mask_overhead(hist, model.score_with_user_vector(vec, all_items))
        tgt_full = full_scores[tgt - 1]
        if tgt_full > -np.inf:
            rank_full = int((full_scores > tgt_full).sum()) + 1
            if rank_full <= args.final_k:
                full_hit[i] = True
                full_ndcg[i] = 1.0 / np.log2(rank_full + 1)

        # ---- two-stage pipeline ----
        merged = merger.recall_and_merge(u, hist, per_source_k=args.recall_k, total_k=None)
        pool = merged.item_ids()
        for cand_k in args.candidate_sizes:
            st = per_size[cand_k]
            ids = pool[:cand_k]
            st["pool"][i] = len(ids)
            if not ids:
                continue
            st["cand_hit"][i] = tgt in set(ids)
            scores = model.score_with_user_vector(vec, np.asarray(ids, dtype=np.int64))
            order = np.argsort(-scores)[: args.final_k]
            top = [ids[j] for j in order.tolist()]
            if tgt in top:
                pos = top.index(tgt) + 1
                st["final_hit"][i] = True
                st["ndcg"][i] = 1.0 / np.log2(pos + 1)
                srcs = {s["name"] for s in merged.candidates[order[pos - 1]].sources}
                key = next(iter(srcs)) if len(srcs) == 1 else "multiple"
                st["contrib"][key] += 1
                if bname in st["contrib_by_bucket"]:
                    st["contrib_by_bucket"][bname][key] += 1

        if (i + 1) % 2000 == 0:
            LOG.info(f"  {i+1}/{n_users} users ({time.perf_counter()-t0:.0f}s)")

    full_recall = float(full_hit.mean())
    full_ndcg_mean = float(full_ndcg.mean())

    rows: list[dict] = []
    source_rows: list[dict] = []
    for cand_k in args.candidate_sizes:
        st = per_size[cand_k]
        pipe_recall = float(st["final_hit"].mean())
        pipe_ndcg = float(st["ndcg"].mean())
        rows.append({
            "candidate_k": cand_k,
            "ranker": args.ranker,
            "num_users": n_users,
            "eval_seed": args.eval_seed,
            "user_subset_hash": uhash,
            "candidate_recall": float(st["cand_hit"].mean()),
            "pipeline_Recall@20": pipe_recall,
            "pipeline_NDCG@20": pipe_ndcg,
            "full_Recall@20": full_recall,
            "full_NDCG@20": full_ndcg_mean,
            "recall_retention": pipe_recall / full_recall if full_recall > 0 else None,
            "ndcg_retention": pipe_ndcg / full_ndcg_mean if full_ndcg_mean > 0 else None,
            "mean_pool_size": float(st["pool"].mean()),
        })
        LOG.info(f"cand_k={cand_k:5d}: pool={st['pool'].mean():7.0f} "
                 f"cand_recall={st['cand_hit'].mean():.4f} "
                 f"R@20={pipe_recall:.4f} (retention {pipe_recall/full_recall:.3f})")

        total_hits = max(sum(st["contrib"].values()), 1)
        for name, c in st["contrib"].items():
            row = {"candidate_k": cand_k, "source": name,
                   "top20_hits": c, "share_of_hits": c / total_hits}
            for b in BUCKET_NAMES.values():
                row[f"{b}_hits"] = st["contrib_by_bucket"][b].get(name, 0)
                row[f"{b}_users"] = bucket_n[b]
            source_rows.append(row)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sout = ROOT / args.source_out
    with open(sout, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(source_rows[0].keys()))
        w.writeheader()
        w.writerows(source_rows)

    header = (f"{'cand_k':>8s}{'pool':>8s}{'cand_recall':>13s}"
              f"{'pipeline R@20':>15s}{'retention':>11s}")
    print(f"\n=== two-stage retention ({args.ranker}, {n_users} users, seed {args.eval_seed}) ===")
    print(f"same-checkpoint full-catalogue Recall@20 = {full_recall:.4f} "
          f"(NDCG@20 = {full_ndcg_mean:.4f})")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['candidate_k']:>8d}{r['mean_pool_size']:>8.0f}{r['candidate_recall']:>13.4f}"
              f"{r['pipeline_Recall@20']:>15.4f}{r['recall_retention']:>11.3f}")

    print("\n=== which recall channel produced the final top-20 hits? ===")
    for r in source_rows:
        if r["candidate_k"] != args.candidate_sizes[-1]:
            continue
        print(f"  {r['source']:<10s} hits={r['top20_hits']:<6d} share={r['share_of_hits']:.3f}")

    save_json({
        "processed_dir": args.processed_dir,
        "ranker": args.ranker,
        "ranker_run": str(registry.specs[args.ranker].run_dir),
        "num_users": int(n_users),
        "eval_seed": int(args.eval_seed),
        "user_subset_hash": uhash,
        "recall_k_per_channel": args.recall_k,
        "final_k": args.final_k,
        "full_catalogue": {"Recall@20": full_recall, "NDCG@20": full_ndcg_mean},
        "tradeoff": rows,
        "source_contribution": source_rows,
        "note": ("pipeline and full-catalogue numbers come from the same checkpoint and "
                 "the same user sample; retention = pipeline / full. Not comparable to "
                 "results/tables/overall.csv, which is a multi-seed full-catalogue mean."),
    }, ROOT / "results" / "tables" / "pipeline_tradeoff.json")
    print(f"\nwrote {out} and {sout}")


if __name__ == "__main__":
    main()
