#!/usr/bin/env python
"""Serving latency benchmark, isolated from the accuracy evaluation.

    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --candidate-sizes 100 500 1000 5000

Accuracy needs many users but only one timing each; latency needs few users but
many repetitions.  This script does the second thing: for each candidate budget
it measures the recall stage once per user and the ranking stage many times,
and reports the **minimum** as well as the median.

The minimum is the number to read on a shared machine: it is the least
contended run and therefore the closest estimate of the true cost.  The median
and p95 are reported too so the noise is visible rather than hidden.

Writes ``results/tables/latency_benchmark.csv``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.pipeline import RankerRegistry, RankerSpec  # noqa: E402
from src.pipeline.rankers import discover_run  # noqa: E402
from src.recall import (  # noqa: E402
    CandidateMerger,
    ItemCFRecall,
    PopularRecall,
    SemanticRecall,
    load_itemcf_index,
)
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.latency")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--itemcf-index", default="artifacts/itemcf_neighbors.npz")
    ap.add_argument("--content-embeddings", default="artifacts/content_embeddings.npy")
    ap.add_argument("--ranker", default="mm_concat")
    ap.add_argument("--candidate-sizes", nargs="*", type=int, default=[100, 200, 500, 1000, 2000])
    ap.add_argument("--include-full-catalogue", action="store_true", default=True,
                    help="also time scoring every item, to show what the pool saves")
    ap.add_argument("--recall-k", type=int, default=500, help="per-channel budget")
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/tables/latency_benchmark.csv")
    args = ap.parse_args()

    from src.serving.app import configure_threads

    configure_threads()  # measure the serving configuration, not the default
    data = ProcessedData.load(ROOT / args.processed_dir)
    strategies = [PopularRecall(data.train_freq, data.num_items)]
    ic_path = ROOT / args.itemcf_index
    if ic_path.exists():
        ic = load_itemcf_index(ic_path)
        strategies.append(ItemCFRecall(ic["neighbors"], ic["sims"], data.num_items))
    emb_path = ROOT / args.content_embeddings
    if emb_path.exists():
        strategies.append(SemanticRecall(np.load(emb_path), data.num_items))
    merger = CandidateMerger(strategies)

    run = discover_run(args.ranker)
    if run is None:
        raise SystemExit(f"no finished run for ranker {args.ranker!r}")
    registry = RankerRegistry(data, {args.ranker: RankerSpec(args.ranker, run)}, device=args.device)
    model = registry.get(args.ranker)
    LOG.info(f"ranker={args.ranker} | channels={merger.source_names} | {data.summary()}")

    users = np.arange(min(args.users, data.num_users), dtype=np.int64)
    sizes = list(args.candidate_sizes)
    if args.include_full_catalogue and data.num_items not in sizes:
        sizes.append(data.num_items)
    rows = []
    for cand_k in sizes:
        recall_ms: list[float] = []
        rank_ms: list[float] = []
        for u in users.tolist():
            hist = [int(x) for x in data.test_history(u)]
            t0 = time.perf_counter()
            merged = merger.recall_and_merge(u, hist, per_source_k=args.recall_k, total_k=None)
            recall_ms.append((time.perf_counter() - t0) * 1000)
            ids = merged.item_ids()[:cand_k]
            if not ids:
                continue
            for _ in range(args.repeats):
                t1 = time.perf_counter()
                _ = model.score_candidates(hist, ids)
                rank_ms.append((time.perf_counter() - t1) * 1000)
        r = np.asarray(rank_ms)
        c = np.asarray(recall_ms)
        row = {
            "ranker": args.ranker,
            "candidate_k": cand_k,
            "users": len(users),
            "rank_repeats": args.repeats,
            "recall_ms_min": float(c.min()),
            "recall_ms_median": float(np.median(c)),
            "rank_ms_min": float(r.min()) if r.size else None,
            "rank_ms_median": float(np.median(r)) if r.size else None,
            "rank_ms_p95": float(np.percentile(r, 95)) if r.size else None,
            "end_to_end_ms_min": float(c.min() + (r.min() if r.size else 0.0)),
        }
        rows.append(row)
        LOG.info(f"cand_k={cand_k:5d} | recall min {row['recall_ms_min']:6.1f} ms | "
                 f"rank min {row['rank_ms_min']:6.2f} ms (median {row['rank_ms_median']:6.2f}, "
                 f"p95 {row['rank_ms_p95']:6.2f})")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_json({
        "ranker": args.ranker,
        "device": args.device,
        "channels": merger.source_names,
        "recall_k_per_channel": args.recall_k,
        "note": ("minimum is the least-contended run and the closest estimate of true cost; "
                 "median and p95 are reported so machine noise is visible"),
        "rows": rows,
    }, ROOT / "results" / "tables" / "latency_benchmark.json")

    header = (f"{'cand_k':>8s}{'recall min':>12s}{'rank min':>10s}"
              f"{'rank med':>10s}{'rank p95':>10s}{'e2e min':>10s}")
    print(f"\n=== serving latency ({args.ranker}, {args.device}, "
          f"{len(users)} users x {args.repeats} repeats) ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['candidate_k']:>8d}{r['recall_ms_min']:>12.1f}{r['rank_ms_min']:>10.2f}"
              f"{r['rank_ms_median']:>10.2f}{r['rank_ms_p95']:>10.2f}"
              f"{r['end_to_end_ms_min']:>10.1f}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
