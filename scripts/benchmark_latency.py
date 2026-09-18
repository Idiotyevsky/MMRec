#!/usr/bin/env python
"""Serving latency benchmark, split by stage.

    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --candidate-sizes 100 500 1000

The ranking stage is not one number.  It is

    history ─► sequence encoder ─► user vector          (encode)
    user vector × candidate embeddings                  (score)

and only the second part depends on the candidate budget.  This script times
them separately, plus the recall stage and the seen-item masking pass, so the
table says which part actually costs anything.

The **full-catalogue** row scores ``np.arange(1, num_items + 1)`` — every real
item.  An earlier version of this script reused the recall union for that row,
which meant the "full catalogue" measurement never scored the full catalogue.

Reading the numbers: this is a **CPU demo benchmark on the machine the script
happens to run on**.  It is intended for *relative* comparison between stages
and candidate budgets, not as a production SLA.  The median is the headline
statistic; min and p95 are reported so the spread is visible.
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
from src.serving.app import configure_threads  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.latency")
FULL = "full_catalogue"


def build_merger(data: ProcessedData, args) -> CandidateMerger:
    strategies = [PopularRecall(data.train_freq, data.num_items)]
    ic_path = ROOT / args.itemcf_index
    if ic_path.exists():
        ic = load_itemcf_index(ic_path)
        strategies.append(ItemCFRecall(ic["neighbors"], ic["sims"], data.num_items))
    else:
        LOG.warning(f"ItemCF index missing at {ic_path}; channel disabled")
    emb_path = ROOT / args.content_embeddings
    if emb_path.exists():
        strategies.append(SemanticRecall(np.load(emb_path), data.num_items))
    else:
        LOG.warning(f"content embeddings missing at {emb_path}; channel disabled")
    return CandidateMerger(strategies)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--itemcf-index", default="artifacts/itemcf_neighbors.npz")
    ap.add_argument("--content-embeddings", default="artifacts/content_embeddings.npy")
    ap.add_argument("--ranker", default="mm_concat")
    ap.add_argument("--candidate-sizes", nargs="*", type=int, default=[100, 200, 500, 1000, 2000])
    ap.add_argument("--recall-k", type=int, default=500, help="per-channel budget")
    ap.add_argument("--users", type=int, default=150)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/tables/latency_benchmark.csv")
    args = ap.parse_args()

    configure_threads()  # measure the serving configuration, not the torch default
    data = ProcessedData.load(ROOT / args.processed_dir)
    merger = build_merger(data, args)

    run = discover_run(args.ranker)
    if run is None:
        raise SystemExit(f"no finished run for ranker {args.ranker!r}")
    registry = RankerRegistry(data, {args.ranker: RankerSpec(args.ranker, run)}, device=args.device)
    model = registry.get(args.ranker)
    full_ids = model.valid_item_ids()
    LOG.info(f"ranker={args.ranker} | channels={merger.source_names} | "
             f"full catalogue = {full_ids.shape[0]} items | {data.summary()}")

    users = np.arange(min(args.users, data.num_users), dtype=np.int64)
    rows = []
    for cand_k in args.candidate_sizes:
        recall_ms: list[float] = []
        encode_ms: list[float] = []
        score_ms: list[float] = []
        mask_ms: list[float] = []
        for u in users.tolist():
            hist = [int(x) for x in data.test_history(u)]

            t0 = time.perf_counter()
            merged = merger.recall_and_merge(u, hist, per_source_k=args.recall_k, total_k=None)
            recall_ms.append((time.perf_counter() - t0) * 1000)
            ids = merged.item_ids()[:cand_k]
            if not len(ids):
                continue

            # encode once per user; score `repeats` times
            t1 = time.perf_counter()
            vec = model.encode_user(hist)
            encode_ms.append((time.perf_counter() - t1) * 1000)
            for _ in range(args.repeats):
                t2 = time.perf_counter()
                scores = model.score_with_user_vector(vec, ids)
                score_ms.append((time.perf_counter() - t2) * 1000)
            t3 = time.perf_counter()
            _ = model.seen_mask_overhead(hist, scores)
            mask_ms.append((time.perf_counter() - t3) * 1000)

        def stat(x, fn=np.median):
            return float(fn(np.asarray(x))) if x else None

        row = {
            "ranker": args.ranker,
            "candidate_k": cand_k,
            "scope": "recall_pool",
            "users": len(users),
            "repeats": args.repeats,
            "recall_ms_median": stat(recall_ms),
            "encode_ms_median": stat(encode_ms),
            "score_ms_median": stat(score_ms),
            "score_ms_min": stat(score_ms, np.min),
            "score_ms_p95": stat(score_ms, lambda a: np.percentile(a, 95)),
            "mask_ms_median": stat(mask_ms),
        }
        row["ranking_ms_median"] = (
            None if row["encode_ms_median"] is None or row["score_ms_median"] is None
            else row["encode_ms_median"] + row["score_ms_median"] + (row["mask_ms_median"] or 0.0)
        )
        row["end_to_end_ms_median"] = (
            None if row["recall_ms_median"] is None or row["ranking_ms_median"] is None
            else row["recall_ms_median"] + row["ranking_ms_median"]
        )
        rows.append(row)
        LOG.info(f"cand_k={cand_k:5d} | recall {row['recall_ms_median']:6.1f} | "
                 f"encode {row['encode_ms_median']:5.2f} | score {row['score_ms_median']:5.2f} | "
                 f"mask {row['mask_ms_median']:5.3f} | e2e {row['end_to_end_ms_median']:6.1f}")

    # ---- true full-catalogue scoring -----------------------------------
    # Every real item, not the recall union.  Scoring and masking are timed
    # separately so "full catalogue" is unambiguous.
    recall_ms, encode_ms, score_ms, mask_ms = [], [], [], []
    for u in users.tolist():
        hist = [int(x) for x in data.test_history(u)]
        t0 = time.perf_counter()
        _ = merger.recall_and_merge(u, hist, per_source_k=args.recall_k, total_k=None)
        recall_ms.append((time.perf_counter() - t0) * 1000)
        t1 = time.perf_counter()
        vec = model.encode_user(hist)
        encode_ms.append((time.perf_counter() - t1) * 1000)
        for _ in range(args.repeats):
            t2 = time.perf_counter()
            scores = model.score_with_user_vector(vec, full_ids)
            score_ms.append((time.perf_counter() - t2) * 1000)
        t3 = time.perf_counter()
        _ = model.seen_mask_overhead(hist, scores)
        mask_ms.append((time.perf_counter() - t3) * 1000)

    def stat(x, fn=np.median):
        return float(fn(np.asarray(x))) if x else None

    full_row = {
        "ranker": args.ranker,
        "candidate_k": int(full_ids.shape[0]),
        "scope": "full_catalogue",
        "users": len(users),
        "repeats": args.repeats,
        "recall_ms_median": stat(recall_ms),
        "encode_ms_median": stat(encode_ms),
        "score_ms_median": stat(score_ms),
        "score_ms_min": stat(score_ms, np.min),
        "score_ms_p95": stat(score_ms, lambda a: np.percentile(a, 95)),
        "mask_ms_median": stat(mask_ms),
    }
    full_row["ranking_ms_median"] = (
        full_row["encode_ms_median"] + full_row["score_ms_median"] + full_row["mask_ms_median"]
    )
    full_row["end_to_end_ms_median"] = full_row["recall_ms_median"] + full_row["ranking_ms_median"]
    rows.append(full_row)
    LOG.info(f"cand_k={full_row['candidate_k']} (FULL CATALOGUE) | "
             f"encode {full_row['encode_ms_median']:.2f} | "
             f"score {full_row['score_ms_median']:.2f} | "
             f"mask {full_row['mask_ms_median']:.3f}")

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
        "num_items": int(data.num_items),
        "full_catalogue_item_count": int(full_ids.shape[0]),
        "users": int(len(users)),
        "repeats": args.repeats,
        "torch_threads": int(__import__("torch").get_num_threads()),
        "headline_statistic": "median",
        "caveat": ("CPU demo benchmark on the current machine; intended for relative "
                   "comparison between stages and candidate budgets, not a production SLA"),
        "rows": rows,
    }, ROOT / "results" / "tables" / "latency_benchmark.json")

    header = (f"{'cand_k':>15s}{'recall':>9s}{'encode':>9s}{'score':>9s}{'mask':>8s}"
              f"{'ranking':>10s}{'e2e':>9s}")
    print(f"\n=== serving latency, ms ({args.ranker}, {args.device}, "
          f"{len(users)} users x {args.repeats} repeats, median) ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        label = "full catalogue" if r["scope"] == "full_catalogue" else str(r["candidate_k"])
        print(f"{label:>15s}{r['recall_ms_median']:>9.1f}{r['encode_ms_median']:>9.2f}"
              f"{r['score_ms_median']:>9.2f}{r['mask_ms_median']:>8.3f}"
              f"{r['ranking_ms_median']:>10.2f}{r['end_to_end_ms_median']:>9.1f}")
    print("\nnote: CPU demo benchmark on this machine, for relative comparison only.")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
