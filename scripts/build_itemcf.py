#!/usr/bin/env python
"""Build the ItemCF neighbour index from training interactions only.

    python scripts/build_itemcf.py --top-m 100

Writes ``artifacts/itemcf_neighbors.npz`` containing, for every item, its
top-M most similar items and the cosine similarity to each.  Serving is then a
gather plus a weighted sum.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.recall.itemcf import build_itemcf_index, save_itemcf_index  # noqa: E402
from src.utils.io import save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--out", default="artifacts/itemcf_neighbors.npz")
    ap.add_argument("--top-m", type=int, default=100)
    ap.add_argument("--chunk-size", type=int, default=512)
    args = ap.parse_args()

    data = ProcessedData.load(ROOT / args.processed_dir)
    print(f"dataset: {data.summary()}")

    t0 = time.perf_counter()
    index = build_itemcf_index(
        flat_items=data.flat_items,
        user_offsets=data.user_offsets,
        train_len=data.train_len,
        num_items=data.num_items,
        top_m=args.top_m,
        chunk_size=args.chunk_size,
    )
    elapsed = time.perf_counter() - t0

    out = ROOT / args.out
    save_itemcf_index(index, out)

    sims = index["sims"]
    nnz = int((sims > 0).sum())
    per_item = (sims > 0).sum(axis=1)
    report = {
        "processed_dir": args.processed_dir,
        "num_items": data.num_items,
        "top_m": args.top_m,
        "pairs_with_nonzero_similarity": nnz,
        "mean_neighbours_per_item": float(per_item[1:].mean()),
        "items_with_no_neighbour": int((per_item[1:] == 0).sum()),
        "max_similarity": float(sims.max()),
        "build_time_s": round(elapsed, 2),
        "out": str(out.relative_to(ROOT)),
        "note": "built from training interactions only; no val/test leakage",
    }
    save_json(report, ROOT / "artifacts" / "itemcf_report.json")
    print(f"built in {elapsed:.1f}s -> {out}")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
