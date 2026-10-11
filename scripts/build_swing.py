#!/usr/bin/env python
"""Build exact Swing from training prefixes; never replaces the ItemCF index."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.dataset import ProcessedData
from src.recall.swing import build_swing_index, save_swing_index
from src.utils.io import save_json
from src.utils.provenance import dataset_fingerprint


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--out", default="artifacts/swing_neighbors.npz")
    ap.add_argument("--report", default="artifacts/swing_report.json")
    ap.add_argument("--top-m", type=int, default=100)
    ap.add_argument("--alpha1", type=float, default=5.0)
    ap.add_argument("--alpha2", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--chunk-size", type=int, default=1024)
    args = ap.parse_args()
    data = ProcessedData.load(ROOT / args.processed_dir)
    print(data.summary(), flush=True)
    start = time.perf_counter()
    index = build_swing_index(data.flat_items, data.user_offsets, data.train_len,
                             data.num_items, args.top_m, args.alpha1, args.alpha2,
                             args.beta, args.chunk_size)
    elapsed = time.perf_counter() - start
    save_swing_index(index, ROOT / args.out, data)
    counts = (index["sims"] > 0).sum(1)
    report = {**vars(args), "dataset_hash": dataset_fingerprint(data),
              "num_items": data.num_items, "num_users": data.num_users,
              "build_time_s": elapsed, "algorithm": "swing",
              "contributing_user_pairs": int(index["contributing_user_pairs"]),
              "pairs_with_nonzero_similarity": int(counts.sum()),
              "mean_neighbours_per_item": float(counts[1:].mean()),
              "items_with_no_neighbour": int((counts[1:] == 0).sum()),
              "note": "Exact train-only, binary interactions, unordered distinct user pairs; "
                      "no sampling, no history caps, no item-popularity normalization."}
    save_json(report, ROOT / args.report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
