#!/usr/bin/env python
"""Latency and candidate yield of the Semantic-ID generative channel.

    python scripts/benchmark_generative.py
    python scripts/benchmark_generative.py --beams 1 5 20 100 --users 200

Autoregressive decoding has a different cost structure from index lookup: it
trades latency for candidates through the beam width, and it cannot return more
candidates than the beam can produce.  This benchmark measures both sides of that
trade so the channel can be placed honestly against the embedding channels:

* ``ms_per_user`` - decode latency at a given beam width (batched, as it would
  run in the service);
* ``mean_candidates`` - what that beam width actually yields after collisions
  are expanded and seen items are removed;
* ``empty_decodings`` - how often the constrained beam dies, which would be a
  silent quality loss if it were not counted.

Timings are medians over repeats, after a warm-up pass, and the process is
single-threaded on CPU by default because that is how serving runs.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.recall.generative import load_generative_recall  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.benchmark_generative")


def measure(strategy, histories, batch_size, repeats):
    """Return (median ms/user, candidates per user, empty decodings)."""
    timings, counts, empties = [], [], 0
    for _ in range(repeats):
        t0 = time.perf_counter()
        produced = 0
        for start in range(0, len(histories), batch_size):
            chunk = histories[start : start + batch_size]
            decoded, stats = strategy.generate_batch(chunk)
            empties += stats["empty_decodings"]
            produced += sum(
                len(strategy._to_candidates(d, histories[start + i], strategy.beam_width))
                for i, d in enumerate(decoded)
            )
        elapsed = time.perf_counter() - t0
        timings.append(elapsed / max(len(histories), 1) * 1000)
        counts.append(produced / max(len(histories), 1))
    return statistics.median(timings), float(np.mean(counts)), empties


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--checkpoint", default="results/runs/genrec_sid/best.pt")
    ap.add_argument("--semantic-ids", default="artifacts/semantic_ids.npz")
    ap.add_argument("--beams", nargs="*", type=int, default=[1, 5, 10, 20, 50, 100])
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-history", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/tables/generative_latency.csv")
    ap.add_argument("--json-out", default="results/tables/generative_latency.json")
    args = ap.parse_args()

    if args.device == "cpu":
        import torch

        torch.set_num_threads(1)

    data = ProcessedData.load(ROOT / args.processed_dir)
    users = np.arange(min(args.users, data.num_users), dtype=np.int64)
    histories = [data.test_history(u) for u in users.tolist()]
    LOG.info(f"benchmarking {len(histories)} users, beams {args.beams}, "
             f"batch {args.batch_size}, device {args.device}")

    rows = []
    for beam in args.beams:
        strat = load_generative_recall(
            ROOT / args.checkpoint, ROOT / args.semantic_ids, device=args.device,
            beam_width=beam, max_history_items=args.max_history,
        )
        # warm-up: first call builds the trie cache and allocates
        strat.generate_batch(histories[: min(8, len(histories))])
        ms, cands, empties = measure(strat, histories, args.batch_size, args.repeats)
        row = {
            "beam_width": beam,
            "users": len(histories),
            "batch_size": args.batch_size,
            "device": args.device,
            "ms_per_user": round(ms, 3),
            "users_per_second": round(1000.0 / ms, 1) if ms > 0 else float("nan"),
            "mean_candidates": round(cands, 2),
            "empty_decodings": empties,
            "empty_rate": round(empties / (len(histories) * args.repeats), 6),
        }
        rows.append(row)
        LOG.info(f"beam {beam:4d} | {row['ms_per_user']:8.2f} ms/user | "
                 f"{row['users_per_second']:8.1f} users/s | "
                 f"{row['mean_candidates']:6.2f} candidates | empty {empties}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== generative decode latency vs beam width ===")
    print(f"{'beam':>5s}{'ms/user':>12s}{'users/s':>10s}{'candidates':>12s}{'empty':>8s}")
    for r in rows:
        print(f"{r['beam_width']:>5d}{r['ms_per_user']:>12.2f}{r['users_per_second']:>10.1f}"
              f"{r['mean_candidates']:>12.2f}{r['empty_decodings']:>8d}")

    save_json({
        "checkpoint": args.checkpoint,
        "semantic_ids": args.semantic_ids,
        "max_history": args.max_history,
        "repeats": args.repeats,
        "rows": rows,
        "note": ("single-threaded CPU unless --device cuda; median over repeats "
                 "after a warm-up pass; candidates are counted after collision "
                 "expansion and seen-item masking"),
    }, ROOT / args.json_out)
    print(f"\nwrote {out} and {ROOT / args.json_out}")


if __name__ == "__main__":
    main()
