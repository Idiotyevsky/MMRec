#!/usr/bin/env python
"""CLI demo of the full serving path: history -> user vector -> ANN -> top-K.

    python scripts/recommend.py --run-dir results/runs/<run_id> --history 12,45,91,102
    python scripts/recommend.py --run-dir results/runs/<run_id> --user-id 7 --top-k 20
    python scripts/recommend.py --run-dir results/runs/<run_id> --demo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serving.recommender import Recommender  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--history", default=None, help="comma separated RAW item ids, oldest first")
    ap.add_argument("--user-id", type=int, default=None, help="use this user's train history")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--demo", action="store_true", help="run a latency demo over 200 users")
    args = ap.parse_args()

    t0 = time.perf_counter()
    rec = Recommender.from_run(args.run_dir, device=args.device)
    print(f"loaded in {time.perf_counter() - t0:.2f}s: {rec.describe()}")
    rec.warm_start()

    if args.demo:
        import numpy as np

        rng = np.random.default_rng(0)
        users = rng.choice(rec.data.num_users, size=200, replace=False)
        lat = []
        for u in users:
            hist = [rec.data.raw_item(int(i)) for i in rec.data.train_items(int(u))]
            t = time.perf_counter()
            rec.recommend(hist, top_k=args.top_k)
            lat.append(time.perf_counter() - t)
        lat = np.asarray(lat) * 1000
        print(f"latency over {lat.size} requests: mean {lat.mean():.2f} ms | "
              f"p50 {np.percentile(lat, 50):.2f} | p95 {np.percentile(lat, 95):.2f} | "
              f"max {lat.max():.2f}")
        return

    if args.user_id is not None:
        u = args.user_id
        hist_internal = rec.data.test_history(u)
        hist_raw = [rec.data.raw_item(int(i)) for i in hist_internal]
        target = rec.data.raw_item(int(rec.data.test_target[u]))
        print(f"user {u} history (raw ids): {hist_raw}")
        print(f"ground-truth next item     : {target}")
    elif args.history:
        hist_raw = [int(x) for x in args.history.split(",") if x.strip()]
        print(f"history (raw ids): {hist_raw}")
    else:
        u = 0
        hist_raw = [rec.data.raw_item(int(i)) for i in rec.data.test_history(u)]
        print(f"(no history given) using user 0: {hist_raw}")

    t0 = time.perf_counter()
    recs = rec.recommend(hist_raw, top_k=args.top_k)
    dt = (time.perf_counter() - t0) * 1000
    print(f"\ntop-{args.top_k} in {dt:.2f} ms")
    for rank, r in enumerate(recs, 1):
        print(f"  {rank:2d}. item {r['item_id']:6d}   score {r.get('score', float('nan')):.4f}")


if __name__ == "__main__":
    main()
