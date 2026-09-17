#!/usr/bin/env python
"""Offline evaluation of the recall layer.

    python scripts/evaluate_recall.py
    python scripts/evaluate_recall.py --max-users 20000 --ks 100 200 500

Measures whether each recall channel (and the merged pool) contains the user's
next item, with the user's own history excluded.  This is *candidate generation*
quality, not ranking quality — the number that matters for the pipeline is how
much headroom the pool leaves for the ranker.

Reported per channel:

    Recall@100 / Recall@200 / Recall@500 / Recall@1000

and, for the merged pool, the same broken down by popularity bucket so it is
visible whether any channel is the only one covering the long tail.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.data.popularity import BUCKET_NAMES  # noqa: E402
from src.recall import (  # noqa: E402
    CandidateMerger,
    ItemCFRecall,
    PopularRecall,
    SemanticRecall,
    load_itemcf_index,
)
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.evaluate_recall")


def build_merger(args, data: ProcessedData) -> CandidateMerger:
    strategies = [PopularRecall(data.train_freq, data.num_items)]
    ic_path = ROOT / args.itemcf_index
    if ic_path.exists():
        ic = load_itemcf_index(ic_path)
        strategies.append(ItemCFRecall(ic["neighbors"], ic["sims"], data.num_items))
    else:
        LOG.warning(f"ItemCF index missing at {ic_path}; skipping that channel")
    emb_path = ROOT / args.content_embeddings
    if emb_path.exists():
        strategies.append(SemanticRecall(np.load(emb_path), data.num_items))
    else:
        LOG.warning(f"content embeddings missing at {emb_path}; skipping that channel")
    return CandidateMerger(strategies)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--itemcf-index", default="artifacts/itemcf_neighbors.npz")
    ap.add_argument("--content-embeddings", default="artifacts/content_embeddings.npy")
    ap.add_argument("--ks", nargs="*", type=int, default=[100, 200, 500, 1000])
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--per-source-k", type=int, default=1000)
    ap.add_argument("--out", default="results/tables/recall_eval.csv")
    ap.add_argument("--json-out", default="results/tables/recall_eval.json")
    args = ap.parse_args()

    data = ProcessedData.load(ROOT / args.processed_dir)
    merger = build_merger(args, data)
    LOG.info(f"channels: {merger.source_names} | {data.summary()}")

    # ProcessedData indexes users 0-based; keep that convention here so the
    # history and the target always belong to the same user.
    users = np.arange(data.num_users, dtype=np.int64)
    if args.max_users:
        users = users[: args.max_users]
    targets = data.test_target[users]

    max_k = max(args.ks)
    per_source_k = max(args.per_source_k, max_k)
    hits = {name: np.zeros((users.shape[0], len(args.ks)), dtype=bool) for name in merger.source_names}
    hits["merged"] = np.zeros((users.shape[0], len(args.ks)), dtype=bool)
    pool_size = np.zeros(users.shape[0], dtype=np.int64)

    t0 = time.perf_counter()
    for i, u in enumerate(users.tolist()):
        hist = [int(x) for x in data.test_history(u)]
        results = merger.recall(u, hist, per_source_k=per_source_k)
        for name, cands in results.items():
            if name.startswith("_"):
                continue
            # candidates are ranked, so the first K are the channel's top-K
            for j, k in enumerate(args.ks):
                hits[name][i, j] = int(targets[i]) in {c.item_id for c in cands[:k]}
        merged = merger.merge(results)
        pool_size[i] = len(merged.candidates)
        for j, k in enumerate(args.ks):
            hits["merged"][i, j] = int(targets[i]) in {c.item_id for c in merged.candidates[:k]}
        if (i + 1) % 10000 == 0:
            LOG.info(f"  {i+1}/{users.shape[0]} users ({time.perf_counter()-t0:.0f}s)")
    elapsed = time.perf_counter() - t0

    rows = []
    for name, mat in hits.items():
        row = {"channel": name, "num_users": int(users.shape[0])}
        for j, k in enumerate(args.ks):
            row[f"Recall@{k}"] = float(mat[:, j].mean())
        rows.append(row)

    # merged pool broken down by popularity bucket
    bucket_rows = []
    for bid, bname in BUCKET_NAMES.items():
        mask = data.popularity_bucket[targets] == bid
        if not mask.any():
            continue
        row = {"bucket": bname, "num_users": int(mask.sum())}
        for j, k in enumerate(args.ks):
            row[f"Recall@{k}"] = float(hits["merged"][mask, j].mean())
        bucket_rows.append(row)

    import csv

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== recall@K by channel (test target, history masked) ===")
    header = f"{'channel':<12s}" + "".join(f"{'R@'+str(k):>10s}" for k in args.ks)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['channel']:<12s}" + "".join(f"{row[f'Recall@{k}']:>10.4f}" for k in args.ks))

    print("\n=== merged pool by popularity bucket ===")
    print(header.replace("channel", "bucket  "))
    for row in bucket_rows:
        print(f"{row['bucket']:<12s}" + "".join(f"{row[f'Recall@{k}']:>10.4f}" for k in args.ks))

    print(f"\nmerged pool size: mean {pool_size.mean():.1f} "
          f"min {pool_size.min()} max {pool_size.max()}")
    print(f"evaluated {users.shape[0]} users in {elapsed:.1f}s "
          f"({elapsed/max(users.shape[0],1)*1000:.1f} ms/user, all channels)")

    save_json({
        "processed_dir": args.processed_dir,
        "num_users": int(users.shape[0]),
        "ks": args.ks,
        "per_source_k": per_source_k,
        "channels": rows,
        "merged_by_bucket": bucket_rows,
        "pool_size": {"mean": float(pool_size.mean()), "min": int(pool_size.min()),
                      "max": int(pool_size.max())},
        "eval_time_s": round(elapsed, 1),
        "note": "candidate generation quality; the user's own history is always excluded",
    }, ROOT / args.json_out)
    print(f"wrote {out} and {ROOT / args.json_out}")


if __name__ == "__main__":
    main()
