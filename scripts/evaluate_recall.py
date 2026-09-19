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
from src.recall.base import RecallStrategy  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.evaluate_recall")


class CachedRecall(RecallStrategy):
    """Serve precomputed candidates for a channel that is cheaper to run in bulk.

    Constrained beam search is roughly an order of magnitude faster when users
    are decoded as a batch, so the generative channel is decoded once up front
    and then presented through the ordinary per-user interface.
    """

    def __init__(self, name: str, cache: dict[int, list], detail: dict | None = None) -> None:
        self.name = name
        self.cache = cache
        self.detail = detail or {}

    def recall(self, user_id, history, top_k):
        return list(self.cache.get(int(user_id), []))[:top_k]

    def is_ready(self):
        return True, f"{len(self.cache)} users precomputed"


def precompute_generative(args, data: ProcessedData, users: np.ndarray):
    """Decode every user's candidates in batches and return a CachedRecall."""
    from src.recall.generative import load_generative_recall

    strat = load_generative_recall(
        ROOT / args.generative_checkpoint,
        ROOT / args.semantic_ids,
        device=args.generative_device,
        beam_width=args.generative_beam,
        max_history_items=args.generative_max_history,
    )
    cache: dict[int, list] = {}
    batch = args.generative_batch
    t0 = time.perf_counter()
    totals, empties = [], 0
    for start in range(0, users.shape[0], batch):
        chunk = users[start : start + batch].tolist()
        histories = [data.test_history(u) for u in chunk]
        decoded, stats = strat.generate_batch(histories)
        empties += stats["empty_decodings"]
        for u, dec in zip(chunk, decoded):
            cands = strat._to_candidates(dec, data.test_history(u), args.per_source_k)
            cache[int(u)] = cands
            totals.append(len(cands))
        if (start // batch + 1) % 20 == 0:
            LOG.info(f"  generative: {start+len(chunk)}/{users.shape[0]} users "
                     f"({time.perf_counter()-t0:.0f}s)")
    elapsed = time.perf_counter() - t0
    detail = {
        "beam_width": args.generative_beam,
        "empty_decodings": int(empties),
        "mean_candidates": float(np.mean(totals)) if totals else 0.0,
        "min_candidates": int(np.min(totals)) if totals else 0,
        "max_candidates": int(np.max(totals)) if totals else 0,
        "users_with_zero_candidates": int(sum(1 for t in totals if t == 0)),
        "decode_time_s": round(elapsed, 1),
        "ms_per_user": round(elapsed / max(users.shape[0], 1) * 1000, 2),
    }
    LOG.info(f"generative channel: {detail}")
    return CachedRecall("generative", cache, detail)


def build_merger(args, data: ProcessedData, users: np.ndarray | None = None) -> CandidateMerger:
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
    detail = None
    if args.generative_checkpoint:
        ck = ROOT / args.generative_checkpoint
        if ck.exists() and users is not None:
            gen = precompute_generative(args, data, users)
            strategies.append(gen)
            detail = gen.detail
        else:
            LOG.warning(f"generative checkpoint missing at {ck}; skipping that channel")
    merger = CandidateMerger(strategies)
    merger.generative_detail = detail
    return merger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--itemcf-index", default="artifacts/itemcf_neighbors.npz")
    ap.add_argument("--content-embeddings", default="artifacts/content_embeddings.npy")
    ap.add_argument("--ks", nargs="*", type=int, default=[100, 200, 500, 1000])
    ap.add_argument("--max-users", type=int, default=None)
    ap.add_argument("--per-source-k", type=int, default=1000)
    ap.add_argument("--generative-checkpoint", default=None,
                    help="path to a trained generative checkpoint; adds the SID channel")
    ap.add_argument("--semantic-ids", default="artifacts/semantic_ids.npz")
    ap.add_argument("--generative-beam", type=int, default=100)
    ap.add_argument("--generative-max-history", type=int, default=20)
    ap.add_argument("--generative-batch", type=int, default=128)
    ap.add_argument("--generative-device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/tables/recall_eval.csv")
    ap.add_argument("--json-out", default="results/tables/recall_eval.json")
    args = ap.parse_args()

    data = ProcessedData.load(ROOT / args.processed_dir)

    # ProcessedData indexes users 0-based; keep that convention here so the
    # history and the target always belong to the same user.
    users = np.arange(data.num_users, dtype=np.int64)
    if args.max_users:
        users = users[: args.max_users]

    merger = build_merger(args, data, users)
    LOG.info(f"channels: {merger.source_names} | {data.summary()}")
    targets = data.test_target[users]

    max_k = max(args.ks)
    per_source_k = max(args.per_source_k, max_k)
    hits = {name: np.zeros((users.shape[0], len(args.ks)), dtype=bool) for name in merger.source_names}
    hits["merged"] = np.zeros((users.shape[0], len(args.ks)), dtype=bool)
    pool_size = np.zeros(users.shape[0], dtype=np.int64)
    # who returned the target, at the widest K, for the overlap analysis
    max_k = max(args.ks)
    found = {name: np.zeros(users.shape[0], dtype=bool) for name in merger.source_names}
    found["merged"] = np.zeros(users.shape[0], dtype=bool)
    cand_counts = {name: np.zeros(users.shape[0], dtype=np.int64) for name in merger.source_names}

    t0 = time.perf_counter()
    for i, u in enumerate(users.tolist()):
        hist = [int(x) for x in data.test_history(u)]
        results = merger.recall(u, hist, per_source_k=per_source_k)
        for name, cands in results.items():
            if name.startswith("_"):
                continue
            # candidates are ranked, so the first K are the channel's top-K
            ids = {c.item_id for c in cands}
            cand_counts[name][i] = len(cands)
            found[name][i] = int(targets[i]) in ids
            for j, k in enumerate(args.ks):
                hits[name][i, j] = int(targets[i]) in {c.item_id for c in cands[:k]}
        merged = merger.merge(results)
        pool_size[i] = len(merged.candidates)
        merged_ids = {c.item_id for c in merged.candidates}
        found["merged"][i] = int(targets[i]) in merged_ids
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
        row["mean_candidates"] = float(cand_counts[name].mean()) if name in cand_counts else float(pool_size.mean())
        rows.append(row)

    # overlap / complementarity at the widest K: which channel is the *only*
    # one that finds the target, and how much of the pool is unique to each
    channel_names = [n for n in merger.source_names if n != "merged"]
    unique_hits, overlaps = {}, {}
    for name in channel_names:
        others = np.zeros(users.shape[0], dtype=bool)
        for other in channel_names:
            if other != name:
                others |= found[other]
        unique_hits[name] = int((found[name] & ~others).sum())
    for i, a in enumerate(channel_names):
        for b in channel_names[i + 1 :]:
            both = int((found[a] & found[b]).sum())
            union = int((found[a] | found[b]).sum())
            overlaps[f"{a}|{b}"] = {
                "both": both,
                "union": union,
                "jaccard": both / union if union else float("nan"),
            }
    total_found = int(found["merged"].sum())

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

    print("\n=== target found at the widest K: overlap and unique contribution ===")
    print(f"{'channel':<12s}{'found':>10s}{'only one':>10s}")
    for name in channel_names:
        n = int(found[name].sum())
        print(f"{name:<12s}{n:>10d}{unique_hits[name]:>10d}")
    for pair, st in overlaps.items():
        print(f"  {pair:<24s} both {st['both']:>7d}  union {st['union']:>7d}  "
              f"Jaccard {st['jaccard']:.3f}")
    print(f"  merged finds {total_found} of {users.shape[0]} targets")

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
        "unique_hits": unique_hits,
        "overlaps": overlaps,
        "merged_targets_found": total_found,
        "generative_detail": getattr(merger, "generative_detail", None),
        "merged_by_bucket": bucket_rows,
        "pool_size": {"mean": float(pool_size.mean()), "min": int(pool_size.min()),
                      "max": int(pool_size.max())},
        "eval_time_s": round(elapsed, 1),
        "note": "candidate generation quality; the user's own history is always excluded",
    }, ROOT / args.json_out)
    print(f"wrote {out} and {ROOT / args.json_out}")


if __name__ == "__main__":
    main()
