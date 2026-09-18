#!/usr/bin/env python
"""Pick a user whose recommendation trace is worth showing in the demo.

    python scripts/find_demo_user.py
    python scripts/find_demo_user.py --candidates 400 --out artifacts/demo_user.json

The GIF and the screenshots must show a request where the pipeline is actually
visible: several recall channels contributing, and a candidate whose rank the
multimodal ranker moved substantially relative to the ID-only baseline.  Rather
than hand-picking (and risking a cherry-picked story), this scans a seeded
sample of users, scores each trace on objective criteria, and writes the winner
to ``artifacts/demo_user.json`` so the capture script is reproducible.

Criteria (all measured, none assumed):
  * history length (more history = more to look at)
  * number of distinct recall sources contributing to the final top-K
  * at least one candidate recalled by ``semantic`` (content recall is the point
    of the project) or by more than one channel
  * largest |rank movement| between the ID-only and multimodal ranker inside the
    same candidate pool
"""

from __future__ import annotations

import argparse
import json
import sys
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

LOG = get_logger("mmrec.demo_user")


def build(args, data: ProcessedData):
    strategies = [PopularRecall(data.train_freq, data.num_items)]
    ic = load_itemcf_index(ROOT / args.itemcf_index)
    strategies.append(ItemCFRecall(ic["neighbors"], ic["sims"], data.num_items))
    emb = np.load(ROOT / args.content_embeddings)
    strategies.append(SemanticRecall(emb, data.num_items))
    merger = CandidateMerger(strategies)

    specs = {}
    for name in ("sasrec", "mm_concat"):
        run = discover_run(name)
        if run is None:
            raise SystemExit(f"no finished run for ranker {name!r}")
        specs[name] = RankerSpec(name, run)
    registry = RankerRegistry(data, specs, device=args.device)
    return merger, registry


def score_user(rec, merger, user_id: int, args) -> dict | None:
    """Objective quality of one user's trace; None if it fails the hard filters."""
    try:
        out = rec.inspect(user_id, recall_k=args.recall_k, top_n=args.top_n,
                          ranker="mm_concat", compare="sasrec")
    except Exception:
        return None

    if out["history_length"] < args.min_history:
        return None

    top = out["top_candidates"]
    if not top:
        return None

    sources = {s for e in top for s in e["sources"]}
    has_semantic = "semantic" in sources
    multi_source = sum(1 for e in top if len(e["sources"]) >= 2)

    max_move = 0
    mover = None
    for e in out["moved_up_by_multimodal"]:
        if e["position_delta"] > max_move:
            max_move, mover = e["position_delta"], e
    for e in out["moved_down_by_multimodal"]:
        if e["position_delta"] > max_move:
            max_move, mover = e["position_delta"], e

    if not has_semantic and multi_source == 0:
        return None
    if max_move < args.min_rank_move:
        return None

    # prefer: movement, then multi-source coverage, then a longer history
    quality = max_move * 2.0 + multi_source * 1.0 + min(out["history_length"], 12) * 0.5
    return {
        "user_id": int(user_id),
        "quality": round(float(quality), 3),
        "history_length": int(out["history_length"]),
        "recall_sources_in_top": sorted(sources),
        "multi_source_candidates": int(multi_source),
        "max_rank_move": int(max_move),
        "moved_item": None if mover is None else {
            "item_id": mover["item_id"],
            "position_delta": int(mover["position_delta"]),
            "sources": mover["sources"],
        },
        "recall_summary": {
            k: out["recall_summary"].get(k)
            for k in ("per_source", "before_dedup", "after_dedup", "duplicates_removed")
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--itemcf-index", default="artifacts/itemcf_neighbors.npz")
    ap.add_argument("--content-embeddings", default="artifacts/content_embeddings.npy")
    ap.add_argument("--candidates", type=int, default=300, help="users to scan")
    ap.add_argument("--scan-seed", type=int, default=7)
    ap.add_argument("--min-history", type=int, default=5)
    ap.add_argument("--min-rank-move", type=int, default=5)
    ap.add_argument("--recall-k", type=int, default=200)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="artifacts/demo_user.json")
    args = ap.parse_args()

    configure_threads()
    data = ProcessedData.load(ROOT / args.processed_dir)
    merger, registry = build(args, data)
    from src.pipeline import TwoStageRecommender

    rec = TwoStageRecommender(data, merger, registry, default_ranker="mm_concat")

    rng = np.random.default_rng(args.scan_seed)
    users = rng.choice(data.num_users, size=min(args.candidates, data.num_users), replace=False)
    LOG.info(f"scanning {users.shape[0]} users (seed {args.scan_seed}) ...")

    scored: list[dict] = []
    for i, u in enumerate(users.tolist()):
        s = score_user(rec, merger, u, args)
        if s is not None:
            scored.append(s)
        if (i + 1) % 50 == 0:
            LOG.info(f"  {i+1}/{users.shape[0]} scanned, {len(scored)} passed the filters")

    if not scored:
        raise SystemExit(
            "no user passed the filters; lower --min-rank-move or --min-history"
        )

    scored.sort(key=lambda s: (-s["quality"], s["user_id"]))
    best = scored[0]
    best.update({
        "scan_seed": int(args.scan_seed),
        "candidates_scanned": int(users.shape[0]),
        "users_passing_filters": len(scored),
        "criteria": {
            "min_history": args.min_history,
            "min_rank_move": args.min_rank_move,
            "needs_semantic_or_multi_source": True,
        },
        "ranker": "mm_concat",
        "compare_ranker": "sasrec",
    })
    save_json(best, ROOT / args.out)

    print("\n=== demo user ===")
    print(json.dumps(best, indent=2))
    print(f"\n{len(scored)} of {users.shape[0]} users passed the filters; "
          f"wrote the best to {args.out}")


if __name__ == "__main__":
    main()
