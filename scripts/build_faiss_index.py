#!/usr/bin/env python
"""Build the offline ANN index from an exported embedding table.

    python scripts/export_embeddings.py --run-dir results/runs/<run_id>
    python scripts/build_faiss_index.py --run-dir results/runs/<run_id>

Writes ``artifacts/item.index`` (+ ``.meta.json``) and reports index build time
and query latency.  Falls back to an exact numpy/torch index when faiss is not
installed; the fallback is exact, only slower.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval.faiss_index import HAS_FAISS, ItemIndex, load_index  # noqa: E402
from src.utils.io import save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="artifacts/item.index")
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--latency-queries", type=int, default=1000)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    emb_path = run_dir / "item_embeddings.npy"
    if not emb_path.exists():
        raise SystemExit(f"{emb_path} not found -- run scripts/export_embeddings.py first")
    emb = np.load(emb_path)

    t0 = time.perf_counter()
    index = ItemIndex(emb, normalize=not args.no_normalize).build()
    build_time = time.perf_counter() - t0
    index.save(ROOT / args.out)

    # ---- latency --------------------------------------------------------
    rng = np.random.default_rng(0)
    q = rng.normal(size=(args.latency_queries, emb.shape[1])).astype(np.float32)
    _ = index.search(q[:8], top_k=args.top_k)  # warm-up
    t0 = time.perf_counter()
    _ = index.search(q, top_k=args.top_k)
    latency_ms = (time.perf_counter() - t0) / args.latency_queries * 1000

    report = {
        "backend": "faiss.IndexFlatIP" if HAS_FAISS else "exact (numpy/torch fallback)",
        "num_items": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "normalized": not args.no_normalize,
        "index_build_time_s": round(build_time, 4),
        "query_latency_ms_per_query": round(latency_ms, 4),
        "queries_per_second": round(1000.0 / max(latency_ms, 1e-9), 1),
        "index_path": str(args.out),
        "run_dir": str(run_dir),
    }
    save_json(report, run_dir / "retrieval_benchmark.json")
    print(f"index: {report['backend']} | {report['num_items']} items x {report['dim']} dims")
    print(f"build {report['index_build_time_s']}s | "
          f"{report['query_latency_ms_per_query']} ms/query "
          f"({report['queries_per_second']} q/s, batch of {args.latency_queries})")

    # sanity: the index must be usable through the public loader
    _ = load_index(ROOT / args.out)
    print("reload check: ok")


if __name__ == "__main__":
    main()
