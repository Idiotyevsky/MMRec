#!/usr/bin/env python
"""Build the content embedding table and its Faiss exact inner-product index.

    python scripts/build_semantic_index.py

The content embedding is ``L2([L2(text) ; L2(image)])`` — each modality block is
normalised before concatenation so that the image block (raw norm ≈ 26) cannot
dominate the text block (raw norm = 1).

The index is ``faiss.IndexFlatIP``: **exact** inner-product retrieval, not an
approximate nearest-neighbour index.
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
from src.recall.semantic import build_content_embeddings  # noqa: E402
from src.retrieval.faiss_index import HAS_FAISS, ItemIndex  # noqa: E402
from src.utils.io import save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--feature-dir", default="data/raw")
    ap.add_argument("--modalities", nargs="*", default=["text", "image"])
    ap.add_argument("--out", default="artifacts/content_embeddings.npy")
    ap.add_argument("--index-out", default="artifacts/content.index")
    args = ap.parse_args()

    data = ProcessedData.load(ROOT / args.processed_dir)
    print(f"dataset: {data.summary()}")

    t0 = time.perf_counter()
    emb = build_content_embeddings(
        feature_dir=ROOT / args.feature_dir,
        row_for_item_dir=ROOT / args.processed_dir,
        num_items=data.num_items,
        modalities=tuple(args.modalities),
    )
    build_time = time.perf_counter() - t0

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)

    t0 = time.perf_counter()
    index = ItemIndex(emb, normalize=False).build()  # embeddings are already unit-norm
    index.save(ROOT / args.index_out)
    index_time = time.perf_counter() - t0

    rng = np.random.default_rng(0)
    q = emb[rng.choice(data.num_items, size=min(512, data.num_items), replace=False) + 1]
    t0 = time.perf_counter()
    _ = index.search(q, top_k=20)
    latency = (time.perf_counter() - t0) / q.shape[0] * 1000

    report = {
        "modalities": list(args.modalities),
        "num_items": data.num_items,
        "dim": int(emb.shape[1]),
        "backend": "faiss.IndexFlatIP" if HAS_FAISS else "exact numpy/torch fallback",
        "retrieval_type": "exact inner-product (NOT approximate nearest neighbour)",
        "normalised": True,
        "build_time_s": round(build_time, 2),
        "index_time_s": round(index_time, 2),
        "query_latency_ms": round(latency, 3),
        "embeddings": str(out.relative_to(ROOT)),
        "index": str((ROOT / args.index_out).relative_to(ROOT)),
        "note": "text and image blocks are L2-normalised before concatenation",
    }
    save_json(report, ROOT / "artifacts" / "semantic_index_report.json")
    print(f"wrote {out} {emb.shape}")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
