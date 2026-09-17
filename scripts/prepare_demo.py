#!/usr/bin/env python
"""Prepare everything the serving demo needs, without retraining.

    python scripts/prepare_demo.py

Checks / builds, in order:

1. processed dataset            (``data/processed/base``)
2. ranker checkpoints           (discovers the newest run per tag; reports what is missing)
3. ItemCF neighbour index       (``artifacts/itemcf_neighbors.npz``)
4. content embeddings + index   (``artifacts/content_embeddings.npy``, ``content.index``)
5. writes ``artifacts/demo_manifest.json`` describing what is ready

Large artifacts stay out of git; this script is how a fresh clone produces them.
Run ``python scripts/train.py --config configs/mm_sasrec_concat.yaml`` first if
the ranker checkpoints are missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.rankers import discover_run  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.prepare_demo")

#: ranker name -> (config tag, human description)
RANKER_TAGS = {
    "sasrec": ("sasrec", "SASRec — ID-only sequential baseline"),
    "mm_concat": ("mm_concat", "MM-SASRec — ID + text + image, concatenation fusion"),
    "mm_gated": ("mm_gated", "MM-SASRec — ID + text + image, gated fusion"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--feature-dir", default="data/raw")
    ap.add_argument("--skip-itemcf", action="store_true")
    ap.add_argument("--skip-semantic", action="store_true")
    ap.add_argument("--top-m", type=int, default=100)
    args = ap.parse_args()

    manifest: dict = {"root": str(ROOT), "steps": {}, "ready": True}
    missing: list[str] = []

    # ---- 1. processed dataset ----
    pdir = ROOT / args.processed_dir
    ok = (pdir / "dataset.npz").exists() and (pdir / "mappings.json").exists()
    manifest["steps"]["dataset"] = {"ok": ok, "path": str(pdir)}
    if not ok:
        missing.append(
            "processed dataset: run "
            "`python -m src.data.preprocess --out data/processed/base`"
        )
    else:
        from src.data.dataset import ProcessedData

        data = ProcessedData.load(pdir)
        LOG.info(f"dataset ok: {data.summary()}")
        manifest["dataset_stats"] = data.stats

    # ---- 2. ranker checkpoints ----
    rankers: dict[str, dict] = {}
    for name, (tag, desc) in RANKER_TAGS.items():
        run = discover_run(tag)
        rankers[name] = {
            "tag": tag, "description": desc,
            "run_dir": str(run.relative_to(ROOT)) if run else None,
            "available": run is not None,
        }
        if run is None:
            cfg = "sasrec.yaml" if tag == "sasrec" else f"{tag}.yaml"
            missing.append(
                f"ranker '{name}': no finished run for tag '{tag}'. "
                f"Train it with `python scripts/train.py --config configs/{cfg}`"
            )
        else:
            LOG.info(f"ranker {name} -> {run.name}")
    available = [n for n, r in rankers.items() if r["available"]]
    manifest["steps"]["rankers"] = {
        "ok": bool(available),
        "available": available,
        "detail": rankers,
    }
    if not available:
        manifest["ready"] = False

    # ---- 3. ItemCF index ----
    itemcf_path = ROOT / "artifacts" / "itemcf_neighbors.npz"
    if args.skip_itemcf:
        manifest["steps"]["itemcf"] = {"ok": itemcf_path.exists(), "skipped": True}
    elif itemcf_path.exists():
        z = np.load(itemcf_path)
        manifest["steps"]["itemcf"] = {"ok": True, "path": str(itemcf_path.relative_to(ROOT)),
                                       "shape": list(z["neighbors"].shape)}
        LOG.info(f"itemcf index present: {z['neighbors'].shape}")
    elif ok:
        import subprocess

        LOG.info("building ItemCF index ...")
        rc = subprocess.call([sys.executable, "scripts/build_itemcf.py", "--top-m", str(args.top_m)],
                             cwd=str(ROOT))
        manifest["steps"]["itemcf"] = {"ok": rc == 0, "built_now": True}
        if rc != 0:
            missing.append("ItemCF index build failed")
    else:
        missing.append("ItemCF index missing (needs the processed dataset)")

    # ---- 4. content embeddings + index ----
    emb_path = ROOT / "artifacts" / "content_embeddings.npy"
    if args.skip_semantic:
        manifest["steps"]["semantic"] = {"ok": emb_path.exists(), "skipped": True}
    elif emb_path.exists():
        emb = np.load(emb_path, mmap_mode="r")
        manifest["steps"]["semantic"] = {"ok": True,
                                         "path": str(emb_path.relative_to(ROOT)),
                                         "shape": list(emb.shape)}
        LOG.info(f"content embeddings present: {emb.shape}")
    elif ok:
        import subprocess

        LOG.info("building content embeddings + index ...")
        rc = subprocess.call([sys.executable, "scripts/build_semantic_index.py"], cwd=str(ROOT))
        manifest["steps"]["semantic"] = {"ok": rc == 0, "built_now": True}
        if rc != 0:
            missing.append("content embedding build failed")
    else:
        missing.append("content embeddings missing (needs the processed dataset)")

    # ---- 5. manifest ----
    manifest["missing"] = missing
    manifest["ready"] = manifest["ready"] and not missing
    out = ROOT / "artifacts" / "demo_manifest.json"
    save_json(manifest, out)

    print("\n=== demo readiness ===")
    for step, info in manifest["steps"].items():
        extra = ""
        if step == "rankers" and info.get("available"):
            extra = f"  [{', '.join(info['available'])}]"
        if info.get("built_now"):
            extra += "  (built now)"
        print(f"  {step:10s} {'OK' if info.get('ok') else 'MISSING'}{extra}")
    if missing:
        print("\nblocking items:")
        for m in missing:
            print(f"  - {m}")
        print(f"\nwrote {out}")
        raise SystemExit(1)
    print(f"\nall demo prerequisites ready -> {out}")


if __name__ == "__main__":
    main()
