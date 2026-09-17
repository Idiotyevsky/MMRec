#!/usr/bin/env python
"""Run the simulated cold-item experiment end to end.

    # 1. build the cold split (once)
    python -m src.data.preprocess --out data/processed/cold10 \
        --cold-ratio 0.1 --cold-seed 42

    # 2. train + evaluate the ID-only and multimodal models on it
    python scripts/run_cold_start.py --gpus 0

The protocol is documented in docs/evaluation_protocol.md.  This script only
orchestrates existing commands; it does not implement a second, divergent
training path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.cold")

JOBS = [
    ("cold_sasrec", "configs/sasrec_cold10.yaml", []),
    ("cold_mm_gated", "configs/mm_sasrec_gated_cold10.yaml", []),
    ("cold_mm_gated_iddrop", "configs/mm_sasrec_gated_iddrop_cold10.yaml", []),
    ("cold_mm_concat", "configs/mm_sasrec_gated_cold10.yaml", ["model.fusion=concat"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="*", type=int, default=[0])
    ap.add_argument("--cold-dir", default="data/processed/cold10")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cold_dir = ROOT / args.cold_dir
    if not cold_dir.exists():
        raise SystemExit(
            f"{cold_dir} not found. Build it first:\n"
            "  python -m src.data.preprocess --out data/processed/cold10 --cold-ratio 0.1 --cold-seed 42"
        )
    data = ProcessedData.load(cold_dir)
    LOG.info(f"cold split: {data.summary()}")
    cold_targets = int(data.is_cold[np.concatenate([data.val_target, data.test_target])].sum())
    LOG.info(f"cold val+test targets: {cold_targets}")

    queues = {g: [] for g in args.gpus}
    for i, (tag, cfg, ov) in enumerate(JOBS):
        g = args.gpus[i % len(args.gpus)]
        queues[g].append((tag, cfg, ov))

    for g, jobs in queues.items():
        qfile = ROOT / f"results/queue_{Path(args.cold_dir).name}_gpu{g}.txt"
        qfile.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for tag, cfg, ov in jobs:
            extra = list(ov)
            if args.seed is not None:
                extra.append(f"seed={args.seed}")
            if args.epochs is not None:
                extra.append(f"epochs={args.epochs}")
            lines.append(f"{tag}|{cfg}|{'@'.join(extra)}")
        qfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd = ["bash", "scripts/run_suite.sh", str(g), str(qfile.relative_to(ROOT))]
        LOG.info(" ".join(cmd))
        if not args.dry_run:
            subprocess.Popen(cmd, cwd=str(ROOT))

    LOG.info("cold-start jobs launched; aggregate with: python analysis/aggregate_results.py")


if __name__ == "__main__":
    main()
