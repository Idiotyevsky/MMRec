#!/usr/bin/env python
"""Run the modality / fusion ablation matrix.

Every variant is produced by overriding a single base config, so the *only*
thing that changes between runs is the variable under study (one clear variable
at a time).

    python scripts/run_ablation.py --list
    python scripts/run_ablation.py --only text_only id_text
    python scripts/run_ablation.py --device cuda --seed 42
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# (tag, base config, extra overrides) -- one variable per row
VARIANTS: list[tuple[str, str, list[str]]] = [
    ("id_only", "configs/mm_sasrec_gated.yaml",
     ["model.modalities={id: true, text: false, image: false, video: false}",
      "model.fusion=concat"]),
    ("text_only", "configs/mm_sasrec_gated.yaml",
     ["model.modalities={id: false, text: true, image: false, video: false}"]),
    ("image_only", "configs/mm_sasrec_gated.yaml",
     ["model.modalities={id: false, text: false, image: true, video: false}"]),
    ("video_only", "configs/mm_sasrec_gated.yaml",
     ["model.modalities={id: false, text: false, image: false, video: true}"]),
    ("id_text", "configs/mm_sasrec_gated.yaml",
     ["model.modalities={id: true, text: true, image: false, video: false}"]),
    ("id_image", "configs/mm_sasrec_gated.yaml",
     ["model.modalities={id: true, text: false, image: true, video: false}"]),
    ("text_image", "configs/mm_sasrec_gated.yaml",
     ["model.modalities={id: false, text: true, image: true, video: false}"]),
    ("id_text_image_concat", "configs/mm_sasrec_concat.yaml", []),
    ("id_text_image_gated", "configs/mm_sasrec_gated.yaml", []),
    ("id_text_image_video_gated", "configs/mm_sasrec_gated.yaml",
     ["model.modalities={id: true, text: true, image: true, video: true}"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="subset of tags to run")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for tag, cfg, ov in VARIANTS:
            print(f"{tag:28s} {cfg}  {' '.join(ov)}")
        return

    todo = [v for v in VARIANTS if args.only is None or v[0] in args.only]
    if not todo:
        raise SystemExit(f"no variant matched {args.only}")

    for tag, cfg, ov in todo:
        cmd = [sys.executable, "scripts/train.py", "--config", cfg, "--tag", tag]
        cmd += list(ov)
        if args.seed is not None:
            cmd.append(f"--seed={args.seed}")
        if args.epochs is not None:
            cmd.append(f"--epochs={args.epochs}")
        if args.device is not None:
            cmd.append(f"--device={args.device}")
        print(">>>", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd, cwd=str(ROOT))
        if rc != 0:
            print(f"!!! {tag} failed with exit code {rc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
