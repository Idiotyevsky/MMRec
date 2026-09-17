#!/usr/bin/env python
"""Export per-item fusion gate weights from a trained MM-SASRec checkpoint.

    python scripts/export_gates.py --run-dir results/runs/mm_gated_xxx

Writes ``gate_weights.npz`` next to the checkpoint with, for every item:

    item_id        internal item id (1..N)
    raw_item_id    original MicroLens item id
    gate_<mod>     fusion weight for that modality (0 for missing/unavailable)
    train_freq     training-only interaction count
    bucket         popularity bucket (-1 PAD, 0 tail, 1 middle, 2 head)
    is_cold        whether the item belongs to the simulated cold set

The gate weights are read from the model, never recomputed by hand, so the
analysis cannot drift away from the model that produced the metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.training.factory import build_model  # noqa: E402
from src.utils.config import Config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = Config(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
    data = ProcessedData.load(ROOT / cfg.data.processed_dir)

    device = torch.device(args.device)
    model = build_model(cfg, data, device=device)
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if not hasattr(model, "gate_weights_for_all_items"):
        raise SystemExit(f"{cfg.model.name} has no fusion gates; nothing to export")

    gates = model.gate_weights_for_all_items()
    payload = {
        "item_id": np.arange(data.num_items + 1, dtype=np.int64),
        "raw_item_id": np.concatenate([[0], data.raw_item_ids]).astype(np.int64),
        "train_freq": data.train_freq.astype(np.int64),
        "bucket": data.popularity_bucket.astype(np.int8),
        "is_cold": data.is_cold.astype(np.int8),
    }
    for m, w in gates.items():
        payload[f"gate_{m}"] = w.numpy().astype(np.float32)

    out = run_dir / "gate_weights.npz"
    np.savez_compressed(out, **payload)
    print(f"wrote {out}")
    print(f"  modalities: {list(gates.keys())}")
    print(f"  items: {data.num_items}, cold: {int(data.is_cold.sum())}")


if __name__ == "__main__":
    main()
