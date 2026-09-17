#!/usr/bin/env python
"""Export the learned item embedding table from a trained run.

    python scripts/export_embeddings.py --run-dir results/runs/<run_id>

Writes ``item_embeddings.npy`` (``(num_items+1, H)``, row 0 = PAD) and
``item_embeddings_meta.json`` into the run directory.  These are the vectors the
Faiss index and the serving layer use; they are read straight from the model, so
retrieval and evaluation cannot disagree.
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
from src.utils.io import save_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = Config(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
    data = ProcessedData.load(ROOT / cfg.data.processed_dir)

    device = torch.device(args.device)
    model = build_model(cfg, data, device=device)
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    with torch.no_grad():
        table = model.all_item_embeddings().cpu().numpy().astype(np.float32)

    np.save(run_dir / "item_embeddings.npy", table)
    save_json(
        {
            "run_id": run_dir.name,
            "model": cfg.model.get("name"),
            "fusion": cfg.model.get("fusion"),
            "modalities": dict(cfg.model.get("modalities", {})),
            "num_items": data.num_items,
            "dim": int(table.shape[1]),
            "checkpoint": args.checkpoint,
            "train_freq_sha": __import__("hashlib").sha256(
                np.ascontiguousarray(data.train_freq).tobytes()
            ).hexdigest()[:16],
        },
        run_dir / "item_embeddings_meta.json",
    )
    print(f"wrote {run_dir/'item_embeddings.npy'} {table.shape}")


if __name__ == "__main__":
    main()
