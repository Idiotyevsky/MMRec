#!/usr/bin/env python
"""Evaluate a saved checkpoint with the full-ranking protocol.

    python scripts/evaluate.py --run-dir results/runs/<run_id> --split test
    python scripts/evaluate.py --run-dir results/runs/<run_id> --split test --cold-only

Prints overall metrics, per-popularity-bucket metrics and (when the dataset has a
cold split) cold-item metrics.  All slices come from one ranking pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import EvalDataset, ProcessedData, collate_eval  # noqa: E402
from src.evaluation.evaluator import FullRankingEvaluator  # noqa: E402
from src.evaluation.slicing import sliced_metrics  # noqa: E402
from src.training.factory import build_model  # noqa: E402
from src.utils.config import Config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cold-only", action="store_true",
                    help="also rank within the cold catalogue only")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = Config(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
    data = ProcessedData.load(ROOT / cfg.data.processed_dir)
    device = torch.device(args.device)

    model = build_model(cfg, data, device=device)
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {run_dir.name} (epoch {ckpt.get('epoch')}, "
          f"params {model.num_parameters():,})")

    name = str(cfg.model.get("name", "")).lower()
    if name in ("bpr", "popular"):
        encode_fn = lambda b: model.encode(b["user_id"].to(device))  # noqa: E731
    else:
        encode_fn = lambda b: model.encode(b["input_ids"].to(device))  # noqa: E731

    max_len = int(cfg.model.get("max_seq_len", 50))
    loader = DataLoader(
        EvalDataset(data, max_len, split=args.split), batch_size=1024,
        shuffle=False, collate_fn=collate_eval,
    )
    ev = FullRankingEvaluator(
        num_items=data.num_items,
        ks=tuple(cfg.evaluation.get("ks", [5, 10, 20])),
        item_chunk_size=int(cfg.evaluation.get("item_chunk_size", 4096)),
        device=device,
    )

    with torch.no_grad():
        item_emb = model.all_item_embeddings()
        result = ev.evaluate(encode_fn, item_emb, loader, desc=args.split)

    out: dict = {"run_id": run_dir.name, "split": args.split, "overall": result.metrics()}
    print("\noverall:")
    for k, v in out["overall"].items():
        print(f"  {k:>12s} {v:.4f}" if isinstance(v, float) else f"  {k:>12s} {v}")

    slices = sliced_metrics(result, data)
    out["slices"] = slices
    print("\nby population:")
    for pop in ("head", "middle", "tail", "cold", "warm"):
        if pop not in slices or slices[pop]["num_users"] == 0:
            continue
        s = slices[pop]
        print(f"  {pop:>7s} n={s['num_users']:>6d}  "
              f"Recall@20={s['Recall@20']:.4f}  NDCG@20={s['NDCG@20']:.4f}")

    if args.cold_only and data.is_cold.any():
        with torch.no_grad():
            cold_res = ev.evaluate(encode_fn, item_emb, loader,
                                   candidate_mask=data.is_cold, desc=f"{args.split}:cold")
        mask = data.is_cold[cold_res.targets]
        out["cold_only_candidates"] = cold_res.subset(mask).metrics()
        print("\nrestricted to cold candidates (cold-target users only):")
        for k, v in out["cold_only_candidates"].items():
            print(f"  {k:>12s} {v:.4f}" if isinstance(v, float) else f"  {k:>12s} {v}")

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
