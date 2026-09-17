#!/usr/bin/env python
"""Re-run final evaluation for finished runs without retraining.

    python scripts/reevaluate.py                  # every run with best.pt
    python scripts/reevaluate.py --only sasrec mm_gated

Used when the *evaluation* changes (e.g. a corrected tie policy) but the trained
weights are still valid.  The ``train`` block of ``metrics.json`` is preserved;
only the evaluation blocks and the stored ranking arrays are rewritten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.evaluation.evaluator import FullRankingEvaluator  # noqa: E402
from src.training.factory import build_model  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.reevaluate")
RUNS = ROOT / "results" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="tag prefixes to process")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--runs-dir", default=str(RUNS))
    args = ap.parse_args()

    device = torch.device(args.device)
    runs_dir = Path(args.runs_dir)
    n_done = 0
    data_cache: dict[str, ProcessedData] = {}

    for d in sorted(runs_dir.iterdir()):
        if not (d.is_dir() and (d / "best.pt").exists() and (d / "config.yaml").exists()):
            continue
        if not (d / "train_summary.json").exists():
            LOG.info(f"skip {d.name}: still training (no train_summary.json)")
            continue
        tag = d.name.rsplit("_", 2)[0]
        if args.only and not any(tag.startswith(p) for p in args.only):
            continue

        cfg = Config(yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")))
        pdir = str(cfg.data.processed_dir)
        if pdir not in data_cache:
            data_cache[pdir] = ProcessedData.load(ROOT / pdir)
        data = data_cache[pdir]

        t0 = time.time()
        model = build_model(cfg, data, device=device)
        ckpt = torch.load(d / "best.pt", map_location=device, weights_only=False)
        state = {k: v for k, v in ckpt["model_state"].items() if k in model.state_dict()}
        model.load_state_dict(state)
        model.eval()

        name = str(cfg.model.get("name", "")).lower()
        if name in ("bpr", "popular"):
            encode_fn = lambda b: model.encode(b["user_id"].to(device))  # noqa: E731
        else:
            encode_fn = lambda b: model.encode(b["input_ids"].to(device))  # noqa: E731

        max_len = int(cfg.model.get("max_seq_len", 50))
        evaluator = FullRankingEvaluator(
            num_items=data.num_items,
            ks=tuple(cfg.evaluation.get("ks", [5, 10, 20])),
            item_chunk_size=int(cfg.evaluation.get("item_chunk_size", 4096)),
            device=device,
        )
        loader = lambda split: _loader(data, max_len, split, int(cfg.training.get("batch_size", 512)))  # noqa: E731

        old = json.loads((d / "metrics.json").read_text(encoding="utf-8")) if (d / "metrics.json").exists() else {}
        results = {"run_id": d.name, "model": cfg.model.get("name"),
                   "seed": old.get("seed", cfg.training.get("seed")), "train": old.get("train", {})}

        with torch.no_grad():
            item_emb = model.all_item_embeddings()
            for split in ("val", "test"):
                res = evaluator.evaluate(encode_fn, item_emb, loader(split))
                results[split] = res.metrics()
                np.savez_compressed(
                    d / f"{split}_ranking.npz",
                    user_ids=res.user_ids, targets=res.targets,
                    rank=res.rank, topk=res.topk,
                )

            if data.is_cold.any():
                res_test = evaluator.evaluate(encode_fn, item_emb, loader("test"))
                is_cold = data.is_cold[res_test.targets]
                cold_full = res_test.subset(is_cold)
                np.savez_compressed(
                    d / "test_ranking_cold.npz",
                    user_ids=cold_full.user_ids, targets=cold_full.targets,
                    rank=cold_full.rank, topk=cold_full.topk,
                )
                res_cold = evaluator.evaluate(encode_fn, item_emb, loader("test"),
                                              candidate_mask=data.is_cold)
                results["cold"] = {
                    "num_users_with_cold_target": int(is_cold.sum()),
                    "num_cold_items": int(data.is_cold.sum()),
                    "full_ranking": cold_full.metrics(),
                    "restricted_to_cold": res_cold.subset(is_cold).metrics(),
                }

        save_json(results, d / "metrics.json")
        LOG.info(
            f"{d.name}: test Recall@20={results['test']['Recall@20']:.4f} "
            f"NDCG@20={results['test']['NDCG@20']:.4f}"
            + (f" | cold Recall@20={results['cold']['full_ranking']['Recall@20']:.4f}"
               if "cold" in results else "")
            + f"  ({time.time()-t0:.1f}s)"
        )
        n_done += 1

    LOG.info(f"re-evaluated {n_done} runs")


def _loader(data, max_len, split, batch_size):
    from torch.utils.data import DataLoader

    from src.data.dataset import EvalDataset, collate_eval

    return DataLoader(
        EvalDataset(data, max_len, split=split),
        batch_size=max(batch_size, 256),
        shuffle=False,
        collate_fn=collate_eval,
    )


if __name__ == "__main__":
    main()
