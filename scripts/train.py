#!/usr/bin/env python
"""Train a model from a YAML config.

    python scripts/train.py --config configs/sasrec.yaml
    python scripts/train.py --config configs/mm_sasrec_gated.yaml --seed 2026 --training.batch_size 256
    python scripts/train.py --config configs/sasrec.yaml --resume results/runs/<id>/last.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ProcessedData  # noqa: E402
from src.evaluation.evaluator import FullRankingEvaluator  # noqa: E402
from src.training.factory import build_model  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import load_config, save_config  # noqa: E402
from src.utils.device import env_fingerprint, get_device  # noqa: E402
from src.utils.io import ensure_dir, run_id, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", default=None, help="explicit run directory (default: results/runs/<run_id>)")
    ap.add_argument("--tag", default=None, help="short tag used in the run id")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-test", action="store_true", help="skip the final test evaluation")
    return ap.parse_known_args()


def main() -> None:
    args, extra = parse_args()
    overrides = list(extra)
    if args.seed is not None:
        overrides.append(f"training.seed={args.seed}")
    if args.epochs is not None:
        overrides.append(f"training.epochs={args.epochs}")
    cfg = load_config(args.config, overrides)

    seed = int(cfg.training.get("seed", 42))
    set_seed(seed)

    device = get_device(args.device or cfg.training.get("device"))
    data = ProcessedData.load(cfg.data.processed_dir)

    tag = args.tag or str(cfg.model.get("name", "model"))
    rid = run_id(tag)
    run_dir = ensure_dir(args.run_dir or Path("results/runs") / rid)

    logger = get_logger(f"mmrec.{rid}", run_dir / "train.log")
    logger.info(f"run_dir={run_dir}")
    logger.info(f"device={device} | {data.summary()}")
    if data.is_cold.any():
        logger.info(f"cold split active: {int(data.is_cold.sum())} items are cold")

    save_config(cfg, run_dir / "config.yaml")
    save_json(env_fingerprint(), run_dir / "environment.txt")

    model = build_model(cfg, data, device=device)
    trainer = Trainer(model, cfg, data, run_dir, device, logger=logger)
    trainer.write_environment()

    t0 = time.time()
    resume_ckpt = None
    if args.resume:
        resume_ckpt = trainer.load_checkpoint(args.resume)

    summary = trainer.fit()
    summary["train_time_s"] = time.time() - t0
    summary["num_parameters"] = int(model.num_parameters())
    summary["device"] = str(device)

    # ---- final evaluation with the best checkpoint ----
    best_path = run_dir / "best.pt"
    if best_path.exists():
        trainer.load_checkpoint(best_path, load_optimizer=False)

    evaluator = FullRankingEvaluator(
        num_items=data.num_items,
        ks=tuple(cfg.evaluation.get("ks", [5, 10, 20])),
        item_chunk_size=int(cfg.evaluation.get("item_chunk_size", 4096)),
        device=device,
    )
    encode_fn = trainer._encode_fn()
    results: dict = {"run_id": rid, "model": cfg.model.get("name"), "seed": seed, "train": summary}

    with torch.no_grad():
        item_emb = model.all_item_embeddings()
        for split in ("val", "test"):
            res = evaluator.evaluate(
                encode_fn=encode_fn,
                item_embeddings=item_emb,
                loader=trainer._eval_loader(split),
                desc=f"final:{split}",
            )
            results[split] = res.metrics()
            np.savez_compressed(
                run_dir / f"{split}_ranking.npz",
                user_ids=res.user_ids,
                targets=res.targets,
                rank=res.rank,
                topk=res.topk,
            )
            logger.info(
                f"{split}: " + " ".join(f"{k}={v:.4f}" for k, v in results[split].items() if "num_users" not in k)
            )

    # cold slices: (a) cold targets inside the FULL ranking, (b) ranking
    # restricted to the cold catalogue.  Both come from real ranking passes.
    if data.is_cold.any():
        res_test = evaluator.evaluate(
            encode_fn=encode_fn,
            item_embeddings=item_emb,
            loader=trainer._eval_loader("test"),
            desc="final:test(full)",
        )
        is_cold_target = data.is_cold[res_test.targets]
        cold_full = res_test.subset(is_cold_target)
        np.savez_compressed(
            run_dir / "test_ranking_cold.npz",
            user_ids=cold_full.user_ids,
            targets=cold_full.targets,
            rank=cold_full.rank,
            topk=cold_full.topk,
        )

        res_cold_only = evaluator.evaluate(
            encode_fn=encode_fn,
            item_embeddings=item_emb,
            loader=trainer._eval_loader("test"),
            candidate_mask=data.is_cold,
            desc="final:test(cold-only candidates)",
        )
        results["cold"] = {
            "num_users_with_cold_target": int(is_cold_target.sum()),
            "num_cold_items": int(data.is_cold.sum()),
            "full_ranking": cold_full.metrics(),
            "restricted_to_cold": res_cold_only.subset(is_cold_target).metrics(),
        }

    save_json(results, run_dir / "metrics.json")
    save_json(results, Path("results") / "latest_metrics.json")
    logger.info(f"done in {summary['train_time_s']:.1f}s -> {run_dir}")


if __name__ == "__main__":
    main()
