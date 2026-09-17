"""Training / validation loop shared by every model.

Responsibilities
----------------
* autoregressive training over all non-padding positions;
* sampled-softmax loss with a negative sampler that can never collide with a
  known interaction (see ``NegativeSampler``);
* AMP, gradient clipping, early stopping, checkpointing, resume;
* writing ``training_log.csv`` + ``metrics.json`` + ``environment.txt`` into the
  run directory so every reported number is traceable to an artifact.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import EvalDataset, ProcessedData, TrainDataset, collate_eval, collate_train
from ..data.negative_sampler import NegativeSampler, build_training_interaction_keys
from ..evaluation.evaluator import FullRankingEvaluator
from ..models.bpr import bpr_loss
from ..models.loss import sampled_softmax_loss
from ..utils.device import env_fingerprint
from ..utils.io import ensure_dir, save_json
from ..utils.provenance import dataset_fingerprint
from ..utils.logging import get_logger


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg,
        data: ProcessedData,
        run_dir: str | Path,
        device: torch.device,
        logger=None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.data = data
        self.run_dir = ensure_dir(run_dir)
        self.device = device
        self.log = logger or get_logger("mmrec.train", self.run_dir / "train.log")

        t = cfg.training
        self.batch_size = int(t.get("batch_size", 512))
        self.epochs = int(t.get("epochs", 200))
        self.max_grad_norm = float(t.get("max_grad_norm", 5.0))
        self.amp = bool(t.get("amp", True)) and device.type == "cuda"
        self.seed = int(t.get("seed", 42))
        self.patience = int(t.get("early_stopping_patience", 20))
        self.min_epochs = int(t.get("min_epochs", 1))
        self.monitor = str(t.get("monitor", "NDCG@10"))
        self.monitor_mode = str(t.get("monitor_mode", "max"))
        self.num_workers = int(t.get("num_workers", 0))
        self.max_seq_len = int(cfg.model.get("max_seq_len", cfg.data.get("max_sequence_length", 50)))

        l = cfg.get("loss", {}) or {}
        default_loss = "bpr" if str(cfg.model.get("name", "")).lower() == "bpr" else "sampled_softmax"
        self.loss_type = str(l.get("type", default_loss)).lower()
        self.num_negatives = int(l.get("num_negatives", 128))
        self.temperature = float(l.get("temperature", 1.0))
        self.in_batch_negatives = bool(l.get("in_batch_negatives", False))

        ns = cfg.get("negative_sampling", {}) or {}
        # train-prefix interactions only: val/test targets must stay reachable.
        self.sampler = NegativeSampler(
            num_items=data.num_items,
            train_freq=data.train_freq,
            all_interactions=build_training_interaction_keys(
                data.flat_items, data.user_offsets, data.train_len, data.num_items
            ),
            mode=str(ns.get("mode", "uniform")),
            popularity_power=float(ns.get("popularity_power", 0.75)),
            seed=self.seed + 1,
        )

        self.evaluator = FullRankingEvaluator(
            num_items=data.num_items,
            ks=tuple(cfg.evaluation.get("ks", [5, 10, 20])),
            item_chunk_size=int(cfg.evaluation.get("item_chunk_size", 4096)),
            device=device,
        )

        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = (
            torch.optim.AdamW(
                params,
                lr=float(t.get("learning_rate", 1e-3)),
                weight_decay=float(t.get("weight_decay", 1e-4)),
                betas=tuple(t.get("betas", (0.9, 0.98))),
            )
            if params
            else None  # e.g. the popularity baseline has no parameters
        )
        self.scheduler = self._build_scheduler() if self.optimizer is not None else None
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp and self.optimizer is not None)

        self.start_epoch = 0
        self.global_step = 0
        self.best_metric = -float("inf") if self.monitor_mode == "max" else float("inf")
        self.best_epoch = -1
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def _build_scheduler(self):
        if self.optimizer is None:
            return None
        t = self.cfg.training
        kind = str(t.get("scheduler", "none")).lower()
        if kind in ("none", "null", ""):
            return None
        if kind == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=int(t.get("epochs", 200)), eta_min=float(t.get("min_lr", 1e-5))
            )
        if kind == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=self.monitor_mode, factor=0.5, patience=max(self.patience // 3, 2)
            )
        if kind == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=int(t.get("step_size", 50)), gamma=0.5)
        raise ValueError(f"Unknown scheduler {kind!r}")

    # ------------------------------------------------------------------
    def _encode_fn(self):
        name = str(self.cfg.model.get("name", "mm_sasrec")).lower()
        if name in ("bpr", "popular", "random"):
            return lambda batch: self.model.encode(batch["user_id"].to(self.device))
        return lambda batch: self.model.encode(batch["input_ids"].to(self.device))

    def _train_loader(self) -> DataLoader:
        ds = TrainDataset(self.data, self.max_seq_len)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_train,
            drop_last=False,
            pin_memory=self.device.type == "cuda",
        )

    def _eval_loader(self, split: str, max_users: int | None = None) -> DataLoader:
        users = None
        if max_users is not None and max_users < self.data.num_users:
            users = np.arange(max_users, dtype=np.int64)
        ds = EvalDataset(self.data, self.max_seq_len, split=split, users=users)
        return DataLoader(
            ds,
            batch_size=max(self.batch_size, 256),
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_eval,
            pin_memory=self.device.type == "cuda",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_positive(target: torch.Tensor) -> torch.Tensor:
        """Pick one random valid target position per row (left padding assumed)."""
        B, L = target.shape
        n_valid = (target > 0).sum(dim=1).clamp(min=1)
        idx = (torch.rand(B, device=target.device) * n_valid.float()).long().clamp(max=(n_valid - 1).clamp(min=0))
        cols = (L - n_valid) + idx  # valid targets occupy the last n_valid columns
        return target.gather(1, cols.view(-1, 1)).squeeze(1)

    def train_epoch(self, epoch: int, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "n": 0, "pos_score": 0.0}
        t0 = time.time()

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            target = batch["target"].to(self.device, non_blocking=True)
            user_ids = batch["user_id"].numpy()

            n_neg = 1 if self.loss_type == "bpr" else self.num_negatives
            # exclude the user's *targets* of this batch from the negatives too,
            # not just the history (which already covers them)
            neg_ids = self.sampler.sample(user_ids, n_neg, exclude_items=target.cpu().numpy())
            neg_ids_t = torch.as_tensor(neg_ids, dtype=torch.long, device=self.device)

            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.amp):
                if self.loss_type == "bpr":
                    # static MF: one user vector, one positive, one negative.
                    # The positive is drawn uniformly from the user's training
                    # items each step, so BPR sees the whole interaction set
                    # instead of a single fixed item per user.
                    pos_items = self._sample_positive(target)
                    user_repr = self.model.encode(
                        torch.as_tensor(user_ids, dtype=torch.long, device=self.device)
                    )
                    pos_emb, _ = self.model.item_embeddings(pos_items)
                    neg_emb, _ = self.model.item_embeddings(neg_ids_t[:, 0])
                    loss = bpr_loss(
                        (user_repr * pos_emb).sum(-1), (user_repr * neg_emb).sum(-1)
                    )
                    pos_score = (user_repr * pos_emb).sum(-1).mean()
                else:
                    # every position is scored with its own hidden state:
                    # h_{b,l} predicts target_{b,l} (standard SASRec objective).
                    seq_repr = self.model.encode_sequence(input_ids)
                    pos_emb, _ = self.model.item_embeddings(target)
                    neg_emb, _ = self.model.item_embeddings(neg_ids_t)
                    ib_emb = None
                    if self.in_batch_negatives and input_ids.shape[0] > 1:
                        # other users' positives: roll by one so no row sees its
                        # own targets (queries and keys differ per row)
                        ib_emb = pos_emb.roll(shifts=1, dims=0).detach()
                    loss = sampled_softmax_loss(
                        seq_repr,
                        pos_emb,
                        neg_emb,
                        target,
                        temperature=self.temperature,
                        in_batch_neg_emb=ib_emb,
                    )
                    with torch.no_grad():
                        valid = (target != 0).to(seq_repr.dtype)
                        pos_score = (
                            ((seq_repr * pos_emb).sum(-1) * valid).sum()
                            / valid.sum().clamp(min=1.0)
                        )

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, step {self.global_step}")

            if self.optimizer is not None:
                self.scaler.scale(loss).backward()
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            totals["loss"] += float(loss.detach()) * input_ids.shape[0]
            totals["pos_score"] += float(pos_score) * input_ids.shape[0]
            totals["n"] += input_ids.shape[0]
            self.global_step += 1

        n = max(totals["n"], 1)
        return {
            "train_loss": totals["loss"] / n,
            "train_pos_score": totals["pos_score"] / n,
            "epoch_time_s": time.time() - t0,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, split: str = "val", max_users: int | None = None) -> dict[str, float]:
        self.model.eval()
        item_emb = self.model.all_item_embeddings()
        result = self.evaluator.evaluate(
            encode_fn=self._encode_fn(),
            item_embeddings=item_emb,
            loader=self._eval_loader(split, max_users=max_users),
            desc=f"eval:{split}",
        )
        m = result.metrics()
        m["num_users"] = int(result.user_ids.shape[0])
        return m

    # ------------------------------------------------------------------
    def _is_better(self, value: float) -> bool:
        if self.monitor_mode == "max":
            return value > self.best_metric + 1e-9
        return value < self.best_metric - 1e-9

    def fit(self) -> dict[str, Any]:
        if self.optimizer is None:
            # parameter-free baseline: evaluate immediately, no training loop
            val = self.validate("val")
            self.best_metric = val[self.monitor]
            self.best_epoch = 0
            row = {"epoch": 0, "lr": 0.0, **{f"val_{k}": v for k, v in val.items()}}
            self.history.append(row)
            with open(self.run_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(list(row.keys()))
                writer.writerow(list(row.values()))
            self.save_checkpoint(self.run_dir / "best.pt", epoch=0, metric=self.best_metric)
            save_json(
                {"best_metric": self.best_metric, "best_epoch": 0, "monitor": self.monitor,
                 "monitor_mode": self.monitor_mode, "history": self.history, "num_parameters": 0},
                self.run_dir / "train_summary.json",
            )
            self.log.info(f"parameter-free model | val {self.monitor} = {self.best_metric:.4f}")
            return {"best_metric": self.best_metric, "best_epoch": 0}

        loader = self._train_loader()
        log_path = self.run_dir / "training_log.csv"
        write_header = not log_path.exists()
        csv_f = open(log_path, "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_f)

        self.log.info(f"training on {self.device} | params={self.model.num_parameters():,}")
        self.log.info(
            f"train users={len(loader.dataset)} batch={self.batch_size} "
            f"neg={self.num_negatives} amp={self.amp} monitor={self.monitor}"
        )

        epochs_without_improvement = 0
        for epoch in range(self.start_epoch, self.epochs):
            tr = self.train_epoch(epoch, loader)
            val = self.validate("val")
            lr = self.optimizer.param_groups[0]["lr"] if self.optimizer is not None else 0.0
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val[self.monitor])
                else:
                    self.scheduler.step()

            row = {"epoch": epoch, "lr": lr, **tr, **{f"val_{k}": v for k, v in val.items()}}
            self.history.append(row)
            if write_header:
                writer.writerow(list(row.keys()))
                write_header = False
            writer.writerow(list(row.values()))
            csv_f.flush()

            improved = self._is_better(val[self.monitor])
            if improved:
                self.best_metric = val[self.monitor]
                self.best_epoch = epoch
                epochs_without_improvement = 0
                self.save_checkpoint(self.run_dir / "best.pt", epoch=epoch, metric=val[self.monitor])
            else:
                epochs_without_improvement += 1

            self.log.info(
                f"epoch {epoch:3d} | loss {tr['train_loss']:.4f} | "
                f"val Recall@20 {val.get('Recall@20', float('nan')):.4f} "
                f"NDCG@20 {val.get('NDCG@20', float('nan')):.4f} | "
                f"{self.monitor} {val[self.monitor]:.4f}"
                f"{' *' if improved else ''} | {tr['epoch_time_s']:.1f}s"
            )

            if epoch + 1 >= self.min_epochs and epochs_without_improvement >= self.patience:
                self.log.info(f"early stopping after {epochs_without_improvement} epochs without improvement")
                break

        csv_f.close()
        self.save_checkpoint(self.run_dir / "last.pt", epoch=epoch, metric=val[self.monitor])
        save_json(
            {
                "best_metric": self.best_metric,
                "best_epoch": self.best_epoch,
                "monitor": self.monitor,
                "monitor_mode": self.monitor_mode,
                "history": self.history,
                "num_parameters": int(self.model.num_parameters()),
            },
            self.run_dir / "train_summary.json",
        )
        return {"best_metric": self.best_metric, "best_epoch": self.best_epoch}

    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path, epoch: int = 0, metric: float | None = None) -> None:
        path = Path(path)
        ensure_dir(path.parent)
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "metric": metric,
            "monitor": self.monitor,
            "config": self.cfg.to_dict() if hasattr(self.cfg, "to_dict") else dict(self.cfg),
            "seed": self.seed,
            "num_items": self.data.num_items,
            "num_users": self.data.num_users,
            "dataset_dir": str(self.data.dir),
            "dataset_hash": self._dataset_hash(),
        }
        torch.save(payload, path)

    def _dataset_hash(self) -> str:
        return dataset_fingerprint(self.data)

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = True) -> dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # never silently load a checkpoint built on a different dataset
        if int(ckpt.get("num_items", -1)) != self.data.num_items:
            raise AssertionError(
                f"checkpoint num_items={ckpt.get('num_items')} != dataset num_items={self.data.num_items}"
            )
        if int(ckpt.get("num_users", -1)) != self.data.num_users:
            raise AssertionError(
                f"checkpoint num_users={ckpt.get('num_users')} != dataset num_users={self.data.num_users}"
            )
        ckpt_hash = ckpt.get("dataset_hash")
        if ckpt_hash is not None and ckpt_hash != self._dataset_hash():
            raise AssertionError(
                "checkpoint was trained on a different item mapping / cold split; "
                "refusing to load it into this dataset"
            )
        # A checkpoint may carry parameters that the current config does not
        # allocate (optional regularisers).  Extra keys are dropped with a
        # warning; *missing* keys are still a hard error, because those mean the
        # checkpoint is genuinely incompatible.
        state = ckpt["model_state"]
        own = self.model.state_dict()
        extra = [k for k in state if k not in own]
        missing = [k for k in own if k not in state]
        if missing:
            raise AssertionError(
                f"checkpoint {path} is missing parameters required by this model: {missing[:5]}"
            )
        if extra:
            self.log.warning(f"dropping {len(extra)} unused checkpoint tensors: {extra[:5]}")
            state = {k: v for k, v in state.items() if k in own}
        self.model.load_state_dict(state)
        if load_optimizer and ckpt.get("optimizer_state") is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if load_optimizer and ckpt.get("scheduler_state") is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.global_step = int(ckpt.get("global_step", 0))
        self.best_metric = float(ckpt.get("best_metric", self.best_metric))
        self.best_epoch = int(ckpt.get("best_epoch", -1))
        self.log.info(
            f"resumed from {path} (epoch {ckpt.get('epoch')}, best {self.monitor}={self.best_metric:.4f})"
        )
        return ckpt

    def write_environment(self) -> None:
        info = env_fingerprint()
        info["run_dir"] = str(self.run_dir)
        info["seed"] = self.seed
        with open(self.run_dir / "environment.txt", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
