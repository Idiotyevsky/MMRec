"""Build models / optimizers from a config + processed dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data.dataset import ProcessedData
from ..models.bpr import BPRMF
from ..models.mm_sasrec import MMSASRec
from ..models.popular import PopularRecommender
from ..models.sasrec import SASRec

MODEL_NAMES = ("popular", "bpr", "sasrec", "mm_sasrec")


def _row_for_item(processed_dir: Path, modalities: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for m in modalities:
        p = processed_dir / f"row_for_item_{m}.npy"
        if p.exists():
            out[m] = np.load(p).astype(np.int64)
    return out


def build_model(cfg, data: ProcessedData, device: torch.device | str = "cpu") -> torch.nn.Module:
    name = str(cfg.model.get("name", "mm_sasrec")).lower()
    m = cfg.model
    cold_mask = data.is_cold if bool(m.get("zero_cold_id", True)) else None

    common = dict(
        hidden_size=int(m.get("hidden_size", 128)),
        dropout=float(m.get("dropout", 0.2)),
        cold_item_mask=cold_mask,
        zero_cold_id=bool(m.get("zero_cold_id", True)),
    )

    if name == "sasrec":
        model = SASRec(
            num_items=data.num_items,
            num_layers=int(m.get("num_layers", 2)),
            num_heads=int(m.get("num_heads", 4)),
            max_seq_len=int(m.get("max_seq_len", 50)),
            dim_feedforward=m.get("dim_feedforward"),
            item_dropout_prob=float(m.get("item_dropout_prob", 0.0)),
            **common,
        )
    elif name == "mm_sasrec":
        modalities = m.get("modalities", {"id": True, "text": True, "image": True, "video": False})
        modalities = {k: bool(v) for k, v in dict(modalities).items()}
        enabled_content = [k for k, v in modalities.items() if v and k != "id"]
        model = MMSASRec(
            num_items=data.num_items,
            num_layers=int(m.get("num_layers", 2)),
            num_heads=int(m.get("num_heads", 4)),
            max_seq_len=int(m.get("max_seq_len", 50)),
            dim_feedforward=m.get("dim_feedforward"),
            modalities=modalities,
            fusion=str(m.get("fusion", "gated")),
            feature_dir=str(cfg.data.get("feature_dir", "data/raw")),
            row_for_item=_row_for_item(Path(cfg.data.processed_dir), enabled_content),
            id_dropout_prob=float(m.get("id_dropout_prob", 0.0)),
            modality_dropout_prob=float(m.get("modality_dropout_prob", 0.0)),
            per_modality_dropout=m.get("per_modality_dropout"),
            normalize_scores=bool(m.get("normalize_scores", False)),
            **common,
        )
    elif name == "bpr":
        model = BPRMF(
            num_users=data.num_users,
            num_items=data.num_items,
            **common,
        )
    elif name == "popular":
        model = PopularRecommender(
            num_items=data.num_items,
            train_freq=data.train_freq,
            cold_item_mask=cold_mask,
        )
    else:
        raise ValueError(f"Unknown model name {name!r}; expected one of {MODEL_NAMES}")

    return model.to(device)


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    t = cfg.training
    lr = float(t.get("learning_rate", 1e-3))
    wd = float(t.get("weight_decay", 1e-4))
    betas = tuple(t.get("betas", (0.9, 0.98)))
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=wd,
        betas=betas,
    )
