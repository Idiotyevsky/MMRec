"""MM-SASRec: multimodal sequential recommendation.

The *only* difference from ``SASRec`` is that the item embedding table is
produced by ``MultimodalItemEncoder`` (ID + content fusion) instead of a plain
``nn.Embedding``.  Everything downstream -- positional encoding, causal
Transformer, dot-product scoring, sampled-softmax loss -- is shared, which keeps
the ID-only vs multimodal comparison clean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .item_encoder import MultimodalItemEncoder
from .sasrec import PAD, SequenceEncoder, mask_pad


class MMSASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        dim_feedforward: int | None = None,
        modalities: dict[str, bool] | None = None,
        fusion: str = "gated",
        feature_dir: str | Path | None = None,
        row_for_item: dict[str, np.ndarray] | None = None,
        id_dropout_prob: float = 0.0,
        modality_dropout_prob: float = 0.0,
        per_modality_dropout: dict[str, float] | None = None,
        cold_item_mask: np.ndarray | None = None,
        zero_cold_id: bool = True,
        normalize_scores: bool = False,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.hidden_size = int(hidden_size)
        self.max_seq_len = int(max_seq_len)
        self.normalize_scores = bool(normalize_scores)

        self.item_encoder = MultimodalItemEncoder(
            num_items=num_items,
            hidden_size=hidden_size,
            modalities=modalities,
            fusion=fusion,
            dropout=dropout,
            feature_dir=feature_dir,
            row_for_item=row_for_item,
            id_dropout_prob=id_dropout_prob,
            modality_dropout_prob=modality_dropout_prob,
            per_modality_dropout=per_modality_dropout,
            cold_item_mask=cold_item_mask,
            zero_cold_id=zero_cold_id,
        )
        self.encoder = SequenceEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            dim_feedforward=dim_feedforward,
        )
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def item_embeddings(self, item_ids: torch.Tensor, generator=None):
        """Return ``(embeddings, gate_weights)`` for arbitrary item ids."""
        return self.item_encoder(item_ids, generator=generator)

    def encode_sequence(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Hidden state of *every* position, ``(B, L, H)`` (training objective)."""
        emb, _ = self.item_encoder(input_ids)
        return self.encoder(self.dropout(emb), input_ids, return_sequence=True)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """User representation for *inference*: the last (most recent) position."""
        return self.encode_sequence(input_ids)[:, -1]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids)

    def score(self, user_repr: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        if self.normalize_scores:
            user_repr = torch.nn.functional.normalize(user_repr, dim=-1)
            item_emb = torch.nn.functional.normalize(item_emb, dim=-1)
        return user_repr @ item_emb.t()

    @torch.no_grad()
    def all_item_embeddings(self) -> torch.Tensor:
        return self.item_encoder.all_item_embeddings()

    @torch.no_grad()
    def gate_weights_for_all_items(self) -> dict[str, torch.Tensor]:
        """Gate weight per item, for interpretability analysis."""
        self.eval()
        device = next(self.parameters()).device
        ids = torch.arange(self.num_items + 1, device=device)
        _, gates = self.item_encoder(ids)
        return {k: v.detach().cpu() for k, v in gates.items()}

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )


__all__ = ["MMSASRec", "mask_pad", "PAD"]
