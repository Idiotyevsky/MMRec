"""BPR-MF: static matrix factorisation baseline.

    L = -log sigmoid( s(u, i+) - s(u, i-) ),  s(u, i) = <e_u, e_i>

Its purpose is to show that *sequential* modelling beats static collaborative
filtering on this dataset, and to provide a collaborative-only reference point
for the cold / long-tail analysis.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

PAD = 0


class BPRMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        hidden_size: int = 128,
        dropout: float = 0.0,
        cold_item_mask: np.ndarray | None = None,
        zero_cold_id: bool = True,
    ) -> None:
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.hidden_size = int(hidden_size)
        self.zero_cold_id = bool(zero_cold_id)

        self.user_embedding = nn.Embedding(self.num_users + 1, hidden_size, padding_idx=PAD)
        self.item_embedding = nn.Embedding(self.num_items + 1, hidden_size, padding_idx=PAD)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.user_embedding.weight[PAD].zero_()
            self.item_embedding.weight[PAD].zero_()

        self.dropout = nn.Dropout(dropout)
        cold = (
            np.zeros(self.num_items + 1, dtype=bool)
            if cold_item_mask is None
            else np.asarray(cold_item_mask, dtype=bool)
        )
        self.register_buffer("cold_item_mask", torch.as_tensor(cold, dtype=torch.bool), persistent=True)

    def user_repr(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.user_embedding(user_ids))

    def encode(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Evaluator entry point: BPR's user representation is a lookup."""
        return self.user_repr(user_ids)

    def item_embeddings(self, item_ids: torch.Tensor, generator=None):
        return self.item_repr(item_ids), {}

    def item_repr(self, item_ids: torch.Tensor) -> torch.Tensor:
        e = self.item_embedding(item_ids)
        if self.zero_cold_id and self.cold_item_mask.any():
            e = e * (~self.cold_item_mask[item_ids]).unsqueeze(-1).to(e.dtype)
        return e

    def forward(
        self, user_ids: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.user_repr(user_ids)
        p = self.item_repr(pos_items)
        n = self.item_repr(neg_items)
        return (u * p).sum(-1), (u * n).sum(-1)

    @torch.no_grad()
    def all_item_embeddings(self) -> torch.Tensor:
        ids = torch.arange(self.num_items + 1, device=self.item_embedding.weight.device)
        table = self.item_repr(ids)
        table[PAD] = 0
        return table

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
