"""Uniform-Random baseline: frozen seeded noise per item, nothing is learned.

It exists for one purpose: the cold-split floor.  An item with zero training
interactions cannot be ranked by a learned ID embedding, so a model that scores
*cold* items only by content must beat chance -- and chance has a closed form.
With a candidate set of ``N`` items, a uniform ranker's expected Recall@k is
``k / N`` (20/1974 on the cold10 split), which is what this model measures on
real data instead of what someone remembered.

The scores are a buffer with ``requires_grad_(False)`` and ``num_parameters()``
is 0: this is not a trained baseline and must never be reported as one.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PAD = 0


class RandomRecommender(nn.Module):
    def __init__(self, num_items: int, seed: int = 42) -> None:
        super().__init__()
        self.num_items = int(num_items)
        gen = torch.Generator().manual_seed(int(seed))
        scores = torch.rand(self.num_items + 1, generator=gen, dtype=torch.float32)
        scores[PAD] = 0.0  # same convention as the other baselines; scoring masks PAD anyway
        self.register_buffer("scores", scores)
        self.hidden_size = 1

    def encode(self, x) -> torch.Tensor:
        if isinstance(x, dict):
            user_ids = x["user_id"]
        else:
            user_ids = x
        return torch.ones(user_ids.shape[0], 1, device=self.scores.device)

    def item_embeddings(self, item_ids: torch.Tensor, generator=None):
        return self.scores[item_ids].unsqueeze(-1), {}

    @torch.no_grad()
    def all_item_embeddings(self) -> torch.Tensor:
        return self.scores.unsqueeze(-1).clone()

    def num_parameters(self, trainable_only: bool = True) -> int:
        return 0
