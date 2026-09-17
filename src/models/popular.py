"""Popularity baseline: ``score(item) = training interaction count``.

Implemented as a degenerate 1-dimensional model so it can reuse the exact same
full-ranking evaluator (and therefore the exact same masking) as every other
model.  It exists as a sanity check: a trained model that fails to beat it has a
bug, not a research finding.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

PAD = 0


class PopularRecommender(nn.Module):
    def __init__(self, num_items: int, train_freq: np.ndarray, cold_item_mask: np.ndarray | None = None) -> None:
        super().__init__()
        self.num_items = int(num_items)
        freq = np.asarray(train_freq, dtype=np.float32)[: self.num_items + 1].copy()
        freq[PAD] = 0.0
        if cold_item_mask is not None:
            # cold items have no training signal -> score 0, exactly like an
            # untrained ID embedding
            freq[np.asarray(cold_item_mask, dtype=bool)] = 0.0
        self.register_buffer("scores", torch.from_numpy(freq))
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
