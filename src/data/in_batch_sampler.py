"""Shared in-batch negatives built only from training positives.

One last non-PAD target per row bounds the pool by batch size (rather than
batch_size * sequence_length). Duplicate item IDs are represented once.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InBatchNegatives:
    item_ids: torch.Tensor  # (K,) unique IDs
    valid_mask: torch.Tensor  # (B, K), True means an allowed negative


class InBatchNegativeSampler:
    """Mask known training interactions; never inspect validation/test labels.

    ``training_interaction_keys`` must come from
    ``build_training_interaction_keys``, not the all-split interaction builder.
    Membership checks are vectorized on the target tensor's device.
    """

    def __init__(self, num_items: int, training_interaction_keys):
        if num_items < 1:
            raise ValueError("num_items must be positive")
        self.num_items = int(num_items)
        self._keys = torch.as_tensor(training_interaction_keys, dtype=torch.long).sort().values

    @torch.no_grad()
    def sample(self, user_ids: torch.Tensor, target: torch.Tensor) -> InBatchNegatives:
        if target.ndim != 2 or user_ids.ndim != 1 or len(user_ids) != len(target):
            raise ValueError("expected target (B,L) and user_ids (B,)")
        if target.dtype != torch.long or user_ids.dtype != torch.long:
            raise ValueError("item and user IDs must be int64")
        if (target < 0).any() or (target > self.num_items).any() or (user_ids < 0).any():
            raise ValueError("item or user ID out of range")
        batch, length = target.shape
        if not batch or not length:
            return InBatchNegatives(target.new_empty(0),
                torch.empty((batch, 0), dtype=torch.bool, device=target.device))
        positions = torch.arange(length, device=target.device).expand_as(target)
        last = positions.masked_fill(target == 0, -1).max(dim=1).values
        positives = target.gather(1, last.clamp_min(0)[:, None]).squeeze(1)
        pool = torch.unique(positives[(last >= 0) & (positives > 0)], sorted=True)
        # Explicitly exclude all targets of this row, even with an empty key
        # table. The complete train-prefix keys also cover truncated histories.
        valid = ~(target[:, :, None] == pool[None, None, :]).any(dim=1)
        valid &= (last >= 0)[:, None]
        if self._keys.numel() and pool.numel():
            self._keys = self._keys.to(target.device)
            queries = user_ids.to(target.device)[:, None] * self.num_items + pool[None, :]
            indices = torch.searchsorted(self._keys, queries.reshape(-1))
            known = self._keys[indices.clamp_max(len(self._keys)-1)] == queries.reshape(-1)
            valid &= ~known.reshape(batch, -1)
        # Drop columns unusable by every user, including the singleton-batch
        # case. A row with no valid negatives simply adds no denominator terms.
        keep = valid.any(dim=0)
        return InBatchNegatives(pool[keep], valid[:, keep])
