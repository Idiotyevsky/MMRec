"""Popularity recall — the always-available fallback channel.

Scores items by their **training-only** interaction count.  This channel is not
meant to be accurate; it exists because every production recommender needs a
channel that can answer for a brand-new user with no history and for items that
no collaborative channel has ever seen.

Leakage: the counts come from ``ProcessedData.train_freq``, which is computed
from training interactions only.  Validation and test interactions never enter.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import RecallCandidate, RecallStrategy


class PopularRecall(RecallStrategy):
    name = "popular"

    def __init__(self, train_freq: np.ndarray, num_items: int | None = None,
                 cold_item_mask: np.ndarray | None = None) -> None:
        freq = np.asarray(train_freq, dtype=np.float64).copy()
        self.num_items = int(num_items if num_items is not None else freq.shape[0] - 1)
        freq = freq[: self.num_items + 1]
        freq[0] = 0.0  # PAD can never be recalled
        if cold_item_mask is not None:
            # cold items have zero training interactions by protocol; keep them
            # at zero so this channel cannot smuggle them in
            freq[np.asarray(cold_item_mask, dtype=bool)] = 0.0
        self.freq = freq
        # descending order, PAD excluded, computed once
        order = np.argsort(-self.freq, kind="stable")
        self.order = order[self.freq[order] > 0]

    # ------------------------------------------------------------------
    def recall(self, user_id: int, history: Sequence[int], top_k: int) -> list[RecallCandidate]:
        seen = self._seen(history)
        # Vectorised skip of the user's history: masking the global order once
        # is O(num_items) in numpy instead of a Python loop per candidate.
        blocked = np.zeros(self.num_items + 1, dtype=bool)
        if seen:
            blocked[np.fromiter(seen, dtype=np.int64, count=len(seen))] = True
        picked = self.order[~blocked[self.order]][:top_k]
        return self._filter(picked, self.freq[picked], seen, top_k)

    # ------------------------------------------------------------------
    def is_ready(self) -> tuple[bool, str]:
        return bool(self.order.size), f"{self.order.size} items with non-zero training frequency"
