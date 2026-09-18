"""Serving-side item metadata: training signal, popularity bucket, cold status.

Two *different* notions of "cold" live here and must never be conflated:

``is_simulated_cold``
    Controlled cold-start **benchmark** membership.  A fixed-seed subset of items
    had *every* training interaction removed by
    ``src/data/preprocess.py --cold-ratio``.  Only exists in the ``cold10``
    dataset, and is what ``results/tables/cold_start.csv`` reports.

``is_zero_train_signal``
    An item that simply has zero observed training interactions in this dataset.
    This is the serving-relevant notion: it is what a genuinely new video looks
    like to the model, and it exists in the base split too (376 items).

Everything is derived from **training statistics only**.  A leak here would be
visible to the user in the UI and would invalidate the cold-start story.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import ProcessedData
from .popularity import BUCKET_NAMES

PAD = 0


@dataclass(frozen=True)
class ItemMeta:
    item_id: int  # internal id
    train_interactions: int
    popularity_bucket: str  # head | middle | tail | simulated_cold
    is_simulated_cold: bool
    is_zero_train_signal: bool
    in_catalogue: bool

    @property
    def exploration_candidate(self) -> bool:
        """Whether this item qualifies for the serving exploration quota.

        Zero training signal is the condition the exposure policy cares about:
        the item has no collaborative evidence, so without reserved exposure it
        would never be shown.
        """
        return bool(self.is_zero_train_signal or self.is_simulated_cold)

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "train_interactions": int(self.train_interactions),
            "popularity_bucket": self.popularity_bucket,
            "is_cold": bool(self.is_simulated_cold),  # kept for API compatibility
            "is_simulated_cold": bool(self.is_simulated_cold),
            "is_zero_train_signal": bool(self.is_zero_train_signal),
            "exploration_candidate": bool(self.exploration_candidate),
            "in_catalogue": bool(self.in_catalogue),
        }


class ItemMetadata:
    """O(1) lookup of training-derived item statistics."""

    def __init__(self, data: ProcessedData) -> None:
        self.num_items = data.num_items
        self.train_freq = np.asarray(data.train_freq, dtype=np.int64)
        self.bucket = np.asarray(data.popularity_bucket, dtype=np.int8)
        self.is_simulated_cold = np.asarray(data.is_cold, dtype=bool)
        self.is_zero_train_signal = self.train_freq == 0
        self.is_zero_train_signal[PAD] = False
        self._bucket_rule = data.stats.get("popularity_buckets", {}).get("rule", "unknown")
        self.has_cold_split = bool(self.is_simulated_cold.any())

    # ------------------------------------------------------------------
    def of(self, item_id: int) -> ItemMeta:
        i = int(item_id)
        in_cat = PAD < i <= self.num_items
        if not in_cat:
            return ItemMeta(i, 0, "unknown", False, False, False)
        sim_cold = bool(self.is_simulated_cold[i])
        bucket = "simulated_cold" if sim_cold else BUCKET_NAMES.get(int(self.bucket[i]), "unknown")
        return ItemMeta(
            item_id=i,
            train_interactions=int(self.train_freq[i]),
            popularity_bucket=bucket,
            is_simulated_cold=sim_cold,
            is_zero_train_signal=bool(self.is_zero_train_signal[i]),
            in_catalogue=True,
        )

    def of_many(self, item_ids) -> list[ItemMeta]:
        return [self.of(i) for i in item_ids]

    def as_dict(self, item_id: int) -> dict:
        return self.of(item_id).as_dict()

    # ------------------------------------------------------------------
    def exploration_flags(self) -> np.ndarray:
        """Boolean array over internal ids: qualifies for the exploration quota."""
        flags = self.is_zero_train_signal | self.is_simulated_cold
        flags[PAD] = False
        return flags

    def summary(self) -> dict:
        counts = {"head": 0, "middle": 0, "tail": 0, "simulated_cold": 0}
        for meta in self.of_many(range(1, self.num_items + 1)):
            counts[meta.popularity_bucket] = counts.get(meta.popularity_bucket, 0) + 1
        return {
            "num_items": self.num_items,
            "bucket_rule": self._bucket_rule,
            "bucket_counts": counts,
            "has_cold_split": self.has_cold_split,
            "cold_items": int(self.is_simulated_cold.sum()),
            "zero_train_frequency_items": int(self.is_zero_train_signal[1:].sum()),
            "exploration_candidates": int(self.exploration_flags()[1:].sum()),
        }
