"""Serving-side item metadata: cold / warm status, popularity bucket, counts.

Every field is derived from **training statistics only**.  This is what the API
and the frontend use to explain a recommendation ("this is a cold item with zero
training interactions"), so a leak here would be visible to the user and would
invalidate the cold-start story.
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
    popularity_bucket: str  # head | middle | tail | cold
    is_cold: bool
    in_catalogue: bool

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "train_interactions": int(self.train_interactions),
            "popularity_bucket": self.popularity_bucket,
            "is_cold": bool(self.is_cold),
            "in_catalogue": bool(self.in_catalogue),
        }


class ItemMetadata:
    """O(1) lookup of training-derived item statistics."""

    def __init__(self, data: ProcessedData) -> None:
        self.num_items = data.num_items
        self.train_freq = np.asarray(data.train_freq, dtype=np.int64)
        self.bucket = np.asarray(data.popularity_bucket, dtype=np.int8)
        self.is_cold = np.asarray(data.is_cold, dtype=bool)
        self._bucket_rule = data.stats.get("popularity_buckets", {}).get("rule", "unknown")
        self.has_cold_split = bool(self.is_cold.any())

    # ------------------------------------------------------------------
    def of(self, item_id: int) -> ItemMeta:
        i = int(item_id)
        in_cat = PAD < i <= self.num_items
        if not in_cat:
            return ItemMeta(i, 0, "unknown", False, False)
        cold = bool(self.is_cold[i])
        bucket = "cold" if cold else BUCKET_NAMES.get(int(self.bucket[i]), "unknown")
        return ItemMeta(i, int(self.train_freq[i]), bucket, cold, True)

    def of_many(self, item_ids) -> list[ItemMeta]:
        return [self.of(i) for i in item_ids]

    def as_dict(self, item_id: int) -> dict:
        return self.of(item_id).as_dict()

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        counts = {"head": 0, "middle": 0, "tail": 0, "cold": 0}
        for meta in self.of_many(range(1, self.num_items + 1)):
            counts[meta.popularity_bucket] = counts.get(meta.popularity_bucket, 0) + 1
        return {
            "num_items": self.num_items,
            "bucket_rule": self._bucket_rule,
            "bucket_counts": counts,
            "has_cold_split": self.has_cold_split,
            "cold_items": int(self.is_cold.sum()),
            "zero_train_frequency_items": int((self.train_freq[1:] == 0).sum()),
        }
