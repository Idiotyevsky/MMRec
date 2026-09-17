"""Recall layer: candidate generation for the two-stage recommender.

A real short-video feed cannot score the whole catalogue for every request, so
the system is split into

    multi-channel recall  ->  candidate merge  ->  multimodal ranker  ->  rerank

This module defines the single interface every recall channel implements, and
the candidate container that is passed downstream.

Id convention
-------------
The recall layer works in **internal item id space** (``1..num_items``, ``0`` is
PAD).  That is the same space the ranker, the ItemCF index and the content index
use, so ids are converted from/to the raw MicroLens ids exactly once, at the
serving boundary (:mod:`src.serving.schemas`).

Every channel must guarantee:

* PAD (0) is never returned;
* items the user has already interacted with are never returned;
* results are ordered by descending score and carry a 1-based ``rank``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

PAD = 0


@dataclass(frozen=True)
class RecallCandidate:
    """One recalled item, produced by one channel."""

    item_id: int  # internal id, 1..num_items
    score: float
    source: str
    rank: int  # 1-based rank inside this channel's own list

    def as_dict(self) -> dict:
        return {"item_id": self.item_id, "score": float(self.score),
                "source": self.source, "rank": int(self.rank)}


@dataclass
class MergedCandidate:
    """A candidate after merging every channel that produced it."""

    item_id: int
    sources: list[dict] = field(default_factory=list)  # [{name, score, rank}, ...]
    merge_score: float = 0.0

    @property
    def source_names(self) -> list[str]:
        return [s["name"] for s in self.sources]

    @property
    def best_rank(self) -> int:
        return min((s["rank"] for s in self.sources), default=10**9)

    def as_dict(self) -> dict:
        return {"item_id": self.item_id, "sources": self.sources,
                "merge_score": float(self.merge_score)}


class RecallStrategy(ABC):
    """Base class for every recall channel."""

    name: str = "base"

    @abstractmethod
    def recall(self, user_id: int, history: Sequence[int], top_k: int) -> list[RecallCandidate]:
        """Return at most ``top_k`` candidates for one user.

        Parameters
        ----------
        user_id : internal user id (1..num_users)
        history : internal item ids, oldest first
        top_k   : maximum number of candidates to return
        """

    # ------------------------------------------------------------------
    def _filter(self, items: np.ndarray, scores: np.ndarray, seen: set[int],
                top_k: int) -> list[RecallCandidate]:
        """Apply the universal guarantees: no PAD, no seen items, no duplicates."""
        out: list[RecallCandidate] = []
        used: set[int] = set()
        for item, score in zip(items.tolist(), scores.tolist()):
            item = int(item)
            if item == PAD or item in seen or item in used:
                continue
            used.add(item)
            out.append(RecallCandidate(item_id=item, score=float(score),
                                       source=self.name, rank=len(out) + 1))
            if len(out) >= top_k:
                break
        return out

    @staticmethod
    def _seen(history: Iterable[int]) -> set[int]:
        return {int(i) for i in history if int(i) != PAD}

    def is_ready(self) -> tuple[bool, str]:
        """Whether the channel can serve requests, with a human reason."""
        return True, "ready"

