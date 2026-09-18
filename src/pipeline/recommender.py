"""Two-stage recommender: multi-channel recall -> merge -> rank -> rerank.

This is the serving architecture.  It is deliberately **separate** from the
strict offline evaluation:

* ``scripts/train.py`` / ``src/evaluation`` answer "how good is the model?"
  with a full-catalogue ranking over all 19 738 items;
* this pipeline answers "how would the system serve a request?" with a few
  hundred recalled candidates.

The two report different numbers and are never mixed in the same table.  See
``docs/evaluation_protocol.md``.

Id convention: everything in this module is **internal** item ids
(``1..num_items``); the serving layer converts to raw MicroLens ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..data.dataset import ProcessedData
from ..data.metadata import ItemMetadata
from ..recall.pipeline import CandidateMerger
from ..rerank.simple import RerankConfig, rerank
from .rankers import RankerRegistry

PAD = 0


@dataclass
class RecommendResult:
    user_id: int
    history: list[int]
    history_mode: str
    ranker: str
    recall_k: int
    final_k: int
    recall_stats: dict
    recommendations: list[dict]
    rerank: dict
    latency_ms: dict = field(default_factory=dict)
    recall_channels: list[dict] = field(default_factory=list)
    target_item: int | None = None
    target_hit: bool | None = None

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "history": self.history,
            "history_mode": self.history_mode,
            "ranker": self.ranker,
            "recall_k": self.recall_k,
            "final_k": self.final_k,
            "recall": self.recall_stats,
            "rerank": self.rerank,
            "recall_channels": self.recall_channels,
            "latency_ms": self.latency_ms,
            "target_item": self.target_item,
            "target_hit": self.target_hit,
            "recommendations": self.recommendations,
        }


class TwoStageRecommender:
    def __init__(
        self,
        data: ProcessedData,
        merger: CandidateMerger,
        registry: RankerRegistry,
        default_ranker: str = "mm_concat",
        history_mode: str = "test",
    ) -> None:
        if history_mode not in ("test", "full"):
            raise ValueError(f"history_mode must be 'test' or 'full', got {history_mode!r}")
        self.data = data
        self.merger = merger
        self.registry = registry
        self.default_ranker = default_ranker
        self.history_mode = history_mode
        self.metadata = ItemMetadata(data)

    # ------------------------------------------------------------------
    def history(self, user_id: int) -> list[int]:
        """History for a **1-based** internal user id.

        ``ProcessedData`` indexes users 0-based; the serving layer uses 1-based
        ids (matching the item convention), so the conversion happens here and
        nowhere else.  Getting this wrong silently serves every user the wrong
        history, which is exactly the bug this docstring exists to prevent.
        """
        idx = self._index(user_id)
        hist = (self.data.test_history(idx) if self.history_mode == "test"
                else self.data.full_history(idx))
        return [int(i) for i in hist]

    def _index(self, user_id: int) -> int:
        u = int(user_id)
        if not (1 <= u <= self.data.num_users):
            raise IndexError(f"user {u} out of range 1..{self.data.num_users}")
        return u - 1

    def target(self, user_id: int) -> int | None:
        """Ground-truth next item, when the history mode makes one available."""
        if self.history_mode != "test":
            return None
        return int(self.data.test_target[self._index(user_id)])

    # ------------------------------------------------------------------
    def recall(
        self,
        user_id: int,
        recall_k: int = 200,
        max_candidates: int | None = None,
        sources: Sequence[str] | None = None,
    ):
        """Run recall and merge for one user.

        ``recall_k`` is the per-channel budget; ``max_candidates`` optionally
        caps the merged pool *before* ranking.  Leaving it at ``None`` means no
        candidate is discarded between recall and ranking, which is the safer
        default — truncating the union measurably drops items that only one
        channel found.
        """
        hist = self.history(user_id)
        return self.merger.recall_and_merge(
            user_id=int(user_id), history=hist,
            per_source_k=recall_k, total_k=max_candidates, sources=sources,
        )

    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        recall_k: int = 200,
        final_k: int = 20,
        ranker: str | None = None,
        cold_exploration: bool = False,
        cold_quota: int = 2,
        max_candidates: int | None = None,
        sources: Sequence[str] | None = None,
    ) -> RecommendResult:
        import time

        t0 = time.perf_counter()
        ranker_name = ranker or self.default_ranker
        hist = self.history(user_id)

        recall_trace: dict = {}
        merged = self.merger.recall_and_merge(
            user_id=int(user_id), history=hist,
            per_source_k=recall_k, total_k=max_candidates, sources=sources,
            trace=recall_trace,
        )
        t_recall = time.perf_counter()

        candidate_ids = merged.item_ids()
        model = self.registry.get(ranker_name)
        scores = (model.score_candidates(hist, candidate_ids)
                  if candidate_ids else np.zeros(0, dtype=np.float32))
        t_rank = time.perf_counter()

        entries = self._entries(hist, candidate_ids, scores, merged, ranker_name)
        cfg = RerankConfig(exploration=cold_exploration, exploration_quota=cold_quota)
        rr = rerank(entries, cfg, final_k=final_k, history=set(hist))
        t_end = time.perf_counter()

        target = self.target(user_id)
        top_ids = [int(e["item_id"]) for e in rr.items]
        return RecommendResult(
            user_id=int(user_id), history=hist, history_mode=self.history_mode,
            ranker=ranker_name, recall_k=recall_k, final_k=final_k,
            recall_stats=merged.stats, recommendations=rr.items, rerank=rr.applied,
            recall_channels=[
                {
                    "name": name,
                    "recalled": merged.stats["per_source"].get(name, 0),
                    "in_pool": merged.stats["source_coverage_in_pool"].get(name, 0),
                    "unique_contribution": merged.stats["unique_contribution"].get(name, 0),
                    "latency_ms": merged.stats.get("per_source_latency_ms", {}).get(name),
                }
                for name in self.merger.source_names
            ],
            latency_ms={
                "recall": round((t_recall - t0) * 1000, 2),
                "rank": round((t_rank - t_recall) * 1000, 2),
                "rerank": round((t_end - t_rank) * 1000, 2),
                "total": round((t_end - t0) * 1000, 2),
            },
            target_item=target,
            target_hit=(target in top_ids) if target is not None else None,
        )

    # ------------------------------------------------------------------
    def _entries(self, hist, candidate_ids, scores, merged, ranker_name: str) -> list[dict]:
        by_id = {c.item_id: c for c in merged.candidates}
        entries: list[dict] = []
        for item, score in zip(candidate_ids, scores.tolist()):
            cand = by_id[int(item)]
            meta = self.metadata.of(int(item))
            entries.append({
                "item_id": int(item),
                "ranking_score": float(score),
                "ranker": ranker_name,
                "merge_score": float(cand.merge_score),
                "sources": cand.source_names,
                "source_trace": cand.sources,
                "recall_rank": {s["name"]: s["rank"] for s in cand.sources},
                "is_cold": meta.is_simulated_cold,
                "is_simulated_cold": meta.is_simulated_cold,
                "is_zero_train_signal": meta.is_zero_train_signal,
                "exploration_candidate": meta.exploration_candidate,
                "popularity_bucket": meta.popularity_bucket,
                "train_interactions": meta.train_interactions,
            })
        entries.sort(key=lambda e: -e["ranking_score"])
        return entries

    # ------------------------------------------------------------------
    def inspect(
        self,
        user_id: int,
        recall_k: int = 200,
        top_n: int = 30,
        ranker: str | None = None,
        compare: str | None = "sasrec",
        cold_exploration: bool = False,
        max_candidates: int | None = None,
    ) -> dict:
        """Full trace of one request, for the Recommendation Inspector page."""
        import time

        t0 = time.perf_counter()
        ranker_name = ranker or self.default_ranker
        hist = self.history(user_id)
        recall_trace: dict = {}
        merged = self.merger.recall_and_merge(
            user_id=int(user_id), history=hist,
            per_source_k=recall_k, total_k=max_candidates, trace=recall_trace,
        )
        candidate_ids = merged.item_ids()
        model = self.registry.get(ranker_name)
        scores = (model.score_candidates(hist, candidate_ids)
                  if candidate_ids else np.zeros(0, dtype=np.float32))

        entries = self._entries(hist, candidate_ids, scores, merged, ranker_name)

        # optional baseline comparison: same candidates, ID-only model
        compare_scores = None
        if compare and compare != ranker_name and compare in self.registry.names:
            base = self.registry.get(compare)
            compare_scores = (base.score_candidates(hist, candidate_ids)
                              if candidate_ids else np.zeros(0, dtype=np.float32))
            by_id = {int(i): float(s) for i, s in zip(candidate_ids, compare_scores.tolist())}
            for e in entries:
                other = by_id.get(int(e["item_id"]))
                e["compare_ranker"] = compare
                e["compare_score"] = other
                e["score_delta"] = None if other is None else float(e["ranking_score"] - other)

        # what the baseline alone would have ranked in the top-N
        baseline_top: list[dict] = []
        if compare_scores is not None:
            order = np.argsort(-np.asarray(compare_scores, dtype=np.float64))[:top_n]
            for pos, j in enumerate(order.tolist(), start=1):
                item = int(candidate_ids[j])
                baseline_top.append({
                    "item_id": item, "baseline_rank": pos,
                    "baseline_score": float(compare_scores[j]),
                    "multimodal_score": float(scores[j]),
                    "delta": float(scores[j] - compare_scores[j]),
                    **self.metadata.as_dict(item),
                })

        target = self.target(user_id)
        merger_names = list(self.merger.source_names)
        top_entries = entries[:top_n]
        moved_up, moved_down = [], []
        if compare_scores is not None:
            # NB: `e["compare_score"] or -1e30` would be wrong -- 0.0 is a legal
            # score and would be treated as missing.  Compare against None.
            def _base_key(e: dict) -> float:
                v = e.get("compare_score")
                return -1e30 if v is None else -float(v)

            ranked_by_base = sorted(entries, key=_base_key)
            base_pos = {int(e["item_id"]): i + 1 for i, e in enumerate(ranked_by_base)}
            # every candidate gets its baseline position and rank delta, so the
            # UI can show movement on the served cards, not only on the extremes
            for i, e in enumerate(entries, start=1):
                pos = base_pos[int(e["item_id"])]
                e["baseline_position"] = pos
                e["rank_delta"] = pos - i  # positive = moved up
            for i, e in enumerate(entries, start=1):
                delta = base_pos[int(e["item_id"])] - i
                if delta >= 5:
                    moved_up.append({**e, "position_delta": delta})
                elif delta <= -5:
                    moved_down.append({**e, "position_delta": -delta})
            # lead with the biggest movers: that is the interesting part of the
            # comparison, and the UI only shows the first few
            moved_up.sort(key=lambda e: -e["position_delta"])
            moved_down.sort(key=lambda e: -e["position_delta"])

        rr = rerank(entries, RerankConfig(exploration=cold_exploration),
                    final_k=20, history=set(hist))

        merged.stats["exploration_in_pool"] = sum(
            1 for e in entries if e.get("exploration_candidate"))

        return {
            "user_id": int(user_id),
            "history": hist,
            "history_mode": self.history_mode,
            "history_length": len(hist),
            "target_item": target,
            "ranker": ranker_name,
            "compare_ranker": compare,
            "recall_summary": {**merged.stats, "candidates_ranked": len(candidate_ids)},
            "recall_channels": [
                {
                    "name": name,
                    "recalled": merged.stats["per_source"].get(name, 0),
                    "in_pool": merged.stats["source_coverage_in_pool"].get(name, 0),
                    "unique_contribution": merged.stats["unique_contribution"].get(name, 0),
                    "latency_ms": merged.stats.get("per_source_latency_ms", {}).get(name),
                    "target_hit": target in {c.item_id for c in merged.candidates
                                             if name in c.source_names},
                }
                for name in merger_names
            ],
            "recall_candidates": [c.as_dict() for c in merged.candidates[:top_n]],
            "top_candidates": top_entries,
            "baseline_top": baseline_top,
            "moved_up_by_multimodal": moved_up[:10],
            "moved_down_by_multimodal": moved_down[:10],
            "final_top_k": rr.items,
            "rerank": rr.applied,
            "latency_ms": {"total": round((time.perf_counter() - t0) * 1000, 2)},
        }

    # ------------------------------------------------------------------
    def system_info(self) -> dict:
        return {
            "dataset": "MicroLens-100K",
            "users": int(self.data.num_users),
            "items": int(self.data.num_items),
            "interactions": int(self.data.stats.get("num_interactions", 0)),
            "default_ranker": self.default_ranker,
            "rankers": self.registry.names,
            "recall_sources": self.merger.source_names,
            "history_mode": self.history_mode,
            "item_metadata": self.metadata.summary(),
            "recall_readiness": self.merger.is_ready(),
        }
