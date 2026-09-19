"""Multi-channel recall + candidate merge.

    popular  ─┐
    itemcf   ─┼─►  merge / dedup / source trace  ─►  ranked candidate pool
    semantic ─┘

The merge step must not throw information away: the Recommendation Inspector
shows *which* channel produced each candidate and at what rank, so every
contributing channel is recorded on the merged candidate.

Ranking the pool is done by **reciprocal rank fusion** (RRF)::

    merge_score(j) = Σ_channels 1 / (k + rank_channel(j)),   k = 60

RRF is used instead of summing raw scores because the three channels produce
incomparable quantities — popularity counts (0..546), ItemCF similarity sums
(0..10) and cosine similarities (-1..1).  RRF only depends on ranks, so no
channel can dominate the pool just because its numbers are larger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .base import MergedCandidate, RecallCandidate, RecallStrategy

DEFAULT_RRF_K = 60


@dataclass
class MergeResult:
    candidates: list[MergedCandidate]
    stats: dict = field(default_factory=dict)

    def item_ids(self) -> list[int]:
        return [c.item_id for c in self.candidates]

    def as_dict(self) -> dict:
        return {"stats": self.stats,
                "candidates": [c.as_dict() for c in self.candidates]}


class CandidateMerger:
    def __init__(
        self,
        strategies: Sequence[RecallStrategy],
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if not strategies:
            raise ValueError("at least one recall strategy is required")
        self.strategies = list(strategies)
        self.rrf_k = int(rrf_k)

    # ------------------------------------------------------------------
    @property
    def source_names(self) -> list[str]:
        return [s.name for s in self.strategies]

    def recall(
        self,
        user_id: int,
        history: Sequence[int],
        per_source_k: int = 200,
        sources: Sequence[str] | None = None,
        trace: dict | None = None,
    ) -> dict[str, list[RecallCandidate]]:
        """Run every requested channel independently.

        When ``trace`` is supplied it is filled with per-channel latency and
        errors, so the UI can show what each channel actually did for this
        request instead of only the merged result.
        """
        import time

        wanted = set(sources) if sources else None
        out: dict[str, list[RecallCandidate]] = {}
        timings: dict[str, float] = {}
        for s in self.strategies:
            if wanted is not None and s.name not in wanted:
                continue
            t0 = time.perf_counter()
            try:
                out[s.name] = s.recall(user_id, history, per_source_k)
            except Exception as exc:  # a dead channel must not kill the request
                out[s.name] = []
                out.setdefault("_errors", []).append(f"{s.name}: {exc}")  # type: ignore[arg-type]
            timings[s.name] = (time.perf_counter() - t0) * 1000
        if trace is not None:
            trace["per_source_latency_ms"] = {k: round(v, 3) for k, v in timings.items()}
        return out

    # ------------------------------------------------------------------
    def merge(self, channel_results: dict[str, list[RecallCandidate]]) -> MergeResult:
        per_source = {k: v for k, v in channel_results.items() if not k.startswith("_")}
        before = sum(len(v) for v in per_source.values())

        merged: dict[int, MergedCandidate] = {}
        for name, cands in per_source.items():
            if not cands:
                continue
            scores = np.asarray([c.score for c in cands], dtype=np.float64)
            lo, hi = float(scores.min()), float(scores.max())
            span = hi - lo
            for c in cands:
                norm = 1.0 if span <= 1e-12 else (c.score - lo) / span
                entry = merged.get(c.item_id)
                if entry is None:
                    entry = MergedCandidate(item_id=c.item_id)
                    merged[c.item_id] = entry
                trace = {
                    "name": name,
                    "score": float(c.score),
                    "rank": int(c.rank),
                    "norm_score": float(norm),
                }
                if c.extra:
                    trace["extra"] = c.extra
                entry.sources.append(trace)
                entry.merge_score += 1.0 / (self.rrf_k + c.rank)

        ordered = sorted(merged.values(), key=lambda c: (-c.merge_score, c.item_id))
        # how many items each channel is the *only* source for: the channel's
        # irreplaceable contribution to the pool, which is the honest way to read
        # a multi-channel recall setup
        unique_contribution = {name: 0 for name in per_source}
        for c in ordered:
            if len(c.sources) == 1:
                unique_contribution[c.sources[0]["name"]] += 1
        stats = {
            "per_source": {k: len(v) for k, v in per_source.items()},
            "unique_contribution": unique_contribution,
            "before_dedup": int(before),
            "after_dedup": int(len(ordered)),
            "duplicates_removed": int(before - len(ordered)),
            "rrf_k": self.rrf_k,
        }
        if "_errors" in channel_results:
            stats["errors"] = channel_results["_errors"]  # type: ignore[assignment]
        return MergeResult(candidates=ordered, stats=stats)

    # ------------------------------------------------------------------
    def recall_and_merge(
        self,
        user_id: int,
        history: Sequence[int],
        per_source_k: int = 200,
        total_k: int | None = None,
        sources: Sequence[str] | None = None,
        trace: dict | None = None,
    ) -> MergeResult:
        result = self.merge(self.recall(user_id, history, per_source_k, sources, trace=trace))
        if total_k is not None and len(result.candidates) > total_k:
            result.candidates = result.candidates[:total_k]
            result.stats["truncated_to"] = int(total_k)
        result.stats["candidates_ranked"] = len(result.candidates)
        coverage: dict[str, int] = {}
        for c in result.candidates:
            for name in c.source_names:
                coverage[name] = coverage.get(name, 0) + 1
        result.stats["source_coverage_in_pool"] = coverage
        if trace is not None:
            result.stats["per_source_latency_ms"] = trace.get("per_source_latency_ms", {})
        return result

    def is_ready(self) -> dict[str, tuple[bool, str]]:
        return {s.name: s.is_ready() for s in self.strategies}
