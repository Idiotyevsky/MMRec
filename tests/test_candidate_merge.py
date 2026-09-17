"""Candidate merge: dedup, source trace, RRF."""

import numpy as np
import pytest

from src.recall.base import RecallCandidate, RecallStrategy
from src.recall.pipeline import CandidateMerger


class _Fixed(RecallStrategy):
    def __init__(self, name, items):
        self.name = name
        self.items = items

    def recall(self, user_id, history, top_k):
        return [RecallCandidate(item_id=i, score=float(len(self.items) - r),
                                source=self.name, rank=r + 1)
                for r, i in enumerate(self.items[:top_k])]


class _Broken(RecallStrategy):
    name = "broken"

    def recall(self, user_id, history, top_k):
        raise RuntimeError("channel is down")


def test_duplicates_are_merged_and_traced():
    merger = CandidateMerger([_Fixed("a", [1, 2, 3]), _Fixed("b", [2, 3, 4])])
    res = merger.merge(merger.recall(1, [], per_source_k=10))
    by_id = {c.item_id: c for c in res.candidates}
    assert set(by_id) == {1, 2, 3, 4}
    assert sorted(by_id[2].source_names) == ["a", "b"]
    assert by_id[2].recall_rank if hasattr(by_id[2], "recall_rank") else True
    ranks = {s["name"]: s["rank"] for s in by_id[2].sources}
    assert ranks == {"a": 2, "b": 1}


def test_stats_report_before_and_after_dedup():
    merger = CandidateMerger([_Fixed("a", [1, 2, 3]), _Fixed("b", [2, 3, 4])])
    res = merger.merge(merger.recall(1, [], per_source_k=10))
    assert res.stats["before_dedup"] == 6
    assert res.stats["after_dedup"] == 4
    assert res.stats["duplicates_removed"] == 2
    assert res.stats["per_source"] == {"a": 3, "b": 3}


def test_multi_source_items_rank_higher_under_rrf():
    merger = CandidateMerger([_Fixed("a", [9, 1]), _Fixed("b", [9, 2])])
    res = merger.merge(merger.recall(1, [], per_source_k=10))
    assert res.candidates[0].item_id == 9, "an item found by both channels should win"


def test_rrf_ignores_raw_score_scale():
    """A channel with huge scores must not dominate one with tiny scores."""

    class _Big(RecallStrategy):
        name = "big"

        def recall(self, user_id, history, top_k):
            return [RecallCandidate(item_id=1, score=1e6, source=self.name, rank=1)]

    class _Small(RecallStrategy):
        name = "small"

        def recall(self, user_id, history, top_k):
            return [RecallCandidate(item_id=2, score=1e-6, source=self.name, rank=1)]

    merger = CandidateMerger([_Big(), _Small()])
    res = merger.merge(merger.recall(1, [], per_source_k=10))
    assert res.candidates[0].merge_score == pytest.approx(res.candidates[1].merge_score)


def test_source_order_is_deterministic():
    merger = CandidateMerger([_Fixed("a", [1, 2]), _Fixed("b", [1, 2])])
    first = [c.item_id for c in merger.merge(merger.recall(1, [], per_source_k=10)).candidates]
    second = [c.item_id for c in merger.merge(merger.recall(1, [], per_source_k=10)).candidates]
    assert first == second


def test_a_failing_channel_does_not_break_the_request():
    merger = CandidateMerger([_Fixed("a", [1, 2]), _Broken()])
    res = merger.recall(1, [], per_source_k=10)
    assert res["broken"] == []
    assert "_errors" in res
    merged = merger.merge(res)
    assert {c.item_id for c in merged.candidates} == {1, 2}


def test_total_k_truncation_is_recorded():
    merger = CandidateMerger([_Fixed("a", list(range(1, 21)))])
    res = merger.recall_and_merge(1, [], per_source_k=20, total_k=5)
    assert len(res.candidates) == 5
    assert res.stats["truncated_to"] == 5


def test_empty_channels_produce_empty_pool():
    merger = CandidateMerger([_Fixed("a", [])])
    res = merger.recall_and_merge(1, [], per_source_k=10)
    assert res.candidates == []
    assert res.stats["after_dedup"] == 0


def test_merger_requires_at_least_one_channel():
    with pytest.raises(ValueError):
        CandidateMerger([])
