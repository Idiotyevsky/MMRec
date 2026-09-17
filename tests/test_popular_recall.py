"""Popular recall channel."""

import numpy as np
import pytest

from src.recall.popular import PopularRecall


@pytest.fixture()
def freq():
    f = np.array([0, 10, 5, 0, 20, 1, 7], dtype=np.int64)  # index 0 = PAD
    return f


def test_returns_items_in_descending_frequency(freq):
    r = PopularRecall(freq, num_items=6)
    out = r.recall(user_id=1, history=[], top_k=4)
    assert [c.item_id for c in out] == [4, 1, 6, 2]
    assert [c.score for c in out] == [20, 10, 7, 5]
    assert [c.rank for c in out] == [1, 2, 3, 4]


def test_pad_is_never_returned(freq):
    r = PopularRecall(freq, num_items=6)
    out = r.recall(1, [], top_k=100)
    assert 0 not in [c.item_id for c in out]


def test_items_with_zero_frequency_are_never_returned(freq):
    r = PopularRecall(freq, num_items=6)
    out = r.recall(1, [], top_k=100)
    assert set(c.item_id for c in out) == {1, 2, 4, 5, 6}


def test_seen_items_are_excluded(freq):
    r = PopularRecall(freq, num_items=6)
    out = r.recall(1, [4, 1], top_k=10)
    ids = [c.item_id for c in out]
    assert 4 not in ids and 1 not in ids
    assert ids == [6, 2, 5]


def test_history_of_only_top_items_still_returns_something(freq):
    r = PopularRecall(freq, num_items=6)
    out = r.recall(1, [4, 1, 6], top_k=10)
    assert [c.item_id for c in out] == [2, 5]


def test_source_name_is_recorded(freq):
    r = PopularRecall(freq, num_items=6)
    assert all(c.source == "popular" for c in r.recall(1, [], top_k=3))


def test_top_k_is_respected(freq):
    r = PopularRecall(freq, num_items=6)
    assert len(r.recall(1, [], top_k=2)) == 2


def test_cold_items_are_never_recalled_by_popularity(freq):
    cold = np.zeros(7, dtype=bool)
    cold[4] = True  # the most popular item is cold -> must not appear
    r = PopularRecall(freq, num_items=6, cold_item_mask=cold)
    ids = [c.item_id for c in r.recall(1, [], top_k=10)]
    assert 4 not in ids


def test_is_ready_reports_state(freq):
    ok, why = PopularRecall(freq, num_items=6).is_ready()
    assert ok and "5 items" in why
