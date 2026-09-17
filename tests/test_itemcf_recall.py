"""ItemCF recall: index construction and online aggregation."""

import numpy as np
import pytest

from src.data.dataset import ProcessedData
from src.recall.itemcf import ItemCFRecall, build_itemcf_index, load_itemcf_index


def test_index_shapes(synthetic_data: ProcessedData):
    idx = build_itemcf_index(
        synthetic_data.flat_items, synthetic_data.user_offsets, synthetic_data.train_len,
        synthetic_data.num_items, top_m=10, chunk_size=32, verbose=False,
    )
    assert idx["neighbors"].shape == (synthetic_data.num_items + 1, 10)
    assert idx["sims"].shape == (synthetic_data.num_items + 1, 10)
    assert (idx["sims"] >= 0).all()
    assert (idx["sims"] <= 1.0 + 1e-5).all()


def test_self_is_never_its_own_neighbour(synthetic_data: ProcessedData):
    idx = build_itemcf_index(
        synthetic_data.flat_items, synthetic_data.user_offsets, synthetic_data.train_len,
        synthetic_data.num_items, top_m=10, chunk_size=32, verbose=False,
    )
    nbr, sims = idx["neighbors"], idx["sims"]
    for item in range(1, synthetic_data.num_items + 1):
        assert item not in nbr[item][sims[item] > 0]


def test_pad_is_never_a_neighbour(synthetic_data: ProcessedData):
    idx = build_itemcf_index(
        synthetic_data.flat_items, synthetic_data.user_offsets, synthetic_data.train_len,
        synthetic_data.num_items, top_m=10, chunk_size=32, verbose=False,
    )
    assert (idx["neighbors"][idx["sims"] > 0] > 0).all()


def test_similarity_is_symmetric_in_construction(synthetic_data: ProcessedData):
    """cos(i,j) must equal cos(j,i) before the top-M truncation."""
    idx = build_itemcf_index(
        synthetic_data.flat_items, synthetic_data.user_offsets, synthetic_data.train_len,
        synthetic_data.num_items, top_m=40, chunk_size=32, verbose=False,
    )
    nbr, sims = idx["neighbors"], idx["sims"]
    checked = 0
    for item in range(1, min(15, synthetic_data.num_items + 1)):
        for n, s in zip(nbr[item].tolist(), sims[item].tolist()):
            if n <= 0 or s <= 0:
                continue
            back = dict(zip(nbr[n].tolist(), sims[n].tolist()))
            if item in back:
                assert back[item] == pytest.approx(s, abs=1e-5)
                checked += 1
    assert checked > 0, "no symmetric pair found; the test is vacuous"


def test_recall_excludes_pad_and_seen(synthetic_data: ProcessedData, synthetic_recall_artifacts):
    idx = load_itemcf_index(synthetic_recall_artifacts["itemcf"])
    r = ItemCFRecall(idx["neighbors"], idx["sims"], synthetic_data.num_items)
    for u in range(10):
        hist = [int(i) for i in synthetic_data.test_history(u)]
        out = r.recall(u, hist, top_k=20)
        ids = [c.item_id for c in out]
        assert 0 not in ids
        assert not (set(ids) & set(hist))
        assert len(ids) == len(set(ids))


def test_recall_is_ordered_by_score(synthetic_data: ProcessedData, synthetic_recall_artifacts):
    idx = load_itemcf_index(synthetic_recall_artifacts["itemcf"])
    r = ItemCFRecall(idx["neighbors"], idx["sims"], synthetic_data.num_items)
    hist = [int(i) for i in synthetic_data.test_history(0)]
    out = r.recall(0, hist, top_k=20)
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_empty_history_returns_nothing(synthetic_data: ProcessedData, synthetic_recall_artifacts):
    idx = load_itemcf_index(synthetic_recall_artifacts["itemcf"])
    r = ItemCFRecall(idx["neighbors"], idx["sims"], synthetic_data.num_items)
    assert r.recall(0, [], top_k=10) == []


def test_recency_decay_prioritises_recent_items(synthetic_data: ProcessedData, synthetic_recall_artifacts):
    idx = load_itemcf_index(synthetic_recall_artifacts["itemcf"])
    r = ItemCFRecall(idx["neighbors"], idx["sims"], synthetic_data.num_items, decay=0.1)
    a = r.score_items([1, 2, 3])
    b = r.score_items([3, 2, 1])
    assert not np.allclose(a, b), "recency weighting has no effect"


def test_top_k_is_respected(synthetic_data: ProcessedData, synthetic_recall_artifacts):
    idx = load_itemcf_index(synthetic_recall_artifacts["itemcf"])
    r = ItemCFRecall(idx["neighbors"], idx["sims"], synthetic_data.num_items)
    hist = [int(i) for i in synthetic_data.test_history(0)]
    assert len(r.recall(0, hist, top_k=5)) <= 5
