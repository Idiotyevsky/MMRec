"""Swing scores, train-only graph construction and recall contract."""
import itertools

import numpy as np
import pytest

from src.recall.swing import (SwingRecall, build_swing_index, load_swing_recall,
                              save_swing_index)


def _build(histories, **kwargs):
    lengths = np.asarray([len(h) for h in histories])
    return build_swing_index(np.concatenate(histories), np.r_[0, lengths.cumsum()],
                             lengths, 8, top_m=8, verbose=False, **kwargs)


def _dense(index):
    out = np.zeros((9, 9))
    for i in range(9):
        out[i, index["neighbors"][i]] = index["sims"][i]
    return out


def test_exact_matches_independent_item_pair_reference():
    histories = [[1, 2, 3], [1, 2, 3, 4], [1, 2, 5], [2, 4, 5], [6]]
    sets = list(map(set, histories))
    expected = np.zeros((9, 9))
    for i, j in itertools.combinations(range(1, 9), 2):
        common_users = [u for u, h in enumerate(sets) if i in h and j in h]
        for u, v in itertools.combinations(common_users, 2):
            expected[i, j] += ((5+len(sets[u]))**(-.3) * (5+len(sets[v]))**(-.3)
                               / (1+len(sets[u] & sets[v])))
        expected[j, i] = expected[i, j]
    result = _dense(_build(histories, chunk_size=2))
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert result[1, 2] > 0
    assert not result[6].any()  # one user cannot produce a Swing pair


def test_pair_overlap_penalty():
    short = _dense(_build([[1, 2], [1, 2]], beta=0))
    long = _dense(_build([[1, 2, 3], [1, 2, 3]], beta=0))
    assert short[1, 2] == pytest.approx(1/3)
    assert long[1, 2] == pytest.approx(1/4)


def test_duplicates_and_pad_do_not_change_binary_graph():
    a = _build([[1, 2], [1, 2, 3]])
    b = _build([[0, 1, 1, 2], [1, 2, 3, 0, 3]])
    np.testing.assert_array_equal(a["neighbors"], b["neighbors"])
    np.testing.assert_array_equal(a["sims"], b["sims"])


def test_no_validation_test_leakage():
    flat = np.asarray([1, 2, 7, 8, 1, 2, 7, 8])
    kwargs = dict(user_offsets=np.array([0, 4, 8]), train_len=np.array([2, 2]),
                  num_items=8, top_m=8, verbose=False)
    a = build_swing_index(flat, **kwargs)
    flat[[2, 3, 6, 7]] = [3, 4, 5, 6]
    b = build_swing_index(flat, **kwargs)
    for key in ("neighbors", "sims", "freq"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not a["sims"][7:].any()


def test_recall_contract_and_determinism():
    a = _build([[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]], chunk_size=1)
    b = _build([[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 4]], chunk_size=3)
    np.testing.assert_array_equal(a["neighbors"], b["neighbors"])
    np.testing.assert_allclose(a["sims"], b["sims"])
    recall = SwingRecall(a["neighbors"], a["sims"], 8)
    result = recall.recall(0, [0, 1, 2], 2)
    assert [r.item_id for r in result] == [3, 4]
    assert [r.rank for r in result] == [1, 2]
    assert all(r.source == "swing" for r in result)
    assert result[0].score >= result[1].score > 0
    assert recall.recall(0, [], 5) == []
    assert recall.recall(0, [1], 0) == []
    assert len(recall.recall(0, [1], 1)) == 1


def test_history_with_no_collaborative_neighbors():
    index = _build([[1, 2], [1, 2], [6]])
    recall = SwingRecall(index["neighbors"], index["sims"], 8)
    assert recall.score_items([6, 7]).dtype == np.float64
    assert recall.recall(0, [6, 7], 1000) == []


def test_index_validation(tmp_path, synthetic_data):
    d = synthetic_data
    index = build_swing_index(d.flat_items, d.user_offsets, d.train_len,
                             d.num_items, top_m=10, verbose=False)
    path = tmp_path / "swing.npz"
    save_swing_index(index, path, d)
    assert load_swing_recall(path, d).name == "swing"
    d.raw_item_ids[[0, 1]] = d.raw_item_ids[[1, 0]]
    with pytest.raises(ValueError, match="mismatch"):
        load_swing_recall(path, d)
    d.raw_item_ids[[0, 1]] = d.raw_item_ids[[1, 0]]
    broken = dict(index, sims=np.full_like(index["sims"], np.nan))
    save_swing_index(broken, path, d)
    with pytest.raises(ValueError, match="shape/values"):
        load_swing_recall(path, d)
    save_swing_index(index, path, d)
    d.train_len[0] -= 1
    with pytest.raises(ValueError, match="mismatch"):
        load_swing_recall(path, d)


@pytest.mark.parametrize("kwargs", [{"alpha2": 0}, {"beta": -1}, {"alpha1": float("nan")},
                                    {"chunk_size": 0}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        _build([[1, 2], [1, 2]], **kwargs)
