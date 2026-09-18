"""Regression tests for the recall-index and benchmark correctness fixes."""

import numpy as np
import pytest

from src.data.dataset import ProcessedData
from src.recall.itemcf import build_itemcf_index, load_itemcf_index


# ----------------------------------------------------------------- ItemCF
def test_itemcf_boundary_follows_train_len(synthetic_data: ProcessedData):
    """The index must be built from the split metadata, not a hard-coded rule.

    ``train_len`` is the contract for which positions are training rows.  If it
    changes, the index has to change with it.
    """
    base = build_itemcf_index(
        synthetic_data.flat_items, synthetic_data.user_offsets, synthetic_data.train_len,
        synthetic_data.num_items, top_m=20, chunk_size=64, verbose=False,
    )
    # pretend every user has one extra training row
    longer = np.minimum(synthetic_data.train_len + 1, np.diff(synthetic_data.user_offsets) - 1)
    changed = build_itemcf_index(
        synthetic_data.flat_items, synthetic_data.user_offsets, longer,
        synthetic_data.num_items, top_m=20, chunk_size=64, verbose=False,
    )
    assert not np.array_equal(base["freq"], changed["freq"]), (
        "changing train_len did not change the item frequencies; the boundary is "
        "probably hard-coded instead of read from the split metadata"
    )


def test_itemcf_boundary_matches_manual_train_slice(synthetic_data: ProcessedData):
    """Frequencies must equal the counts over exactly the declared train rows."""
    idx = build_itemcf_index(
        synthetic_data.flat_items, synthetic_data.user_offsets, synthetic_data.train_len,
        synthetic_data.num_items, top_m=10, chunk_size=64, verbose=False,
    )
    manual = np.zeros(synthetic_data.num_items + 1, dtype=np.int64)
    for u in range(synthetic_data.num_users):
        start = int(synthetic_data.user_offsets[u])
        end = start + int(synthetic_data.train_len[u])
        seg = synthetic_data.flat_items[start:end]
        if seg.size:
            manual += np.bincount(seg, minlength=synthetic_data.num_items + 1)[
                : synthetic_data.num_items + 1
            ]
    assert np.array_equal(idx["freq"], manual)


def test_min_cooccurrence_is_a_real_threshold(synthetic_data: ProcessedData):
    """A higher threshold must never produce more neighbours, only fewer."""
    loose = build_itemcf_index(
        synthetic_data.flat_items, synthetic_data.user_offsets, synthetic_data.train_len,
        synthetic_data.num_items, top_m=20, chunk_size=64, min_cooccurrence=1, verbose=False,
    )
    strict = build_itemcf_index(
        synthetic_data.flat_items, synthetic_data.user_offsets, synthetic_data.train_len,
        synthetic_data.num_items, top_m=20, chunk_size=64, min_cooccurrence=3, verbose=False,
    )
    n_loose = int((loose["sims"] > 0).sum())
    n_strict = int((strict["sims"] > 0).sum())
    assert n_strict <= n_loose
    assert strict["n_filtered_by_min_cooccurrence"] > 0, (
        "min_cooccurrence=3 filtered nothing; the parameter is not wired to the "
        "raw co-occurrence count"
    )


def test_min_cooccurrence_threshold_semantics(synthetic_data: ProcessedData):
    """Every surviving pair must be supported by >= threshold shared users."""
    threshold = 4
    strict = build_itemcf_index(
        synthetic_data.flat_items, synthetic_data.user_offsets, synthetic_data.train_len,
        synthetic_data.num_items, top_m=20, chunk_size=64,
        min_cooccurrence=threshold, verbose=False,
    )
    # rebuild the raw co-occurrence counts independently
    from scipy import sparse

    users, items = [], []
    for u in range(synthetic_data.num_users):
        start = int(synthetic_data.user_offsets[u])
        end = start + int(synthetic_data.train_len[u])
        for it in synthetic_data.flat_items[start:end].tolist():
            if it > 0:
                users.append(u)
                items.append(it)
    X = sparse.csr_matrix(
        (np.ones(len(items), dtype=np.float32), (users, items)),
        shape=(synthetic_data.num_users, synthetic_data.num_items + 1),
    )
    X.sum_duplicates()
    X.data[:] = 1.0
    checked = 0
    for item in range(1, synthetic_data.num_items + 1):
        for n, s in zip(strict["neighbors"][item].tolist(), strict["sims"][item].tolist()):
            if n <= 0 or s <= 0:
                continue
            shared = float(X[:, item].multiply(X[:, n]).sum())
            assert shared >= threshold, f"pair ({item},{n}) has only {shared} shared users"
            checked += 1
    assert checked > 0, "no surviving pairs to check; the test is vacuous"


def test_index_roundtrip_keeps_new_field(tmp_path, synthetic_data: ProcessedData):
    idx = build_itemcf_index(
        synthetic_data.flat_items, synthetic_data.user_offsets, synthetic_data.train_len,
        synthetic_data.num_items, top_m=5, chunk_size=64, min_cooccurrence=2, verbose=False,
    )
    path = tmp_path / "itemcf.npz"
    np.savez_compressed(path, **idx)
    loaded = load_itemcf_index(path)
    assert "n_filtered_by_min_cooccurrence" in loaded
    assert np.array_equal(loaded["neighbors"], idx["neighbors"])


# ------------------------------------------------------- latency benchmark
def test_full_catalogue_uses_every_valid_item(synthetic_data: ProcessedData, synthetic_run):
    """The full-catalogue row must score num_items items, not the recall union."""
    from src.pipeline import RankerRegistry, RankerSpec

    registry = RankerRegistry(
        synthetic_data, {"sasrec": RankerSpec("sasrec", synthetic_run)}, device="cpu"
    )
    model = registry.get("sasrec")
    ids = model.valid_item_ids()
    assert ids.shape[0] == synthetic_data.num_items
    assert ids.min() == 1 and ids.max() == synthetic_data.num_items
    assert 0 not in ids, "PAD must not be part of the full-catalogue scoring set"


def test_encode_and_score_are_separable(synthetic_data: ProcessedData, synthetic_run):
    """Encoding is candidate-independent; scoring is not."""
    from src.pipeline import RankerRegistry, RankerSpec

    registry = RankerRegistry(
        synthetic_data, {"sasrec": RankerSpec("sasrec", synthetic_run)}, device="cpu"
    )
    model = registry.get("sasrec")
    hist = [int(i) for i in synthetic_data.test_history(0)]
    vec = model.encode_user(hist)
    n = synthetic_data.num_items
    small = model.score_with_user_vector(vec, np.arange(1, 6, dtype=np.int64))
    big = model.score_with_user_vector(vec, np.arange(1, n + 1, dtype=np.int64))
    assert small.shape == (5,)
    assert big.shape == (n,)
    assert np.allclose(small, big[:5]), "scoring must not depend on the candidate list"


def test_seen_mask_marks_history_and_pad(synthetic_data: ProcessedData, synthetic_run):
    from src.pipeline import RankerRegistry, RankerSpec

    registry = RankerRegistry(
        synthetic_data, {"sasrec": RankerSpec("sasrec", synthetic_run)}, device="cpu"
    )
    model = registry.get("sasrec")
    hist = [int(i) for i in synthetic_data.test_history(0)]
    n = synthetic_data.num_items
    # full-catalogue layout: scores[i - 1] belongs to internal id i
    scores = np.zeros(n, dtype=np.float32)
    masked = model.seen_mask_overhead(hist, scores)
    for i in hist:
        assert masked[int(i) - 1] == -np.inf
    untouched = [i for i in range(1, n + 1) if i not in set(hist)]
    assert np.all(masked[np.asarray(untouched) - 1] == 0.0)

    # positional layout: scores[k] belongs to item_ids[k]
    item_ids = np.array([9, 4, 7, 2, 11], dtype=np.int64)
    pos = model.seen_mask_overhead([4, 11], np.zeros(5, dtype=np.float32), item_ids=item_ids)
    assert pos[1] == -np.inf and pos[4] == -np.inf
    assert pos[0] == 0.0 and pos[2] == 0.0 and pos[3] == 0.0
