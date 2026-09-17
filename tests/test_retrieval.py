"""Retrieval index and serving path."""

import numpy as np
import pytest

from src.retrieval.faiss_index import HAS_FAISS, ItemIndex, build_index


@pytest.fixture()
def emb():
    rng = np.random.default_rng(0)
    e = rng.normal(size=(30, 8)).astype(np.float32)
    e[0] = 0.0  # PAD
    return e


def test_index_returns_nearest_first(emb):
    idx = build_index(emb, normalize=True)
    q = emb[7:8].copy()
    ids, scores = idx.search(q, top_k=5)
    assert ids[0, 0] == 7
    assert scores[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert np.all(np.diff(scores[0]) <= 1e-6)


def test_pad_is_never_returned(emb):
    idx = build_index(emb, normalize=True)
    q = np.zeros((3, 8), dtype=np.float32)  # zero query -> all similarities 0
    ids, _ = idx.search(q, top_k=10)
    assert (ids > 0).all()


def test_exclusions_are_respected(emb):
    idx = build_index(emb, normalize=True)
    q = emb[7:8].copy()
    ex = np.array([[7, 3, 9]], dtype=np.int64)
    ids, _ = idx.search(q, top_k=5, exclude=ex)
    assert 7 not in ids[0]
    assert 3 not in ids[0]
    assert 9 not in ids[0]
    assert ids.shape[1] == 5


def test_exclusion_larger_than_overfetch_still_works(emb):
    """Excluding many items must not silently shrink the result list."""
    idx = build_index(emb, normalize=True)
    q = emb[7:8].copy()
    ex = np.arange(1, 25, dtype=np.int64).reshape(1, -1)  # exclude almost everything
    ids, _ = idx.search(q, top_k=5, exclude=ex)
    assert ids.shape[1] == 5
    assert set(ids[0].tolist()) <= set(range(25, 30))


def test_normalization_makes_it_cosine(emb):
    idx = ItemIndex(emb, normalize=True)
    q = np.ones((1, 8), dtype=np.float32) * 3.0  # scale must not matter
    q2 = np.ones((1, 8), dtype=np.float32) * 0.1
    i1, s1 = idx.search(q, top_k=3)
    i2, s2 = idx.search(q2, top_k=3)
    assert np.array_equal(i1, i2)
    assert np.allclose(s1, s2, atol=1e-5)


def test_save_and_load(tmp_path, emb):
    idx = build_index(emb, normalize=True)
    idx.save(tmp_path / "item.index")
    from src.retrieval.faiss_index import load_index

    loaded = load_index(tmp_path / "item.index")
    assert loaded is not None


def test_fallback_and_faiss_agree(emb):
    """The exact fallback must produce the same top-k as the faiss index."""
    idx = build_index(emb, normalize=True)
    q = emb[3:6].copy()
    faiss_ids, faiss_scores = idx.search(q, top_k=5)

    fallback = ItemIndex(emb, normalize=True)
    fallback._index = None  # force the exact path
    np_ids, np_scores = fallback.search(q, top_k=5)

    assert np.array_equal(faiss_ids, np_ids)
    assert np.allclose(faiss_scores, np_scores, atol=1e-5)


def test_backend_reported():
    assert isinstance(HAS_FAISS, bool)
