"""Semantic (content) recall channel."""

import numpy as np
import pytest

from src.data.dataset import ProcessedData
from src.recall.semantic import SemanticRecall, build_content_embeddings


def test_embeddings_are_unit_norm(synthetic_data: ProcessedData, synthetic_dir):
    emb = build_content_embeddings(synthetic_dir, synthetic_dir, synthetic_data.num_items,
                                   modalities=("text", "image"))
    norms = np.linalg.norm(emb[1:], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert np.all(emb[0] == 0), "PAD must stay zero"


def test_blocks_are_balanced(synthetic_data: ProcessedData, synthetic_dir):
    """Each modality block must be unit-norm before concatenation.

    Without this the image block (raw norm ~26 on the real data) would dominate
    the text block (norm 1) and the channel would silently become image-only.
    """
    emb = build_content_embeddings(synthetic_dir, synthetic_dir, synthetic_data.num_items,
                                   modalities=("text", "image"))
    # the split point is the text feature dimension, which the synthetic
    # generator is free to choose
    text_dim = int(np.load(synthetic_dir / "text_feat.npy", mmap_mode="r").shape[1])
    assert np.allclose(np.linalg.norm(emb[1:, :text_dim], axis=1), 1.0 / np.sqrt(2), atol=1e-5)
    assert np.allclose(np.linalg.norm(emb[1:, text_dim:], axis=1), 1.0 / np.sqrt(2), atol=1e-5)


def test_recall_excludes_pad_and_seen(synthetic_data: ProcessedData, synthetic_recall_artifacts):
    emb = np.load(synthetic_recall_artifacts["content"])
    r = SemanticRecall(emb, synthetic_data.num_items)
    for u in range(10):
        hist = [int(i) for i in synthetic_data.test_history(u)]
        out = r.recall(u, hist, top_k=20)
        ids = [c.item_id for c in out]
        assert 0 not in ids
        assert not (set(ids) & set(hist))
        assert len(ids) == len(set(ids))


def test_recall_is_ordered(synthetic_data: ProcessedData, synthetic_recall_artifacts):
    emb = np.load(synthetic_recall_artifacts["content"])
    r = SemanticRecall(emb, synthetic_data.num_items)
    hist = [int(i) for i in synthetic_data.test_history(0)]
    out = r.recall(0, hist, top_k=20)
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)


def test_query_is_recency_weighted(synthetic_data: ProcessedData, synthetic_recall_artifacts):
    emb = np.load(synthetic_recall_artifacts["content"])
    r = SemanticRecall(emb, synthetic_data.num_items, decay=0.1)
    q1 = r.query_vector([1, 2, 3])
    q2 = r.query_vector([3, 2, 1])
    assert not np.allclose(q1, q2)


def test_empty_history_returns_nothing(synthetic_data: ProcessedData, synthetic_recall_artifacts):
    emb = np.load(synthetic_recall_artifacts["content"])
    r = SemanticRecall(emb, synthetic_data.num_items)
    assert r.recall(0, [], top_k=10) == []


def test_target_can_be_retrieved_from_content_alone(synthetic_data: ProcessedData, synthetic_recall_artifacts):
    """The whole point of the channel: it never looks at collaborative signal."""
    emb = np.load(synthetic_recall_artifacts["content"])
    r = SemanticRecall(emb, synthetic_data.num_items)
    hits = 0
    for u in range(synthetic_data.num_users):
        hist = [int(i) for i in synthetic_data.test_history(u)]
        ids = {c.item_id for c in r.recall(u, hist, top_k=50)}
        hits += int(synthetic_data.test_target[u]) in ids
    assert hits > 0, "content recall retrieved no target at all"


def test_index_is_exact_not_approximate(synthetic_data: ProcessedData, synthetic_recall_artifacts):
    """IndexFlatIP must return the true top-k of the inner product."""
    emb = np.load(synthetic_recall_artifacts["content"])
    r = SemanticRecall(emb, synthetic_data.num_items)
    hist = [int(i) for i in synthetic_data.test_history(0)]
    out = r.recall(0, hist, top_k=10)
    q = r.query_vector(hist)
    seen = set(hist)
    scores = emb @ q
    scores[0] = -np.inf
    for i in seen:
        scores[i] = -np.inf
    expected = np.argsort(-scores)[:10].tolist()
    assert [c.item_id for c in out] == expected
