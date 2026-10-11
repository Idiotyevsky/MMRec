"""MMR correctness, missing content and serving policy composition."""
import copy

import numpy as np
import pytest

from src.rerank import RerankConfig, content_diversity, rerank


def _entries():
    return [{"item_id": i, "ranking_score": score, "exploration_candidate": i == 4,
             "sources": ["semantic", "two_tower"]} for i,score in enumerate([10,9.5,9,8], 1)]


def _embeddings():
    return np.array([[0,0], [1,0], [1,0], [0,1], [.6,.8]], dtype=np.float32)


def test_diverse_item_beats_near_duplicate_and_trace_is_preserved():
    entries = _entries()
    original = copy.deepcopy(entries)
    result = rerank(entries, RerankConfig(mmr=True, mmr_lambda=.5), 2,
                    content_embeddings=_embeddings())
    assert [e["item_id"] for e in result.items] == [1,3]
    assert entries == original  # request-local data is never mutated
    assert result.items[1]["ranking_score"] == 9
    assert result.items[1]["ranking_position"] == 3
    assert result.items[1]["rerank_delta"] == 1
    assert result.items[1]["sources"] == ["semantic", "two_tower"]
    assert result.applied["mmr"]["after"]["ild"] > result.applied["mmr"]["before"]["ild"]


def test_matches_independent_greedy_reference():
    rng = np.random.default_rng(7)
    table = np.r_[np.zeros((1,7)), rng.normal(size=(12,7))]
    entries = [{"item_id": i+1, "ranking_score": 12-i} for i in range(12)]
    v = table[1:] / np.linalg.norm(table[1:], axis=1, keepdims=True)
    rel = np.linspace(1,0,12)
    selected = []
    for _ in range(5):
        scores = [(.6*rel[i] - .4*max(v[i] @ v[s] for s in selected) if selected else .6*rel[i])
                  if i not in selected else -np.inf for i in range(12)]
        selected.append(int(np.argmax(scores)))
    actual = rerank(entries, RerankConfig(mmr=True, mmr_lambda=.6), 5, content_embeddings=table)
    assert [e["item_id"] for e in actual.items] == [s+1 for s in selected]


def test_lambda_one_and_disabled_preserve_ranking_even_with_ties():
    entries = _entries()
    entries[1]["ranking_score"] = entries[0]["ranking_score"]
    off = rerank(entries, RerankConfig(), 4)
    identity = rerank(entries, RerankConfig(mmr=True, mmr_lambda=1), 4)
    assert identity.items == off.items
    assert identity.applied["mmr"]["fallback_reason"] == "lambda_one_identity"


def test_pad_seen_duplicates_window_and_short_list():
    entries = [{"item_id": 0, "ranking_score": 99}, *_entries(), _entries()[0]]
    result = rerank(entries, RerankConfig(mmr=True, mmr_lambda=0, mmr_candidate_k=2), 2,
                    history={4}, content_embeddings=_embeddings())
    assert [e["item_id"] for e in result.items] == [1,2]
    assert result.applied["seen_filtered"] == 1
    assert result.applied["deduped"] == 1
    assert len(rerank(_entries(), RerankConfig(mmr=True), 10, content_embeddings=_embeddings()).items) == 4
    assert rerank(_entries(), RerankConfig(mmr=True), 0).items == []
    assert rerank([], RerankConfig(mmr=True), 20).items == []


@pytest.mark.parametrize("table,reason", [(None, "content_embeddings_unavailable"),
    (np.zeros((5,2)), "all_content_unavailable"), (np.zeros((2,2)), "content_embedding_mapping_invalid")])
def test_explicit_fallback(table, reason):
    result = rerank(_entries(), RerankConfig(mmr=True), 2, content_embeddings=table)
    assert [e["item_id"] for e in result.items] == [1,2]
    assert result.applied["mmr"]["fallback_reason"] == reason
    assert not result.applied["mmr"]["applied"]


def test_missing_features_do_not_get_a_diversity_bonus():
    table = _embeddings()
    table[2] = np.nan
    result = rerank(_entries(), RerankConfig(mmr=True, mmr_lambda=.5), 2, content_embeddings=table)
    assert [e["item_id"] for e in result.items] == [1,3]
    assert result.applied["mmr"]["missing_feature_items"] == 1
    assert all(np.isfinite(e["mmr_score"]) for e in result.items)
    diversity = content_diversity(table[[1,2,3]])
    assert diversity["valid_pair_fraction"] == pytest.approx(1/3)
    assert diversity["ild"] == pytest.approx(1)


def test_exploration_quota_retains_selected_set_and_injection_flags():
    entries = _entries()
    base = rerank(entries, RerankConfig(exploration=True, exploration_quota=1), 2)
    actual = rerank(entries, RerankConfig(exploration=True, exploration_quota=1,
                                        mmr=True, mmr_lambda=.5), 2, content_embeddings=_embeddings())
    assert {e["item_id"] for e in actual.items} == {e["item_id"] for e in base.items}
    assert [e["item_id"] for e in actual.items if e["exploration"]] == [4]
    assert actual.applied["mmr"]["mode"] == "quota_order_only"
    assert [e["final_rank"] for e in actual.items] == [1,2]


@pytest.mark.parametrize("kwargs", [{"mmr_lambda": -.1}, {"mmr_lambda": 1.1},
                                    {"mmr_lambda": float("nan")}, {"mmr_candidate_k": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        RerankConfig(**kwargs)
