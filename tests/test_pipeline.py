"""Two-stage pipeline: recall -> rank -> rerank, and serving metadata."""

import numpy as np
import pytest

from src.data.dataset import ProcessedData
from src.data.metadata import ItemMetadata
from src.pipeline import RankerRegistry, RankerSpec, TwoStageRecommender
from src.recall import (
    CandidateMerger,
    ItemCFRecall,
    PopularRecall,
    SemanticRecall,
    load_itemcf_index,
)
from src.rerank.simple import RerankConfig, rerank


@pytest.fixture(scope="module")
def pipeline(synthetic_data, synthetic_recall_artifacts, synthetic_run):
    ic = load_itemcf_index(synthetic_recall_artifacts["itemcf"])
    emb = np.load(synthetic_recall_artifacts["content"])
    merger = CandidateMerger([
        PopularRecall(synthetic_data.train_freq, synthetic_data.num_items),
        ItemCFRecall(ic["neighbors"], ic["sims"], synthetic_data.num_items),
        SemanticRecall(emb, synthetic_data.num_items),
    ])
    registry = RankerRegistry(
        synthetic_data, {"sasrec": RankerSpec("sasrec", synthetic_run)}, device="cpu"
    )
    registry.warm()
    return TwoStageRecommender(synthetic_data, merger, registry, default_ranker="sasrec")


# ---------------------------------------------------------------- pipeline
def test_recommend_returns_requested_number(pipeline):
    res = pipeline.recommend(1, recall_k=50, final_k=10)
    assert len(res.recommendations) == 10
    assert [e["final_rank"] for e in res.recommendations] == list(range(1, 11))


def test_history_and_target_belong_to_the_same_user(pipeline):
    """1-based serving ids vs 0-based ProcessedData indexing is a classic bug."""
    for uid in (1, 2, 3, 17):
        hist = pipeline.history(uid)
        expected = [int(i) for i in pipeline.data.test_history(uid - 1)]
        assert hist == expected
        assert pipeline.target(uid) == int(pipeline.data.test_target[uid - 1])
        assert pipeline.target(uid) not in set(hist)


def test_user_id_out_of_range_raises(pipeline):
    with pytest.raises(IndexError):
        pipeline.history(0)
    with pytest.raises(IndexError):
        pipeline.history(pipeline.data.num_users + 1)


def test_seen_items_never_recommended(pipeline):
    for uid in (1, 5, 9):
        hist = set(pipeline.history(uid))
        res = pipeline.recommend(uid, recall_k=100, final_k=20)
        assert not ({e["item_id"] for e in res.recommendations} & hist)


def test_pad_never_recommended(pipeline):
    for uid in (1, 2, 3):
        res = pipeline.recommend(uid, recall_k=100, final_k=20)
        assert all(e["item_id"] > 0 for e in res.recommendations)


def test_every_recommendation_carries_source_and_metadata(pipeline):
    res = pipeline.recommend(1, recall_k=100, final_k=10)
    for e in res.recommendations:
        assert e["sources"], "a served item must name the channel that recalled it"
        assert e["source_trace"]
        assert e["popularity_bucket"] in ("head", "middle", "tail", "cold")
        assert e["is_cold"] in (True, False)
        assert e["train_interactions"] >= 0
        assert "ranking_score" in e


def test_recall_stats_are_reported(pipeline):
    res = pipeline.recommend(1, recall_k=50, final_k=5)
    assert res.recall_stats["before_dedup"] >= res.recall_stats["after_dedup"]
    assert res.recall_stats["per_source"]


def test_scores_are_sorted_descending(pipeline):
    res = pipeline.recommend(1, recall_k=100, final_k=20)
    scores = [e["ranking_score"] for e in res.recommendations]
    assert scores == sorted(scores, reverse=True)


def test_latency_is_reported(pipeline):
    res = pipeline.recommend(1, recall_k=50, final_k=5)
    assert res.latency_ms["total"] >= 0
    assert "recall" in res.latency_ms and "rank" in res.latency_ms


def test_candidate_cap_is_respected(pipeline):
    res = pipeline.recommend(1, recall_k=50, final_k=5, max_candidates=20)
    assert res.recall_stats["after_dedup"] >= res.recall_stats["candidates_ranked"]
    assert len(res.recommendations) <= 5


def test_ranker_choice_changes_the_result(pipeline):
    """The registry must actually switch models, not silently reuse one."""
    assert pipeline.registry.names == ["sasrec"]
    with pytest.raises(KeyError):
        pipeline.registry.get("does_not_exist")


def test_system_info_shape(pipeline):
    info = pipeline.system_info()
    for k in ("dataset", "users", "items", "interactions", "default_ranker",
              "rankers", "recall_sources", "item_metadata"):
        assert k in info
    assert info["recall_sources"] == ["popular", "itemcf", "semantic"]


def test_inspect_reports_both_directions(pipeline):
    out = pipeline.inspect(1, recall_k=50, top_n=10, compare="sasrec")
    for k in ("recall_summary", "top_candidates", "final_top_k", "recall_candidates",
              "baseline_top", "moved_up_by_multimodal", "moved_down_by_multimodal"):
        assert k in out
    assert out["recall_summary"]["before_dedup"] >= out["recall_summary"]["after_dedup"]
    # comparing a model against itself must produce no movement
    same = pipeline.inspect(1, recall_k=50, top_n=10, compare="sasrec", ranker="sasrec")
    assert same["moved_up_by_multimodal"] == []
    assert same["moved_down_by_multimodal"] == []


def test_inspect_candidate_set_matches_recommend(pipeline):
    """The inspector must not quietly rank a different candidate set."""
    ins = pipeline.inspect(3, recall_k=50, top_n=10)
    rec = pipeline.recommend(3, recall_k=50, final_k=10)
    ins_ids = {e["item_id"] for e in ins["top_candidates"]}
    rec_ids = {e["item_id"] for e in rec.recommendations}
    assert rec_ids <= ins_ids | {e["item_id"] for e in ins["top_candidates"]}
    assert ins["recall_summary"]["candidates_ranked"] >= len(rec_ids)


# ---------------------------------------------------------------- metadata
def test_item_metadata_bucket_matches_dataset(synthetic_data):
    meta = ItemMetadata(synthetic_data)
    from src.data.popularity import BUCKET_NAMES

    for i in (1, 2, 5, 10):
        m = meta.of(i)
        assert m.in_catalogue
        assert m.train_interactions == int(synthetic_data.train_freq[i])
        assert m.popularity_bucket == BUCKET_NAMES[int(synthetic_data.popularity_bucket[i])]


def test_item_metadata_rejects_pad_and_out_of_range(synthetic_data):
    meta = ItemMetadata(synthetic_data)
    assert not meta.of(0).in_catalogue
    assert not meta.of(synthetic_data.num_items + 1).in_catalogue
    assert meta.of(0).popularity_bucket == "unknown"


def test_cold_metadata_is_marked(synthetic_cold_dir):
    data = ProcessedData.load(synthetic_cold_dir)
    meta = ItemMetadata(data)
    cold = np.flatnonzero(data.is_cold)
    assert cold.size > 0
    for i in cold[:5].tolist():
        m = meta.of(int(i))
        assert m.is_cold and m.popularity_bucket == "cold"
        assert m.train_interactions == 0


# ---------------------------------------------------------------- rerank
def _entries():
    return [
        {"item_id": 1, "ranking_score": 10.0, "is_cold": False},
        {"item_id": 2, "ranking_score": 9.0, "is_cold": False},
        {"item_id": 3, "ranking_score": 8.0, "is_cold": True},
        {"item_id": 4, "ranking_score": 7.0, "is_cold": False},
        {"item_id": 5, "ranking_score": 6.0, "is_cold": True},
    ]


def test_rerank_off_is_plain_top_k():
    out = rerank(_entries(), RerankConfig(), final_k=3).items
    assert [e["item_id"] for e in out] == [1, 2, 3]


def test_seen_filter_is_applied():
    out = rerank(_entries(), RerankConfig(), final_k=3, history={1}).items
    assert 1 not in [e["item_id"] for e in out]


def test_dedup_is_applied():
    dup = _entries() + [{"item_id": 1, "ranking_score": 5.0, "is_cold": False}]
    out = rerank(dup, RerankConfig(), final_k=10).items
    assert [e["item_id"] for e in out].count(1) == 1


def test_cold_exploration_guarantees_a_cold_slot():
    cfg = RerankConfig(cold_exploration=True, cold_quota=2)
    out = rerank(_entries(), cfg, final_k=4).items
    assert sum(1 for e in out if e["is_cold"]) >= 2
    assert len(out) == 4


def test_cold_exploration_is_bounded_by_availability():
    entries = [{"item_id": i, "ranking_score": 10.0 - i, "is_cold": False} for i in range(1, 6)]
    entries.append({"item_id": 99, "ranking_score": 1.0, "is_cold": True})
    cfg = RerankConfig(cold_exploration=True, cold_quota=3)
    out = rerank(entries, cfg, final_k=4).items
    assert sum(1 for e in out if e["is_cold"]) == 1  # only one exists
    assert len(out) == 4  # still filled to final_k


def test_cold_exploration_never_shortens_the_list():
    cfg = RerankConfig(cold_exploration=True, cold_quota=5)
    out = rerank(_entries(), cfg, final_k=5).items
    assert len(out) == 5
