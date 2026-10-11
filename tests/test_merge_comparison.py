"""Reject invalid RRF/snake comparisons before publishing a result table."""

import copy
import json

import pytest

from analysis.compare_merge_results import compare


def _artifact(method):
    return {
        "processed_dir": "data/processed/base", "ranker": "mm_concat",
        "ranker_run": "results/runs/example", "num_users": 10,
        "eval_seed": 42, "user_subset_hash": "same-users",
        "recall_k_per_channel": 1000,
        "recall_sources": ["popular", "itemcf", "semantic", "two_tower"],
        "final_k": 20, "merge_method": method,
        "full_catalogue": {"Recall@20": 0.1, "NDCG@20": 0.05},
        "tradeoff": [{"candidate_k": 100, "candidate_recall": 0.2,
                      "pipeline_Recall@20": 0.1, "pipeline_NDCG@20": 0.05,
                      "recall_retention": 1.0, "mean_pool_size": 100.0}],
    }


def _write(tmp_path, a, b):
    paths = [tmp_path / "rrf.json", tmp_path / "snake.json"]
    for path, value in zip(paths, [a, b]):
        path.write_text(json.dumps(value))
    return paths


def test_comparison_uses_artifacts(tmp_path):
    a, b = _artifact("rrf"), _artifact("snake")
    b["tradeoff"][0]["pipeline_NDCG@20"] = 0.06
    rows, _ = compare(*_write(tmp_path, a, b))
    assert rows[0]["delta_pipeline_NDCG@20"] == pytest.approx(0.01)
    assert rows[0]["delta_candidate_recall"] == 0


@pytest.mark.parametrize("key,value", [
    ("user_subset_hash", "different-users"),
    ("ranker_run", "results/runs/different"),
    ("recall_k_per_channel", 500),
    ("recall_sources", ["popular", "semantic"]),
    ("full_catalogue", {"Recall@20": 0.2, "NDCG@20": 0.05}),
    ("merge_method", "rrf"),
])
def test_comparison_rejects_protocol_mismatch(tmp_path, key, value):
    a, b = _artifact("rrf"), _artifact("snake")
    b[key] = value
    with pytest.raises(ValueError):
        compare(*_write(tmp_path, a, b))


def test_comparison_rejects_budget_mismatch(tmp_path):
    a, b = _artifact("rrf"), _artifact("snake")
    b["tradeoff"] = copy.deepcopy(a["tradeoff"])
    b["tradeoff"][0]["candidate_k"] = 200
    with pytest.raises(ValueError, match="budgets"):
        compare(*_write(tmp_path, a, b))
