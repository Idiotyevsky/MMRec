"""Serving API contract tests.

Runs against the synthetic dataset and a real (tiny) trained checkpoint, so the
full path config -> factory -> checkpoint -> pipeline -> HTTP is exercised.
"""

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.pipeline import RankerSpec  # noqa: E402
from src.serving.app import create_app  # noqa: E402
from src.serving.service import ShortRecService  # noqa: E402


@pytest.fixture(scope="module")
def client(synthetic_dir, synthetic_recall_artifacts, synthetic_run, tmp_path_factory):
    root = tmp_path_factory.mktemp("serving_root")
    # the service resolves artifacts relative to `root`
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "artifacts" / "itemcf_neighbors.npz",
                        **dict(np.load(synthetic_recall_artifacts["itemcf"])))
    np.save(root / "artifacts" / "content_embeddings.npy",
            np.load(synthetic_recall_artifacts["content"]))
    (root / "data").mkdir(parents=True, exist_ok=True)

    service = ShortRecService(
        processed_dir=str(synthetic_dir),
        cold_processed_dir="does/not/exist",
        device="cpu",
        default_ranker="sasrec",
        root=root,
        ranker_specs={"sasrec": RankerSpec("sasrec", synthetic_run, description="ID-only")},
        feature_dir=str(synthetic_dir),
    )
    with TestClient(create_app(service=service)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_system_schema(client):
    r = client.get("/system")
    assert r.status_code == 200
    body = r.json()
    for key in ("dataset", "users", "items", "interactions", "default_ranker",
                "rankers", "recall_sources", "recall_readiness", "item_metadata",
                "history_mode"):
        assert key in body, f"/system must expose {key}"
    assert body["recall_sources"] == ["popular", "itemcf", "semantic"]
    assert isinstance(body["rankers"], list) and body["rankers"]
    assert body["rankers"][0]["name"] == "sasrec"


def test_user_schema(client):
    r = client.get("/users/1")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == 1
    assert body["history_length"] == len(body["history"])
    for h in body["history"]:
        assert {"item_id", "popularity_bucket", "is_cold", "train_interactions"} <= set(h)


def test_user_out_of_range_is_404(client):
    assert client.get("/users/0").status_code == 404
    assert client.get("/users/999999").status_code == 404


def test_recall_schema(client):
    r = client.get("/users/1/recall", params={"top_k": 20})
    assert r.status_code == 200
    body = r.json()
    assert body["stats"]["before_dedup"] >= body["stats"]["after_dedup"]
    assert body["candidates"]
    c0 = body["candidates"][0]
    assert {"item_id", "sources", "merge_score", "popularity_bucket", "is_cold"} <= set(c0)
    assert c0["sources"], "merged candidates must keep their source trace"


def test_recall_source_filter(client):
    r = client.get("/users/1/recall", params={"top_k": 20, "sources": "popular"})
    assert r.status_code == 200
    for c in r.json()["candidates"]:
        assert [s["name"] for s in c["sources"]] == ["popular"]


def test_recommend_schema_and_ids(client):
    r = client.get("/users/1/recommend", params={"top_k": 5})
    assert r.status_code == 200
    body = r.json()
    for key in ("user_id", "history", "recall", "rerank", "latency_ms",
                "recommendations", "target_item", "ranker"):
        assert key in body
    assert len(body["recommendations"]) <= 5
    seen = set(body["history"])
    for rec in body["recommendations"]:
        # MicroLens raw item ids start at 0, so 0 is a valid served id
        assert rec["item_id"] >= 0
        assert rec["item_id"] not in seen, "the API must not serve a seen item"
        assert rec["sources"]
        assert rec["popularity_bucket"] in ("head", "middle", "tail", "cold")
        assert isinstance(rec["ranking_score"], float)
        assert rec["final_rank"] >= 1


def test_recommend_rejects_unknown_ranker(client):
    r = client.get("/users/1/recommend", params={"ranker": "nope"})
    assert r.status_code == 400


def test_cold_exploration_parameter_is_accepted(client):
    r = client.get("/users/1/recommend",
                   params={"cold_exploration": True, "cold_quota": 1, "top_k": 5})
    assert r.status_code == 200


def test_inspect_schema(client):
    r = client.get("/users/1/inspect", params={"top_n": 5, "compare": "sasrec"})
    assert r.status_code == 200
    body = r.json()
    for key in ("history", "history_items", "recall_summary", "recall_candidates",
                "top_candidates", "baseline_top", "final_top_k", "rerank",
                "moved_up_by_multimodal", "moved_down_by_multimodal"):
        assert key in body, f"/inspect must expose {key}"
    assert body["recall_summary"]["candidates_ranked"] > 0


def test_inspect_cold_dataset_is_absent_gracefully(client):
    """/cold endpoints must degrade, not crash, when the cold split is missing."""
    assert client.get("/cold/summary").status_code == 200
    assert client.get("/cold/items").json() == []


def test_models_endpoint(client):
    r = client.get("/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert models and models[0]["name"] == "sasrec"
    assert "description" in models[0]


def test_legacy_recommend_endpoint(client):
    r = client.post("/recommend", json={"history": [1, 2, 3], "top_k": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "items" in body and "latency_ms" in body
    assert len(body["items"]) <= 3
