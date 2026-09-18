"""Media API contract: manifest, per-item lookup, video serving, degradation."""

import json

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.pipeline import RankerSpec  # noqa: E402
from src.serving.service import ShortRecService  # noqa: E402

MP4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 64


def _write_manifest(root, user_id=1):
    media = root / "data" / "demo_media"
    media.mkdir(parents=True, exist_ok=True)
    (media / "8278.mp4").write_bytes(MP4)
    manifest = {
        "source": "MicroLens official media (test)",
        "verified_mapping": True,
        "mapping_report": {"item_mapping_is_bijection": True, "matched_fraction": 0.999853},
        "user_id": user_id,
        "ranker": "sasrec",
        "items": {
            "0": {
                "item_id": 0, "official_video_id": 8278, "rank": 1,
                "title": "a real title", "sources": ["itemcf"],
                "baseline_rank": 230, "rank_delta": 229,
                "popularity_bucket": "tail", "train_interactions": 9,
                "available": True, "verified": True,
                "local_path": "data/demo_media/8278.mp4", "bytes": len(MP4),
            },
            "1": {
                "item_id": 1, "official_video_id": 7382, "rank": 2,
                "available": False, "verified": False,
                "local_path": "data/demo_media/does_not_exist.mp4",
            },
        },
    }
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    (root / "artifacts" / "demo_media_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _client(root, synthetic_dir, synthetic_recall_artifacts, synthetic_run, with_media=True):
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / "artifacts" / "itemcf_neighbors.npz",
                        **dict(np.load(synthetic_recall_artifacts["itemcf"])))
    np.save(root / "artifacts" / "content_embeddings.npy",
            np.load(synthetic_recall_artifacts["content"]))
    if with_media:
        _write_manifest(root)

    service = ShortRecService(
        processed_dir=str(synthetic_dir),
        cold_processed_dir="does/not/exist",
        device="cpu",
        default_ranker="sasrec",
        root=root,
        ranker_specs={"sasrec": RankerSpec("sasrec", synthetic_run, description="ID-only")},
        feature_dir=str(synthetic_dir),
    )
    from src.serving.app import create_app

    return TestClient(create_app(service=service))


@pytest.fixture(scope="module")
def media_client(synthetic_dir, synthetic_recall_artifacts, synthetic_run, tmp_path_factory):
    root = tmp_path_factory.mktemp("media_root")
    with _client(root, synthetic_dir, synthetic_recall_artifacts, synthetic_run) as c:
        yield c


@pytest.fixture(scope="module")
def no_media_client(synthetic_dir, synthetic_recall_artifacts, synthetic_run, tmp_path_factory):
    root = tmp_path_factory.mktemp("no_media_root")
    with _client(root, synthetic_dir, synthetic_recall_artifacts, synthetic_run, with_media=False) as c:
        yield c


# ---------------------------------------------------------------- manifest
def test_manifest_endpoint(media_client):
    """The manifest is keyed by the raw MicroLens item id the API exposes."""
    r = media_client.get("/media/manifest")
    assert r.status_code == 200
    body = r.json()
    assert body["prepared"] is True
    assert body["verified_mapping"] is True
    assert body["mapping_report"]["item_mapping_is_bijection"] is True
    assert body["num_items"] == 2
    assert body["num_available"] == 1, "only the item with a real file counts as available"


def test_manifest_reports_unprepared_without_media(no_media_client):
    r = no_media_client.get("/media/manifest")
    assert r.status_code == 200
    body = r.json()
    assert body["prepared"] is False
    assert "prepare_media_demo" in body["hint"]


def test_missing_media_directory_does_not_break_serving(no_media_client):
    """The recommendation service must work with no media prepared at all."""
    assert no_media_client.get("/health").status_code == 200
    assert no_media_client.get("/system").status_code == 200
    assert no_media_client.get("/feed").json()["items"] == []


# ---------------------------------------------------------------- feed
def test_feed_contains_only_playable_items(media_client):
    r = media_client.get("/feed")
    assert r.status_code == 200
    body = r.json()
    assert body["prepared"] is True
    assert [i["item_id"] for i in body["items"]] == [0]
    item = body["items"][0]
    assert item["media_url"] == "/media/video/0"
    assert item["rank_delta"] == 229
    assert item["title"] == "a real title"


def test_feed_carries_zero_train_flag(media_client):
    item = media_client.get("/feed").json()["items"][0]
    assert "is_zero_train_signal" in item


# ---------------------------------------------------------------- per item
def test_item_media_for_available_item(media_client):
    r = media_client.get("/items/0/media")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True and body["verified"] is True
    assert body["official_video_id"] == 8278


def test_item_media_for_unavailable_item(media_client):
    body = media_client.get("/items/1/media").json()
    assert body["available"] is False
    assert body["media_url"] is None


def test_item_media_for_unknown_item(media_client):
    body = media_client.get("/items/999999/media").json()
    assert body["available"] is False
    assert "reason" in body


def test_item_media_rejects_non_integer(media_client):
    assert media_client.get("/items/abc/media").status_code == 422


# ---------------------------------------------------------------- video
def test_video_endpoint_serves_the_manifest_file(media_client):
    r = media_client.get("/media/video/0")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("video/mp4")
    assert r.content.startswith(b"\x00\x00\x00\x20ftyp")


def test_video_endpoint_404_for_unavailable_item(media_client):
    assert media_client.get("/media/video/1").status_code == 404


def test_video_endpoint_404_for_unknown_item(media_client):
    assert media_client.get("/media/video/999999").status_code == 404


def test_video_endpoint_cannot_traverse(media_client):
    """A non-integer id is rejected by the path type, not by string handling."""
    assert media_client.get("/media/video/..%2f..%2fetc%2fpasswd").status_code in (404, 422)


def test_video_endpoint_supports_range_requests(media_client):
    r = media_client.get("/media/video/0", headers={"Range": "bytes=0-7"})
    assert r.status_code in (200, 206)
    assert len(r.content) <= len(MP4)
