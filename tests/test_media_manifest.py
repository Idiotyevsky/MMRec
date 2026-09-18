"""Media library: manifest integrity, id safety, graceful degradation."""

import json

import pytest

from src.media.library import MediaLibrary


@pytest.fixture()
def media_root(tmp_path):
    """A tiny manifest plus real (dummy) mp4 files on disk."""
    media = tmp_path / "data" / "demo_media"
    media.mkdir(parents=True)
    (media / "8278.mp4").write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32)
    (media / "7382.mp4").write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32)
    manifest = {
        "source": "MicroLens official media (test)",
        "verified_mapping": True,
        "mapping_report": {"item_mapping_is_bijection": True},
        "user_id": 68317,
        "ranker": "mm_concat",
        "items": {
            "18111": {
                "item_id": 18111, "official_video_id": 8278, "rank": 1,
                "title": "take stock of husband and wife quarrel",
                "sources": ["itemcf"], "baseline_rank": 230, "rank_delta": 229,
                "popularity_bucket": "tail", "train_interactions": 9,
                "available": True, "verified": True,
                "local_path": "data/demo_media/8278.mp4", "bytes": 44,
            },
            "17597": {
                "item_id": 17597, "official_video_id": 7382, "rank": 2,
                "title": "2 minutes up high", "sources": ["itemcf"],
                "baseline_rank": 147, "rank_delta": 145,
                "popularity_bucket": "tail", "train_interactions": 9,
                "available": True, "verified": True,
                "local_path": "data/demo_media/7382.mp4", "bytes": 44,
            },
            "99999": {
                "item_id": 99999, "official_video_id": 1, "rank": 3,
                "available": True, "verified": True,
                "local_path": "data/demo_media/missing.mp4", "bytes": 0,
            },
        },
    }
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    path = tmp_path / "artifacts" / "demo_media_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, path


def test_manifest_loads_and_reports_prepared(media_root):
    root, path = media_root
    lib = MediaLibrary(manifest_path=path, root=root)
    assert lib.prepared
    s = lib.manifest_summary()
    assert s["prepared"] and s["verified_mapping"]
    assert s["user_id"] == 68317
    assert s["num_items"] == 3


def test_missing_file_is_reported_unavailable(media_root):
    """A manifest entry whose file vanished must not be advertised as playable."""
    root, path = media_root
    lib = MediaLibrary(manifest_path=path, root=root)
    rec = lib.get(99999)
    assert rec is not None
    assert rec.available is False
    assert lib.video_path(99999) is None


def test_feed_only_contains_playable_items(media_root):
    root, path = media_root
    lib = MediaLibrary(manifest_path=path, root=root)
    feed = lib.feed()
    assert [f["item_id"] for f in feed] == [18111, 17597]
    assert all(f["media_url"] for f in feed)


def test_unknown_item_returns_none(media_root):
    root, path = media_root
    lib = MediaLibrary(manifest_path=path, root=root)
    assert lib.get(123456) is None
    assert lib.video_path(123456) is None


def test_video_path_is_confined_to_the_media_directory(media_root):
    root, path = media_root
    lib = MediaLibrary(manifest_path=path, root=root)
    p = lib.video_path(18111)
    assert p is not None
    assert (root / "data" / "demo_media").resolve() in p.parents


def test_path_traversal_in_the_manifest_is_rejected(media_root):
    """Even a malicious manifest path must not escape data/demo_media."""
    root, path = media_root
    payload = json.loads(path.read_text())
    payload["items"]["5"] = {
        "item_id": 5, "available": True, "verified": True,
        "local_path": "data/demo_media/../../etc/passwd",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    lib = MediaLibrary(manifest_path=path, root=root)
    assert lib.video_path(5) is None


def test_absent_manifest_degrades_gracefully(tmp_path):
    lib = MediaLibrary(manifest_path=tmp_path / "nope.json", root=tmp_path)
    assert lib.prepared is False
    assert lib.feed() == []
    s = lib.manifest_summary()
    assert s["prepared"] is False
    assert "prepare_media_demo" in s["hint"]


def test_corrupt_manifest_does_not_raise(tmp_path):
    p = tmp_path / "artifacts" / "demo_media_manifest.json"
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    lib = MediaLibrary(manifest_path=p, root=tmp_path)
    assert lib.prepared is False


def test_titles_are_real_catalogue_metadata(media_root):
    root, path = media_root
    lib = MediaLibrary(manifest_path=path, root=root)
    rec = lib.get(18111)
    assert rec.title and isinstance(rec.title, str)
    assert rec.official_video_id == 8278


def test_manifest_ids_are_item_ids_not_filenames(media_root):
    """The manifest must keep both id spaces, since they are a permutation."""
    root, path = media_root
    lib = MediaLibrary(manifest_path=path, root=root)
    rec = lib.get(18111)
    assert rec.item_id == 18111
    assert rec.official_video_id == 8278
    assert rec.item_id != rec.official_video_id
