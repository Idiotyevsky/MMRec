"""Serving-side access to prepared demo media.

The library is a thin, safe lookup over ``artifacts/demo_media_manifest.json``:

* a request only ever names an **integer item id**;
* the file path is taken from the manifest, never built from user input;
* anything not in the manifest is reported as unavailable rather than guessed;
* a missing manifest or missing media directory degrades to "not prepared"
  instead of breaking the recommendation service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "artifacts" / "demo_media_manifest.json"


@dataclass(frozen=True)
class MediaRecord:
    item_id: int
    official_video_id: int | None
    available: bool
    verified: bool
    local_path: Path | None
    title: str | None
    rank: int | None
    sources: list[str]
    baseline_rank: int | None
    rank_delta: int | None
    popularity_bucket: str | None
    train_interactions: int | None
    bytes: int | None
    likes: int | None = None
    views: int | None = None

    def as_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "available": self.available,
            "verified": self.verified,
            "official_video_id": self.official_video_id,
            "title": self.title,
            "rank": self.rank,
            "sources": self.sources,
            "baseline_rank": self.baseline_rank,
            "rank_delta": self.rank_delta,
            "popularity_bucket": self.popularity_bucket,
            "train_interactions": self.train_interactions,
            "bytes": self.bytes,
            "likes": self.likes,
            "views": self.views,
            "media_url": f"/media/video/{self.item_id}" if self.available else None,
        }


class MediaLibrary:
    def __init__(self, manifest_path: str | Path | None = None, root: Path | None = None) -> None:
        self.root = Path(root) if root else ROOT
        # resolve the manifest relative to `root` so a test fixture (or an
        # alternate deployment root) does not silently read the repo's manifest
        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.root / "artifacts" / "demo_media_manifest.json"
        self.manifest: dict = {}
        self._by_item: dict[int, MediaRecord] = {}
        self.load()

    # ------------------------------------------------------------------
    def load(self) -> None:
        self.manifest = {}
        self._by_item = {}
        if not self.manifest_path.exists():
            return
        try:
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for key, rec in (self.manifest.get("items") or {}).items():
            try:
                item_id = int(key)
            except (TypeError, ValueError):
                continue
            path = rec.get("local_path")
            abs_path = (self.root / path) if path else None
            available = bool(rec.get("available")) and abs_path is not None and abs_path.exists()
            self._by_item[item_id] = MediaRecord(
                item_id=item_id,
                official_video_id=rec.get("official_video_id"),
                available=available,
                verified=bool(rec.get("verified")) and available,
                local_path=abs_path if available else None,
                title=rec.get("title"),
                rank=rec.get("rank"),
                sources=list(rec.get("sources") or []),
                baseline_rank=rec.get("baseline_rank"),
                rank_delta=rec.get("rank_delta"),
                popularity_bucket=rec.get("popularity_bucket"),
                train_interactions=rec.get("train_interactions"),
                bytes=rec.get("bytes"),
                likes=rec.get("likes"),
                views=rec.get("views"),
            )

    # ------------------------------------------------------------------
    @property
    def prepared(self) -> bool:
        return bool(self._by_item)

    def get(self, item_id: int) -> MediaRecord | None:
        """Exact manifest lookup.  ``item_id`` must be an int."""
        return self._by_item.get(int(item_id))

    def video_path(self, item_id: int) -> Path | None:
        """Path for an available, verified video; ``None`` otherwise.

        The path always comes from the manifest, so a crafted item id cannot
        escape the media directory.
        """
        rec = self.get(item_id)
        if rec is None or not rec.available or rec.local_path is None:
            return None
        resolved = rec.local_path.resolve()
        media_root = (self.root / "data" / "demo_media").resolve()
        if media_root not in resolved.parents:
            return None
        return resolved

    def manifest_summary(self) -> dict:
        if not self.prepared:
            return {
                "prepared": False,
                "hint": "run `python scripts/prepare_media_demo.py` to download demo media",
            }
        items = sorted(self._by_item.values(), key=lambda r: (r.rank is None, r.rank or 0))
        return {
            "prepared": True,
            "source": self.manifest.get("source"),
            "verified_mapping": self.manifest.get("verified_mapping"),
            "mapping_report": self.manifest.get("mapping_report"),
            "user_id": self.manifest.get("user_id"),
            "ranker": self.manifest.get("ranker"),
            "num_items": len(items),
            "num_available": sum(1 for r in items if r.available),
            "items": [r.as_dict() for r in items],
            "note": self.manifest.get("note"),
        }

    def feed(self, user_id: int | None = None) -> list[dict]:
        """Playable items only, in recommendation order."""
        items = sorted(self._by_item.values(), key=lambda r: (r.rank is None, r.rank or 0))
        return [r.as_dict() for r in items if r.available]
