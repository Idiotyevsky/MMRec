"""Random access into the official MicroLens video archive.

The official 100K video archive is a **split zip**: 222 parts of 2 GB
(``.z01`` .. ``.z222``) plus a final ``.zip`` part, ~477 GB in total, holding
``MicroLens-100k_videos/<videoID>.mp4`` for ``videoID = 1..19738``.  Downloading
it is neither practical nor necessary: a single video is 20-50 MB and we only
need a handful for the demo.

The central directory lives in the last part, so it can be read with two range
requests (the end-of-central-directory record, then the directory itself), after
which any individual video can be fetched with one more range request and
decompressed locally.  Total download for the index: ~2 MB.
"""

from __future__ import annotations

import json
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://recsys.westlake.edu.cn/MicroLens-100k-Dataset"
ZIP_NAME = "MicroLens-100k_videos.zip"
NUM_PARTS = 223  # disks 0..222, where disk 222 is the .zip part
EOCD_SIG = b"PK\x05\x06"
CD_SIG = b"PK\x01\x02"
USER_AGENT = "Mozilla/5.0 (compatible; ShortRec-demo/1.0)"


@dataclass(frozen=True)
class VideoEntry:
    video_id: int
    disk: int
    local_header_offset: int
    compressed_size: int
    uncompressed_size: int
    crc32: int

    def as_dict(self) -> dict:
        return {
            "video_id": self.video_id, "disk": self.disk,
            "local_header_offset": self.local_header_offset,
            "compressed_size": self.compressed_size,
            "uncompressed_size": self.uncompressed_size,
            "crc32": self.crc32,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VideoEntry":
        return cls(int(d["video_id"]), int(d["disk"]), int(d["local_header_offset"]),
                   int(d["compressed_size"]), int(d["uncompressed_size"]), int(d["crc32"]))


def part_url(disk: int) -> str:
    """URL of the archive part holding a given disk index."""
    if disk == NUM_PARTS - 1:
        return f"{BASE_URL}/{ZIP_NAME}"
    return f"{BASE_URL}/MicroLens-100k_videos.z{disk + 1:02d}"


def _fetch(url: str, start: int | None = None, end: int | None = None,
           timeout: int = 300) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}" if end is not None else f"bytes={start}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _remote_size(url: str, timeout: int = 120) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.headers["Content-Length"])


class MediaArchive:
    """Read-only, range-based accessor for the official split-zip archive."""

    def __init__(self, index_path: str | Path | None = None,
                 base_url: str = BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self.index_path = Path(index_path) if index_path else None
        self.entries: dict[int, VideoEntry] = {}
        self.index_meta: dict = {}

    # ------------------------------------------------------------------
    def _url(self, disk: int) -> str:
        if disk == NUM_PARTS - 1:
            return f"{self.base_url}/{ZIP_NAME}"
        return f"{self.base_url}/MicroLens-100k_videos.z{disk + 1:02d}"

    def build_index(self, verbose: bool = True) -> dict:
        """Read the central directory with two range requests."""
        zip_url = self._url(NUM_PARTS - 1)
        size = _remote_size(zip_url)
        if verbose:
            print(f"  last part size: {size} bytes")
        tail_len = min(65536 + 22, size)
        tail = _fetch(zip_url, size - tail_len, size - 1)
        i = tail.rfind(EOCD_SIG)
        if i < 0:
            raise RuntimeError("end-of-central-directory record not found; not a zip archive?")
        (_, this_disk, cd_disk, _, n_total, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", tail[i : i + 22])
        if verbose:
            print(f"  entries: {n_total}, central directory disk: {cd_disk}, size: {cd_size}")
        cd = _fetch(zip_url, cd_off, cd_off + cd_size - 1)

        entries: dict[int, VideoEntry] = {}
        pos = 0
        while pos < len(cd) - 4 and cd[pos : pos + 4] == CD_SIG:
            (_, _, _, _, _, _, _, crc, csize, usize, fnlen, extralen, commentlen,
             disk, _, _, lho) = struct.unpack("<IHHHHHHIIIHHHHHII", cd[pos : pos + 46])
            name = cd[pos + 46 : pos + 46 + fnlen].decode("utf-8", "replace")
            if name.endswith(".mp4"):
                stem = name.split("/")[-1][:-4]
                if stem.isdigit():
                    vid = int(stem)
                    entries[vid] = VideoEntry(vid, disk, lho, csize, usize, crc)
            pos += 46 + fnlen + extralen + commentlen

        self.entries = entries
        self.index_meta = {
            "archive": ZIP_NAME,
            "base_url": self.base_url,
            "num_parts": NUM_PARTS,
            "num_videos": len(entries),
            "video_id_range": [min(entries), max(entries)] if entries else None,
            "total_compressed_bytes": int(sum(e.compressed_size for e in entries.values())),
            "index_method": "HTTP range read of the end-of-central-directory + central directory",
        }
        return self.index_meta

    # ------------------------------------------------------------------
    def save_index(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.index_meta,
                   "entries": [e.as_dict() for e in self.entries.values()]}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load_index(self, path: str | Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.index_meta = payload.get("meta", {})
        self.entries = {int(e["video_id"]): VideoEntry.from_dict(e) for e in payload["entries"]}
        return bool(self.entries)

    def ensure_index(self, index_path: str | Path, verbose: bool = True) -> dict:
        if self.load_index(index_path):
            return self.index_meta
        meta = self.build_index(verbose=verbose)
        self.save_index(index_path)
        return meta

    # ------------------------------------------------------------------
    def fetch_video(self, video_id: int, timeout: int = 600) -> bytes:
        """Download and decompress one video.  Verifies size and CRC32."""
        entry = self.entries.get(int(video_id))
        if entry is None:
            raise KeyError(f"video {video_id} is not in the archive index")

        url = self._url(entry.disk)
        start = entry.local_header_offset
        end = start + 30 + 4096  # local header is small; over-read is trimmed below
        head = _fetch(url, start, start + 30 + 1024 - 1, timeout=timeout)
        if head[:4] != b"PK\x03\x04":
            raise RuntimeError(f"video {video_id}: bad local header at offset {start}")
        fnlen, extralen = struct.unpack("<HH", head[26:30])
        data_start = start + 30 + fnlen + extralen
        end = data_start + entry.compressed_size - 1
        raw = _fetch(url, data_start, end, timeout=timeout)
        if len(raw) != entry.compressed_size:
            raise RuntimeError(
                f"video {video_id}: fetched {len(raw)} bytes, expected {entry.compressed_size}"
            )

        method = 8  # deflate is the only method the archive uses
        out = zlib.decompressobj(-15).decompress(raw)
        if len(out) != entry.uncompressed_size:
            raise RuntimeError(
                f"video {video_id}: decompressed {len(out)} bytes, expected "
                f"{entry.uncompressed_size}"
            )
        if zlib.crc32(out) & 0xFFFFFFFF != entry.crc32:
            raise RuntimeError(f"video {video_id}: CRC mismatch")
        if out[4:8] != b"ftyp":
            raise RuntimeError(f"video {video_id}: output is not an ISO media file")
        return out

    def download_video(self, video_id: int, dest: str | Path) -> dict:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = self.fetch_video(video_id)
        dest.write_bytes(data)
        return {"video_id": int(video_id), "path": str(dest), "bytes": len(data),
                "verified": True}

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        return dict(self.index_meta)
