#!/usr/bin/env python
"""Prepare real MicroLens media for the playable feed demo.

    python scripts/prepare_media_demo.py                    # demo user, top-10
    python scripts/prepare_media_demo.py --user-id 68317 --top-k 8
    python scripts/prepare_media_demo.py --fetch-official   # download official metadata first

What it does
------------
1. ensures the official metadata files are present
   (``MicroLens-100k_pairs.csv``, ``..._title_en.csv``, ``..._likes_and_views.txt``);
2. recovers the ``hf item id -> official video id`` permutation from exact
   millisecond timestamps and refuses to continue if it is not a bijection;
3. runs the existing recommendation pipeline for one user and reads the real
   trace (recall sources, baseline rank, multimodal rank);
4. downloads **only** the videos for those items, straight out of the official
   split archive using HTTP range requests, verifying size, CRC32 and MP4 magic;
5. writes ``artifacts/demo_media_manifest.json`` with the full mapping evidence.

Nothing here guesses: an item whose mapping cannot be verified is marked
``available: false`` and the UI falls back to a metadata-only card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.media.archive import MediaArchive  # noqa: E402
from src.media.mapping import (  # noqa: E402
    LIKES_TXT,
    OFFICIAL_DIR,
    PAIRS_CSV,
    TITLES_CSV,
    build_mapping,
    load_likes_views,
    load_titles,
    save_mapping,
)
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.media")

OFFICIAL_FILES = {
    "MicroLens-100k_pairs.csv": PAIRS_CSV,
    "MicroLens-100k_title_en.csv": TITLES_CSV,
    "MicroLens-100k_likes_and_views.txt": LIKES_TXT,
}
BASE = "https://recsys.westlake.edu.cn/MicroLens-100k-Dataset"
MANIFEST = ROOT / "artifacts" / "demo_media_manifest.json"
ARCHIVE_INDEX = ROOT / "artifacts" / "media_archive_index.json"


def fetch_official(force: bool = False) -> None:
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, dest in OFFICIAL_FILES.items():
        if dest.exists() and not force:
            LOG.info(f"official metadata present: {name}")
            continue
        url = f"{BASE}/{name}"
        LOG.info(f"downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        LOG.info(f"  -> {dest.relative_to(ROOT)} ({dest.stat().st_size/1024**2:.1f} MB)")


def _ffmpeg() -> str | None:
    import shutil
    return shutil.which("ffmpeg")


def probe_video(path: Path) -> dict:
    """Codec/size/duration via ffprobe; empty dict if ffprobe is unavailable."""
    import shutil
    import subprocess

    if not shutil.which("ffprobe"):
        return {}
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height,duration",
             "-of", "default=noprint_wrappers=1", str(path)],
            text=True, timeout=60,
        )
    except Exception:
        return {}
    info = {}
    for line in out.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    return info


def transcode_for_playback(src: Path, dest: Path, clip_seconds: float,
                           max_height: int) -> dict:
    """Re-encode to H.264 so a browser can actually play it.

    The official MicroLens videos are **HEVC/H.265**, which Chromium and Firefox
    cannot decode (the demuxer fails outright).  Serving the original file would
    render a black rectangle.  A real delivery pipeline transcodes for the web;
    this does the same thing, and the manifest records both the source and the
    playback codec so the step is visible rather than hidden.
    """
    import subprocess

    ffmpeg = _ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; cannot transcode for browser playback")
    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(src)]
    if clip_seconds and clip_seconds > 0:
        cmd += ["-t", str(clip_seconds)]
    cmd += [
        "-vf", f"scale=-2:{max_height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
        str(dest),
    ]
    subprocess.run(cmd, check=True, timeout=900)
    info = probe_video(dest)
    if info.get("codec_name") != "h264" or dest.stat().st_size == 0:
        raise RuntimeError(f"transcode produced an unusable file: {info}")
    return info


def recommendation_trace(user_id: int, ranker: str, recall_k: int, top_k: int) -> dict:
    """Real trace for one user, through the same service the API serves.

    Going through ``ShortRecService`` matters: it returns **raw** MicroLens item
    ids, which is the id space the media manifest and the API use.  Calling the
    pipeline directly would return internal ids and silently shift every lookup
    by one.
    """
    from src.serving.app import configure_threads
    from src.serving.service import ShortRecService

    configure_threads()
    service = ShortRecService(
        processed_dir="data/processed/base",
        cold_processed_dir="data/processed/cold10",
        device="cpu",
        default_ranker=ranker,
    )
    if "sasrec" not in service.registry.names:
        raise SystemExit("the ID-only baseline run is missing; needed for rank movement")
    return service.inspect(user_id, recall_k=recall_k, top_n=top_k, ranker=ranker, compare="sasrec")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user-id", type=int, default=None,
                    help="default: the user in artifacts/demo_user.json")
    ap.add_argument("--ranker", default="mm_concat")
    ap.add_argument("--top-k", type=int, default=10, help="how many served items to fetch")
    ap.add_argument("--recall-k", type=int, default=200)
    ap.add_argument("--out-dir", default="data/demo_media")
    ap.add_argument("--fetch-official", action="store_true",
                    help="download the official metadata files before mapping")
    ap.add_argument("--force", action="store_true", help="re-download videos that exist")
    ap.add_argument("--no-transcode", action="store_true",
                    help="serve the original HEVC files (browsers cannot play them)")
    ap.add_argument("--clip-seconds", type=float, default=15.0,
                    help="length of the playable clip (0 = full video)")
    ap.add_argument("--max-height", type=int, default=720)
    args = ap.parse_args()

    if args.fetch_official:
        fetch_official()
    missing = [n for n, p in OFFICIAL_FILES.items() if not p.exists()]
    if missing:
        raise SystemExit(
            "official metadata missing: " + ", ".join(missing) + "\n"
            "Run `python scripts/prepare_media_demo.py --fetch-official` first."
        )

    # ---- 1. verified id mapping --------------------------------------
    LOG.info("recovering hf item id -> official video id mapping ...")
    lut, mapping_report = build_mapping()
    save_mapping(lut, mapping_report)
    LOG.info(
        f"mapping verified: {mapping_report['unambiguous_timestamp_matches']} timestamp "
        f"matches ({mapping_report['matched_fraction']:.4%}), "
        f"{mapping_report['items_mapped']} items, bijection="
        f"{mapping_report['item_mapping_is_bijection']}"
    )

    # ---- 2. real recommendation trace --------------------------------
    user_id = args.user_id
    if user_id is None:
        demo = ROOT / "artifacts" / "demo_user.json"
        user_id = int(json.loads(demo.read_text(encoding="utf-8"))["user_id"]) if demo.exists() else 7
    LOG.info(f"running the {args.ranker} pipeline for user {user_id} ...")
    trace = recommendation_trace(user_id, args.ranker, args.recall_k, args.top_k)

    # ---- 3. resolve media --------------------------------------------
    titles = load_titles()
    likes_views = load_likes_views()
    archive = MediaArchive()
    archive.ensure_index(ARCHIVE_INDEX)
    LOG.info(f"archive index: {len(archive.entries)} videos across "
             f"{archive.index_meta['num_parts']} parts")

    out_dir = ROOT / args.out_dir
    items: dict[str, dict] = {}
    n_ok = n_fail = 0
    for rank, entry in enumerate(trace["final_top_k"][: args.top_k], start=1):
        raw_id = int(entry["item_id"])          # raw MicroLens item id
        video_id = int(lut[raw_id])             # lut is indexed by raw id
        record: dict = {
            "item_id": raw_id,
            "official_video_id": video_id,
            "rank": rank,
            "title": titles.get(video_id),
            "sources": entry["sources"],
            "recall_rank": entry.get("recall_rank", {}),
            "mm_rank": rank,
            "baseline_rank": entry.get("baseline_position"),
            "rank_delta": entry.get("rank_delta"),
            "popularity_bucket": entry["popularity_bucket"],
            "train_interactions": entry["train_interactions"],
            "is_zero_train_signal": entry.get("is_zero_train_signal", False),
            "available": False,
            "verified": False,
        }
        if video_id in likes_views:
            likes, views = likes_views[video_id]
            record["likes"] = likes
            record["views"] = views
        source = out_dir / f"{video_id}.source.mp4"
        dest = out_dir / f"{video_id}.mp4"
        try:
            if not source.exists() or args.force:
                archive.download_video(video_id, source)
            src_info = probe_video(source)
            record["source_codec"] = src_info.get("codec_name")
            record["source_bytes"] = source.stat().st_size

            if args.no_transcode:
                if dest.exists() and not args.force:
                    dest.unlink()
                dest = source
                record["playback_codec"] = src_info.get("codec_name")
                record["note"] = ("original file served as-is; browsers cannot decode "
                                  "HEVC, so playback may fail")
            else:
                if not dest.exists() or args.force:
                    play_info = transcode_for_playback(source, dest, args.clip_seconds,
                                                       args.max_height)
                else:
                    play_info = probe_video(dest)
                record["playback_codec"] = play_info.get("codec_name")
                record["playback_duration_s"] = float(play_info.get("duration") or 0.0)
                record["clip_seconds"] = args.clip_seconds
                record["note"] = ("transcoded from the original HEVC to H.264 for browser "
                                  "playback; the clip is the first "
                                  f"{args.clip_seconds:g}s of the source video")

            data = dest.read_bytes()
            record.update({"available": True, "verified": True,
                           "local_path": str(dest.relative_to(ROOT)),
                           "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()[:16]})
            n_ok += 1
        except Exception as exc:  # a missing video must not fail the whole demo
            record["error"] = str(exc)
            n_fail += 1
            LOG.warning(f"  item {raw_id} (video {video_id}): {exc}")
        items[str(raw_id)] = record
        status = "ok" if record["available"] else "UNAVAILABLE"
        LOG.info(f"  rank {rank:2d} item {raw_id:6d} -> video {video_id:6d}  "
                 f"{status:12s} {(record.get('title') or '')[:48]}")

    manifest = {
        "source": "MicroLens official media (recsys.westlake.edu.cn)",
        "source_url": f"{BASE}/{MediaArchive()._url(0).rsplit('/', 1)[-1]}",
        "media_kind": "raw mp4 playback media for the demo feed",
        "verified_mapping": mapping_report["item_mapping_is_bijection"],
        "mapping_source": "MicroLens-100k_pairs.csv joined on exact millisecond timestamps",
        "mapping_report": mapping_report,
        "archive_index": archive.index_meta,
        "user_id": user_id,
        "ranker": args.ranker,
        "top_k": args.top_k,
        "num_items": len(items),
        "num_available": n_ok,
        "num_unavailable": n_fail,
        "items": items,
        "playback": {
            "transcoded": not args.no_transcode,
            "reason": ("the official videos are HEVC/H.265, which browsers cannot decode; "
                       "the playable file is an H.264 transcode of the first "
                       f"{args.clip_seconds:g}s" if not args.no_transcode
                       else "original files served as-is (HEVC; playback may fail)"),
            "clip_seconds": args.clip_seconds,
            "max_height": args.max_height,
        },
        "note": (
            "item_id is the raw MicroLens item id used by the API and the UI; "
            "official_video_id is the MicroLens video identifier, which is also "
            "the mp4 filename stem. "
            "These are raw playback videos for the demo feed; the ranking model "
            "uses ID + text + image features, not the video stream."
        ),
    }
    save_json(manifest, MANIFEST)
    print(f"\nwrote {MANIFEST.relative_to(ROOT)}: {n_ok} playable, {n_fail} unavailable")
    print(f"media in {out_dir.relative_to(ROOT)} "
          f"({sum(p.stat().st_size for p in out_dir.glob('*.mp4'))/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
