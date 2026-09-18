# Media provenance — how a recommendation is matched to a real video

The playable feed (`/watch`) renders recommended items as real MicroLens videos.
This document records exactly how that mapping was established, why it could not
be assumed, and how to re-verify it.

```
python scripts/verify_media_mapping.py     # re-runs every check below
```

## 1. Why a mapping is necessary

The model is trained on the HuggingFace re-upload `sisuo/Microlens_100k`
(`microlens_100k.inter` plus pre-extracted features). That re-upload
**re-indexed both users and items** into a contiguous 0-based range:

```
microlens_100k.inter    userID 0..99999      itemID 0..19737
official release        user   36121...      video  1..19738
```

The two id spaces are **not** the same. Binding a video by assuming
`item_id == filename` would attach the wrong video to every recommendation, so
the permutation has to be recovered from evidence.

## 2. What the official release provides

`https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/`

| file | contents |
|---|---|
| `MicroLens-100k_pairs.csv` | `user, item, timestamp` — 719 405 rows |
| `MicroLens-100k_title_en.csv` | official English titles, keyed by video id 1..19738 |
| `MicroLens-100k_likes_and_views.txt` | `videoID, likes, views` |
| `MicroLens-100k_videos.zip` + `.z01`..`.z222` | the video archive (223 parts, ~477 GB) |

The pairs file has exactly the same row count as the modelling file, and the same
timestamp range — both describe the same interactions.

## 3. Recovering the permutation

`src/media/mapping.py` joins the two files on the **exact millisecond
timestamp**, which is unique per interaction in both:

```
HF (user, item, ts)  ──ts──►  official (user, item, ts)
```

Only rows whose timestamp is unique in *both* files are used, so the join is
unambiguous. The result is accepted only if it is a **function** (one HF item maps
to exactly one official id), **total** (all 19 738 items covered) and
**injective** (no two HF items share an official id). Otherwise the script raises
and nothing downstream runs.

## 4. Verification

| check | result |
|---|---|
| exact millisecond timestamp matches | **719 299 / 719 405 (99.985 %)** |
| item mapping is a bijection | 19 738 HF items → 19 738 distinct video ids, zero collisions |
| user mapping is a bijection | 100 000 HF users → 100 000 distinct official users |
| official video id range | 1 .. 19738 |

A wrong relation would be many-to-many almost immediately: with ~36 timestamp
matches per item, the chance that every match of an item lands on one official id
*and* that no two items collide is negligible.

### Independent cross-check

Timestamps alone would be circular if the two files shared a bug. As an
independent signal, the per-item mean `x_label` (from the modelling file) is
correlated with the official per-video `views` (from a different file), weighted
by interaction count:

| mapping | correlation |
|---|---|
| correct | **+0.0078** |
| shuffled control | +0.0001 |
| ratio | **59×** |

The absolute correlation is small — `x_label` is an engagement bucket, not a view
count — but the correct mapping is two orders of magnitude above the control.

> A nearest-neighbour comparison between the HuggingFace video features (1024-d)
> and the official VideoMAE features (768-d) was **inconclusive**: different
> encoders produce different neighbourhoods, so it neither confirms nor refutes
> the mapping. It is recorded here because it was run and did not help.

## 5. Extracting only the needed videos

Downloading 477 GB is unnecessary. The archive is a split zip whose central
directory lives in the final `.zip` part, so `src/media/archive.py`:

1. range-reads the last 64 KB to find the end-of-central-directory record;
2. range-reads the ~2.2 MB central directory;
3. for each needed video, range-reads its local header + compressed bytes and
   inflates them locally.

Total index cost: ~2 MB. Per video: one range request, verified against the
declared size, CRC32 and the `ftyp` MP4 magic. `artifacts/media_archive_index.json`
caches the parsed index so step 1–2 happen once.

Video files inside the archive are named `MicroLens-100k_videos/<videoID>.mp4`,
so the recovered official id **is** the filename stem.

## 6. Codec conversion

The official videos are **HEVC/H.265** (verified with `ffprobe`). Chromium and
Firefox cannot decode HEVC — the demuxer fails outright with
`DEMUXER_ERROR_COULD_NOT_OPEN`, and the page renders a black rectangle.

`scripts/prepare_media_demo.py` therefore transcodes the first 15 s of each needed
video to H.264 (`libx264`, CRF 26, 720p, no audio, `+faststart`) and records both
codecs in the manifest:

```json
"playback": {
  "transcoded": true,
  "reason": "the official videos are HEVC/H.265, which browsers cannot decode; ...",
  "clip_seconds": 15.0,
  "max_height": 720
}
```

The original file is kept next to it as `<videoID>.source.mp4`, so the served
artefact is never confused with the source. This is the same step a delivery
pipeline performs; it is documented rather than hidden.

## 7. What the manifest records

`artifacts/demo_media_manifest.json`:

```json
{
  "source": "MicroLens official media (recsys.westlake.edu.cn)",
  "verified_mapping": true,
  "mapping_source": "MicroLens-100k_pairs.csv joined on exact millisecond timestamps",
  "mapping_report": { "...": "every number in section 4" },
  "archive_index": { "...": "part count, id range, total bytes" },
  "playback": { "...": "transcode settings and reason" },
  "items": {
    "16981": {
      "item_id": 16981,            // raw MicroLens item id (API / UI id space)
      "official_video_id": 19695,  // archive filename stem
      "title": "...",              // official catalogue title
      "rank": 19, "baseline_rank": 579, "rank_delta": 560,
      "sources": ["semantic"],
      "source_codec": "hevc", "playback_codec": "h264",
      "available": true, "verified": true,
      "local_path": "data/demo_media/19695.mp4",
      "sha256": "..."
    }
  }
}
```

Both id spaces are kept because they are a permutation: `item_id` is what the
API and UI use, `official_video_id` is what the archive uses.

## 8. Licensing and redistribution

The videos are **downloaded locally** from the official MicroLens source and are
**not redistributed** in this repository:

* `data/demo_media/` is in `.gitignore`;
* the README states the same thing;
* the committed demo GIF is a short screen recording of the running application,
  used to illustrate the system.

The archive index (`artifacts/media_archive_index.json`) contains only byte
offsets and sizes — no media content.

## 9. Reproducing

```bash
python scripts/prepare_media_demo.py --fetch-official   # official metadata, ~20 MB
python scripts/prepare_media_demo.py --user-id 68317 --top-k 20
python scripts/verify_media_mapping.py                  # exits non-zero on failure
```

`--no-transcode` serves the original HEVC files (playback will fail in a
browser); `--clip-seconds 0` keeps the full video.
