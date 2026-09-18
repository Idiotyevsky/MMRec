"""Map ShortRec internal item ids to official MicroLens media identifiers.

Why this module exists
----------------------
The dataset used for modelling is the HuggingFace re-upload
``sisuo/Microlens_100k``.  That re-upload **re-indexed both users and items** to a
contiguous 0-based range, so its ``itemID`` is *not* the official MicroLens
``videoID``.  Binding a video to an item by assuming ``item_id == filename`` would
therefore be wrong.

The official release (``recsys.westlake.edu.cn``) ships
``MicroLens-100k_pairs.csv`` with the original ``user, item, timestamp`` triples.
Both files describe the same interactions, so the permutation is recoverable by
joining on the **exact millisecond timestamp**:

    HF (user, item, ts)  --ts-->  official (user, item, ts)

Verification (``scripts/verify_media_mapping.py`` runs it and records the result):

* 719 299 of 719 405 interactions match on an exact millisecond timestamp;
* the induced relation is a **perfect bijection** on both axes — 100 000 HF users
  map to 100 000 distinct official users, 19 738 HF items to 19 738 distinct
  official video ids, with zero collisions.  A wrong relation would be
  many-to-many almost immediately;
* as an independent check, the per-item mean ``x_label`` correlates with the
  official ``likes`` under the correct mapping 24x more strongly than under a
  shuffled control.

Video files in the official archive are named ``<videoID>.mp4`` with
``videoID`` in ``1..19738``, so the recovered id is directly the filename stem.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_DIR = ROOT / "data" / "raw" / "official"
PAIRS_CSV = OFFICIAL_DIR / "MicroLens-100k_pairs.csv"
TITLES_CSV = OFFICIAL_DIR / "MicroLens-100k_title_en.csv"
LIKES_TXT = OFFICIAL_DIR / "MicroLens-100k_likes_and_views.txt"
MAPPING_NPZ = ROOT / "artifacts" / "item_id_mapping.npz"
MAPPING_JSON = ROOT / "artifacts" / "item_id_mapping.json"

NUM_ITEMS = 19738


def build_mapping(
    inter_path: str | Path = ROOT / "data" / "raw" / "microlens_100k.inter",
    pairs_path: str | Path = PAIRS_CSV,
    num_items: int = NUM_ITEMS,
) -> tuple[np.ndarray, dict]:
    """Recover ``hf_item_id -> official video_id`` from exact timestamps.

    Returns ``(lut, report)`` where ``lut[i]`` is the official video id of HF item
    ``i`` and ``report`` documents the evidence.  Raises if the relation is not a
    function, because that would mean the two files do not describe the same
    interactions and no mapping can be trusted.
    """
    import pandas as pd

    inter_path, pairs_path = Path(inter_path), Path(pairs_path)
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"official pairs file not found at {pairs_path}. Download it with "
            "`python scripts/prepare_media_demo.py --fetch-official` or from "
            "https://recsys.westlake.edu.cn/MicroLens-100k-Dataset/"
        )

    ours = pd.read_csv(inter_path, sep="\t").rename(columns={"userID": "user", "itemID": "item"})
    off = pd.read_csv(pairs_path)

    cu, co = ours.timestamp.value_counts(), off.timestamp.value_counts()
    uniq = set(cu[cu == 1].index) & set(co[co == 1].index)
    a = ours[ours.timestamp.isin(uniq)][["user", "item", "timestamp"]].sort_values("timestamp").reset_index(drop=True)
    b = off[off.timestamp.isin(uniq)][["user", "item", "timestamp"]].sort_values("timestamp").reset_index(drop=True)
    if len(a) == 0 or not np.array_equal(a.timestamp.values, b.timestamp.values):
        raise RuntimeError("timestamp join produced no aligned rows; cannot verify the mapping")

    lut = np.full(num_items, -1, dtype=np.int64)
    collisions = 0
    for hf_item, official_item in zip(a.item.values, b.item.values):
        if lut[hf_item] == -1:
            lut[hf_item] = official_item
        elif lut[hf_item] != official_item:
            collisions += 1

    if collisions or (lut < 0).any():
        raise RuntimeError(
            f"mapping is not a bijection ({collisions} conflicting rows, "
            f"{int((lut < 0).sum())} unmapped items); refusing to use it"
        )
    if len(set(lut.tolist())) != num_items:
        raise RuntimeError("mapping is not injective; refusing to use it")

    user_pairs = pd.DataFrame({"hf": a.user.values, "off": b.user.values}).drop_duplicates()
    report = {
        "method": "exact millisecond timestamp join",
        "hf_interactions": int(len(ours)),
        "official_interactions": int(len(off)),
        "unambiguous_timestamp_matches": int(len(a)),
        "matched_fraction": round(len(a) / len(ours), 6),
        "items_mapped": int((lut >= 0).sum()),
        "official_video_id_range": [int(lut.min()), int(lut.max())],
        "item_mapping_is_bijection": True,
        "user_pairs_distinct_hf": int(user_pairs.hf.nunique()),
        "user_mapping_is_bijection": bool(user_pairs.hf.nunique() == len(user_pairs)),
        "sources": {
            "hf_interactions": str(inter_path.relative_to(ROOT)),
            "official_pairs": str(pairs_path.relative_to(ROOT)),
        },
    }
    return lut, report


def load_or_build_mapping(force: bool = False) -> tuple[np.ndarray, dict]:
    """Cached accessor: build the mapping once, reuse the artifact afterwards."""
    if not force and MAPPING_NPZ.exists():
        npz = np.load(MAPPING_NPZ)
        lut = npz["hf_item_to_official_video_id"]
        report = json.loads(MAPPING_JSON.read_text(encoding="utf-8")) if MAPPING_JSON.exists() else {}
        return lut, report
    lut, report = build_mapping()
    save_mapping(lut, report)
    return lut, report


def save_mapping(lut: np.ndarray, report: dict) -> None:
    MAPPING_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MAPPING_NPZ, hf_item_to_official_video_id=lut)
    MAPPING_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_titles(path: str | Path = TITLES_CSV) -> dict[int, str]:
    """Official English titles, keyed by official video id.

    These are **real** catalogue metadata, not inferred from features.
    """
    import csv

    path = Path(path)
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                out[int(row[0])] = row[1].strip()
            except ValueError:
                continue
    return out


def load_likes_views(path: str | Path = LIKES_TXT) -> dict[int, tuple[int, int]]:
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[int, tuple[int, int]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                out[int(parts[0])] = (int(parts[1]), int(parts[2]))
            except ValueError:
                continue
    return out
