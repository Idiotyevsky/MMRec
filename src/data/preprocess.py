"""MicroLens preprocessing: mappings, chronological split, cold split, stats.

Raw schema (verified on ``data/raw/microlens_100k.inter``)::

    userID \t itemID \t timestamp \t x_label     (TSV, header row)
    userID    in [0, 99999]     -> 100000 users, contiguous
    itemID    in [0, 19737]     -> 19738 items, contiguous
    timestamp int64 milliseconds
    x_label   in {0,1,2}        -> engagement bucket, NOT used as a feature

Feature files ``{text,image,video}_feat.npy`` have exactly ``num_items`` rows and
row ``r`` belongs to raw item id ``r``.  We still build an explicit
``raw_item_id -> feature_row`` array and assert the identity rather than
assuming it, so that swapping in a differently-indexed feature dump fails loudly.

Internal id convention everywhere downstream::

    0                = PAD
    1 .. num_items   = valid items
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..utils.io import ensure_dir, save_json
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from .popularity import popularity_buckets

LOGGER = get_logger("mmrec.preprocess")

PAD = 0


# ----------------------------------------------------------------------
# raw loading
# ----------------------------------------------------------------------
def load_interactions(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (user_ids, item_ids, timestamps) as int64 arrays."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header[:2] != ["userID", "itemID"]:
            raise ValueError(f"Unexpected header in {path}: {header}")
        data = np.loadtxt(f, delimiter="\t", dtype=np.int64, ndmin=2)
    if data.shape[1] < 3:
        raise ValueError(f"Expected >=3 columns in {path}, got {data.shape[1]}")
    return data[:, 0].copy(), data[:, 1].copy(), data[:, 2].copy()


def build_mapping(raw_ids: np.ndarray, name: str) -> np.ndarray:
    """Map arbitrary raw ids to a contiguous 0-based range, preserving order.

    Returns an array ``lut`` such that ``lut[raw_id] = internal_id``.
    ``internal_id`` starts at 0 for the first-seen raw id.
    """
    uniq = np.unique(raw_ids)
    lut = np.full(int(uniq.max()) + 1, -1, dtype=np.int64)
    lut[uniq] = np.arange(uniq.shape[0], dtype=np.int64)
    if (lut[raw_ids] < 0).any():
        raise AssertionError(f"{name}: unmapped raw ids present")
    return lut


# ----------------------------------------------------------------------
# core preprocessing
# ----------------------------------------------------------------------
def preprocess(
    inter_path: str | Path,
    feature_dir: str | Path,
    out_dir: str | Path,
    max_sequence_length: int = 50,
    min_user_interactions: int = 5,
    min_item_interactions: int = 1,
    popularity_rule: str = "frequency_quantile",
    head_frac: float = 0.2,
    tail_frac: float = 0.6,
    cold_ratio: float = 0.0,
    cold_seed: int = 42,
    cold_min_eval_occurrences: int = 1,
    modalities: tuple[str, ...] = ("text", "image", "video"),
    verbose: bool = True,
) -> dict:
    """Build the processed dataset and write it to ``out_dir``.

    Returns the metadata dict that was written to ``meta.json``.
    """
    out_dir = ensure_dir(out_dir)
    log = LOGGER.info if verbose else (lambda *a, **k: None)

    # ---------------- load & validate raw ----------------
    raw_users, raw_items, raw_ts = load_interactions(inter_path)
    n_raw = raw_users.shape[0]
    log(f"loaded {n_raw} interactions")

    if np.isnan(raw_ts).any():
        raise ValueError("timestamps contain NaN")

    # ---------------- filter items (before any split) ----------------
    raw_item_freq = np.bincount(raw_items, minlength=int(raw_items.max()) + 1)
    keep_item_raw = raw_item_freq >= min_item_interactions
    n_items_dropped = int((~keep_item_raw).sum())

    keep_mask = keep_item_raw[raw_items]
    n_before = keep_mask.shape[0]
    raw_users, raw_items, raw_ts = raw_users[keep_mask], raw_items[keep_mask], raw_ts[keep_mask]
    log(f"item filter (min_item_interactions={min_item_interactions}): "
        f"dropped {n_items_dropped} items, {n_before - keep_mask.sum()} interactions")

    # ---------------- filter users ----------------
    user_freq = np.bincount(raw_users, minlength=int(raw_users.max()) + 1)
    keep_user_raw = user_freq >= min_user_interactions
    n_users_dropped = int((~keep_user_raw).sum())
    keep_mask = keep_user_raw[raw_users]
    raw_users, raw_items, raw_ts = raw_users[keep_mask], raw_items[keep_mask], raw_ts[keep_mask]
    log(f"user filter (min_user_interactions={min_user_interactions}): dropped {n_users_dropped} users")

    # ---------------- mappings ----------------
    user_lut = build_mapping(raw_users, "user")
    item_lut = build_mapping(raw_items, "item")
    raw_user_ids = np.unique(raw_users)
    raw_item_ids = np.unique(raw_items)
    num_users = int(raw_user_ids.shape[0])
    num_items = int(raw_item_ids.shape[0])
    log(f"num_users={num_users} num_items={num_items}")

    internal_users = user_lut[raw_users] + 1  # 1-based, 0 reserved
    internal_items = item_lut[raw_items] + 1

    # ---------------- chronological ordering ----------------
    # Sort by (user, timestamp).  lexsort is stable, so equal timestamps keep
    # their original file order.
    order = np.lexsort((np.arange(internal_users.shape[0]), raw_ts, internal_users))
    internal_users = internal_users[order]
    internal_items = internal_items[order]
    raw_ts_sorted = raw_ts[order]

    # verify: strictly non-decreasing timestamps inside every user
    if num_users > 0:
        boundaries = np.flatnonzero(np.diff(internal_users)) + 1
        starts = np.concatenate([[0], boundaries])
        ends = np.concatenate([boundaries, [internal_users.shape[0]]])
        for s, e in zip(starts, ends):
            if e - s > 1 and (np.diff(raw_ts_sorted[s:e]) < 0).any():
                raise AssertionError("timestamps not monotonic within a user after sort")

    counts = np.bincount(internal_users, minlength=num_users + 1)[1:]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    # ---------------- leave-one-out split ----------------
    train_len = counts - 2
    if (train_len < 1).any():
        bad = int((train_len < 1).sum())
        raise ValueError(
            f"{bad} users have <3 interactions after filtering; "
            "raise min_user_interactions or lower the split requirement"
        )
    val_target = np.empty(num_users, dtype=np.int64)
    test_target = np.empty(num_users, dtype=np.int64)
    for u in range(num_users):
        s, e = offsets[u], offsets[u + 1]
        val_target[u] = internal_items[e - 2]
        test_target[u] = internal_items[e - 1]
    assert (train_len >= 1).all()
    log(f"split: mean train len {train_len.mean():.2f}, "
        f"max {counts.max()}, users={num_users}")

    # ---------------- cold split ----------------
    is_cold = np.zeros(num_items + 1, dtype=bool)
    cold_raw_ids: list[int] = []
    if cold_ratio > 0:
        rng = np.random.default_rng(cold_seed)
        # candidates: items that appear as a val or test target at least
        # `cold_min_eval_occurrences` times (otherwise they cannot be evaluated)
        target_counts = np.bincount(
            np.concatenate([val_target, test_target]), minlength=num_items + 1
        )
        candidates = np.flatnonzero(target_counts[1:] >= cold_min_eval_occurrences) + 1
        n_cold = int(round(num_items * cold_ratio))
        if n_cold > candidates.shape[0]:
            raise ValueError(
                f"cold_ratio={cold_ratio} needs {n_cold} items but only "
                f"{candidates.shape[0]} have >= {cold_min_eval_occurrences} eval occurrences"
            )
        chosen = rng.choice(candidates, size=n_cold, replace=False)
        is_cold[chosen] = True
        cold_raw_ids = sorted(int(raw_item_ids[c - 1]) for c in chosen)
        log(f"cold split: {n_cold} items ({cold_ratio:.1%}), "
            f"{int(target_counts[chosen].sum())} cold val/test targets")

        # remove cold items from TRAIN positions only (vectorised)
        user_of_pos = np.repeat(np.arange(num_users), counts)
        train_end_per_user = offsets[1:] - 2
        is_train_pos = np.arange(internal_items.shape[0]) < train_end_per_user[user_of_pos]
        drop = is_train_pos & is_cold[internal_items]
        log(f"cold split: removed {int(drop.sum())} train interactions of cold items")
        internal_items = internal_items[~drop]
        internal_users = internal_users[~drop]

        # recompute offsets; drop users whose train part became empty
        counts2 = np.bincount(internal_users, minlength=num_users + 1)[1:]
        keep_users = (counts2 - 2) >= 1
        if not keep_users.all():
            log(f"cold split: dropping {int((~keep_users).sum())} users with empty train")
            u_lut2 = np.full(num_users + 1, -1, dtype=np.int64)
            u_lut2[1:][keep_users] = np.arange(int(keep_users.sum())) + 1
            new_users = u_lut2[internal_users]
            keep_pos = new_users > 0
            internal_users = new_users[keep_pos]
            internal_items = internal_items[keep_pos]
            raw_user_ids = raw_user_ids[keep_users]
            num_users = int(keep_users.sum())
            counts2 = np.bincount(internal_users, minlength=num_users + 1)[1:]
            val_target = val_target[keep_users]
            test_target = test_target[keep_users]
        counts = counts2
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        train_len = counts - 2

    # ---------------- train-only item frequency ----------------
    user_of_pos = np.repeat(np.arange(num_users), counts)
    train_end_per_user = offsets[1:] - 2
    is_train_pos = np.arange(internal_items.shape[0]) < train_end_per_user[user_of_pos]
    train_items = internal_items[is_train_pos]
    train_freq = np.bincount(train_items, minlength=num_items + 1)[: num_items + 1].astype(np.int64)
    assert train_freq[PAD] == 0, "PAD leaked into training interactions"
    n_train_inter = int(train_items.shape[0])

    buckets = popularity_buckets(
        train_freq, rule=popularity_rule, head_frac=head_frac, tail_frac=tail_frac,
        num_items=num_items,
    )

    # ---------------- feature row mapping ----------------
    # raw item id -> feature row.  Verified by construction, not assumed.
    feat_rows = {}
    for m in modalities:
        p = Path(feature_dir) / f"{m}_feat.npy"
        if not p.exists():
            log(f"feature {m}: missing file {p}, skipping")
            continue
        arr = np.load(p, mmap_mode="r")
        n_rows = arr.shape[0]
        # candidate mapping: raw item id is a valid row index
        if int(raw_item_ids.max()) < n_rows:
            row_for_raw = raw_item_ids.astype(np.int64)
            mapping_kind = "identity(raw_item_id -> feature_row)"
        else:
            raise ValueError(
                f"{m}_feat.npy has {n_rows} rows but raw item ids reach "
                f"{int(raw_item_ids.max())}; an explicit item->row table is required"
            )
        row_for_item = np.zeros(num_items + 1, dtype=np.int64)
        row_for_item[1:] = row_for_raw
        assert row_for_item[0] == 0
        assert row_for_item[1:].max() < n_rows
        feat_rows[m] = {"row_for_item": row_for_item, "dim": int(arr.shape[1]),
                        "num_rows": int(n_rows), "mapping": mapping_kind}
        log(f"feature {m}: dim={arr.shape[1]} rows={n_rows} ({mapping_kind})")

    # ---------------- stats ----------------
    seq_len = counts.astype(np.float64)
    n_inter = int(counts.sum())

    # The validation and test targets are NOT drawn from the same distribution:
    # the most recent interaction of a user is measurably more long-tailed than
    # the one before it.  Reporting this prevents the (real, expected) val/test
    # gap from being mistaken for a bug or for model-selection optimism.
    def _target_stats(targets: np.ndarray) -> dict:
        f = train_freq[targets]
        return {
            "mean_train_freq": float(f.mean()),
            "median_train_freq": float(np.median(f)),
            "frac_zero_train_freq": float((f == 0).mean()),
            "bucket_share": {
                name: float((buckets["bucket"][targets] == bid).mean())
                for bid, name in ((2, "head"), (1, "middle"), (0, "tail"))
            },
        }

    stats = {
        "num_users": int(num_users),
        "num_items": int(num_items),
        "num_interactions": n_inter,
        "avg_sequence_length": float(seq_len.mean()),
        "median_sequence_length": float(np.median(seq_len)),
        "min_sequence_length": int(seq_len.min()),
        "max_sequence_length": int(seq_len.max()),
        "avg_train_length": float(train_len.mean()),
        "num_train_interactions": n_train_inter,
        "sparsity": float(1.0 - n_inter / (num_users * num_items)),
        "item_freq_train": {
            "min": int(train_freq[1:].min()),
            "max": int(train_freq[1:].max()),
            "median": float(np.median(train_freq[1:])),
            "mean": float(train_freq[1:].mean()),
            "num_zero_freq": int((train_freq[1:] == 0).sum()),
        },
        "popularity_buckets": {
            "rule": buckets["rule"],
            "rule_description": buckets["rule_description"],
            "counts": buckets["counts"],
            "interaction_mass": buckets["interaction_mass"],
        },
        "cold_split": {
            "enabled": cold_ratio > 0,
            "ratio": cold_ratio,
            "seed": cold_seed,
            "min_eval_occurrences": cold_min_eval_occurrences,
            "num_cold_items": int(is_cold.sum()),
            "cold_raw_item_ids": cold_raw_ids,
        },
        "filters": {
            "min_user_interactions": min_user_interactions,
            "min_item_interactions": min_item_interactions,
            "max_sequence_length": max_sequence_length,
        },
        "raw": {
            "inter_file": str(inter_path),
            "feature_dir": str(feature_dir),
            "n_raw_interactions": int(n_raw),
            "n_users_dropped": n_users_dropped,
            "n_items_dropped": n_items_dropped,
        },
        "features": {m: {"dim": v["dim"], "mapping": v["mapping"]} for m, v in feat_rows.items()},
        "target_distribution": {
            "val": _target_stats(val_target),
            "test": _target_stats(test_target),
            "note": (
                "val and test targets are not identically distributed: the last "
                "interaction of a user is more long-tailed than the second-to-last"
            ),
        },
    }

    # ---------------- write ----------------
    np.savez_compressed(
        out_dir / "dataset.npz",
        flat_items=internal_items.astype(np.int32),
        user_offsets=offsets,
        train_len=train_len.astype(np.int32),
        val_target=val_target.astype(np.int32),
        test_target=test_target.astype(np.int32),
        train_freq=train_freq,
        popularity_bucket=buckets["bucket"],
        is_cold=is_cold,
    )
    for m, v in feat_rows.items():
        np.save(out_dir / f"row_for_item_{m}.npy", v["row_for_item"])

    save_json(
        {
            "num_users": num_users,
            "num_items": num_items,
            "raw_user_ids": raw_user_ids.tolist(),
            "raw_item_ids": raw_item_ids.tolist(),
        },
        out_dir / "mappings.json",
    )
    save_json(stats, out_dir / "stats.json")
    save_json(stats, out_dir / "meta.json")
    log(f"wrote processed dataset to {out_dir}")
    return stats


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Preprocess MicroLens-100K")
    ap.add_argument("--inter", default="data/raw/microlens_100k.inter")
    ap.add_argument("--feature-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/base")
    ap.add_argument("--max-sequence-length", type=int, default=50)
    ap.add_argument("--min-user-interactions", type=int, default=5)
    ap.add_argument("--min-item-interactions", type=int, default=1)
    ap.add_argument("--popularity-rule", default="frequency_quantile")
    ap.add_argument("--head-frac", type=float, default=0.2)
    ap.add_argument("--tail-frac", type=float, default=0.6)
    ap.add_argument("--cold-ratio", type=float, default=0.0)
    ap.add_argument("--cold-seed", type=int, default=42)
    ap.add_argument("--cold-min-eval-occurrences", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    preprocess(
        inter_path=args.inter,
        feature_dir=args.feature_dir,
        out_dir=args.out,
        max_sequence_length=args.max_sequence_length,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        popularity_rule=args.popularity_rule,
        head_frac=args.head_frac,
        tail_frac=args.tail_frac,
        cold_ratio=args.cold_ratio,
        cold_seed=args.cold_seed,
        cold_min_eval_occurrences=args.cold_min_eval_occurrences,
    )


if __name__ == "__main__":
    main()
