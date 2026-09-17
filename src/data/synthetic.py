"""Generate a tiny synthetic dataset in the *processed* format.

Used by unit tests and ``scripts/smoke_test.py`` so that the whole pipeline
(data -> model -> training -> evaluation -> checkpoint) can be exercised without
the real MicroLens files.  The generator writes exactly the same artifacts as
``src.data.preprocess``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..utils.io import ensure_dir, save_json
from .popularity import popularity_buckets


def make_synthetic_dataset(
    out_dir: str | Path,
    num_users: int = 60,
    num_items: int = 40,
    min_len: int = 5,
    max_len: int = 12,
    text_dim: int = 16,
    image_dim: int = 24,
    video_dim: int = 8,
    cold_ratio: float = 0.0,
    seed: int = 0,
) -> Path:
    out_dir = ensure_dir(out_dir)
    rng = np.random.default_rng(seed)

    # popularity-skewed item distribution so long-tail buckets are meaningful
    item_p = rng.dirichlet(np.full(num_items, 0.3))
    lengths = rng.integers(min_len, max_len + 1, size=num_users)
    if max_len > num_items:
        raise ValueError("synthetic generator needs num_items >= max_len")

    # each user draws distinct items from a popularity-skewed distribution, so
    # (user, item) pairs stay unique (as in the real data) while popular items
    # remain statistically learnable
    sequences: list[np.ndarray] = []
    for u in range(num_users):
        n = int(lengths[u])
        seq = rng.choice(num_items, size=n, replace=False, p=item_p)
        sequences.append(seq.astype(np.int64))

    flat = np.concatenate(sequences) + 1  # 0 = PAD
    counts = np.array([s.shape[0] for s in sequences], dtype=np.int64)
    user_of = np.repeat(np.arange(num_users, dtype=np.int64), counts)
    val_target = np.array([sequences[u][-2] for u in range(num_users)], dtype=np.int64) + 1
    test_target = np.array([sequences[u][-1] for u in range(num_users)], dtype=np.int64) + 1

    is_cold = np.zeros(num_items + 1, dtype=bool)
    if cold_ratio > 0:
        n_cold = max(1, int(round(num_items * cold_ratio)))
        is_cold[rng.choice(num_items, size=n_cold, replace=False) + 1] = True

    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    train_end = offsets[1:] - 2
    is_train = np.arange(flat.shape[0]) < train_end[user_of]

    if cold_ratio > 0:
        drop = is_train & is_cold[flat]
        flat = flat[~drop]
        user_of = user_of[~drop]
        counts2 = np.bincount(user_of, minlength=num_users)[:num_users]
        keep_users = (counts2 - 2) >= 1
        lut = np.full(num_users, -1, dtype=np.int64)
        lut[keep_users] = np.arange(int(keep_users.sum()))
        new_user = lut[user_of]
        keep = new_user >= 0
        flat = flat[keep]
        user_of = new_user[keep]
        num_users = int(keep_users.sum())
        counts = np.bincount(user_of, minlength=num_users)[:num_users]
        val_target = val_target[keep_users]
        test_target = test_target[keep_users]

    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    train_len = counts - 2
    train_end = offsets[1:] - 2
    is_train = np.arange(flat.shape[0]) < train_end[user_of]
    flat_internal = flat
    train_freq = np.bincount(flat_internal[is_train], minlength=num_items + 1)[: num_items + 1]
    buckets = popularity_buckets(train_freq, rule="frequency_quantile", num_items=num_items)

    np.savez_compressed(
        out_dir / "dataset.npz",
        flat_items=flat_internal.astype(np.int32),
        user_offsets=offsets,
        train_len=train_len.astype(np.int32),
        val_target=val_target.astype(np.int32),
        test_target=test_target.astype(np.int32),
        train_freq=train_freq.astype(np.int64),
        popularity_bucket=buckets["bucket"],
        is_cold=is_cold,
    )

    # features: item i (raw 0-based) -> row i, matching the real data layout
    text = rng.normal(size=(num_items, text_dim)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    image = rng.normal(size=(num_items, image_dim)).astype(np.float32)
    video = rng.normal(size=(num_items, video_dim)).astype(np.float32)
    np.save(out_dir / "text_feat.npy", text)
    np.save(out_dir / "image_feat.npy", image)
    np.save(out_dir / "video_feat.npy", video)

    lut_arr = np.zeros(num_items + 1, dtype=np.int64)
    lut_arr[1:] = np.arange(num_items, dtype=np.int64)
    for m in ("text", "image", "video"):
        np.save(out_dir / f"row_for_item_{m}.npy", lut_arr)

    save_json(
        {"num_users": num_users, "num_items": num_items,
         "raw_user_ids": list(range(num_users)), "raw_item_ids": list(range(num_items))},
        out_dir / "mappings.json",
    )
    stats = {
        "num_users": num_users,
        "num_items": num_items,
        "num_interactions": int(offsets[-1]),
        "avg_sequence_length": float(np.diff(offsets).mean()),
        "median_sequence_length": float(np.median(np.diff(offsets))),
        "min_sequence_length": int(np.diff(offsets).min()),
        "max_sequence_length": int(np.diff(offsets).max()),
        "avg_train_length": float(train_len.mean()),
        "num_train_interactions": int(is_train.sum()),
        "sparsity": float(1.0 - offsets[-1] / (num_users * num_items)),
        "item_freq_train": {"min": int(train_freq[1:].min()), "max": int(train_freq[1:].max()),
                            "median": float(np.median(train_freq[1:])),
                            "mean": float(train_freq[1:].mean()),
                            "num_zero_freq": int((train_freq[1:] == 0).sum())},
        "popularity_buckets": {"rule": buckets["rule"], "rule_description": buckets["rule_description"],
                               "counts": buckets["counts"], "interaction_mass": buckets["interaction_mass"]},
        "cold_split": {"enabled": cold_ratio > 0, "ratio": cold_ratio, "seed": seed,
                       "min_eval_occurrences": 1, "num_cold_items": int(is_cold.sum()),
                       "cold_raw_item_ids": []},
        "filters": {"min_user_interactions": min_len, "min_item_interactions": 1,
                    "max_sequence_length": max_len},
        "raw": {"inter_file": "synthetic", "feature_dir": str(out_dir),
                "n_raw_interactions": int(flat.shape[0]), "n_users_dropped": 0, "n_items_dropped": 0},
        "features": {m: {"dim": d, "mapping": "identity(raw_item_id -> feature_row)"}
                     for m, d in (("text", text_dim), ("image", image_dim), ("video", video_dim))},
        "synthetic": True,
    }
    save_json(stats, out_dir / "stats.json")
    save_json(stats, out_dir / "meta.json")
    return out_dir
