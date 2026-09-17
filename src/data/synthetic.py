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
    structure: str = "iid",
    seed: int = 0,
) -> Path:
    """``structure`` controls whether the next item depends on the current one.

    ``iid`` draws every interaction independently from a popularity-skewed
    distribution.  That makes *Popular* the Bayes-optimal predictor, so it can
    never be used to check "SASRec beats Popular" -- there is no sequential
    signal to learn.  ``sequential`` walks forward from a popularity-skewed start
    item, so the next item really does depend on the previous one.
    """
    out_dir = ensure_dir(out_dir)
    rng = np.random.default_rng(seed)

    # popularity-skewed item distribution so long-tail buckets are meaningful
    item_p = rng.dirichlet(np.full(num_items, 0.3))
    lengths = rng.integers(min_len, max_len + 1, size=num_users)
    if max_len > num_items:
        raise ValueError("synthetic generator needs num_items >= max_len")
    if structure not in ("iid", "sequential"):
        raise ValueError(f"unknown structure {structure!r}")

    # each user visits distinct items, so (user, item) pairs stay unique exactly
    # as in the real data
    sequences: list[np.ndarray] = []
    for u in range(num_users):
        n = int(lengths[u])
        if structure == "iid":
            seq = rng.choice(num_items, size=n, replace=False, p=item_p)
        else:
            # forward walk: the successor of item i is i+1..i+3 whenever those are
            # still free, which is a genuinely learnable sequential pattern
            used = [int(rng.choice(num_items, p=item_p))]
            seen = {used[0]}
            while len(used) < n:
                cur = used[-1]
                cands = [cur + s for s in (1, 2, 3) if cur + s < num_items and cur + s not in seen]
                if cands:
                    nxt = int(rng.choice(cands))
                else:  # walk ran out of room: fall back to an unused item
                    pool = np.array([i for i in range(num_items) if i not in seen], dtype=np.int64)
                    w = item_p[pool]
                    nxt = int(rng.choice(pool, p=w / w.sum()))
                used.append(nxt)
                seen.add(nxt)
            seq = np.array(used, dtype=np.int64)
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


def make_deterministic_dataset(
    out_dir: str | Path,
    num_users: int = 24,
    num_items: int = 40,
    window: int = 20,
    text_dim: int = 16,
    image_dim: int = 16,
    video_dim: int = 8,
    seed: int = 0,
) -> Path:
    """Chains ``i, i+1, i+2, ...``: every user owns a window of the chain.

    The successor of an item is a deterministic function of that item, and the
    windows overlap so that *every* transition ``x -> x+1`` is observed in some
    user's training prefix -- including the ones that are another user's
    validation target.  A correct autoregressive model can therefore drive the
    loss to ~0 and rank the true successor first; one trained with a broken
    objective (a single hidden state scored against every target) cannot, which
    is why this fixture is the end-to-end proof of the training objective.
    """
    out_dir = ensure_dir(out_dir)
    rng = np.random.default_rng(seed)
    if not 2 < window <= num_items:
        raise ValueError("need 2 < window <= num_items")

    cycle = np.arange(1, num_items + 1, dtype=np.int64)
    n_starts = num_items - window + 1
    length = window

    seqs = []
    for u in range(num_users):
        start = int(u % n_starts)
        seqs.append(cycle[start : start + window].copy())

    counts = np.full(num_users, length, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    flat = np.concatenate(seqs).astype(np.int32)
    train_len = counts - 2
    val_target = np.array([s[-2] for s in seqs], dtype=np.int64)
    test_target = np.array([s[-1] for s in seqs], dtype=np.int64)

    is_train = np.arange(flat.shape[0]) < (offsets[1:] - 2)[
        np.repeat(np.arange(num_users), counts)
    ]
    train_freq = np.bincount(flat[is_train], minlength=num_items + 1)[: num_items + 1]
    buckets = popularity_buckets(train_freq, rule="frequency_quantile", num_items=num_items)

    np.savez_compressed(
        out_dir / "dataset.npz",
        flat_items=flat,
        user_offsets=offsets,
        train_len=train_len.astype(np.int32),
        val_target=val_target.astype(np.int32),
        test_target=test_target.astype(np.int32),
        train_freq=train_freq.astype(np.int64),
        popularity_bucket=buckets["bucket"],
        is_cold=np.zeros(num_items + 1, dtype=bool),
    )

    text = rng.normal(size=(num_items, text_dim)).astype(np.float32)
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    np.save(out_dir / "text_feat.npy", text)
    np.save(out_dir / "image_feat.npy", rng.normal(size=(num_items, image_dim)).astype(np.float32))
    np.save(out_dir / "video_feat.npy", rng.normal(size=(num_items, video_dim)).astype(np.float32))
    lut = np.zeros(num_items + 1, dtype=np.int64)
    lut[1:] = np.arange(num_items, dtype=np.int64)
    for m in ("text", "image", "video"):
        np.save(out_dir / f"row_for_item_{m}.npy", lut)

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
        "cold_split": {"enabled": False, "ratio": 0.0, "seed": seed,
                       "min_eval_occurrences": 1, "num_cold_items": 0, "cold_raw_item_ids": []},
        "filters": {"min_user_interactions": 1, "min_item_interactions": 1,
                    "max_sequence_length": int(np.diff(offsets).max())},
        "raw": {"inter_file": "synthetic-deterministic", "feature_dir": str(out_dir),
                "n_raw_interactions": int(flat.shape[0]), "n_users_dropped": 0, "n_items_dropped": 0},
        "features": {m: {"dim": d, "mapping": "identity(raw_item_id -> feature_row)"}
                     for m, d in (("text", text_dim), ("image", image_dim), ("video", video_dim))},
        "synthetic": True,
        "deterministic_transitions": True,
        "chain": cycle.tolist(),
    }
    save_json(stats, out_dir / "stats.json")
    save_json(stats, out_dir / "meta.json")
    return out_dir
