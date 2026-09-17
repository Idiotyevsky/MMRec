"""Slice a single ranking pass by cold / popularity bucket.

All slices come from one ``EvalResult``, so a cold-start table and an overall
table produced from the same run are guaranteed to be the same ranking.
"""

from __future__ import annotations

import numpy as np

from ..data.popularity import BUCKET_NAMES
from .evaluator import EvalResult


def target_masks(data, targets: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks over the evaluated users, keyed by population."""
    targets = np.asarray(targets, dtype=np.int64)
    masks: dict[str, np.ndarray] = {"all": np.ones(targets.shape[0], dtype=bool)}
    if data.is_cold.any():
        masks["cold"] = data.is_cold[targets]
        masks["warm"] = ~data.is_cold[targets]
    for bid, name in BUCKET_NAMES.items():
        masks[name] = data.popularity_bucket[targets] == bid
    return masks


def sliced_metrics(result: EvalResult, data, ks=(5, 10, 20)) -> dict[str, dict]:
    """Per-population metric bundles from one ranking pass."""
    masks = target_masks(data, result.targets)
    out: dict[str, dict] = {}
    for name, m in masks.items():
        sub = result.subset(m)
        bundle = {f"Recall@{k}": sub.recall_at(k) for k in ks}
        bundle.update({f"NDCG@{k}": sub.ndcg_at(k) for k in ks})
        bundle["num_users"] = int(m.sum())
        out[name] = bundle
    return out


def load_ranking(path) -> dict:
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}
