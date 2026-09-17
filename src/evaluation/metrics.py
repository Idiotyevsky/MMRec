"""Ranking metrics for single-target (leave-one-out) sequential recommendation.

All functions take a 1-based ``rank`` array (``inf`` / a large sentinel when the
target was not retrieved) so that a single ranking pass can be sliced into
overall / cold / per-popularity-bucket reports without recomputing scores.

Tie policy: the evaluator computes the **average rank**

    rank = 1 + #{items scoring strictly higher} + (#{items scoring equal} - 1) / 2

For a uniquely-scored target this is the ordinary rank.  For a target tied with
``n`` other items it is the midpoint, which is the only tie policy that does not
silently favour or punish a model.  This matters concretely for cold-item
evaluation: an ID-only model gives every cold item a score of exactly 0, and an
optimistic rank would report a perfect hit for every one of them.
"""

from __future__ import annotations

import numpy as np

#: sentinel rank for a target that is not reachable in the candidate set
NOT_RETRIEVED = float("inf")


def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def recall_at_k(rank: np.ndarray, k: int) -> float:
    """Fraction of users whose single ground-truth item is ranked in the top k."""
    r = _as_array(rank)
    if r.size == 0:
        return float("nan")
    return float((r <= k).mean())


def hit_rate_at_k(rank: np.ndarray, k: int) -> float:
    """Identical to Recall for a single ground truth; kept for reporting."""
    return recall_at_k(rank, k)


def ndcg_at_k(rank: np.ndarray, k: int) -> float:
    """Normalised DCG at k for a single relevant item (IDCG = 1)."""
    r = _as_array(rank)
    if r.size == 0:
        return float("nan")
    gains = np.where(r <= k, 1.0 / np.log2(r + 1.0), 0.0)
    return float(gains.mean())


def mrr_at_k(rank: np.ndarray, k: int) -> float:
    """Mean reciprocal rank at k."""
    r = _as_array(rank)
    if r.size == 0:
        return float("nan")
    rr = np.where(r <= k, 1.0 / r, 0.0)
    return float(rr.mean())


def coverage_at_k(topk: np.ndarray, num_items: int) -> float:
    """Fraction of the catalogue that appears in at least one user's top-k."""
    if topk.size == 0:
        return 0.0
    distinct = np.unique(topk[topk > 0]).size
    return float(distinct / max(num_items, 1))


def metric_bundle(rank: np.ndarray, topk: np.ndarray, num_items: int, ks=(5, 10, 20)) -> dict:
    out: dict[str, float] = {}
    for k in ks:
        out[f"Recall@{k}"] = recall_at_k(rank, k)
        out[f"NDCG@{k}"] = ndcg_at_k(rank, k)
        out[f"MRR@{k}"] = mrr_at_k(rank, k)
        out[f"HitRate@{k}"] = hit_rate_at_k(rank, k)
    out["Coverage@20"] = coverage_at_k(topk, num_items)
    out["num_users"] = int(np.asarray(rank).size)
    return out
