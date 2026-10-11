"""Exact, train-only Swing I2I index with ItemCF-compatible online aggregation.

For distinct users u < v and items i != j in their common training items:

    sim(i,j) += (alpha1+|Iu|)^(-beta) * (alpha1+|Iv|)^(-beta)
                / (alpha2+|Iu intersect Iv|)

Unordered user pairs are counted once, without item-popularity normalization.
No user/item subsampling or interaction caps are applied. Sparse user-overlap
blocks bound intermediate memory; only pairs sharing >=2 items can contribute.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from .itemcf import ItemCFRecall, load_itemcf_index, save_itemcf_index
from ..utils.provenance import dataset_fingerprint


def build_swing_index(flat_items, user_offsets, train_len, num_items,
                      top_m=100, alpha1=5.0, alpha2=1.0, beta=0.3,
                      chunk_size=1024, verbose=True):
    """Build symmetric Swing scores before per-item top-M truncation."""
    from scipy import sparse

    if (num_items < 1 or top_m < 1 or chunk_size < 1 or
            not np.isfinite([alpha1, alpha2, beta]).all() or
            alpha1 < 0 or alpha2 <= 0 or beta < 0):
        raise ValueError("invalid Swing dimensions or smoothing parameters")
    flat = np.asarray(flat_items, dtype=np.int64)
    offsets = np.asarray(user_offsets, dtype=np.int64)
    lengths = np.asarray(train_len, dtype=np.int64)
    if (offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or
            offsets[-1] != len(flat) or (np.diff(offsets) < 0).any() or
            lengths.shape != (len(offsets)-1,) or (lengths < 0).any() or
            (lengths > np.diff(offsets)).any()):
        raise ValueError("invalid training-prefix boundaries")
    histories = []
    for start, length in zip(offsets[:-1], lengths):
        items = np.unique(flat[start:start+length])
        if (items < 0).any() or (items > num_items).any():
            raise ValueError("training item id out of range")
        histories.append(items[items > 0])
    sizes = np.asarray([len(h) for h in histories], dtype=np.int64)
    items = np.concatenate(histories)
    users = np.repeat(np.arange(len(histories)), sizes)
    x = sparse.csr_matrix((np.ones(len(items), dtype=np.int32), (users, items)),
                          shape=(len(histories), num_items+1))
    xt = x.T.tocsc()
    sets = [set(h.tolist()) for h in histories]
    weights = np.zeros(len(sizes), dtype=np.float64)
    nonempty = sizes > 0
    weights[nonempty] = (alpha1 + sizes[nonempty]) ** (-beta)
    scores = [defaultdict(float) for _ in range(num_items+1)]
    n_pairs = 0
    for start in range(0, len(histories), chunk_size):
        overlap = (x[start:start+chunk_size] @ xt).tocoo()
        keep = (overlap.data >= 2) & (overlap.col > overlap.row + start)
        for u, v, count in zip(overlap.row[keep] + start, overlap.col[keep], overlap.data[keep]):
            common = sorted(sets[u] & sets[v])
            contribution = float(weights[u] * weights[v] / (alpha2 + count))
            for i, j in combinations(common, 2):
                scores[i][j] += contribution
                scores[j][i] += contribution
        n_pairs += int(keep.sum())
        if verbose and (start // chunk_size) % 10 == 0:
            print(f"swing: {min(start+chunk_size,len(histories))}/{len(histories)} users; "
                  f"{n_pairs} contributing user pairs", flush=True)
    neighbors = np.zeros((num_items+1, top_m), dtype=np.int32)
    sims = np.zeros_like(neighbors, dtype=np.float32)
    for item, values in enumerate(scores):
        # Explicit id tie-break makes export independent of sparse traversal.
        pairs = sorted(values.items(), key=lambda pair: (-pair[1], pair[0]))[:top_m]
        if pairs:
            ids, vals = zip(*pairs)
            neighbors[item, :len(pairs)] = ids
            sims[item, :len(pairs)] = vals
    return {"neighbors": neighbors, "sims": sims,
            "freq": np.asarray(x.sum(axis=0)).ravel().astype(np.int64),
            "algorithm": np.asarray("swing"), "alpha1": np.float64(alpha1),
            "alpha2": np.float64(alpha2), "beta": np.float64(beta),
            "contributing_user_pairs": np.int64(n_pairs)}


class SwingRecall(ItemCFRecall):
    """Same recency-weighted neighbor aggregation, distinct source trace."""
    name = "swing"

    def recall(self, user_id, history, top_k):
        if top_k <= 0:
            return []
        return super().recall(user_id, history, top_k)


def save_swing_index(index, path: str | Path, data):
    save_itemcf_index({**index, "dataset_hash": np.asarray(dataset_fingerprint(data)),
                       "raw_item_ids": np.asarray(data.raw_item_ids, dtype=np.int64)}, path)


def load_swing_recall(path: str | Path, data, **kwargs):
    index = load_itemcf_index(path)
    if (str(index.get("algorithm", "")) != "swing" or
            str(index.get("dataset_hash", "")) != dataset_fingerprint(data) or
            not np.array_equal(index.get("raw_item_ids"), data.raw_item_ids)):
        raise ValueError("Swing index dataset/algorithm mismatch")
    neighbors, sims = index["neighbors"], index["sims"]
    if (neighbors.ndim != 2 or neighbors.shape != sims.shape or
            neighbors.shape[0] != data.num_items+1 or neighbors.shape[1] < 1 or
            not np.issubdtype(neighbors.dtype, np.integer) or
            (neighbors < 0).any() or (neighbors > data.num_items).any() or
            not np.isfinite(sims).all() or (sims < 0).any() or
            np.any(sims[neighbors == 0]) or np.any(sims[0])):
        raise ValueError("invalid Swing index shape/values")
    return SwingRecall(neighbors, sims, data.num_items, **kwargs)
