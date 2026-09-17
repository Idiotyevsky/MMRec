"""Item-based collaborative filtering recall.

Classic ItemCF with cosine-normalised co-occurrence:

    sim(i, j) = |U_i ∩ U_j| / sqrt(|U_i| * |U_j|)

built from **training interactions only**.  The top-``M`` neighbours of every
item are pre-computed offline into a small artifact so that serving is a gather
plus a weighted sum — no similarity computation at request time.

Why this channel exists: it is the workhorse of industrial recall.  It finds
items that co-occur with what the user watched, which popularity recall cannot
do and content recall may miss when the collaborative pattern is strong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .base import RecallCandidate, RecallStrategy

PAD = 0
DEFAULT_TOP_M = 100
DEFAULT_DECAY = 0.9


# ----------------------------------------------------------------------
# offline index construction
# ----------------------------------------------------------------------
def build_itemcf_index(
    flat_items: np.ndarray,
    user_offsets: np.ndarray,
    train_len: np.ndarray,
    num_items: int,
    top_m: int = DEFAULT_TOP_M,
    chunk_size: int = 512,
    min_cooccurrence: int = 1,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Compute the top-``M`` cosine-similar items for every item.

    The user-item matrix is normalised per item by ``1/sqrt(freq)`` so that a
    plain ``XᵀX`` yields the cosine similarity directly.  The product is
    evaluated in item chunks because a dense ``num_items × num_items`` matrix
    would be 1.5 GB at this catalogue size.
    """
    from scipy import sparse

    flat = np.asarray(flat_items, dtype=np.int64)
    offsets = np.asarray(user_offsets, dtype=np.int64)
    tlen = np.asarray(train_len, dtype=np.int64)
    num_users = offsets.shape[0] - 1

    # ---- gather training positions only (no val/test leakage) ----
    user_of = np.repeat(np.arange(num_users, dtype=np.int64), np.diff(offsets))
    pos = np.arange(flat.shape[0])
    train_end = offsets[1:] - 2
    is_train = pos < train_end[user_of]
    items = flat[is_train]
    users = user_of[is_train]
    keep = items > PAD
    items, users = items[keep], users[keep]

    X = sparse.csr_matrix(
        (np.ones(items.shape[0], dtype=np.float32), (users, items)),
        shape=(num_users, num_items + 1),
    )
    X.sum_duplicates()
    X.data[:] = 1.0  # binary: a user counts once per item
    freq = np.asarray(X.sum(axis=0)).ravel()
    inv = np.zeros_like(freq, dtype=np.float32)
    nz = freq > 0
    inv[nz] = 1.0 / np.sqrt(freq[nz])
    Xn = X @ sparse.diags(inv)  # column-normalised
    Xn = Xn.tocsc()

    neighbors = np.zeros((num_items + 1, top_m), dtype=np.int32)
    sims = np.zeros((num_items + 1, top_m), dtype=np.float32)

    for start in range(1, num_items + 1, chunk_size):
        end = min(start + chunk_size, num_items + 1)
        block = (Xn.T @ Xn[:, start:end]).toarray()  # (num_items+1, end-start)
        block[:PAD + 1, :] = 0.0
        for col in range(end - start):
            item = start + col
            block[item, col] = 0.0  # never recommend an item as its own neighbour
        if min_cooccurrence > 1:
            block[block < 1e-12] = 0.0
        k = min(top_m, block.shape[0])
        idx = np.argpartition(-block, kth=k - 1, axis=0)[:k]
        vals = np.take_along_axis(block, idx, axis=0)
        order = np.argsort(-vals, axis=0)
        idx = np.take_along_axis(idx, order, axis=0)
        vals = np.take_along_axis(vals, order, axis=0)
        neighbors[start:end, :k] = idx.T.astype(np.int32)
        sims[start:end, :k] = vals.T.astype(np.float32)
        if verbose and (start // chunk_size) % 10 == 0:
            print(f"  itemcf: {end - 1}/{num_items} items")

    sims[neighbors == PAD] = 0.0
    return {"neighbors": neighbors, "sims": sims, "freq": freq.astype(np.int64)}


def save_itemcf_index(index: dict[str, np.ndarray], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **index)


def load_itemcf_index(path: str | Path) -> dict[str, np.ndarray]:
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


# ----------------------------------------------------------------------
# online recall
# ----------------------------------------------------------------------
class ItemCFRecall(RecallStrategy):
    name = "itemcf"

    def __init__(
        self,
        neighbors: np.ndarray,
        sims: np.ndarray,
        num_items: int | None = None,
        decay: float = DEFAULT_DECAY,
        max_history: int = 50,
    ) -> None:
        self.neighbors = np.asarray(neighbors, dtype=np.int64)
        self.sims = np.asarray(sims, dtype=np.float32)
        self.num_items = int(num_items if num_items is not None else self.neighbors.shape[0] - 1)
        self.decay = float(decay)
        self.max_history = int(max_history)

    # ------------------------------------------------------------------
    def score_items(self, history: Sequence[int]) -> np.ndarray:
        """Aggregate neighbour similarities into a dense score vector.

        ``score(j) = Σ_i w(i) · sim(i, j)`` where ``w`` gives the most recent
        history item the highest weight (``decay`` per step back).
        """
        scores = np.zeros(self.num_items + 1, dtype=np.float64)
        hist = [int(i) for i in history if int(i) > PAD][-self.max_history :]
        if not hist:
            return scores
        hist = np.asarray(hist, dtype=np.int64)
        weights = self.decay ** np.arange(hist.shape[0] - 1, -1, -1)
        nbr = self.neighbors[hist]  # (H, M)
        sim = self.sims[hist] * weights[:, None]
        flat_items = nbr.ravel()
        flat_scores = sim.ravel()
        valid = flat_items > PAD
        scores = np.bincount(flat_items[valid], weights=flat_scores[valid],
                             minlength=self.num_items + 1)[: self.num_items + 1]
        return scores

    def recall(self, user_id: int, history: Sequence[int], top_k: int) -> list[RecallCandidate]:
        scores = self.score_items(history)
        seen = self._seen(history)
        if seen:
            scores[np.fromiter(seen, dtype=np.int64)] = -np.inf
        scores[PAD] = -np.inf
        k = min(top_k, self.num_items)
        idx = np.argpartition(-scores, kth=k - 1)[:k] if k < scores.shape[0] else np.arange(scores.shape[0])
        idx = idx[np.argsort(-scores[idx], kind="stable")]
        idx = idx[np.isfinite(scores[idx]) & (scores[idx] > 0)]
        return self._filter(idx, scores[idx], seen, top_k)

    def is_ready(self) -> tuple[bool, str]:
        nnz = int((self.sims > 0).sum())
        return nnz > 0, f"{nnz} (item, neighbour) pairs with non-zero similarity"
