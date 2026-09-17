"""Offline ANN index over learned item embeddings.

Faiss is an **optional** dependency: if it is unavailable the module falls back
to an exact ``torch.topk`` search so that the retrieval path never blocks the
rest of the project.  The fallback is exact rather than approximate, so recall
numbers are comparable; only latency differs.

Item 0 (PAD) is stored as a zero vector and explicitly excluded from results.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:  # pragma: no cover - environment dependent
    import faiss  # type: ignore

    HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    HAS_FAISS = False

PAD = 0


class ItemIndex:
    """Inner-product index over L2-normalised item embeddings (= cosine)."""

    def __init__(self, embeddings: np.ndarray, normalize: bool = True) -> None:
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got {emb.shape}")
        self.raw = emb.copy()
        self.dim = emb.shape[1]
        self.normalize = normalize
        self._index = None
        self._torch_table = None
        if normalize:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
        self.embeddings = emb
        self.embeddings[PAD] = 0.0

    # ------------------------------------------------------------------
    def build(self) -> "ItemIndex":
        if HAS_FAISS:
            index = faiss.IndexFlatIP(self.dim)
            index.add(np.ascontiguousarray(self.embeddings))
            self._index = index
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if HAS_FAISS:
            if self._index is None:
                self.build()
            faiss.write_index(self._index, str(path))
            meta = path.with_suffix(path.suffix + ".meta.json")
            meta.write_text(
                json.dumps({"dim": self.dim, "num_items": int(self.embeddings.shape[0]),
                            "normalize": self.normalize, "backend": "faiss.IndexFlatIP"}),
                encoding="utf-8",
            )
        else:
            np.save(path.with_suffix(".npy"), self.embeddings)
            path.with_suffix(path.suffix + ".meta.json").write_text(
                json.dumps({"dim": self.dim, "num_items": int(self.embeddings.shape[0]),
                            "normalize": self.normalize, "backend": "numpy"}),
                encoding="utf-8",
            )

    # ------------------------------------------------------------------
    def search(self, queries: np.ndarray, top_k: int = 20, exclude=None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(item_ids, scores)`` of shape ``(B, top_k)``.

        ``exclude`` is an optional ``(B, n_exclude)`` array of item ids to remove
        (e.g. the user's watch history).
        """
        q = np.asarray(queries, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if self.normalize:
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            q = q / norms

        ex = None
        if exclude is not None:
            ex = np.asarray(exclude, dtype=np.int64)
            if ex.ndim == 1:
                ex = ex[:, None]
        n_ex = 0 if ex is None else ex.shape[1]

        if self._index is not None:
            # faiss cannot express per-query exclusions; over-fetch and filter
            fetch = int(min(self.embeddings.shape[0], max(top_k + n_ex + 8, 4 * top_k)))
            raw_scores, raw_ids = self._index.search(np.ascontiguousarray(q), fetch)
            out_ids = np.zeros((raw_ids.shape[0], top_k), dtype=np.int64)
            out_scores = np.full((raw_ids.shape[0], top_k), -np.inf, dtype=np.float32)
            for b in range(raw_ids.shape[0]):
                banned = {PAD}
                if ex is not None and b < ex.shape[0]:
                    banned |= set(ex[b].tolist())
                keep = [i for i, it in enumerate(raw_ids[b]) if int(it) not in banned]
                if len(keep) < top_k:  # over-fetch was not enough: fall back to exact
                    return self._exact_search(q, top_k, ex)
                keep = np.asarray(keep[:top_k], dtype=np.int64)
                out_ids[b] = raw_ids[b][keep]
                out_scores[b] = raw_scores[b][keep].astype(np.float32)
            return out_ids, out_scores

        return self._exact_search(q, top_k, ex)

    # ------------------------------------------------------------------
    def _exact_search(self, q: np.ndarray, top_k: int, ex: np.ndarray | None):
        """Exact inner-product search with masking, used as the fallback path."""
        import torch

        if self._torch_table is None:
            self._torch_table = torch.from_numpy(self.embeddings)
        sims = torch.from_numpy(q) @ self._torch_table.t()
        sims[:, PAD] = float("-inf")
        if ex is not None:
            ex_t = torch.as_tensor(ex, dtype=torch.long, device=sims.device)
            rows = torch.arange(sims.shape[0], device=sims.device).unsqueeze(1).expand_as(ex_t)
            sims[rows, ex_t] = float("-inf")
        k = min(top_k, sims.shape[1] - 1)
        scores, ids = torch.topk(sims, k=k, dim=1)
        return ids.numpy().astype(np.int64), scores.numpy().astype(np.float32)


def build_index(embeddings: np.ndarray, normalize: bool = True) -> ItemIndex:
    return ItemIndex(embeddings, normalize=normalize).build()


def load_index(path: str | Path):
    """Return a faiss index object, or a numpy array when faiss is missing."""
    path = Path(path)
    if HAS_FAISS and path.exists():
        return faiss.read_index(str(path))
    npy = path.with_suffix(".npy")
    if npy.exists():
        return np.load(npy)
    raise FileNotFoundError(f"no index at {path} (or {npy})")


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t0)
