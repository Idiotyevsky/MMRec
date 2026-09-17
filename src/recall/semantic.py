"""Content / semantic recall over pre-extracted multimodal item features.

This is the channel that makes the whole project worth building: it can retrieve
an item that has **never been interacted with**, because it never looks at
collaborative signal.  It only needs the item's text and cover-image features.

Pipeline::

    item content embedding  =  L2( [ L2(text) ; L2(image) ] )
    user query              =  L2( Σ_i w(i) · e_i )        w = recency decay
    query  ->  Faiss exact inner-product search  ->  top-K items

Terminology note: the index is ``faiss.IndexFlatIP``, which performs **exact**
inner-product retrieval.  It is not an approximate nearest-neighbour index; that
would require IVF/HNSW.  Latency numbers here are therefore exact-search
latencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..retrieval.faiss_index import ItemIndex
from .base import RecallCandidate, RecallStrategy

PAD = 0
DEFAULT_DECAY = 0.9


# ----------------------------------------------------------------------
# offline embedding construction
# ----------------------------------------------------------------------
def build_content_embeddings(
    feature_dir: str | Path,
    row_for_item_dir: str | Path,
    num_items: int,
    modalities: tuple[str, ...] = ("text", "image"),
    l2_normalise: bool = True,
) -> np.ndarray:
    """Concatenate the requested modalities into one content embedding per item.

    Each modality block is L2-normalised **before** concatenation.  This matters:
    raw text features are already unit-norm while raw image features have norm
    ≈ 26, so concatenating them un-normalised would let the image block dominate
    the inner product and silently turn this into an image-only channel.
    """
    feature_dir = Path(feature_dir)
    row_for_item_dir = Path(row_for_item_dir)
    blocks: list[np.ndarray] = []
    for m in modalities:
        arr = np.load(feature_dir / f"{m}_feat.npy", mmap_mode="r")
        lut_path = row_for_item_dir / f"row_for_item_{m}.npy"
        if not lut_path.exists():
            lut_path = feature_dir / f"row_for_item_{m}.npy"
        lut = np.load(lut_path).astype(np.int64)
        block = np.zeros((num_items + 1, arr.shape[1]), dtype=np.float32)
        block[1:] = np.asarray(arr[lut[1:]], dtype=np.float32)
        block[0] = 0.0
        if l2_normalise:
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            block = block / np.maximum(norms, 1e-8)
        blocks.append(block)

    emb = np.concatenate(blocks, axis=1)
    if l2_normalise:
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-8)
    emb[PAD] = 0.0
    return emb.astype(np.float32)


# ----------------------------------------------------------------------
# online recall
# ----------------------------------------------------------------------
class SemanticRecall(RecallStrategy):
    name = "semantic"

    def __init__(
        self,
        embeddings: np.ndarray,
        num_items: int | None = None,
        decay: float = DEFAULT_DECAY,
        max_history: int = 50,
        normalize: bool = True,
        index: ItemIndex | None = None,
    ) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.num_items = int(num_items if num_items is not None else self.embeddings.shape[0] - 1)
        self.decay = float(decay)
        self.max_history = int(max_history)
        self.index = index or ItemIndex(self.embeddings, normalize=normalize).build()

    # ------------------------------------------------------------------
    def query_vector(self, history: Sequence[int]) -> np.ndarray:
        """Recency-weighted mean of the history's content embeddings."""
        hist = [int(i) for i in history if int(i) > PAD][-self.max_history :]
        if not hist:
            return np.zeros((self.embeddings.shape[1],), dtype=np.float32)
        hist = np.asarray(hist, dtype=np.int64)
        weights = self.decay ** np.arange(hist.shape[0] - 1, -1, -1)
        emb = self.embeddings[hist]  # (H, D)
        q = (emb * weights[:, None]).sum(axis=0) / max(weights.sum(), 1e-8)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        return q.astype(np.float32)

    def score_items(self, history: Sequence[int], candidate_items: np.ndarray) -> np.ndarray:
        """Exact inner product between the user query and specific candidates."""
        q = self.query_vector(history)
        if not np.any(q):
            return np.zeros(candidate_items.shape[0], dtype=np.float32)
        return self.embeddings[candidate_items] @ q

    def recall(self, user_id: int, history: Sequence[int], top_k: int) -> list[RecallCandidate]:
        seen = self._seen(history)
        if not seen and not np.any(self.query_vector(history)):
            return []
        q = self.query_vector(history)[None, :]
        # over-fetch so that history exclusions cannot shrink the result set
        fetch = min(self.num_items, top_k + len(seen) + 16)
        items, scores = self.index.search(q, top_k=fetch,
                                          exclude=np.fromiter(seen, dtype=np.int64,
                                                              count=len(seen)).reshape(1, -1) if seen else None)
        return self._filter(items[0], scores[0], seen, top_k)

    def is_ready(self) -> tuple[bool, str]:
        ok = bool(np.any(self.embeddings))
        return ok, f"content embeddings {self.embeddings.shape[0]} x {self.embeddings.shape[1]}"
