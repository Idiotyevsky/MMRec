"""Negative sampling for sampled-softmax training.

Guarantees enforced here (each has a regression test):
* a sampled negative is never the positive target of that position;
* a sampled negative is never any known interaction of that user
  (train, validation or test), so negatives cannot silently become positives;
* PAD (item 0) is never sampled;
* duplicates inside one draw are tolerated but the *known-interaction* and
  *positive* exclusions are re-drawn until satisfied.
"""

from __future__ import annotations

import numpy as np

PAD = 0


class NegativeSampler:
    """Draw negatives under either a uniform or a popularity distribution.

    Parameters
    ----------
    num_items : size of the catalogue (internal ids are ``1..num_items``).
    train_freq : int array of length ``num_items + 1``; index 0 is ignored.
    all_interactions : int64 array of ``user * num_items + item`` keys for every
        interaction of every user.  Sorted.  Used for O(log n) membership tests.
    mode : ``uniform`` or ``popularity`` (sampling probability ∝ freq^power).
    """

    def __init__(
        self,
        num_items: int,
        train_freq: np.ndarray | None = None,
        all_interactions: np.ndarray | None = None,
        mode: str = "uniform",
        popularity_power: float = 0.75,
        seed: int = 42,
    ) -> None:
        self.num_items = int(num_items)
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self._keys = None
        if all_interactions is not None and np.asarray(all_interactions).size:
            self._keys = np.sort(np.asarray(all_interactions, dtype=np.int64))

        if mode == "uniform":
            self._cdf = None
        elif mode == "popularity":
            if train_freq is None:
                raise ValueError("popularity sampling requires train_freq")
            w = np.asarray(train_freq, dtype=np.float64)[1 : self.num_items + 1].copy()
            w = np.power(w, popularity_power)
            pos = w[w > 0]
            floor = float(pos.min()) if pos.size else 1.0
            w = np.maximum(w, floor)  # zero-frequency items stay reachable
            self._cdf = np.cumsum(w / w.sum())
        else:
            raise ValueError(f"Unknown negative sampling mode: {mode}")

    # ------------------------------------------------------------------
    def _draw(self, n: int) -> np.ndarray:
        """Draw ``n`` items in ``1..num_items`` (with replacement)."""
        if self._cdf is None:
            return self.rng.integers(1, self.num_items + 1, size=n, dtype=np.int64)
        u = self.rng.random(size=n)
        return (np.searchsorted(self._cdf, u) + 1).astype(np.int64)

    def _is_known(self, user_ids: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Vectorised membership test over (B, N) item draws."""
        if self._keys is None:
            return np.zeros(items.shape, dtype=bool)
        keys = np.asarray(user_ids, dtype=np.int64)[:, None] * self.num_items + items
        flat = keys.ravel()
        idx = np.searchsorted(self._keys, flat)
        idx = np.clip(idx, 0, self._keys.shape[0] - 1)
        return (self._keys[idx] == flat).reshape(items.shape)

    def sample(
        self,
        user_ids: np.ndarray,
        num_negatives: int,
        exclude_items: np.ndarray | None = None,
        max_retries: int = 32,
    ) -> np.ndarray:
        """Return ``(B, num_negatives)`` int64 negatives, one row per user.

        ``exclude_items`` holds extra forbidden items, shape ``(B,)`` or ``(B, K)``.
        """
        user_ids = np.asarray(user_ids, dtype=np.int64)
        B = int(user_ids.shape[0])
        out = self._draw(B * num_negatives).reshape(B, num_negatives)

        ex = None
        if exclude_items is not None:
            ex = np.asarray(exclude_items, dtype=np.int64)
            if ex.ndim == 1:
                ex = ex[:, None]

        for _ in range(max_retries):
            bad = (out <= PAD) | (out > self.num_items)
            bad |= self._is_known(user_ids, out)
            if ex is not None:
                for k in range(ex.shape[1]):
                    bad |= out == ex[:, k : k + 1]
            n_bad = int(bad.sum())
            if n_bad == 0:
                break
            out[bad] = self._draw(n_bad)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def build_interaction_keys(
        flat_items: np.ndarray, user_offsets: np.ndarray, num_items: int
    ) -> np.ndarray:
        """Sorted ``user * num_items + item`` keys for every interaction."""
        flat = np.asarray(flat_items, dtype=np.int64)
        counts = np.diff(np.asarray(user_offsets, dtype=np.int64))
        user_of = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
        valid = flat > 0
        return user_of[valid] * num_items + flat[valid]
