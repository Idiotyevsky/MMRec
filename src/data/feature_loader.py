"""Multimodal feature loading with explicit availability masks.

Design notes
------------
* Raw feature files are memory-mapped, never copied into RAM.
* Rows are addressed by *internal* item id (0 = PAD, 1..N = items) via a
  caller-supplied ``row_for_item`` array.  The mapping is built by the
  preprocessing step and validated there; this class only asserts consistency.
* A modality row that is all-zero (or non-finite) is treated as *missing*
  rather than as a legitimate semantic vector.  ``missing != zero`` is the
  whole point: a zero vector silently entering a fusion layer teaches the model
  that "no image" looks like a specific real image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MODALITIES = ("text", "image", "video")


def _row_is_valid(row: np.ndarray) -> bool:
    if not np.isfinite(row).all():
        return False
    return bool(np.any(row != 0.0))


class MultimodalFeatures:
    """Lazy, memory-mapped access to pre-extracted item features."""

    def __init__(
        self,
        data_dir: str | Path,
        modalities: tuple[str, ...] = MODALITIES,
        row_for_item: np.ndarray | None = None,
        num_items: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.modalities = tuple(modalities)
        self.arrays: dict[str, np.ndarray] = {}
        self.dims: dict[str, int] = {}

        for m in self.modalities:
            p = self.data_dir / f"{m}_feat.npy"
            if not p.exists():
                continue
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 2:
                raise ValueError(f"{p} must be 2-D, got shape {arr.shape}")
            self.arrays[m] = arr
            self.dims[m] = int(arr.shape[1])

        if not self.arrays:
            raise FileNotFoundError(
                f"No modality feature files found under {self.data_dir} "
                f"(looked for {[f'{m}_feat.npy' for m in self.modalities]})"
            )

        if row_for_item is None:
            n_rows = next(iter(self.arrays.values())).shape[0]
            if num_items is not None and num_items + 1 != n_rows:
                raise ValueError(
                    f"num_items+1 ({num_items + 1}) != feature rows ({n_rows}); "
                    "explicit row_for_item mapping is required."
                )
            row_for_item = np.arange(n_rows, dtype=np.int64)
        self.row_for_item = np.asarray(row_for_item, dtype=np.int64)

        # availability: computed through the row mapping (NOT by assuming that
        # internal id i lives at feature row i), and PAD is always unavailable
        self._availability: dict[str, np.ndarray] = {}
        for m, arr in self.arrays.items():
            n = self.row_for_item.shape[0]
            rows = np.zeros(n, dtype=np.int64)
            valid = np.zeros(n, dtype=bool)
            in_range = self.row_for_item < arr.shape[0]
            rows[in_range] = self.row_for_item[in_range]
            if in_range.any():
                sub = np.asarray(arr[rows[in_range]])
                ok = np.isfinite(sub).all(axis=1) & (np.abs(sub).sum(axis=1) > 0)
                valid[in_range] = ok
            valid[0] = False
            self._availability[m] = valid

    # ------------------------------------------------------------------
    @property
    def available_modalities(self) -> tuple[str, ...]:
        return tuple(self.arrays.keys())

    def availability(self, modality: str) -> np.ndarray:
        return self._availability[modality]

    def missing_rate(self) -> dict[str, float]:
        out = {}
        for m in self.arrays:
            a = self._availability[m]
            out[m] = float(1.0 - a[1:].mean()) if a.shape[0] > 1 else 0.0
        return out

    def fetch(self, modality: str, internal_ids: np.ndarray) -> np.ndarray:
        """Return float32 features for internal ids, with PAD -> zeros."""
        arr = self.arrays[modality]
        ids = np.asarray(internal_ids, dtype=np.int64)
        rows = np.zeros(ids.shape, dtype=np.int64)
        valid = ids > 0
        rows[valid] = self.row_for_item[ids[valid]]
        valid &= rows < arr.shape[0]
        out = np.zeros((*ids.shape, arr.shape[1]), dtype=np.float32)
        if valid.any():
            out[valid] = np.asarray(arr[rows[valid]], dtype=np.float32)
        return out

    def mask(self, modality: str, internal_ids: np.ndarray) -> np.ndarray:
        """Boolean availability for internal ids (PAD -> False)."""
        a = self._availability[modality]
        ids = np.asarray(internal_ids, dtype=np.int64)
        out = np.zeros(ids.shape, dtype=bool)
        valid = (ids > 0) & (ids < a.shape[0])
        out[valid] = a[ids[valid]]
        return out
