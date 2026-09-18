"""Torch datasets over the processed MicroLens arrays.

Split convention (leave-one-out, strictly chronological)::

    user history:  i1  i2  i3  i4  i5  i6  i7        (sorted by timestamp)
    train items :  i1  i2  i3  i4  i5
    val target  :  i6
    test target :  i7

Training uses the standard autoregressive shift over the train items::

    input  = [i1 i2 i3 i4]
    target = [i2 i3 i4 i5]

so every non-padding position contributes a loss term.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PAD = 0


@dataclass
class ProcessedData:
    """Everything the training / evaluation code needs, loaded once."""

    dir: Path
    num_users: int
    num_items: int
    flat_items: np.ndarray
    user_offsets: np.ndarray
    train_len: np.ndarray
    val_target: np.ndarray
    test_target: np.ndarray
    train_freq: np.ndarray
    popularity_bucket: np.ndarray
    is_cold: np.ndarray
    raw_item_ids: np.ndarray
    stats: dict

    # ------------------------------------------------------------------
    @property
    def is_simulated_cold(self) -> np.ndarray:
        """Alias for ``is_cold`` with the semantics spelled out.

        The on-disk field is still called ``is_cold`` (existing artifacts stay
        valid), but in code and in the UI this is specifically the *simulated
        cold-start benchmark* split, not "an item with no training signal".
        """
        return self.is_cold

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "ProcessedData":
        path = Path(path)
        npz = np.load(path / "dataset.npz")
        mappings = json.loads((path / "mappings.json").read_text(encoding="utf-8"))
        stats = json.loads((path / "stats.json").read_text(encoding="utf-8"))
        data = cls(
            dir=path,
            num_users=int(mappings["num_users"]),
            num_items=int(mappings["num_items"]),
            flat_items=npz["flat_items"].astype(np.int64),
            user_offsets=npz["user_offsets"].astype(np.int64),
            train_len=npz["train_len"].astype(np.int64),
            val_target=npz["val_target"].astype(np.int64),
            test_target=npz["test_target"].astype(np.int64),
            train_freq=npz["train_freq"].astype(np.int64),
            popularity_bucket=npz["popularity_bucket"].astype(np.int8),
            is_cold=npz["is_cold"].astype(bool),
            raw_item_ids=np.asarray(mappings["raw_item_ids"], dtype=np.int64),
            stats=stats,
        )
        data.validate()
        return data

    def validate(self) -> None:
        assert self.user_offsets.shape[0] == self.num_users + 1
        assert self.flat_items.shape[0] == self.user_offsets[-1]
        assert self.train_freq.shape[0] == self.num_items + 1
        assert self.popularity_bucket.shape[0] == self.num_items + 1
        assert self.is_cold.shape[0] == self.num_items + 1
        assert self.raw_item_ids.shape[0] == self.num_items
        assert self.train_freq[PAD] == 0, "PAD must have zero training frequency"
        assert self.flat_items.min() >= PAD
        assert self.flat_items.max() <= self.num_items
        # every user's train part must be non-empty
        assert (self.train_len >= 1).all(), "some users have an empty train history"
        # the evaluation targets must not appear anywhere in the history
        self.assert_no_target_leak()

    def assert_no_target_leak(self) -> None:
        """Targets must be strictly after the history used to predict them."""
        for u in range(self.num_users):
            s = self.user_offsets[u]
            t = int(self.train_len[u])
            history = set(self.flat_items[s : s + t + 1].tolist())  # train + val target
            if int(self.val_target[u]) in set(self.flat_items[s : s + t].tolist()):
                raise AssertionError(f"user {u}: val target leaks into train history")
            if int(self.test_target[u]) in history:
                raise AssertionError(f"user {u}: test target leaks into train+val history")
        # all interactions of a user are distinct by construction
        if not self.is_cold.any():
            for u in range(0, self.num_users, max(1, self.num_users // 500)):
                s, e = self.user_offsets[u], self.user_offsets[u + 1]
                seg = self.flat_items[s:e]
                if len(set(seg.tolist())) != seg.shape[0]:
                    raise AssertionError(f"user {u}: duplicate interactions in history")

    # ------------------------------------------------------------------
    def train_items(self, u: int) -> np.ndarray:
        s = self.user_offsets[u]
        return self.flat_items[s : s + int(self.train_len[u])]

    def val_history(self, u: int) -> np.ndarray:
        return self.train_items(u)

    def test_history(self, u: int) -> np.ndarray:
        s = self.user_offsets[u]
        return self.flat_items[s : s + int(self.train_len[u]) + 1]

    def full_history(self, u: int) -> np.ndarray:
        s, e = self.user_offsets[u], self.user_offsets[u + 1]
        return self.flat_items[s:e]

    def raw_item(self, internal_id: int) -> int:
        return int(self.raw_item_ids[internal_id - 1])

    def summary(self) -> str:
        s = self.stats
        return (
            f"{s['num_users']} users | {s['num_items']} items | "
            f"{s['num_interactions']} interactions | "
            f"avg len {s['avg_sequence_length']:.2f} | "
            f"simulated cold items {int(self.is_cold.sum())}"
        )


# ----------------------------------------------------------------------
def _left_pad(seq: np.ndarray, max_len: int) -> np.ndarray:
    """Right-align ``seq`` inside a zero array of length ``max_len``."""
    out = np.zeros(max_len, dtype=np.int64)
    if seq.shape[0] > max_len:
        seq = seq[-max_len:]
    out[max_len - seq.shape[0] :] = seq
    return out


class TrainDataset(Dataset):
    """One autoregressive sample per user (all positions share the loss)."""

    def __init__(self, data: ProcessedData, max_len: int, users: np.ndarray | None = None) -> None:
        self.data = data
        self.max_len = int(max_len)
        self.users = (
            np.arange(data.num_users, dtype=np.int64) if users is None else np.asarray(users, dtype=np.int64)
        )

    def __len__(self) -> int:
        return int(self.users.shape[0])

    def __getitem__(self, idx: int) -> dict:
        u = int(self.users[idx])
        items = self.data.train_items(u)
        inp = items[:-1]
        tgt = items[1:]
        if inp.shape[0] == 0:  # train_len == 1
            inp = items.copy()
            tgt = items.copy()
        return {
            "input_ids": _left_pad(inp, self.max_len),
            "target": _left_pad(tgt, self.max_len),
            "user_id": u,
        }


class EvalDataset(Dataset):
    """One sample per user; the history is everything strictly before the target."""

    def __init__(
        self,
        data: ProcessedData,
        max_len: int,
        split: str = "test",
        users: np.ndarray | None = None,
    ) -> None:
        if split not in ("val", "test"):
            raise ValueError(f"split must be 'val' or 'test', got {split!r}")
        self.data = data
        self.max_len = int(max_len)
        self.split = split
        self.users = (
            np.arange(data.num_users, dtype=np.int64) if users is None else np.asarray(users, dtype=np.int64)
        )
        self.targets = data.val_target if split == "val" else data.test_target

    def __len__(self) -> int:
        return int(self.users.shape[0])

    def __getitem__(self, idx: int) -> dict:
        u = int(self.users[idx])
        hist = self.data.val_history(u) if self.split == "val" else self.data.test_history(u)
        return {
            "input_ids": _left_pad(hist, self.max_len),
            "user_id": u,
            "target": int(self.targets[u]),
        }


def collate_train(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.as_tensor(np.stack([b["input_ids"] for b in batch]), dtype=torch.long),
        "target": torch.as_tensor(np.stack([b["target"] for b in batch]), dtype=torch.long),
        "user_id": torch.as_tensor([b["user_id"] for b in batch], dtype=torch.long),
    }


def collate_eval(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.as_tensor(np.stack([b["input_ids"] for b in batch]), dtype=torch.long),
        "user_id": torch.as_tensor([b["user_id"] for b in batch], dtype=torch.long),
        "target": torch.as_tensor([b["target"] for b in batch], dtype=torch.long),
    }
