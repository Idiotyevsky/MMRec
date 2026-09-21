"""Content-aware two-tower retrieval model.

The item tower deliberately has no item-id embedding: a new catalogue item can
therefore be encoded from its text/image features before it has interactions.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _project(dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout))


class ContentItemTower(nn.Module):
    """Text/image-only item encoder with explicit availability bits."""
    def __init__(self, num_items: int, hidden_size: int = 128, dropout: float = .1,
                 feature_dir: str | Path | None = None, row_for_item: dict[str, np.ndarray] | None = None,
                 modalities: dict[str, bool] | None = None, normalize_embeddings: bool = True) -> None:
        super().__init__()
        self.num_items, self.hidden_size = int(num_items), int(hidden_size)
        self.normalize_embeddings = bool(normalize_embeddings)
        enabled = dict(modalities or {"text": True, "image": True})
        self.enabled = [m for m in ("text", "image") if enabled.get(m, False)]
        if not self.enabled:
            raise ValueError("ContentItemTower requires text and/or image; item-id embeddings are intentionally unsupported")
        if feature_dir is None:
            raise ValueError("feature_dir is required for content-aware retrieval")
        self.projections = nn.ModuleDict()
        for m in self.enabled:
            path = Path(feature_dir) / f"{m}_feat.npy"
            if not path.exists():
                raise ValueError(f"enabled modality {m!r} has no file at {path}")
            arr = np.load(path, mmap_mode="r")
            lut = (row_for_item or {}).get(m)
            if lut is None:
                candidate = Path(feature_dir) / f"row_for_item_{m}.npy"
                lut = np.load(candidate) if candidate.exists() else np.arange(self.num_items + 1)
            lut = np.asarray(lut, dtype=np.int64)
            if lut.shape != (self.num_items + 1,) or lut[1:].max(initial=0) >= arr.shape[0]:
                raise ValueError(f"invalid row mapping for {m}")
            feats = np.zeros((self.num_items + 1, arr.shape[1]), dtype=np.float32)
            feats[1:] = np.asarray(arr[lut[1:]], dtype=np.float32)
            available = np.isfinite(feats).all(1) & (np.abs(feats).sum(1) > 0)
            available[0] = False
            self.register_buffer(f"{m}_features", torch.from_numpy(feats), persistent=False)
            self.register_buffer(f"{m}_available", torch.from_numpy(available), persistent=True)
            self.projections[m] = _project(int(arr.shape[1]), hidden_size, dropout)
        # zero projected unavailable inputs plus a bit, so absent != a genuine zero vector.
        self.fusion = nn.Sequential(nn.Linear(len(self.enabled) * hidden_size + len(self.enabled), hidden_size),
                                    nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size))

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        pad = item_ids.eq(0)
        pieces, bits = [], []
        for m in self.enabled:
            mask = getattr(self, f"{m}_available")[item_ids] & ~pad
            e = self.projections[m](getattr(self, f"{m}_features")[item_ids])
            pieces.append(e * mask.unsqueeze(-1).to(e.dtype))
            bits.append(mask.unsqueeze(-1).to(e.dtype))
        out = self.fusion(torch.cat(pieces + bits, dim=-1))
        out = out * (~pad).unsqueeze(-1).to(out.dtype)
        return F.normalize(out, dim=-1, eps=1e-8) if self.normalize_embeddings else out

    @torch.no_grad()
    def all_item_embeddings(self, batch_size: int = 8192) -> torch.Tensor:
        was_training = self.training
        self.eval()
        ids = torch.arange(self.num_items + 1, device=next(self.parameters()).device)
        out = [self(ids[i:i + batch_size]) for i in range(0, len(ids), batch_size)]
        if was_training: self.train()
        table = torch.cat(out)
        table[0] = 0
        return table


class UserTower(nn.Module):
    def __init__(self, hidden_size: int = 128, dropout: float = .1, recency_decay: float = .9,
                 normalize_embeddings: bool = True) -> None:
        super().__init__()
        self.recency_decay, self.normalize_embeddings = float(recency_decay), bool(normalize_embeddings)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size))

    def forward(self, item_vectors: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Encode full left-padded histories; output is one vector per row."""
        valid = item_ids.ne(0)
        length = valid.sum(1)
        positions = torch.arange(item_ids.shape[1], device=item_ids.device).unsqueeze(0)
        exponent = (positions - (item_ids.shape[1] - 1)).clamp(max=0)
        weights = (self.recency_decay ** (-exponent)).to(item_vectors.dtype) * valid
        pooled = (item_vectors * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True).clamp_min(1e-8)
        out = self.mlp(pooled) * length.ne(0).unsqueeze(-1).to(pooled.dtype)
        return F.normalize(out, dim=-1, eps=1e-8) if self.normalize_embeddings else out


class TwoTowerRetriever(nn.Module):
    """Trainer-compatible learned retrieval model (all outputs are 128-D by default)."""
    def __init__(self, num_items: int, hidden_size: int = 128, max_history: int = 50, dropout: float = .1,
                 recency_decay: float = .9, feature_dir=None, row_for_item=None, modalities=None,
                 normalize_embeddings: bool = True) -> None:
        super().__init__()
        self.num_items, self.hidden_size, self.max_seq_len = int(num_items), int(hidden_size), int(max_history)
        self.item_tower = ContentItemTower(num_items, hidden_size, dropout, feature_dir, row_for_item, modalities,
                                           normalize_embeddings)
        self.user_tower = UserTower(hidden_size, dropout, recency_decay, normalize_embeddings)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor: return self.item_tower(item_ids)
    def item_embeddings(self, item_ids: torch.Tensor, generator=None): return self.encode_items(item_ids), {}
    def encode_history(self, input_ids: torch.Tensor) -> torch.Tensor: return self.user_tower(self.encode_items(input_ids), input_ids)
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor: return self.encode_history(input_ids)
    def encode_sequence(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Every position predicts the shifted target using only its own prefix.
        vectors = self.encode_items(input_ids)
        outputs = [self.user_tower(vectors[:, :i + 1], input_ids[:, :i + 1]) for i in range(input_ids.shape[1])]
        return torch.stack(outputs, dim=1)
    def score(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor: return user_vec @ item_vec.transpose(-1, -2)
    @torch.no_grad()
    def all_item_embeddings(self) -> torch.Tensor: return self.item_tower.all_item_embeddings()
    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)
