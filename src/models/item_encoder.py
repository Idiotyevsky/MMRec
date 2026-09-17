"""Unified multimodal item encoder.

    item_id ──► Embedding ──────────────────────┐
    text    ──► LayerNorm→Linear→GELU→Dropout ──┤
    image   ──► LayerNorm→Linear→GELU→Dropout ──┼─► Fusion ─► e_i ∈ R^H
    video   ──► LayerNorm→Linear→GELU→Dropout ──┘

Design points
-------------
* Every modality keeps its own input dimension; nothing is forced to match.
* Raw features are never modified on disk.  Normalisation happens inside the
  projection (``LayerNorm`` first), which is exactly what the spec asks for.
* Availability is explicit.  A missing modality gets zero weight and its
  embedding is zeroed, so it cannot be confused with a real zero vector.
* ``id_dropout`` / modality dropout are training-only regularisers that simulate
  items whose collaborative signal is unreliable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .fusion import build_fusion

CONTENT_MODALITIES = ("text", "image", "video")
ALL_MODALITIES = ("id", "text", "image", "video")


def _projection(input_dim: int, hidden_size: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_size),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class MultimodalItemEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 128,
        modalities: dict[str, bool] | None = None,
        fusion: str = "gated",
        dropout: float = 0.2,
        feature_dir: str | Path | None = None,
        row_for_item: dict[str, np.ndarray] | None = None,
        id_dropout_prob: float = 0.0,
        modality_dropout_prob: float = 0.0,
        per_modality_dropout: dict[str, float] | None = None,
        cold_item_mask: np.ndarray | None = None,
        zero_cold_id: bool = True,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.hidden_size = int(hidden_size)
        self.dropout_p = float(dropout)
        self.fusion_kind = fusion

        modalities = dict(modalities or {"id": True, "text": True, "image": True, "video": False})
        self.enabled = [m for m in ALL_MODALITIES if modalities.get(m, False)]
        if not self.enabled:
            raise ValueError("at least one modality must be enabled")
        self.use_id = "id" in self.enabled

        # ---- ID branch ----
        if self.use_id:
            self.id_embedding = nn.Embedding(self.num_items + 1, hidden_size, padding_idx=0)
            nn.init.normal_(self.id_embedding.weight, std=0.02)
            with torch.no_grad():
                self.id_embedding.weight[0].zero_()

        # ---- content branches ----
        self.projections = nn.ModuleDict()
        self.feature_dims: dict[str, int] = {}
        if feature_dir is not None:
            self._load_features(feature_dir, row_for_item)

        for m in self.enabled:
            if m == "id":
                continue
            if m not in self.feature_dims:
                raise ValueError(
                    f"modality {m!r} is enabled but no feature file was found under {feature_dir}"
                )
            self.projections[m] = _projection(self.feature_dims[m], hidden_size, dropout)

        # ---- dropout regularisation ----
        per_mod = dict(per_modality_dropout or {})
        if id_dropout_prob:
            per_mod["id"] = per_mod.get("id", 0.0) + float(id_dropout_prob)
        for m in CONTENT_MODALITIES:
            if modality_dropout_prob:
                per_mod[m] = per_mod.get(m, 0.0) + float(modality_dropout_prob)
        self.modality_dropout = {m: float(min(max(per_mod.get(m, 0.0), 0.0), 1.0)) for m in self.enabled}

        # ---- cold items ----
        cold = (
            np.zeros(self.num_items + 1, dtype=bool)
            if cold_item_mask is None
            else np.asarray(cold_item_mask, dtype=bool)
        )
        self.register_buffer("cold_item_mask", torch.as_tensor(cold, dtype=torch.bool), persistent=True)
        self.zero_cold_id = bool(zero_cold_id)

        self.fusion = build_fusion(fusion, self.enabled, hidden_size, dropout)

    # ------------------------------------------------------------------
    def _load_features(self, feature_dir: str | Path, row_for_item: dict | None) -> None:
        feature_dir = Path(feature_dir)
        for m in self.enabled:
            if m == "id":
                continue
            p = feature_dir / f"{m}_feat.npy"
            if not p.exists():
                continue
            arr = np.load(p, mmap_mode="r")
            dim = int(arr.shape[1])
            lut = None
            if row_for_item is not None and m in row_for_item:
                lut = np.asarray(row_for_item[m], dtype=np.int64)
            elif (feature_dir / f"row_for_item_{m}.npy").exists():
                lut = np.load(feature_dir / f"row_for_item_{m}.npy").astype(np.int64)
            elif arr.shape[0] == self.num_items + 1:
                lut = np.arange(self.num_items + 1, dtype=np.int64)
            elif arr.shape[0] == self.num_items:
                lut = np.zeros(self.num_items + 1, dtype=np.int64)
                lut[1:] = np.arange(self.num_items, dtype=np.int64)
            else:
                raise ValueError(
                    f"{m}_feat.npy has {arr.shape[0]} rows; cannot align to "
                    f"{self.num_items} items without an explicit mapping"
                )
            if lut.shape[0] != self.num_items + 1:
                raise ValueError(f"row mapping for {m} has {lut.shape[0]} entries, expected {self.num_items + 1}")
            if lut[1:].max() >= arr.shape[0]:
                raise ValueError(f"row mapping for {m} points past the end of the feature file")
            feats = np.zeros((self.num_items + 1, dim), dtype=np.float32)
            feats[1:] = np.asarray(arr[lut[1:]], dtype=np.float32)
            # availability: non-finite or all-zero rows are "missing"
            valid = np.isfinite(feats).all(axis=1) & (np.abs(feats).sum(axis=1) > 0)
            valid[0] = False
            self.feature_dims[m] = dim
            self.register_buffer(f"{m}_features", torch.from_numpy(feats), persistent=False)
            self.register_buffer(f"{m}_available", torch.from_numpy(valid), persistent=True)

    # ------------------------------------------------------------------
    def _modality_tensors(self, item_ids: torch.Tensor) -> tuple[dict, dict]:
        emb: dict[str, torch.Tensor] = {}
        mask: dict[str, torch.Tensor] = {}
        pad = item_ids == 0

        if self.use_id:
            e = self.id_embedding(item_ids)
            m = ~pad
            if self.zero_cold_id:
                m = m & ~self.cold_item_mask[item_ids]
            emb["id"] = e
            mask["id"] = m

        for m in self.enabled:
            if m == "id":
                continue
            feats = getattr(self, f"{m}_features")[item_ids]
            emb[m] = self.projections[m](feats)
            mask[m] = getattr(self, f"{m}_available")[item_ids] & ~pad
        return emb, mask

    def _apply_modality_dropout(self, emb: dict, mask: dict, generator=None) -> dict:
        """Training-time dropout of modality availability.

        Guarantees that a *valid* item never loses every modality: if all
        available modalities are dropped, the ID branch (when present) is
        restored, otherwise the strongest content branch is restored.
        """
        if not self.training or not any(self.modality_dropout.values()):
            return mask
        out = dict(mask)
        for m, p in self.modality_dropout.items():
            if p <= 0:
                continue
            keep = torch.rand(mask[m].shape, device=mask[m].device, generator=generator) >= p
            out[m] = mask[m] & keep

        stacked = torch.stack([out[m] for m in self.enabled], dim=0)
        none_left = ~stacked.any(dim=0)
        pad = ~torch.stack([mask[m] for m in self.enabled], dim=0).any(dim=0)
        restore = none_left & ~pad
        if restore.any():
            target = "id" if self.use_id else self.enabled[0]
            out[target] = out[target] | restore
        for m in self.enabled:
            emb[m] = emb[m] * out[m].unsqueeze(-1).to(emb[m].dtype)
        return out

    # ------------------------------------------------------------------
    def forward(
        self, item_ids: torch.Tensor, generator=None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        emb, mask = self._modality_tensors(item_ids)
        mask = self._apply_modality_dropout(emb, mask, generator=generator)
        fused, gates = self.fusion(emb, mask)
        return fused, gates

    # ------------------------------------------------------------------
    @torch.no_grad()
    def all_item_embeddings(self, batch_size: int = 8192) -> torch.Tensor:
        """Dense ``(num_items + 1, H)`` embedding table used for scoring.

        Evaluated without dropout so the table is deterministic.
        """
        was_training = self.training
        self.eval()
        out = []
        device = self.id_embedding.weight.device if self.use_id else next(self.parameters()).device
        ids = torch.arange(self.num_items + 1, device=device)
        for i in range(0, ids.shape[0], batch_size):
            chunk = ids[i : i + batch_size]
            e, _ = self.forward(chunk)
            out.append(e)
        table = torch.cat(out, dim=0)
        table[0] = 0
        if was_training:
            self.train()
        return table

    def extra_repr(self) -> str:
        return (
            f"num_items={self.num_items}, hidden={self.hidden_size}, "
            f"modalities={self.enabled}, fusion={self.fusion_kind}, "
            f"dropout={self.modality_dropout}"
        )
