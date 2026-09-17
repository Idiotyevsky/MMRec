"""Fusion modules for combining ID / text / image / video item representations.

Every fusion module receives
    embeddings : dict[str, Tensor]  (B, L, H) or (N, H)   -> already projected
    masks      : dict[str, BoolTensor] same leading shape -> True = modality present

and must return ``(fused, gate_weights)`` where ``gate_weights`` is a dict with
the same modality keys, shape ``(...,)`` (no hidden dim).  Missing modalities
must receive exactly zero weight -- ``missing != zero`` is a correctness
requirement, not a nicety.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

def _neg_inf(dtype: torch.dtype) -> float:
    """Finite-but-unreachable logit.

    ``-1e9`` overflows float16 (which tops out near 65504) and would corrupt AMP
    training, so the sentinel is derived from the dtype itself.  ``finfo.min``
    is used rather than ``-inf`` so that ``0 * value`` stays well-defined.
    """
    return float(torch.finfo(dtype).min)


def _align_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a ``(..., )`` availability mask to the trailing dim of ``ref``."""
    if mask.dim() == ref.dim():
        return mask
    if mask.dim() == ref.dim() - 1:
        return mask.unsqueeze(-1)
    raise ValueError(f"mask shape {tuple(mask.shape)} incompatible with embedding {tuple(ref.shape)}")


class ConcatFusion(nn.Module):
    """Concatenate modality embeddings, append availability bits, then MLP.

    Availability bits are concatenated so the MLP can distinguish "this item has
    no image" from "this item's image projects to the zero vector".  Missing
    modalities are zero-filled to keep a fixed input width.
    """

    def __init__(self, modality_order: list[str], hidden_size: int, dropout: float = 0.1,
                 use_mask_bits: bool = True) -> None:
        super().__init__()
        self.modality_order = list(modality_order)
        self.use_mask_bits = use_mask_bits
        in_dim = hidden_size * len(self.modality_order) + (
            len(self.modality_order) if use_mask_bits else 0
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self, embeddings: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts = []
        for m in self.modality_order:
            e = embeddings[m]
            mk_e = _align_mask(masks[m], e)
            parts.append(e * mk_e.to(e.dtype))
            if self.use_mask_bits:
                parts.append(mk_e.to(e.dtype))  # one bit per modality, not per dim
        fused = self.mlp(torch.cat(parts, dim=-1))
        fused = self.norm(fused)
        # a fully-missing position (PAD) must stay exactly zero
        any_avail = torch.stack([masks[m].bool() for m in self.modality_order], dim=0).any(dim=0)
        fused = fused * any_avail.unsqueeze(-1).to(fused.dtype)

        gates: dict[str, torch.Tensor] = {}
        total = any_avail.to(fused.dtype)
        for m in self.modality_order:
            gates[m] = masks[m].to(fused.dtype) / total.clamp(min=1.0)
        return fused, gates


class GatedFusion(nn.Module):
    """Per-item adaptive weighting of modalities (softmax over available ones).

        z     = [e_1 ; e_2 ; ... ; e_K]
        alpha = softmax(MLP(z) + log(mask))
        e     = sum_k alpha_k * e_k

    Masking is applied *before* the softmax by adding ``-inf`` to unavailable
    modalities, so the weights of the available modalities still sum to one and
    a missing modality receives exactly zero weight.
    """

    def __init__(self, modality_order: list[str], hidden_size: int, dropout: float = 0.1,
                 hidden_mult: int = 2) -> None:
        super().__init__()
        self.modality_order = list(modality_order)
        k = len(self.modality_order)
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_size * k, hidden_size * hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * hidden_mult, k),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self, embeddings: dict[str, torch.Tensor], masks: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Zero out unavailable modalities *before* the gate MLP: otherwise the
        # content of a missing modality would still influence the weights of the
        # available ones through the shared first linear layer.
        zeroed = {m: embeddings[m] * _align_mask(masks[m], embeddings[m]).to(embeddings[m].dtype)
                  for m in self.modality_order}
        z = torch.cat([zeroed[m] for m in self.modality_order], dim=-1)
        logits = self.gate_mlp(z)  # (..., K)

        mask_stack = torch.stack([masks[m].bool() for m in self.modality_order], dim=-1)
        any_avail = mask_stack.any(dim=-1)
        logits = logits.masked_fill(~mask_stack, _neg_inf(logits.dtype))
        # avoid a fully-masked row producing NaN; those rows are zeroed afterwards
        safe = torch.where(any_avail.unsqueeze(-1), logits, torch.zeros_like(logits))
        alpha = torch.softmax(safe, dim=-1)

        fused = torch.zeros_like(embeddings[self.modality_order[0]])
        gates: dict[str, torch.Tensor] = {}
        for k, m in enumerate(self.modality_order):
            a = alpha[..., k]
            gates[m] = a * any_avail.to(a.dtype)
            fused = fused + a.unsqueeze(-1) * zeroed[m]
        fused = self.norm(fused) * any_avail.unsqueeze(-1).to(fused.dtype)
        return fused, gates


def build_fusion(kind: str, modality_order: list[str], hidden_size: int, dropout: float):
    kind = kind.lower()
    if kind == "concat":
        return ConcatFusion(modality_order, hidden_size, dropout)
    if kind == "gated":
        return GatedFusion(modality_order, hidden_size, dropout)
    raise ValueError(f"Unknown fusion kind: {kind!r} (expected 'concat' or 'gated')")


__all__ = ["ConcatFusion", "GatedFusion", "build_fusion", "F"]
