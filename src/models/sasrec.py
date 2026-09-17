"""ID-only SASRec: the primary sequential baseline.

Correctness notes
-----------------
* Left padding: position ``L-1`` is always the most recent item, so the user
  representation is ``out[:, -1]``.
* Causal mask: additive ``-inf`` above the diagonal, so position ``i`` can never
  attend to position ``j > i``.  Verified by ``tests/test_attention_mask.py``.
* Padding is excluded twice: via ``src_key_padding_mask`` (keys) and by taking
  the last position, which is guaranteed non-padding.
* Item 0 is PAD and is never a valid candidate; ``mask_pad`` sets its score to
  ``-inf`` so it can never enter a top-k list.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

PAD = 0


class SequenceEncoder(nn.Module):
    """Causal Transformer over item embeddings (shared by SASRec and MM-SASRec)."""

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward or hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype = torch.bool) -> torch.Tensor:
        """Boolean mask, ``True`` where attention is forbidden (strictly upper)."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Combined causal + padding mask, ``(B*num_heads, L, L)`` bool.

        ``True`` means "forbidden".  We deliberately build a per-batch 3D mask
        rather than passing ``src_key_padding_mask`` because left-padded queries
        would then attend to *nothing*: softmax over an all-masked row yields
        NaN, which PyTorch's fused attention propagates into the real positions.
        Allowing each padding query to attend only to itself keeps every row
        finite; padding outputs are never read.
        """
        B, L = input_ids.shape
        pad = input_ids == PAD
        causal = self.causal_mask(L, input_ids.device).unsqueeze(0)  # (1, L, L)
        eye = torch.eye(L, dtype=torch.bool, device=input_ids.device).unsqueeze(0)
        key_pad = pad.unsqueeze(1).expand(B, L, L) & ~eye  # (B, L, L)
        mask = causal | key_pad
        return mask.unsqueeze(1).expand(B, self.num_heads, L, L).reshape(B * self.num_heads, L, L)

    def forward(
        self, item_emb: torch.Tensor, input_ids: torch.Tensor, return_sequence: bool = False
    ) -> torch.Tensor:
        """``item_emb``: (B, L, H).

        Returns the user representation ``(B, H)`` -- the hidden state of the
        last position, which is the most recent item because padding is on the
        left -- or the full ``(B, L, H)`` sequence when ``return_sequence``.
        """
        B, L, _ = item_emb.shape
        if L > self.max_seq_len:
            raise ValueError(f"sequence length {L} exceeds max_seq_len {self.max_seq_len}")
        pos = torch.arange(L, device=item_emb.device).unsqueeze(0).expand(B, L)
        x = self.dropout(item_emb + self.position_embedding(pos))
        out = self.encoder(x, mask=self.attention_mask(input_ids))
        out = self.final_norm(out)
        if return_sequence:
            return out
        return out[:, -1]


def mask_pad(scores: torch.Tensor) -> torch.Tensor:
    """Set the PAD column (index 0) to ``-inf`` so it can never be ranked."""
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    return scores.index_fill(1, torch.tensor([PAD], device=scores.device), float("-inf"))


class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        max_seq_len: int = 50,
        dim_feedforward: int | None = None,
        cold_item_mask=None,
        zero_cold_id: bool = True,
        item_dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.hidden_size = int(hidden_size)
        self.max_seq_len = int(max_seq_len)
        self.zero_cold_id = bool(zero_cold_id)
        # Training-only control: replace the whole item embedding with a learned
        # <mask> vector.  This separates "dropout regularises" from "content
        # information helps" when reading the ID-dropout ablation.
        self.item_dropout_prob = float(item_dropout_prob)

        self.item_embedding = nn.Embedding(self.num_items + 1, hidden_size, padding_idx=PAD)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[PAD].zero_()
        # only allocate the <mask> vector when the regulariser is actually used,
        # so checkpoints stay compatible across configs that do not enable it
        if self.item_dropout_prob > 0:
            self.mask_embedding = nn.Parameter(torch.zeros(hidden_size))

        cold = (
            np.zeros(self.num_items + 1, dtype=bool)
            if cold_item_mask is None
            else np.asarray(cold_item_mask, dtype=bool)
        )
        self.register_buffer("cold_item_mask", torch.as_tensor(cold, dtype=torch.bool), persistent=True)

        self.encoder = SequenceEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            dim_feedforward=dim_feedforward,
        )
        self.dropout = nn.Dropout(dropout)

    def item_embeddings(self, item_ids: torch.Tensor, generator=None):
        """Uniform interface with MMSASRec: returns ``(embeddings, gates)``.

        Cold items have no collaborative signal by protocol, so their ID
        embedding is zeroed at inference for every model that owns an ID branch.
        """
        e = self.item_embedding(item_ids)
        if self.zero_cold_id and self.cold_item_mask.any():
            e = e * (~self.cold_item_mask[item_ids]).unsqueeze(-1).to(e.dtype)
        if self.training and self.item_dropout_prob > 0 and hasattr(self, "mask_embedding"):
            drop = torch.rand(item_ids.shape, device=item_ids.device) < self.item_dropout_prob
            drop &= item_ids > PAD
            e = torch.where(drop.unsqueeze(-1), self.mask_embedding.expand_as(e), e)
        return e, {}

    def encode_sequence(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Hidden state of *every* position, ``(B, L, H)``.

        Training must use this: with left padding, position ``l`` has only seen
        tokens ``<= l``, so ``sequence_repr[:, l]`` is the state that predicts
        ``target[:, l]``.  Collapsing to the last state would score every target
        from the same vector, which is a different (and much easier) objective.
        """
        emb, _ = self.item_embeddings(input_ids)
        return self.encoder(self.dropout(emb), input_ids, return_sequence=True)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """User representation for *inference*: the last (most recent) position."""
        return self.encode_sequence(input_ids)[:, -1]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids)

    def score(self, user_repr: torch.Tensor, item_emb: torch.Tensor) -> torch.Tensor:
        """``item_emb``: (N, H) -> logits (B, N)."""
        return user_repr @ item_emb.t()

    @torch.no_grad()
    def all_item_embeddings(self) -> torch.Tensor:
        ids = torch.arange(self.num_items + 1, device=self.item_embedding.weight.device)
        table = self.item_embeddings(ids)[0].detach()
        table[PAD] = 0
        return table

    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = self.parameters()
        return sum(p.numel() for p in ps if (p.requires_grad or not trainable_only))
