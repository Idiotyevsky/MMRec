"""Training objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

PAD = 0


def sampled_softmax_logits(
    sequence_repr: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    temperature: float = 1.0,
    in_batch_neg_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-position logits ``(B, L, 1 + N [+ K])``; column 0 is the positive.

    Split out from the loss so tests can assert on the *scores* themselves
    (e.g. that a future token cannot change an earlier position's ranking).
    """
    if sequence_repr.dim() != 3:
        raise ValueError(
            "sequence_repr must be (B, L, H); a (B, H) user vector would be "
            f"broadcast over all positions.  Got {tuple(sequence_repr.shape)}."
        )
    if sequence_repr.shape[:2] != pos_emb.shape[:2]:
        raise ValueError(
            f"sequence_repr {tuple(sequence_repr.shape)} and pos_emb "
            f"{tuple(pos_emb.shape)} must agree on (B, L)"
        )
    tau = max(float(temperature), 1e-6)

    s_pos = (sequence_repr * pos_emb).sum(-1) / tau  # (B, L)
    s_neg = torch.einsum("blh,bnh->bln", sequence_repr, neg_emb) / tau  # (B, L, N)
    logits = torch.cat([s_pos.unsqueeze(-1), s_neg], dim=-1)  # (B, L, 1 + N)

    if in_batch_neg_emb is not None and in_batch_neg_emb.shape[1] > 0:
        s_ib = torch.einsum("blh,bkh->blk", sequence_repr, in_batch_neg_emb) / tau
        logits = torch.cat([logits, s_ib], dim=-1)
    return logits


def sampled_softmax_loss(
    sequence_repr: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 1.0,
    in_batch_neg_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Position-wise sampled-softmax (sampled cross-entropy) next-item loss.

    Every position ``l`` is scored with *its own* hidden state::

        s+_{b,l}   = h_{b,l} . e+_{b,l}
        s-_{b,l,n} = h_{b,l} . e-_{b,n}
        L_{b,l}    = -log softmax([s+, s-] )_0

    Parameters
    ----------
    sequence_repr : (B, L, H) per-position hidden states (``encode_sequence``).
    pos_emb       : (B, L, H) embeddings of the positive targets.
    neg_emb       : (B, N, H) negatives shared by the whole sequence of a user.
    target        : (B, L) ground-truth item ids; ``target == PAD`` marks a
        position with no supervision and is excluded from the average.
    in_batch_neg_emb : optional (B, K, H) extra negatives.  The caller must
        guarantee they are not the user's own positives; misuse silently makes
        the task easier.

    A single ``(B, H)`` representation is rejected on purpose -- broadcasting the
    last hidden state over all positions would score ``h_L`` against
    ``target_1 ... target_L``, which is not the SASRec objective.
    """
    logits = sampled_softmax_logits(
        sequence_repr, pos_emb, neg_emb, temperature, in_batch_neg_emb
    )
    B, L = logits.shape[:2]
    zeros = torch.zeros((B, L), dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), zeros.reshape(-1), reduction="none"
    ).view(B, L)

    mask = (target != PAD).to(loss.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def bce_loss_with_negatives(
    sequence_repr: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Position-wise BCE variant (one negative per position), for reference."""
    s_pos = (sequence_repr * pos_emb).sum(-1)
    s_neg = (sequence_repr * neg_emb).sum(-1)
    loss = -(F.logsigmoid(s_pos) + F.logsigmoid(-s_neg))
    mask = (target != PAD).to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)
