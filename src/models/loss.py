"""Training objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

PAD = 0


def sampled_softmax_loss(
    user_repr: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    in_batch_pos_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sampled-softmax (a.k.a. sampled cross-entropy) next-item loss.

        L = -log  exp(s+) / ( exp(s+) + sum_j exp(s_j-) )

    Scores are divided by ``temperature``.  The loss is averaged over positions
    whose *input* token is not PAD, matching the standard SASRec shift
    convention (the first item of a sequence is never predicted).

    Parameters
    ----------
    user_repr : (B, H)
    pos_emb   : (B, L, H) embeddings of the positive targets
    neg_emb   : (B, N, H) shared negatives for the whole sequence of each user
    input_ids : (B, L) used only to build the loss mask
    in_batch_pos_emb : optional (B, K, H) extra negatives (other users' targets)
    """
    B, L, _ = pos_emb.shape
    tau = max(float(temperature), 1e-6)

    s_pos = (user_repr.unsqueeze(1) * pos_emb).sum(-1) / tau  # (B, L)
    s_neg = torch.bmm(neg_emb, user_repr.unsqueeze(-1)).squeeze(-1) / tau  # (B, N)

    logits = torch.cat([s_pos.unsqueeze(-1), s_neg.unsqueeze(1).expand(B, L, -1)], dim=-1)

    if in_batch_pos_emb is not None and in_batch_pos_emb.shape[1] > 0:
        s_ib = torch.bmm(
            in_batch_pos_emb, user_repr.unsqueeze(-1)
        ).squeeze(-1) / tau  # (B, K)
        logits = torch.cat([logits, s_ib.unsqueeze(1).expand(B, L, -1)], dim=-1)

    target = torch.zeros(logits.shape[:2], dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none"
    ).view(B, L)

    mask = (input_ids != PAD).to(loss.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def bce_loss_with_negatives(
    user_repr: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy variant (one negative per position) for reference."""
    s_pos = (user_repr.unsqueeze(1) * pos_emb).sum(-1)
    s_neg = (user_repr.unsqueeze(1) * neg_emb).sum(-1)
    loss = -(F.logsigmoid(s_pos) + F.logsigmoid(-s_neg))
    mask = (input_ids != PAD).to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)
