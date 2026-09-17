"""Full-ranking evaluator.

Protocol (see ``docs/evaluation_protocol.md``)
---------------------------------------------
For every evaluation user we score *the entire catalogue* (minus the PAD slot),
mask every item the user has already interacted with in the history that was fed
to the model, and compute the exact rank of the ground-truth target.

* The ground truth is never masked -- ``_mask_seen`` asserts this.
* Candidate dot-products are computed in item chunks (``item_chunk_size``,
  default 4096): the ``(batch, chunk)`` product is the largest temporary, while
  the batch-level ``(batch, num_items)`` score matrix is retained and filled
  chunk by chunk.  Ranking is therefore over the complete catalogue and exact --
  no candidate pre-filtering, no approximate top-k.
* Per-user ranks are returned, so cold-start / long-tail / per-bucket reports are
  slices of a single ranking pass and are guaranteed to come from the same
  scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from .metrics import NOT_RETRIEVED, metric_bundle  # noqa: F401

PAD = 0


@dataclass
class EvalResult:
    user_ids: np.ndarray
    targets: np.ndarray
    rank: np.ndarray  # 1-based rank, NOT_RETRIEVED when outside top_max_k
    topk: np.ndarray  # (B, max_k) internal item ids
    num_items: int
    ks: tuple[int, ...] = (5, 10, 20)
    extra: dict = field(default_factory=dict)

    @property
    def max_k(self) -> int:
        return int(self.topk.shape[1])

    def subset(self, mask: np.ndarray) -> "EvalResult":
        mask = np.asarray(mask, dtype=bool)
        return EvalResult(
            user_ids=self.user_ids[mask],
            targets=self.targets[mask],
            rank=self.rank[mask],
            topk=self.topk[mask],
            num_items=self.num_items,
            ks=self.ks,
            extra=self.extra,
        )

    def metrics(self) -> dict:
        return metric_bundle(self.rank, self.topk, self.num_items, self.ks)

    def recall_at(self, k: int) -> float:
        return float((self.rank <= k).mean()) if self.rank.size else float("nan")

    def ndcg_at(self, k: int) -> float:
        if not self.rank.size:
            return float("nan")
        r = self.rank.astype(np.float64)
        return float(np.where(r <= k, 1.0 / np.log2(r + 1.0), 0.0).mean())


class FullRankingEvaluator:
    def __init__(
        self,
        num_items: int,
        ks: tuple[int, ...] = (5, 10, 20),
        item_chunk_size: int = 4096,
        device: torch.device | str | None = None,
    ) -> None:
        self.num_items = int(num_items)
        self.ks = tuple(ks)
        self.max_k = max(self.ks)
        self.item_chunk_size = int(item_chunk_size)
        self.device = torch.device(device) if device is not None else None

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _score_batch(
        self,
        user_repr: torch.Tensor,
        item_emb: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Chunked ``user_repr @ item_emb.T`` -> (B, num_items+1)."""
        B = user_repr.shape[0]
        N = item_emb.shape[0]
        scores = torch.full((B, N), float("-inf"), device=user_repr.device, dtype=user_repr.dtype)
        for s in range(0, N, self.item_chunk_size):
            e = min(s + self.item_chunk_size, N)
            chunk = item_emb[s:e]
            scores[:, s:e] = user_repr @ chunk.t()
        scores[:, PAD] = float("-inf")
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.unsqueeze(0), float("-inf"))
        return scores

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        encode_fn: Callable[[dict], torch.Tensor],
        item_embeddings: torch.Tensor,
        loader,
        candidate_mask: np.ndarray | None = None,
        desc: str | None = None,
    ) -> EvalResult:
        """Run a full-ranking evaluation pass.

        Parameters
        ----------
        encode_fn : maps a collated batch to ``(B, H)`` user representations.
        item_embeddings : ``(num_items + 1, H)`` dense scoring table.
        loader : iterable of collated eval batches with ``input_ids``,
            ``user_id`` and ``target``.
        candidate_mask : optional ``(num_items + 1,)`` bool array restricting the
            candidate set (used for the cold-only restricted ranking).
        """
        device = self.device or item_embeddings.device
        item_embeddings = item_embeddings.to(device)

        all_users, all_targets, all_ranks, all_topk = [], [], [], []
        iterator = loader
        if desc:
            try:
                import sys

                from tqdm import tqdm

                if sys.stderr.isatty():  # keep background logs clean
                    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
            except Exception:  # pragma: no cover
                iterator = loader

        cmask_t = None
        if candidate_mask is not None:
            cmask_t = torch.as_tensor(np.asarray(candidate_mask), dtype=torch.bool, device=device)

        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            target = batch["target"].to(device)
            user_repr = encode_fn(batch).to(device)

            scores = self._score_batch(user_repr, item_embeddings, cmask_t)
            scores = self._mask_seen(scores, input_ids, target)

            tgt_score = scores.gather(1, target.view(-1, 1))
            finite_tgt = torch.isfinite(tgt_score).flatten()
            if candidate_mask is None:
                # with the full catalogue as candidates the target must ALWAYS be
                # scoreable -- otherwise seen-item masking swallowed it
                if not finite_tgt.all():
                    bad = (~finite_tgt).nonzero().flatten()[:5].tolist()
                    raise AssertionError(
                        f"ground-truth target was masked or non-finite for rows {bad}; "
                        "this is a masking bug (the target must never be excluded)"
                    )
            else:
                # a restricted candidate set may legitimately exclude the target;
                # those rows get rank = +inf and never count towards Recall@k
                in_candidates = cmask_t[target]
                if not torch.isfinite(tgt_score[in_candidates]).all():
                    raise AssertionError(
                        "a target inside the candidate set was masked or non-finite"
                    )
                tgt_score = torch.where(
                    in_candidates.view(-1, 1), tgt_score,
                    torch.full_like(tgt_score, float("-inf")),
                )
            rank = self._average_rank(scores, target)
            if candidate_mask is not None:
                # a target outside the candidate set is unreachable by definition;
                # without this it would inherit a small rank from the tiny
                # candidate list and be counted as a hit
                rank = torch.where(
                    cmask_t[target], rank, torch.full_like(rank, float(NOT_RETRIEVED))
                )

            # never ask for more entries than there are finite candidates,
            # otherwise torch.topk would return masked (PAD) columns
            n_finite = int(torch.isfinite(scores).sum(dim=1).min().item())
            k = min(self.max_k, n_finite)
            topk = torch.topk(scores, k=k, dim=1).indices
            if k < self.max_k:  # pad with PAD so the array width stays constant
                pad = torch.zeros((scores.shape[0], self.max_k - k), dtype=topk.dtype, device=topk.device)
                topk = torch.cat([topk, pad], dim=1)

            all_users.append(batch["user_id"].cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_ranks.append(rank.cpu().numpy().astype(np.float64))
            all_topk.append(topk.cpu().numpy())

        return EvalResult(
            user_ids=np.concatenate(all_users),
            targets=np.concatenate(all_targets),
            rank=np.concatenate(all_ranks).astype(np.float64),
            topk=np.concatenate(all_topk),
            num_items=self.num_items,
            ks=self.ks,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _average_rank(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Tie-neutral rank of ``target`` within ``scores`` (see metrics.py)."""
        tgt = scores.gather(1, target.view(-1, 1))
        n_greater = (scores > tgt).sum(dim=1)
        n_equal = (scores == tgt).sum(dim=1)
        return 1.0 + n_greater.to(scores.dtype) + (n_equal.to(scores.dtype) - 1.0) / 2.0

    @staticmethod
    def _mask_seen(scores: torch.Tensor, input_ids: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Set the score of every item in the user's history to ``-inf``.

        The ground-truth target is protected: it can never be masked.
        """
        seen = torch.zeros_like(scores, dtype=torch.bool)
        valid = input_ids > PAD
        seen.scatter_(1, input_ids.clamp(min=0), valid)
        seen.scatter_(1, target.view(-1, 1), False)  # never mask the target
        return scores.masked_fill(seen, float("-inf"))


def sentinel_rank(rank: np.ndarray, max_k: int) -> np.ndarray:
    """Keep ranks as-is; ``NOT_RETRIEVED`` values simply never satisfy ``<= k``."""
    return np.asarray(rank, dtype=np.float64)
