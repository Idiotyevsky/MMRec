"""Generative Semantic-ID recall.

The item catalogue is represented by Semantic IDs (RQ-VAE codes over multimodal
content), and the next item is produced by an autoregressive decoder instead of
by scoring a vector against an index.  This module exposes that as an ordinary
:class:`~src.recall.base.RecallStrategy`, so it plugs into the existing merge and
evaluation paths unchanged.

Three properties are enforced here, because they are the ways this channel
silently breaks:

1. **Every returned item is real.** Decoding is prefix-constrained by the
   Semantic-ID trie, so the model can only emit code tuples that exist in the
   catalogue; ``empty_decodings`` counts the (rare) cases where the beam dies,
   and it is reported rather than hidden.
2. **A Semantic ID may map to several items.** Collisions are expanded, not
   dropped, and the collision size is recorded on the candidate.
3. **Seen items and PAD never appear**, matching every other channel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..models.semantic_id import SemanticIDMapper
from .base import RecallCandidate, RecallStrategy

PAD = 0


class GenerativeRecall(RecallStrategy):
    name = "generative"

    def __init__(
        self,
        model: torch.nn.Module,
        mapper: SemanticIDMapper,
        num_items: int | None = None,
        beam_width: int = 20,
        max_history_items: int = 20,
        device: torch.device | str = "cpu",
        batch_size: int = 64,
    ) -> None:
        self.model = model.eval()
        self.mapper = mapper
        self.num_items = int(num_items if num_items is not None else mapper.num_items)
        self.beam_width = int(beam_width)
        self.max_history_items = int(max_history_items)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.last_stats: dict = {}

    # ------------------------------------------------------------------
    def _history_tokens(self, history: Sequence[int]) -> list[int]:
        items = [int(i) for i in history if int(i) > PAD][-self.max_history_items :]
        toks: list[int] = []
        for it in items:
            if 1 <= it <= self.num_items:
                toks.extend(self.mapper.item_tokens(it))
        return toks

    @torch.no_grad()
    def generate_batch(
        self, histories: Sequence[Sequence[int]]
    ) -> tuple[list[list[tuple[tuple[int, ...], float]]], dict]:
        """Run constrained beam search for several users at once."""
        token_histories = [self._history_tokens(h) for h in histories]
        decoded = self.model.generate(
            token_histories,
            self.mapper,
            beam_width=self.beam_width,
            batch_size=self.batch_size,
        )
        empty = sum(1 for d in decoded if not d)
        stats = {
            "beam_width": self.beam_width,
            "empty_decodings": int(empty),
            "users": len(histories),
            "mean_candidates_per_user": float(np.mean([len(d) for d in decoded])) if decoded else 0.0,
        }
        return decoded, stats

    # ------------------------------------------------------------------
    def recall(self, user_id: int, history: Sequence[int], top_k: int) -> list[RecallCandidate]:
        decoded, stats = self.generate_batch([history])
        self.last_stats = stats
        return self._to_candidates(decoded[0], history, top_k)

    def _to_candidates(
        self,
        decoded: list[tuple[tuple[int, ...], float]],
        history: Sequence[int],
        top_k: int,
    ) -> list[RecallCandidate]:
        seen = self._seen(history)
        out: list[RecallCandidate] = []
        used: set[int] = set()
        for beam_rank, (sid, logprob) in enumerate(decoded, start=1):
            items = self.mapper.sid_to_items.get(tuple(sid), [])
            for item in items:
                item = int(item)
                if item == PAD or item in seen or item in used:
                    continue
                used.add(item)
                out.append(RecallCandidate(
                    item_id=item,
                    score=float(logprob),
                    source=self.name,
                    rank=len(out) + 1,
                    extra={
                        "semantic_id": list(int(c) for c in sid),
                        "beam_rank": beam_rank,
                        "generation_logprob": float(logprob),
                        "collision_size": len(items),
                    },
                ))
                if len(out) >= top_k:
                    return out
        return out

    # ------------------------------------------------------------------
    def is_ready(self) -> tuple[bool, str]:
        ok = bool(self.mapper.num_items)
        return ok, (f"{self.mapper.num_items} items, {self.mapper.num_levels} levels x "
                    f"{self.mapper.codebook_size} codes, beam {self.beam_width}")


def load_generative_recall(
    checkpoint: str | Path,
    semantic_ids: str | Path,
    meta_path: str | Path | None = None,
    device: str = "cpu",
    beam_width: int = 20,
    max_history_items: int = 20,
) -> GenerativeRecall:
    """Rebuild a :class:`GenerativeRecall` from a trained checkpoint.

    The checkpoint records the Semantic-ID hash it was trained on; loading it
    against a different Semantic-ID table is refused, because the token ids would
    silently refer to different items.
    """
    import hashlib
    import json

    from ..models.generative_rec import GenerativeRecommender

    ROOT = Path(__file__).resolve().parents[2]
    ckpt_path = Path(checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    sid_path = Path(semantic_ids)
    if not sid_path.is_absolute():
        sid_path = ROOT / sid_path

    npz = np.load(sid_path)
    codes = npz["codes"].astype(np.int64)
    meta_p = Path(meta_path) if meta_path else sid_path.with_name("semantic_ids_meta.json")
    if not Path(meta_p).is_absolute():
        meta_p = ROOT / meta_p
    if "codebook_size" in npz:
        codebook_size = int(npz["codebook_size"])
        num_items = int(npz["num_items"])
    else:
        meta = json.loads(Path(meta_p).read_text(encoding="utf-8"))
        codebook_size = int(meta["codebook_size"])
        num_items = int(meta["num_items"])

    sid_hash = hashlib.sha256(np.ascontiguousarray(codes).tobytes()).hexdigest()[:16]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    stored = (ckpt.get("semantic_id_hash") or (ckpt.get("args") or {}).get("semantic_id_hash"))
    if stored is not None and stored != sid_hash:
        raise AssertionError(
            f"checkpoint was trained on Semantic IDs {stored} but {sid_path} hashes to "
            f"{sid_hash}; refusing to load"
        )

    args = ckpt.get("args") or {}
    num_levels = int(codes.shape[1])
    if "max_items" not in args:
        raise AssertionError(
            f"{ckpt_path} does not record max_items; it was not written by "
            "scripts/train_generative_rec.py and its sequence length is unknown"
        )
    train_max_items = int(args["max_items"])
    # Take the sequence length from the checkpoint itself.  Recomputing it from
    # the formula would silently diverge the moment the training script changes
    # how it sizes the positional embedding, and the failure would surface as a
    # shape error only at load time.
    state = ckpt["model_state"]
    pos_key = "encoder.position_embedding.weight"
    if pos_key not in state:
        raise AssertionError(f"{ckpt_path} has no {pos_key}; cannot infer sequence length")
    max_seq_len = int(state[pos_key].shape[0])
    if max_seq_len < 1 + num_levels:
        raise AssertionError(f"checkpoint sequence length {max_seq_len} is too small")
    model = GenerativeRecommender(
        vocab_size=3 + num_levels * codebook_size,
        num_levels=num_levels,
        hidden_size=int(args.get("hidden_size", 192)),
        num_layers=int(args.get("num_layers", 3)),
        num_heads=int(args.get("num_heads", 6)),
        dropout=0.0,
        max_seq_len=max_seq_len,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mapper = SemanticIDMapper(codes, num_items=num_items, codebook_size=codebook_size)
    return GenerativeRecall(
        model=model, mapper=mapper, num_items=num_items, beam_width=beam_width,
        max_history_items=min(max_history_items, train_max_items), device=device,
    )
