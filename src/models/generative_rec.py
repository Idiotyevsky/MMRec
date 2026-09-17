"""Semantic-ID generative recommendation (TIGER-style, deliberately small).

The item vocabulary is replaced by semantic code tokens, so the model is a
*generative retriever*: it emits the Semantic ID of the next item instead of
scoring a vector against a catalogue.  Decoding is constrained by a prefix trie
(``PrefixConstraint``), which is what stops the model from hallucinating items
that do not exist.

This is intentionally a small causal Transformer (a few million parameters),
not an LLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sasrec import SequenceEncoder
from .semantic_id import BOS, EOS, PAD, PrefixConstraint, SemanticIDMapper  # noqa: F401


def _flatten_tokens(history) -> list[int]:
    """Accept either a flat token list or a list of per-item token lists."""
    out: list[int] = []
    for t in history:
        if isinstance(t, (list, tuple)):
            out.extend(int(x) for x in t)
        else:
            out.append(int(t))
    return out


class GenerativeRecommender(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_levels: int,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.num_levels = int(num_levels)
        self.hidden_size = int(hidden_size)
        self.max_seq_len = int(max_seq_len)

        self.token_embedding = nn.Embedding(self.vocab_size, hidden_size, padding_idx=PAD)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[PAD].zero_()

        self.encoder = SequenceEncoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            dim_feedforward=dim_feedforward,
        )
        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(hidden_size, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # weight tying

    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.token_embedding(input_ids))
        h = self.encoder(emb, input_ids, return_sequence=True)
        return self.lm_head(h)

    def loss(self, input_ids: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_ids)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=PAD,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        histories: list[list[int]],
        mapper: SemanticIDMapper,
        beam_width: int = 8,
        batch_size: int = 64,
        max_history_tokens: int | None = None,
    ) -> list[list[tuple[tuple[int, ...], float]]]:
        """Constrained decoding of the next Semantic ID for each history.

        ``histories`` holds raw *token* sequences (no BOS).  Returns, per history,
        a ranked list of ``(semantic_id, cumulative_logprob)``; only real items
        can appear because every step is restricted to valid prefixes.
        """
        self.eval()
        constraint = PrefixConstraint(mapper)
        results: list[list[tuple[tuple[int, ...], float]]] = []
        limit = max_history_tokens or (self.max_seq_len - self.num_levels - 1)
        limit = max(limit, 1)
        histories = [_flatten_tokens(h) for h in histories]

        for start in range(0, len(histories), batch_size):
            chunk = histories[start : start + batch_size]
            results.extend(
                self._generate_chunk(chunk, constraint, mapper, beam_width, limit)
            )
        return results

    def _generate_chunk(self, chunk, constraint, mapper, beam_width, limit):
        device = next(self.parameters()).device
        B = len(chunk)
        beams: list[list[tuple[list[int], float]]] = []
        for hist in chunk:
            seq = [BOS] + list(hist)[-limit:]
            beams.append([(seq, 0.0)])

        for _step in range(self.num_levels):
            # batch all live beams of this chunk into one forward pass
            rows, meta = [], []
            for b in range(B):
                for i, (tokens, lp) in enumerate(beams[b]):
                    rows.append(tokens)
                    meta.append((b, i))
            if not rows:
                break
            L = max(len(r) for r in rows)
            ids = torch.zeros((len(rows), L), dtype=torch.long, device=device)
            for j, r in enumerate(rows):
                ids[j, L - len(r) :] = torch.tensor(r, dtype=torch.long, device=device)
            logits = self.forward(ids)[:, -1, :]
            logprobs = F.log_softmax(logits.float(), dim=-1)

            new_beams: list[list[tuple[list[int], float]]] = [[] for _ in range(B)]
            for j, (b, i) in enumerate(meta):
                tokens, lp = rows[j], beams[b][i][1]
                # only the *current item's* partially generated codes constrain
                # the next step; the history is already a complete SID sequence
                partial = tokens[1:]
                rem = len(partial) % self.num_levels
                prefix = partial[len(partial) - rem :] if rem else []
                allowed = constraint.allowed(prefix)
                if not allowed:
                    continue
                allow_t = torch.tensor(allowed, dtype=torch.long, device=device)
                vals = logprobs[j].index_select(0, allow_t)
                k = min(beam_width, vals.shape[0])
                top = torch.topk(vals, k=k)
                for v, idx in zip(top.values.tolist(), top.indices.tolist()):
                    new_beams[b].append((tokens + [allowed[idx]], lp + v))

            for b in range(B):
                if not new_beams[b]:
                    continue
                new_beams[b].sort(key=lambda x: -x[1])
                beams[b] = new_beams[b][:beam_width]

        out = []
        for b in range(B):
            cands = []
            for tokens, lp in beams[b]:
                body = tokens[1:]
                sid = mapper.sid_from_tokens(body[-self.num_levels :]) if body else None
                if sid is not None:
                    cands.append((sid, lp))
            cands.sort(key=lambda x: -x[1])
            out.append(cands)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
