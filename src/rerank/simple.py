"""Lightweight serving-side reranking.

Deliberately *not* a learned reranker.  It implements the three policies every
feed needs and nothing more:

1. **seen filter** — never show an item the user already watched;
2. **dedup** — never show the same item twice;
3. **exploration quota** — reserve a few slots for items with **no training
   signal** (``exploration_candidate``), which a score-only ordering would
   never surface.

The quota is an *exposure policy*, not a model improvement.  Offline ranking
metrics are measured with it **off**; it exists because the cold-item experiment
shows content-only items are systematically out-scored by warm items in the full
catalogue, so without a quota they would receive no impressions at all.

Which items count as exploration targets is decided upstream by
``ItemMetadata`` (``is_zero_train_signal`` or simulated cold), not here — the
reranker only enforces the policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RerankConfig:
    filter_seen: bool = True
    dedup: bool = True
    #: reserve slots for exploration candidates (items with no training signal)
    exploration: bool = False
    #: minimum number of exploration candidates guaranteed in the output
    exploration_quota: int = 2

    def as_dict(self) -> dict:
        return {
            "filter_seen": self.filter_seen,
            "dedup": self.dedup,
            "exploration": self.exploration,
            "exploration_quota": self.exploration_quota,
        }


@dataclass
class RerankResult:
    items: list[dict]
    applied: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"applied": self.applied, "items": self.items}


def rerank(
    scored_items: list[dict],
    config: RerankConfig,
    final_k: int,
    history: set[int] | None = None,
) -> RerankResult:
    """Apply the serving policies to an already-scored candidate list.

    ``scored_items`` must be ordered by descending ranking score and each entry
    must carry ``item_id``, ``ranking_score`` and ``exploration_candidate``.

    ``entry["exploration"]`` is set to ``True`` only for items that were
    **injected by the quota**, i.e. items that would not have made the cut on
    their own score.  An exploration candidate that ranks highly on merit is not
    marked, so the flag always means "this slot was reserved for it".
    """
    history = history or set()
    applied = {"seen_filtered": 0, "deduped": 0, "exploration_injected": 0,
               "exploration_available": 0, "config": config.as_dict()}

    kept: list[dict] = []
    seen_ids: set[int] = set()
    for entry in scored_items:
        item = int(entry["item_id"])
        if config.filter_seen and item in history:
            applied["seen_filtered"] += 1
            continue
        if config.dedup and item in seen_ids:
            applied["deduped"] += 1
            continue
        seen_ids.add(item)
        kept.append(entry)

    exploration_pool = [e for e in kept if e.get("exploration_candidate")]
    applied["exploration_available"] = len(exploration_pool)
    injected_ids: set[int] = set()

    if not config.exploration or config.exploration_quota <= 0:
        out = kept[:final_k]
    else:
        quota = min(int(config.exploration_quota), final_k)
        # reserve `quota` slots; the rest go to the best-scoring candidates
        # overall, and exploration candidates may also qualify on merit
        head = kept[: max(final_k - quota, 0)]
        head_ids = {int(e["item_id"]) for e in head}
        injected = [e for e in exploration_pool if int(e["item_id"]) not in head_ids][:quota]
        injected_ids = {int(e["item_id"]) for e in injected}
        out = head + injected
        applied["exploration_injected"] = len(injected)
        if len(out) < final_k:  # fewer exploration items than the quota: top up
            chosen = {int(e["item_id"]) for e in out}
            for e in kept:
                if len(out) >= final_k:
                    break
                if int(e["item_id"]) not in chosen:
                    out.append(e)
                    chosen.add(int(e["item_id"]))
        out.sort(key=lambda e: -float(e["ranking_score"]))

    for pos, entry in enumerate(out, start=1):
        entry["final_rank"] = pos
        # exact: only items the quota actually injected are flagged
        entry["exploration"] = int(entry["item_id"]) in injected_ids
    applied["exploration_injected_ids"] = sorted(injected_ids)
    return RerankResult(items=out[:final_k], applied=applied)
