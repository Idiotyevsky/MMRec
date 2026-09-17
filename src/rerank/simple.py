"""Lightweight serving-side reranking.

Deliberately *not* a learned reranker.  It implements the three policies every
feed needs and nothing more:

1. **seen filter** — never show an item the user already watched;
2. **dedup** — never show the same item twice;
3. **cold exploration quota** — guarantee that at least ``cold_quota`` cold
   items appear in the final list.

The quota is an *exposure policy*, not a model improvement.  Offline ranking
metrics are measured with it **off**; it exists because the cold-item experiment
shows content-only cold items are systematically out-scored by warm items in the
full catalogue, so without a quota they would receive no impressions at all.
Turning it on trades a little relevance for cold-item exposure and catalogue
coverage — both of which are reported so the trade-off is visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RerankConfig:
    filter_seen: bool = True
    dedup: bool = True
    cold_exploration: bool = False
    cold_quota: int = 2  # minimum number of cold items guaranteed in the output

    def as_dict(self) -> dict:
        return {
            "filter_seen": self.filter_seen,
            "dedup": self.dedup,
            "cold_exploration": self.cold_exploration,
            "cold_quota": self.cold_quota,
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
    must carry ``item_id``, ``is_cold`` and ``ranking_score``.
    """
    history = history or set()
    applied = {"seen_filtered": 0, "deduped": 0, "cold_injected": 0,
               "cold_available": 0, "config": config.as_dict()}

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

    cold_pool = [e for e in kept if e.get("is_cold")]
    applied["cold_available"] = len(cold_pool)

    if not config.cold_exploration or config.cold_quota <= 0:
        out = kept[:final_k]
    else:
        quota = min(int(config.cold_quota), final_k)
        # reserve `quota` slots for cold items; the rest go to the best-scoring
        # candidates overall (cold items may also qualify on merit)
        head = kept[: max(final_k - quota, 0)]
        head_ids = {int(e["item_id"]) for e in head}
        injected = [e for e in cold_pool if int(e["item_id"]) not in head_ids][:quota]
        applied["cold_injected"] = len(injected)
        out = head + injected
        if len(out) < final_k:  # not enough cold items: top up on merit
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
        entry["exploration"] = bool(entry.get("is_cold") and config.cold_exploration
                                    and pos > final_k - config.cold_quota)
    return RerankResult(items=out[:final_k], applied=applied)
