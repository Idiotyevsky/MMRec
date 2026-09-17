"""Item popularity statistics computed on TRAINING interactions only.

Leakage policy: every popularity number used as a feature, a bucketing rule, or
a negative-sampling distribution is derived exclusively from the training split.
Using full-lifetime counts would leak the existence of validation/test
interactions into the model and into the analysis.
"""

from __future__ import annotations

import numpy as np


def compute_item_frequency(
    sequences: list[np.ndarray] | np.ndarray,
    num_items: int,
    offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Count occurrences of every internal item id (index 0 = PAD, stays 0).

    Parameters
    ----------
    sequences : either a flat int array plus ``offsets``, or a list of arrays.
    """
    freq = np.zeros(num_items + 1, dtype=np.int64)
    if offsets is not None:
        flat = np.asarray(sequences, dtype=np.int64)
        off = np.asarray(offsets, dtype=np.int64)
        if flat.size:
            valid = flat > 0
            freq += np.bincount(flat[valid], minlength=num_items + 1)[: num_items + 1]
        _ = off
    else:
        for seq in sequences:
            seq = np.asarray(seq, dtype=np.int64)
            if seq.size == 0:
                continue
            valid = seq > 0
            if valid.any():
                freq += np.bincount(seq[valid], minlength=num_items + 1)[: num_items + 1]
    return freq


def popularity_buckets(
    train_freq: np.ndarray,
    rule: str = "frequency_quantile",
    head_frac: float = 0.2,
    tail_frac: float = 0.6,
    num_items: int | None = None,
) -> dict:
    """Assign every item to head / middle / tail.

    Two rules are supported so the main result can be robustness-checked:

    ``frequency_quantile``
        Split the *item catalogue* by frequency rank: the top ``head_frac`` of
        items (by frequency) are head, the bottom ``tail_frac`` are tail.
        This makes the tail a large but low-activity part of the catalogue,
        which is what "long tail" means for a content platform.

    ``interaction_mass``
        Split the *interaction mass*: head items are the smallest set of items
        covering ``head_frac`` of all interactions, tail items are the items
        that together cover only ``tail_frac`` of interactions (counted from
        the least popular end).

    Items with zero training interactions are always tail.
    """
    freq = np.asarray(train_freq, dtype=np.int64)
    if num_items is None:
        num_items = freq.shape[0] - 1
    item_freq = freq[1 : num_items + 1]

    bucket = np.full(num_items + 1, -1, dtype=np.int8)  # -1 = PAD
    bucket[1:] = 1  # default middle

    if rule == "frequency_quantile":
        order = np.argsort(-item_freq, kind="stable")
        n = num_items
        n_head = int(round(n * head_frac))
        n_tail = int(round(n * tail_frac))
        bucket[order[:n_head] + 1] = 2  # head
        if n_tail > 0:
            bucket[order[n - n_tail :] + 1] = 0  # tail
        # zero-frequency items are unconditionally tail
        bucket[1:][item_freq == 0] = 0
        rule_desc = (
            f"frequency_quantile: head=top {head_frac:.0%} items by train freq, "
            f"tail=bottom {tail_frac:.0%} items by train freq, middle=rest"
        )
    elif rule == "interaction_mass":
        total = item_freq.sum()
        order = np.argsort(-item_freq, kind="stable")
        cum = np.cumsum(item_freq[order]) / max(total, 1)
        n_head = int(np.searchsorted(cum, head_frac) + 1)
        bucket[order[:n_head] + 1] = 2
        order_asc = order[::-1]
        cum_asc = np.cumsum(item_freq[order_asc]) / max(total, 1)
        n_tail = int(np.searchsorted(cum_asc, tail_frac) + 1)
        bucket[order_asc[:n_tail] + 1] = 0
        bucket[1:][item_freq == 0] = 0
        rule_desc = (
            f"interaction_mass: head=smallest item set covering {head_frac:.0%} of "
            f"train interactions, tail=item set covering {tail_frac:.0%} of train "
            f"interactions from the least popular end"
        )
    else:
        raise ValueError(f"Unknown popularity rule: {rule}")

    counts = {name: int((bucket == v).sum()) for v, name in [(2, "head"), (1, "middle"), (0, "tail")]}
    mass = {
        name: float(item_freq[bucket[1:] == v].sum() / max(item_freq.sum(), 1))
        for v, name in [(2, "head"), (1, "middle"), (0, "tail")]
    }
    return {
        "bucket": bucket,
        "rule": rule,
        "rule_description": rule_desc,
        "counts": counts,
        "interaction_mass": mass,
    }


BUCKET_NAMES = {2: "head", 1: "middle", 0: "tail"}
BUCKET_IDS = {"head": 2, "middle": 1, "tail": 0}
