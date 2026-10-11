"""Greedy maximal marginal relevance using fixed content cosine similarity.

Relevance is min-max scaled over the ranking window. Lambda=1 exactly
preserves the incoming ranking, including ties. Complexity is O(window*K*D),
without a full catalogue or window-square similarity matrix.
"""
from __future__ import annotations

import numpy as np


def content_diversity(vectors: np.ndarray) -> dict:
    """ILD over available pairs; missing vectors are never counted as diverse."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norm = np.linalg.norm(np.where(np.isfinite(vectors), vectors, 0), axis=1)
    valid = np.isfinite(vectors).all(axis=1) & np.isfinite(norm) & (norm > 1e-8)
    v = vectors[valid] / norm[valid, None]
    pairs = len(v) * (len(v)-1) // 2
    total = len(vectors) * (len(vectors)-1) // 2
    if not pairs:
        return {"ild": None, "mean_cosine": None, "near_duplicate_pair_share": None,
                "valid_pair_fraction": 0.0}
    sim = np.clip(np.einsum("id,jd->ij", v, v, optimize=False), -1.0, 1.0)
    values = sim[np.triu_indices(len(v), k=1)]
    mean = float(values.mean())
    return {"ild": 1-mean, "mean_cosine": mean,
            "near_duplicate_pair_share": float((values >= .9).mean()),
            "valid_pair_fraction": pairs / total if total else 0.0}


def mmr_order(entries: list[dict], embeddings: np.ndarray | None, final_k: int,
              lambda_: float = .8, candidate_k: int = 100) -> tuple[list[dict], dict]:
    """Choose top-K from a relevance window, then append unchosen entries.

    Returns new dictionaries for chosen items with selection-time MMR evidence.
    Unknown content has conservative redundancy 1 after the first selection,
    rather than receiving a novelty bonus for a missing vector. Missing/invalid
    tables fall back explicitly to the original order.
    """
    if not np.isfinite(lambda_) or not 0 <= lambda_ <= 1 or candidate_k < 1:
        raise ValueError("MMR requires lambda in [0,1] and a positive candidate window")
    count = min(len(entries), max(candidate_k, final_k))
    info = {"requested": True, "applied": False, "lambda": float(lambda_),
            "candidate_window": int(candidate_k), "candidate_count": count,
            "changed_positions": 0, "missing_feature_items": 0,
            "fallback_reason": None}
    if final_k <= 0 or not entries:
        info["fallback_reason"] = "empty_output"
        return list(entries), info
    if lambda_ == 1:
        info["fallback_reason"] = "lambda_one_identity"
        return list(entries), info
    if embeddings is None:
        info["fallback_reason"] = "content_embeddings_unavailable"
        return list(entries), info
    table = np.asarray(embeddings)
    ids = np.asarray([int(e["item_id"]) for e in entries[:count]], dtype=np.int64)
    if table.ndim != 2 or table.shape[1] == 0 or (ids <= 0).any() or (ids >= len(table)).any():
        info["fallback_reason"] = "content_embedding_mapping_invalid"
        return list(entries), info
    v = np.asarray(table[ids], dtype=np.float32)
    norm = np.linalg.norm(np.where(np.isfinite(v), v, 0), axis=1)
    valid = np.isfinite(v).all(axis=1) & np.isfinite(norm) & (norm > 1e-8)
    info["missing_feature_items"] = int((~valid).sum())
    if not valid.any():
        info["fallback_reason"] = "all_content_unavailable"
        return list(entries), info
    v = np.where(valid[:, None], v, 0) / np.maximum(norm[:, None], 1e-8)
    relevance = np.asarray([float(e["ranking_score"]) for e in entries[:count]], dtype=np.float64)
    if not np.isfinite(relevance).all():
        info["fallback_reason"] = "nonfinite_ranking_scores"
        return list(entries), info
    lo, hi = relevance.min(), relevance.max()
    relevance = (relevance-lo)/(hi-lo) if hi > lo else np.ones(count)
    available = np.ones(count, dtype=bool)
    redundancy = np.zeros(count, dtype=np.float64)
    selected, chosen = [], []
    for step in range(min(final_k, count)):
        objective = lambda_*relevance - (1-lambda_)*redundancy
        objective[~available] = -np.inf
        index = int(np.argmax(objective))  # deterministic original-order tie break
        selected.append({**entries[index], "mmr_score": float(objective[index]),
                         "mmr_relevance": float(relevance[index]),
                         "mmr_redundancy": float(redundancy[index])})
        chosen.append(index)
        available[index] = False
        similarity = np.clip(np.einsum("ij,j->i", v, v[index], optimize=False), -1.0, 1.0)
        similarity[~valid] = 1.0
        if not valid[index]:
            similarity[:] = 1.0
        redundancy = similarity.astype(np.float64) if step == 0 else np.maximum(redundancy, similarity)
    info["applied"] = True
    info["changed_positions"] = sum(e["item_id"] != entries[i]["item_id"] for i,e in enumerate(selected))
    info["before"] = content_diversity(table[ids[:min(final_k,count)]])
    info["after"] = content_diversity(table[[int(e["item_id"]) for e in selected]])
    chosen_set = set(chosen)
    return selected + [e for i,e in enumerate(entries) if i not in chosen_set], info
