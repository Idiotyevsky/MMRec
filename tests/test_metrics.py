"""Metrics must be numerically exact on hand-computed rankings."""

import math

import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    coverage_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_hand_computed():
    rank = np.array([1, 2, 3, 4, 100])
    assert recall_at_k(rank, 1) == pytest.approx(1 / 5)
    assert recall_at_k(rank, 3) == pytest.approx(3 / 5)
    assert recall_at_k(rank, 4) == pytest.approx(4 / 5)
    assert recall_at_k(rank, 100) == pytest.approx(1.0)


def test_ndcg_hand_computed():
    rank = np.array([1, 2, 3, 4, 100])
    expected3 = (1.0 + 1 / math.log2(3) + 1 / math.log2(4)) / 5
    assert ndcg_at_k(rank, 3) == pytest.approx(expected3, rel=1e-12)
    expected1 = 1.0 / 5
    assert ndcg_at_k(rank, 1) == pytest.approx(expected1, rel=1e-12)
    assert ndcg_at_k(np.array([1]), 1) == pytest.approx(1.0)


def test_mrr_hand_computed():
    rank = np.array([1, 2, 3, 4, 100])
    expected = (1.0 + 0.5 + 1 / 3 + 0.25) / 5
    assert mrr_at_k(rank, 3) == pytest.approx((1.0 + 0.5 + 1 / 3) / 5, rel=1e-12)
    assert mrr_at_k(rank, 4) == pytest.approx(expected, rel=1e-12)


def test_ndcg_perfect_ranking_is_one():
    rank = np.ones(10)
    assert recall_at_k(rank, 5) == 1.0
    assert ndcg_at_k(rank, 5) == pytest.approx(1.0)
    assert mrr_at_k(rank, 5) == pytest.approx(1.0)
    assert hit_rate_at_k(rank, 5) == pytest.approx(1.0)


def test_ndcg_never_exceeds_one():
    rng = np.random.default_rng(0)
    rank = rng.integers(1, 500, size=2000)
    for k in (1, 5, 10, 20, 100):
        assert 0.0 <= ndcg_at_k(rank, k) <= 1.0
        assert 0.0 <= recall_at_k(rank, k) <= 1.0
        assert 0.0 <= mrr_at_k(rank, k) <= 1.0


def test_recall_monotonic_in_k():
    rng = np.random.default_rng(1)
    rank = rng.integers(1, 300, size=1000)
    prev = -1.0
    for k in (1, 5, 10, 20, 50):
        v = recall_at_k(rank, k)
        assert v >= prev - 1e-12
        prev = v


def test_ndcg_matches_manual_formula():
    rank = np.array([7, 2, 30, 1, 20])
    for k in (5, 10, 20):
        manual = np.mean([1 / math.log2(r + 1) if r <= k else 0.0 for r in rank])
        assert ndcg_at_k(rank, k) == pytest.approx(manual, rel=1e-12)


def test_coverage_at_k():
    topk = np.array([[1, 2, 3], [3, 4, 5], [1, 1, 1]])
    assert coverage_at_k(topk, num_items=10) == pytest.approx(5 / 10)


def test_not_retrieved_sentinel_never_counts():
    from src.evaluation.metrics import NOT_RETRIEVED

    rank = np.array([1, NOT_RETRIEVED, NOT_RETRIEVED, NOT_RETRIEVED])
    assert recall_at_k(rank, 20) == pytest.approx(0.25)
    assert ndcg_at_k(rank, 20) == pytest.approx(0.25)


def test_tie_policy_is_tie_neutral():
    """Averaged ranks: a block of tied items shares the midpoint.

    Concretely: if 1974 cold items all score exactly 0 and the target is one of
    them, the target's rank must be ~987, not 1.
    """
    from src.evaluation.evaluator import FullRankingEvaluator

    B, N = 1, 1975
    scores = torch.full((B, N), -float("inf"))
    scores[0, 1:] = 0.0  # 1974 tied candidates, PAD excluded
    target = torch.tensor([7])
    rank = FullRankingEvaluator._average_rank(scores, target)
    assert rank.item() == pytest.approx(1 + (1974 - 1) / 2, abs=1e-6)


def test_unique_target_gets_ordinary_rank():
    from src.evaluation.evaluator import FullRankingEvaluator

    scores = torch.tensor([[float("-inf"), 0.5, 0.9, 0.1, 0.7]])
    target = torch.tensor([2])  # score 0.9 is the highest
    assert FullRankingEvaluator._average_rank(scores, target).item() == 1.0
    target = torch.tensor([4])  # 0.7 is beaten only by 0.9
    assert FullRankingEvaluator._average_rank(scores, target).item() == 2.0
