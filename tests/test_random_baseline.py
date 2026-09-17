"""The uniform-Random baseline: a frozen ranker whose cold recall has a closed form.

It exists so that the cold-split floor is *measured* rather than remembered.
With ``N`` candidates, a uniform ranker has expected Recall@k = k/N and mean
rank (N+1)/2; both are checked here on the evaluator the real runs use.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EvalDataset, ProcessedData, collate_eval
from src.data.synthetic import make_synthetic_dataset
from src.evaluation.evaluator import FullRankingEvaluator
from src.models.random_model import RandomRecommender
from src.training.factory import build_model
from src.utils.config import Config


def _loader(data, split, bs=32):
    return DataLoader(
        EvalDataset(data, 50, split=split), batch_size=bs, shuffle=False, collate_fn=collate_eval
    )


def test_scores_are_frozen_and_parameter_free():
    model = RandomRecommender(num_items=50, seed=42)
    assert model.num_parameters() == 0
    assert list(model.parameters()) == []
    assert not model.scores.requires_grad


def test_seed_determines_the_scores():
    a = RandomRecommender(50, seed=42).scores
    b = RandomRecommender(50, seed=42).scores
    c = RandomRecommender(50, seed=7).scores
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_pad_slot_is_never_a_candidate(tmp_path):
    """PAD may only appear as the evaluator's tail filler, never as a ranked item."""
    d = make_synthetic_dataset(tmp_path / "s", num_users=20, num_items=30, seed=0)
    data = ProcessedData.load(d)
    model = RandomRecommender(data.num_items, seed=42)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20), item_chunk_size=512)
    res = ev.evaluate(lambda b: model.encode(b["user_id"]), model.all_item_embeddings(), _loader(data, "test"))
    assert (res.targets != 0).all()
    for row in res.topk:
        pad = row == 0
        if pad.any():
            first = int(np.argmax(pad))
            assert (row[first:] == 0).all(), f"PAD ranked before a real item: {row}"


def test_factory_builds_it(tmp_path):
    d = make_synthetic_dataset(tmp_path / "s", num_users=20, num_items=30, seed=0)
    data = ProcessedData.load(d)
    cfg = Config({"model": {"name": "random"}, "training": {"seed": 7}})
    model = build_model(cfg, data, device="cpu")
    assert isinstance(model, RandomRecommender)
    assert model.num_parameters() == 0
    assert torch.equal(model.scores, RandomRecommender(data.num_items, seed=7).scores)


def test_cold_only_ranking_matches_the_k_over_n_floor(tmp_path):
    """Cold items have no training interactions, so a learned ranker cannot see
    them; a uniform ranker must land on exactly the chance level."""
    d = make_synthetic_dataset(
        tmp_path / "cold", num_users=150, num_items=200, cold_ratio=0.5, seed=3
    )
    data = ProcessedData.load(d)
    n_cold = int(data.is_cold.sum())
    assert n_cold >= 50

    model = RandomRecommender(data.num_items, seed=42)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20), item_chunk_size=512)
    res = ev.evaluate(
        lambda b: model.encode(b["user_id"]),
        model.all_item_embeddings(),
        _loader(data, "test"),
        candidate_mask=data.is_cold,
    )
    cold = res.subset(data.is_cold[res.targets])
    n = len(cold.rank)
    assert n >= 50 and cold.rank.min() >= 1 and cold.rank.max() <= n_cold

    # ranks of a uniform ranker over n_cold candidates: mean (N+1)/2,
    # so the sample mean has standard error N / sqrt(12 n)
    se_rank = n_cold / np.sqrt(12 * n)
    assert abs(cold.rank.mean() - (n_cold + 1) / 2) < 4 * se_rank

    # Recall@k = k/N with binomial noise
    recall20 = cold.metrics()["Recall@20"]
    se_recall = np.sqrt((20 / n_cold) * (1 - 20 / n_cold) / n)
    assert abs(recall20 - 20 / n_cold) < 4 * max(se_recall, 1e-9)
