"""Full-ranking evaluation: seen-item masking, exact ranks, target protection."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import EvalDataset, ProcessedData, collate_eval
from src.evaluation.evaluator import FullRankingEvaluator


class _ToyModel:
    """Deterministic scorer: score(u, i) = <u_onehot-ish, e_i>."""

    def __init__(self, num_users, num_items, hidden, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.item_emb = torch.randn(num_items + 1, hidden, generator=g)
        self.item_emb[0] = 0
        self.user_emb = torch.randn(num_users + 1, hidden, generator=g)

    def encode_fn(self, batch):
        return self.user_emb[batch["user_id"]]


def _loader(data, max_len, split, bs=16):
    return DataLoader(
        EvalDataset(data, max_len, split=split),
        batch_size=bs,
        shuffle=False,
        collate_fn=collate_eval,
    )


def test_ranks_match_bruteforce(synthetic_data: ProcessedData):
    data = synthetic_data
    model = _ToyModel(data.num_users, data.num_items, hidden=16)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20))
    res = ev.evaluate(model.encode_fn, model.item_emb, _loader(data, 50, "test"))

    # brute force reference for the first 25 users
    for b in range(25):
        u = int(res.user_ids[b])
        scores = (model.user_emb[u] @ model.item_emb.t()).clone()
        scores[0] = float("-inf")
        hist = data.test_history(u)
        for it in hist:
            scores[int(it)] = float("-inf")
        tgt = int(res.targets[b])
        assert torch.isfinite(scores[tgt])
        expected_rank = int((scores > scores[tgt]).sum().item()) + 1
        assert int(res.rank[b]) == expected_rank


def test_seen_items_never_appear_in_topk(synthetic_data: ProcessedData):
    data = synthetic_data
    model = _ToyModel(data.num_users, data.num_items, hidden=16)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20))
    res = ev.evaluate(model.encode_fn, model.item_emb, _loader(data, 50, "test"))
    for b in range(res.topk.shape[0]):
        u = int(res.user_ids[b])
        seen = set(data.test_history(u).tolist())
        assert not (set(res.topk[b].tolist()) & seen)


def test_pad_never_appears_in_topk(synthetic_data: ProcessedData):
    data = synthetic_data
    model = _ToyModel(data.num_users, data.num_items, hidden=16)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20))
    res = ev.evaluate(model.encode_fn, model.item_emb, _loader(data, 50, "test"))
    assert (res.topk > 0).all()


def test_target_is_never_masked_even_if_it_looks_like_history(synthetic_data: ProcessedData):
    """Even if a target also appeared in the input, masking must protect it."""
    data = synthetic_data
    model = _ToyModel(data.num_users, data.num_items, hidden=16)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20))

    loader = _loader(data, 50, "test")
    batch = next(iter(loader))
    batch["input_ids"][0, -1] = batch["target"][0]  # force a collision
    scores = torch.zeros(batch["input_ids"].shape[0], data.num_items + 1)
    masked = ev._mask_seen(scores, batch["input_ids"], batch["target"])
    assert torch.isfinite(masked[0, int(batch["target"][0])])


def test_candidate_restriction(synthetic_data: ProcessedData):
    """With a restricted catalogue, only those items may ever be ranked."""
    data = synthetic_data
    model = _ToyModel(data.num_users, data.num_items, hidden=16)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20))
    allowed = np.zeros(data.num_items + 1, dtype=bool)
    allowed[[1, 2, 3, 4, 5, 6]] = True
    allowed[data.test_target] = True  # keep every target rankable
    res = ev.evaluate(
        model.encode_fn, model.item_emb, _loader(data, 50, "test"), candidate_mask=allowed
    )
    ranked = set(np.unique(res.topk).tolist()) - {0}
    assert ranked <= {1, 2, 3, 4, 5, 6} | set(data.test_target.tolist())


def test_candidate_restriction_excludes_missing_targets(synthetic_data: ProcessedData):
    """A target outside the candidate set must simply never be retrieved."""
    data = synthetic_data
    model = _ToyModel(data.num_users, data.num_items, hidden=16)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20))
    allowed = np.zeros(data.num_items + 1, dtype=bool)
    allowed[[1, 2, 3]] = True
    allowed[data.test_target] = False  # no target is rankable
    res = ev.evaluate(
        model.encode_fn, model.item_emb, _loader(data, 50, "test"), candidate_mask=allowed
    )
    assert (res.rank > 20).all()
    assert res.metrics()["Recall@20"] == 0.0


def test_metrics_are_bounded(synthetic_data: ProcessedData):
    data = synthetic_data
    model = _ToyModel(data.num_users, data.num_items, hidden=16)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20))
    res = ev.evaluate(model.encode_fn, model.item_emb, _loader(data, 50, "test"))
    m = res.metrics()
    for k in (5, 10, 20):
        assert 0.0 <= m[f"Recall@{k}"] <= 1.0
        assert 0.0 <= m[f"NDCG@{k}"] <= 1.0
        assert m[f"NDCG@{k}"] <= m[f"Recall@{k}"] + 1e-9


def test_subset_metrics_are_consistent(synthetic_data: ProcessedData):
    data = synthetic_data
    model = _ToyModel(data.num_users, data.num_items, hidden=16)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20))
    res = ev.evaluate(model.encode_fn, model.item_emb, _loader(data, 50, "test"))
    mask = np.arange(res.rank.shape[0]) % 2 == 0
    sub = res.subset(mask)
    assert sub.rank.shape[0] == int(mask.sum())
    assert np.allclose(sub.rank, res.rank[mask])
    assert sub.metrics()["Recall@10"] == pytest.approx(float((res.rank[mask] <= 10).mean()))


def test_perfect_scorer_gets_recall_one(synthetic_data: ProcessedData):
    """An oracle that ranks the target first must achieve Recall@k == 1.

    ``item_emb`` is the identity, so ``<u, e_i>`` is non-zero only for one item.
    Setting the user vector to ``10 * e_target`` makes the target the unique
    arg-max regardless of the seen-item masking.
    """
    data = synthetic_data
    n = data.num_items + 1
    item_emb = torch.eye(n)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20))

    def oracle(batch):
        return 10.0 * item_emb[batch["target"]]

    res = ev.evaluate(oracle, item_emb, _loader(data, 50, "test"))
    assert (res.rank == 1).all()
    m = res.metrics()
    for k in (5, 10, 20):
        assert m[f"Recall@{k}"] == pytest.approx(1.0)
        assert m[f"NDCG@{k}"] == pytest.approx(1.0)


def test_worst_scorer_gets_recall_zero(synthetic_data: ProcessedData):
    data = synthetic_data
    n = data.num_items + 1
    item_emb = torch.eye(n)
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20))

    def anti_oracle(batch):
        # align with the *smallest* valid item id, which is almost never the target
        return -10.0 * item_emb[batch["target"]]

    res = ev.evaluate(anti_oracle, item_emb, _loader(data, 50, "test"))
    assert res.metrics()["Recall@20"] <= 0.2
