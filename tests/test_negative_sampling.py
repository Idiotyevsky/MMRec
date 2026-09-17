"""Negatives must never be the positive target, a known interaction, or PAD."""

import numpy as np

from src.data.dataset import ProcessedData
from src.data.negative_sampler import NegativeSampler


def _sampler(data: ProcessedData, mode: str = "uniform") -> NegativeSampler:
    return NegativeSampler(
        num_items=data.num_items,
        train_freq=data.train_freq,
        all_interactions=NegativeSampler.build_interaction_keys(
            data.flat_items, data.user_offsets, data.num_items
        ),
        mode=mode,
        seed=7,
    )


def _known_sets(data: ProcessedData) -> list[set]:
    out = []
    for u in range(data.num_users):
        s, e = data.user_offsets[u], data.user_offsets[u + 1]
        out.append(set(data.flat_items[s:e].tolist()))
    return out


def test_never_samples_pad(synthetic_data: ProcessedData):
    s = _sampler(synthetic_data)
    users = np.arange(synthetic_data.num_users)
    neg = s.sample(users, 64)
    assert (neg > 0).all(), "PAD sampled as a negative"


def test_never_samples_a_known_interaction(synthetic_data: ProcessedData):
    data = synthetic_data
    s = _sampler(data)
    known = _known_sets(data)
    users = np.arange(data.num_users)
    neg = s.sample(users, 64)
    for b, u in enumerate(users):
        assert not (set(neg[b].tolist()) & known[u]), f"negative collided with user {u}'s history"


def test_never_samples_the_positive_target(synthetic_data: ProcessedData):
    data = synthetic_data
    s = _sampler(data)
    users = np.arange(data.num_users)
    targets = data.test_target[users]
    neg = s.sample(users, 64, exclude_items=targets)
    for b in range(users.shape[0]):
        assert targets[b] not in set(neg[b].tolist())


def test_popularity_mode_respects_exclusions(synthetic_data: ProcessedData):
    data = synthetic_data
    s = _sampler(data, mode="popularity")
    known = _known_sets(data)
    users = np.arange(data.num_users)
    neg = s.sample(users, 64)
    assert (neg > 0).all()
    for b, u in enumerate(users):
        assert not (set(neg[b].tolist()) & known[u])


def test_popularity_mode_is_skewed_towards_frequent_items(synthetic_data: ProcessedData):
    data = synthetic_data
    s = _sampler(data, mode="popularity")
    neg = s.sample(np.arange(data.num_users), 256).ravel()
    freq = data.train_freq[neg]
    uniform = _sampler(data, mode="uniform").sample(np.arange(data.num_users), 256).ravel()
    assert freq.mean() > data.train_freq[uniform].mean()


def test_draws_are_reproducible_for_a_fixed_seed(synthetic_data: ProcessedData):
    a = _sampler(synthetic_data).sample(np.arange(10), 32)
    b = _sampler(synthetic_data).sample(np.arange(10), 32)
    assert np.array_equal(a, b)


def test_shapes(synthetic_data: ProcessedData):
    s = _sampler(synthetic_data)
    neg = s.sample(np.arange(7), 16)
    assert neg.shape == (7, 16)
    assert neg.dtype == np.int64
