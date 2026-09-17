"""Negatives must never be the positive target, a *train* interaction, or PAD.

The second guarantee is exactly the one that must **not** be strengthened: the
training sampler may only know the interactions the model has already observed
(the train prefix).  Validation and test targets are the future, and excluding
them would leak the label into training.
"""

import numpy as np

from src.data.dataset import ProcessedData
from src.data.negative_sampler import NegativeSampler, build_training_interaction_keys


def _sampler(data: ProcessedData, mode: str = "uniform") -> NegativeSampler:
    """Exactly the sampler the Trainer builds: train-prefix keys only."""
    return NegativeSampler(
        num_items=data.num_items,
        train_freq=data.train_freq,
        all_interactions=build_training_interaction_keys(
            data.flat_items, data.user_offsets, data.train_len, data.num_items
        ),
        mode=mode,
        seed=7,
    )


def _train_sets(data: ProcessedData) -> list[set]:
    return [set(data.train_items(u).tolist()) for u in range(data.num_users)]


def test_never_samples_pad(synthetic_data: ProcessedData):
    s = _sampler(synthetic_data)
    users = np.arange(synthetic_data.num_users)
    neg = s.sample(users, 64)
    assert (neg > 0).all(), "PAD sampled as a negative"


def test_never_samples_a_known_interaction(synthetic_data: ProcessedData):
    data = synthetic_data
    s = _sampler(data)
    known = _train_sets(data)
    users = np.arange(data.num_users)
    neg = s.sample(users, 64)
    for b, u in enumerate(users):
        assert not (set(neg[b].tolist()) & known[u]), f"negative collided with user {u}'s training history"


def test_never_samples_the_positive_target(synthetic_data: ProcessedData):
    """``exclude_items`` is the per-batch target column the Trainer passes."""
    data = synthetic_data
    s = _sampler(data)
    users = np.arange(data.num_users)
    per_user = [data.train_items(u)[1:] for u in users]
    targets = np.zeros((users.shape[0], max(len(t) for t in per_user)), dtype=np.int64)
    for b, t in enumerate(per_user):
        targets[b, : t.shape[0]] = t
    neg = s.sample(users, 64, exclude_items=targets)
    for b in range(users.shape[0]):
        assert not (set(neg[b].tolist()) & set(per_user[b].tolist()))


def test_popularity_mode_respects_exclusions(synthetic_data: ProcessedData):
    data = synthetic_data
    s = _sampler(data, mode="popularity")
    known = _train_sets(data)
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


# ----------------------------------------------------------------------
# the training sampler must not know the future
# ----------------------------------------------------------------------
def test_train_keys_contain_no_validation_or_test_target(synthetic_data: ProcessedData):
    data = synthetic_data
    keys = set(
        build_training_interaction_keys(
            data.flat_items, data.user_offsets, data.train_len, data.num_items
        ).tolist()
    )
    for u in range(data.num_users):
        for target in (int(data.val_target[u]), int(data.test_target[u])):
            assert u * data.num_items + target not in keys, (
                f"user {u}: future target {target} is excluded from negative sampling"
            )


def test_train_keys_contain_every_training_interaction(synthetic_data: ProcessedData):
    data = synthetic_data
    keys = set(
        build_training_interaction_keys(
            data.flat_items, data.user_offsets, data.train_len, data.num_items
        ).tolist()
    )
    for u in range(data.num_users):
        for item in data.train_items(u).tolist():
            assert u * data.num_items + int(item) in keys


def test_future_targets_stay_reachable_as_negatives(synthetic_data: ProcessedData):
    """A val/test target must remain a legal negative -- heavily drawn here.

    If the sampler excluded future targets, the training objective would be
    scoring against a candidate set shaped by the label.
    """
    data = synthetic_data
    s = _sampler(data)
    users = np.arange(data.num_users)
    draws = s.sample(users, 4096)
    for b, u in enumerate(users):
        drawn = set(draws[b].tolist())
        for target in (int(data.val_target[u]), int(data.test_target[u])):
            if target in _train_sets(data)[u]:
                continue  # already seen in training, legitimately excluded
            assert target in drawn, (
                f"user {u}: future target {target} is unreachable as a negative"
            )


def test_sampler_ignores_future_positives_that_are_unknown_in_training(tmp_path):
    """Direct construction: user history train=[1,2,3], val=4, test=5.

    Nothing about 4 or 5 is observable during training, so both must remain
    drawable; 1,2,3 must never be drawn.
    """
    flat = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    offsets = np.array([0, 5], dtype=np.int64)
    train_len = np.array([3], dtype=np.int64)
    num_items = 6
    s = NegativeSampler(
        num_items=num_items,
        all_interactions=build_training_interaction_keys(flat, offsets, train_len, num_items),
        mode="uniform",
        seed=0,
    )
    draws = s.sample(np.array([0]), 4096)[0]
    seen = set(draws.tolist())
    assert not (seen & {1, 2, 3}), "a training history item was sampled as a negative"
    assert 4 in seen and 5 in seen, f"future targets were excluded: {sorted(seen)}"


def test_trainer_wires_the_train_prefix_sampler(synthetic_data, synthetic_dir, tmp_path):
    """Integration: the keys the Trainer builds must match the train prefix."""
    import torch

    from src.training.factory import build_model
    from src.training.trainer import Trainer
    from src.utils.config import Config

    data = synthetic_data
    cfg = Config({
        "model": {"name": "sasrec", "hidden_size": 8, "num_layers": 1, "num_heads": 2,
                  "dropout": 0.0, "max_seq_len": 12, "zero_cold_id": True},
        "data": {"processed_dir": str(synthetic_dir), "feature_dir": str(synthetic_dir),
                 "max_sequence_length": 12},
        "loss": {"num_negatives": 8, "temperature": 1.0},
        "negative_sampling": {"mode": "uniform"},
        "training": {"batch_size": 8, "learning_rate": 0.01, "max_grad_norm": 5.0,
                     "amp": False, "epochs": 1, "scheduler": "none", "monitor": "NDCG@10",
                     "monitor_mode": "max", "seed": 0, "num_workers": 0},
        "evaluation": {"full_ranking": True, "item_chunk_size": 128, "ks": [5, 10, 20]},
    })
    model = build_model(cfg, data, device=torch.device("cpu"))
    trainer = Trainer(model, cfg, data, tmp_path / "run", torch.device("cpu"))
    keys = set(np.asarray(trainer.sampler._keys).tolist())
    expected = set(
        build_training_interaction_keys(
            data.flat_items, data.user_offsets, data.train_len, data.num_items
        ).tolist()
    )
    assert keys == expected
    all_interactions = set(
        NegativeSampler.build_interaction_keys(data.flat_items, data.user_offsets, data.num_items).tolist()
    )
    assert keys < all_interactions, "the fixture does not exercise the difference"
