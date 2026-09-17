"""Cold-item protocol: no cold item may be learnable through the training split."""

import numpy as np

from src.data.dataset import ProcessedData


def test_cold_items_exist_and_are_a_reasonable_share(synthetic_cold_dir):
    d = ProcessedData.load(synthetic_cold_dir)
    n_cold = int(d.is_cold.sum())
    assert n_cold > 0
    assert 0.05 <= n_cold / d.num_items <= 0.30


def test_cold_items_have_zero_training_frequency(synthetic_cold_dir):
    d = ProcessedData.load(synthetic_cold_dir)
    cold_ids = np.flatnonzero(d.is_cold)
    assert (d.train_freq[cold_ids] == 0).all(), "a cold item received training interactions"


def test_cold_items_are_absent_from_every_training_sequence(synthetic_cold_dir):
    d = ProcessedData.load(synthetic_cold_dir)
    for u in range(d.num_users):
        s = d.user_offsets[u]
        train = d.flat_items[s : s + int(d.train_len[u])]
        assert not d.is_cold[train].any(), f"user {u} has a cold item in the training history"


def test_cold_items_still_have_evaluation_targets(synthetic_cold_dir):
    d = ProcessedData.load(synthetic_cold_dir)
    targets = np.concatenate([d.val_target, d.test_target])
    n_cold_targets = int(d.is_cold[targets].sum())
    assert n_cold_targets > 0, "cold split produced no evaluable cold targets"
    assert n_cold_targets >= 0.5 * int(d.is_cold.sum())


def test_non_cold_items_are_untouched(synthetic_cold_dir):
    d = ProcessedData.load(synthetic_cold_dir)
    hot_ids = np.flatnonzero(~d.is_cold)[1:]
    assert (d.train_freq[hot_ids] > 0).any()


def test_cold_mask_is_false_for_pad_and_aligned(synthetic_cold_dir):
    d = ProcessedData.load(synthetic_cold_dir)
    assert not d.is_cold[0]
    assert d.is_cold.shape[0] == d.num_items + 1


def test_base_dataset_has_no_cold_items(synthetic_data):
    assert not synthetic_data.is_cold.any()


def test_cold_split_is_deterministic(synthetic_cold_dir, tmp_path):
    """Same seed => identical cold set."""
    from src.data.synthetic import make_synthetic_dataset

    a = make_synthetic_dataset(tmp_path / "a", num_users=80, num_items=50, cold_ratio=0.15, seed=1)
    b = make_synthetic_dataset(tmp_path / "b", num_users=80, num_items=50, cold_ratio=0.15, seed=1)
    da, db = ProcessedData.load(a), ProcessedData.load(b)
    assert np.array_equal(da.is_cold, db.is_cold)
    assert np.array_equal(da.flat_items, db.flat_items)
