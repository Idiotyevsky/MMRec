"""The chronological split must be strictly ordered with no future leakage."""

import numpy as np
import pytest

from src.data.dataset import EvalDataset, ProcessedData, TrainDataset


def test_train_val_test_are_chronologically_ordered(synthetic_data: ProcessedData):
    d = synthetic_data
    for u in range(d.num_users):
        hist = d.full_history(u)
        t = int(d.train_len[u])
        train = hist[:t]
        val = int(d.val_target[u])
        test = int(d.test_target[u])
        assert train.shape[0] >= 1
        # val target is the second-to-last, test target the last interaction
        assert val == hist[t]
        assert test == hist[t + 1]
        # no target may appear in the history that predicts it
        assert val not in set(train.tolist())
        assert test not in set(hist[: t + 1].tolist())


def test_eval_histories_match_the_protocol(synthetic_data: ProcessedData):
    d = synthetic_data
    for u in range(min(20, d.num_users)):
        vh = d.val_history(u)
        th = d.test_history(u)
        assert vh.shape[0] == int(d.train_len[u])
        assert th.shape[0] == int(d.train_len[u]) + 1
        assert th[-1] == d.val_target[u]  # val target is history for test
        assert d.test_target[u] not in set(th.tolist())


def test_train_dataset_shifts_by_one(synthetic_data: ProcessedData):
    ds = TrainDataset(synthetic_data, max_len=50)
    for i in range(min(20, len(ds))):
        s = ds[i]
        inp, tgt = s["input_ids"], s["target"]
        real_in = inp[inp > 0]
        real_tgt = tgt[tgt > 0]
        assert real_in.shape[0] == real_tgt.shape[0]
        # target is the input shifted one step to the left
        assert np.array_equal(real_tgt[:-1], real_in[1:])


def test_left_padding(synthetic_data: ProcessedData):
    ds = TrainDataset(synthetic_data, max_len=8)
    s = ds[0]
    inp = s["input_ids"]
    assert inp.shape[0] == 8
    # padding is on the left: once a non-zero appears, no zero may follow
    first_nonzero = np.flatnonzero(inp > 0)
    if first_nonzero.size:
        assert (inp[first_nonzero[0] :] > 0).all()


def test_long_sequence_is_truncated_to_most_recent(synthetic_data: ProcessedData):
    d = synthetic_data
    ds = EvalDataset(d, max_len=4, split="test")
    for i in range(min(20, len(ds))):
        s = ds[i]
        u = int(s["user_id"])
        hist = d.test_history(u)
        expected = hist[-4:]
        got = s["input_ids"][s["input_ids"] > 0]
        assert np.array_equal(got, expected)


def test_no_target_leak_assertion_catches_a_bad_dataset(synthetic_data: ProcessedData):
    """A dataset whose target equals a history item must fail validation."""
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in synthetic_data.dir.iterdir():
            (tmp / f.name).write_bytes(f.read_bytes())
        npz = dict(np.load(tmp / "dataset.npz"))
        # force the val target of user 0 to equal its first train item
        t = int(npz["train_len"][0])
        s = int(npz["user_offsets"][0])
        npz["val_target"][0] = npz["flat_items"][s]
        np.savez_compressed(tmp / "dataset.npz", **npz)
        with pytest.raises(AssertionError):
            ProcessedData.load(tmp)
