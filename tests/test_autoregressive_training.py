"""The training objective must be *position-wise* autoregressive.

The bug these tests exist for: ``model.encode(input_ids)`` returns ``(B, H)``
(the last position).  If that tensor is fed to the loss, ``h_L`` is broadcast and
scored against *every* target of the sequence::

    h_last -> i2, h_last -> i3, h_last -> i4, h_last -> i5   (wrong)

instead of the SASRec objective::

    h1 -> i2, h2 -> i3, h3 -> i4, h4 -> i5                   (correct)

The old objective is much easier (the model only has to predict the *set* of
next items, not each one from the right state) and it silently changes what the
whole project measures.  The causal-mask tests in ``test_attention_mask.py`` do
**not** catch it, because the mask is correct -- the mistake is in which hidden
state is paired with which target.

Covered here:
* A: every position contributes its own hidden state, end to end through Trainer
* B: a changed future suffix cannot alter an earlier position's scores
* C: correct alignment beats the "repeat the last state" objective
* D: the loss mask keys on the target, not on the input
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.loss import sampled_softmax_logits, sampled_softmax_loss
from src.models.sasrec import SASRec
from src.training.trainer import Trainer
from src.utils.config import Config
from src.utils.seed import set_seed

H = 8
L = 5
PAD = 0


# ----------------------------------------------------------------------
# A. every position is scored with its own hidden state
# ----------------------------------------------------------------------
def test_loss_rejects_a_broadcast_user_vector():
    """``(B, H)`` is exactly the shape that produced the bug -- refuse it."""
    seq = torch.randn(2, H)  # the old ``model.encode(input_ids)`` output
    pos = torch.randn(2, L, H)
    neg = torch.randn(2, 4, H)
    target = torch.ones(2, L, dtype=torch.long)
    with pytest.raises(ValueError):
        sampled_softmax_loss(seq, pos, neg, target)
    with pytest.raises(ValueError):
        sampled_softmax_logits(seq, pos, neg)


def test_early_positions_change_the_loss():
    torch.manual_seed(0)
    seq = torch.randn(3, L, H)
    pos = torch.randn(3, L, H)
    neg = torch.randn(3, 6, H)
    target = torch.ones(3, L, dtype=torch.long)

    base = sampled_softmax_loss(seq, pos, neg, target)
    moved = seq.clone()
    moved[:, 0] += 3.0  # touch position 0 only
    assert not torch.allclose(base, sampled_softmax_loss(moved, pos, neg, target)), (
        "position 0 does not influence the loss"
    )

    # ... and the last state alone must NOT reproduce the correct loss
    last_only = seq[:, -1:].expand_as(seq)
    assert not torch.allclose(
        base, sampled_softmax_loss(last_only, pos, neg, target), atol=1e-4
    ), "the loss is invariant to using only the last position (broadcast bug)"


def test_position_zero_is_used_with_identical_targets():
    """With identical positives/negatives per position, only h varies.

    Scores must differ across positions; if a single state were broadcast, every
    position would produce the same cross-entropy.
    """
    torch.manual_seed(1)
    seq = torch.randn(1, L, H)
    pos = torch.randn(1, 1, H).expand(1, L, H).contiguous()
    neg = torch.randn(1, 3, H)

    logits = sampled_softmax_logits(seq, pos, neg)
    per_position = logits[..., 0]  # positive score of each position
    assert not torch.allclose(per_position, per_position[:, :1].expand_as(per_position))


class _Spy(nn.Module):
    """Wraps a model and records the shapes handed to the loss path."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.seq_shapes: list[tuple] = []

    def encode_sequence(self, input_ids):
        h = self.inner.encode_sequence(input_ids)
        self.seq_shapes.append(tuple(h.shape))
        return h

    def encode(self, input_ids):
        return self.inner.encode(input_ids)

    def item_embeddings(self, item_ids, generator=None):
        return self.inner.item_embeddings(item_ids, generator)

    def all_item_embeddings(self):
        return self.inner.all_item_embeddings()

    def num_parameters(self, *a, **k):
        return self.inner.num_parameters(*a, **k)


def _cfg(processed_dir, **model_kw) -> Config:
    model = {
        "name": "sasrec",
        "hidden_size": 16,
        "num_layers": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "max_seq_len": 12,
        "zero_cold_id": True,
    }
    model.update(model_kw)
    return Config({
        "model": model,
        "data": {"processed_dir": str(processed_dir), "feature_dir": str(processed_dir),
                 "max_sequence_length": 12},
        "loss": {"type": "sampled_softmax", "num_negatives": 8, "temperature": 1.0},
        "negative_sampling": {"mode": "uniform"},
        "training": {"batch_size": 16, "learning_rate": 0.01, "weight_decay": 0.0,
                     "max_grad_norm": 5.0, "amp": False, "epochs": 1,
                     "early_stopping_patience": 10, "scheduler": "none",
                     "monitor": "NDCG@10", "monitor_mode": "max", "seed": 0,
                     "num_workers": 0},
        "evaluation": {"full_ranking": True, "item_chunk_size": 512, "ks": [5, 10, 20]},
    })


def test_trainer_scores_every_position_with_its_own_state(synthetic_data, synthetic_dir, tmp_path):
    """End-to-end: the tensor handed to the loss must be ``(B, L, H)``."""
    set_seed(0)
    cfg = _cfg(synthetic_dir)
    device = torch.device("cpu")
    model = SASRec(num_items=synthetic_data.num_items, hidden_size=16, num_layers=1,
                   num_heads=2, dropout=0.0, max_seq_len=cfg.model.max_seq_len)
    spy = _Spy(model)
    trainer = Trainer(spy, cfg, synthetic_data, tmp_path / "run", device)
    trainer.train_epoch(0, trainer._train_loader())

    assert spy.seq_shapes, "encode_sequence was never called during training"
    for shape in spy.seq_shapes:
        assert len(shape) == 3, f"loss received a {shape} tensor, expected (B, L, H)"
        assert shape[1] == cfg.model.max_seq_len
        assert shape[2] == cfg.model.hidden_size


# ----------------------------------------------------------------------
# B. a future token cannot change an earlier position's prediction
# ----------------------------------------------------------------------
def _model(seed: int = 0) -> SASRec:
    torch.manual_seed(seed)
    m = SASRec(num_items=64, hidden_size=16, num_layers=2, num_heads=2,
               dropout=0.0, max_seq_len=12)
    m.eval()
    return m


def test_changing_the_future_suffix_leaves_earlier_scores_identical():
    """Full chain: model -> sequence representation -> loss scoring.

    Positions 0 and 1 must be scored identically before and after rewriting the
    suffix, because their hidden states cannot see it.
    """
    torch.manual_seed(3)
    m = _model()
    neg = torch.randn(1, 7, 16)

    ids_a = torch.tensor([[0, 0, 5, 9, 12, 18]])
    ids_b = ids_a.clone()
    ids_b[0, 3] = 40  # rewrite everything from position 3 onward
    ids_b[0, 4] = 41
    ids_b[0, 5] = 42

    # score the *same* reference item at every position in both runs
    ref = m.item_embeddings(torch.tensor([[33]]))[0].expand(1, ids_a.shape[1], 16).contiguous()

    with torch.no_grad():
        logits_a = sampled_softmax_logits(m.encode_sequence(ids_a), ref, neg)
        logits_b = sampled_softmax_logits(m.encode_sequence(ids_b), ref, neg)

    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-6), (
        "a later token changed the score of an earlier position"
    )
    assert not torch.allclose(logits_a[:, 3:], logits_b[:, 3:], atol=1e-6), (
        "the test is vacuous: rewriting the suffix changed nothing at all"
    )


# ----------------------------------------------------------------------
# C. alignment: h_l must be paired with target_l
# ----------------------------------------------------------------------
def test_correct_alignment_is_easier_than_repeating_the_last_state():
    """Handcrafted: h1 knows i2, h2 knows i3, h3 knows i4 -- but not each other."""
    torch.manual_seed(0)
    d = 32
    # near-orthogonal item embeddings
    e_pos = torch.nn.functional.normalize(torch.randn(4, d), dim=-1)
    neg = torch.nn.functional.normalize(torch.randn(1, 64, d), dim=-1)

    # h_l = e_target_l  (perfect per-position prediction), L = 3
    seq = e_pos[1:].unsqueeze(0)                      # (1, 3, d): predicts i2,i3,i4
    pos = e_pos[1:].unsqueeze(0)
    target = torch.ones(1, 3, dtype=torch.long)       # all three positions supervised

    correct = sampled_softmax_loss(seq, pos, neg, target)
    broadcast = sampled_softmax_loss(seq[:, -1:].expand_as(seq), pos, neg, target)

    assert correct < broadcast, (
        "the position-wise objective should be easier than broadcasting the "
        f"last state (correct={correct:.4f}, broadcast={broadcast:.4f})"
    )
    # h3 matches i4 but must be a poor predictor of i3 and i2
    wrong = pos.clone()
    wrong[0, 0] = e_pos[3]  # swap: score i4 at the position whose target is i2
    assert sampled_softmax_loss(seq, wrong, neg, target) > correct


def test_teacher_forcing_alignment_matches_the_dataset_shift(synthetic_data):
    """The Trainer's targets must be exactly ``items[1:]`` -- i.e. target_l follows input_l."""
    from src.data.dataset import TrainDataset

    ds = TrainDataset(synthetic_data, max_len=12)
    for idx in range(min(8, len(ds))):
        sample = ds[idx]
        u = sample["user_id"]
        items = synthetic_data.train_items(u)
        inp, tgt = sample["input_ids"], sample["target"]
        valid = tgt != PAD
        assert np.array_equal(inp[valid], items[:-1][-valid.sum():]), f"input misaligned for user {u}"
        assert np.array_equal(tgt[valid], items[1:][-valid.sum():]), f"target misaligned for user {u}"


# ----------------------------------------------------------------------
# D. the loss mask is defined by the target
# ----------------------------------------------------------------------
def test_padded_targets_are_excluded_from_the_loss():
    torch.manual_seed(0)
    seq = torch.randn(1, L, H)
    pos = torch.randn(1, L, H)
    neg = torch.randn(1, 4, H)

    full = torch.ones(1, L, dtype=torch.long)
    none = torch.zeros(1, L, dtype=torch.long)
    assert sampled_softmax_loss(seq, pos, neg, none) == 0.0
    assert sampled_softmax_loss(seq, pos, neg, full) > 0.0

    # masking positions 0 and 1 must equal the mean over positions 2..4,
    # not the mean over all five
    part = full.clone()
    part[0, :2] = 0
    manual = sampled_softmax_loss(seq[:, 2:], pos[:, 2:], neg, full[:, 2:])
    assert torch.allclose(sampled_softmax_loss(seq, pos, neg, part), manual, atol=1e-6)
