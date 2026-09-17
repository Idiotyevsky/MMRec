"""End-to-end proof that the training objective is the SASRec one.

On a dataset whose transitions are deterministic (``i -> i+1``), a correctly
trained model must be able to:
* drive the training loss to ~0,
* rank the true successor first,
* and -- the discriminating check -- be *better* under the position-wise
  objective than under the "broadcast the last hidden state" objective.  A model
  trained with the broadcast bug satisfies the first two but fails the third,
  because its intermediate hidden states are never asked to predict anything.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.dataset import ProcessedData
from src.data.synthetic import make_deterministic_dataset
from src.evaluation.evaluator import FullRankingEvaluator
from src.models.loss import sampled_softmax_loss
from src.training.factory import build_model
from src.training.trainer import Trainer
from src.utils.config import Config
from src.utils.seed import set_seed


@pytest.fixture(scope="module")
def deterministic_dir(tmp_path_factory):
    return make_deterministic_dataset(
        tmp_path_factory.mktemp("deterministic"), num_users=24, num_items=40, window=20, seed=0
    )


@pytest.fixture(scope="module")
def deterministic_data(deterministic_dir):
    return ProcessedData.load(deterministic_dir)


def _val_transition_is_learnable(data) -> np.ndarray:
    """Users whose val transition ``x -> x+1`` also occurs in *some* train prefix.

    The very last windows of the chain are the only ones without coverage; a
    model cannot be blamed for a transition it could never observe.
    """
    seen = set()
    for u in range(data.num_users):
        tr = data.train_items(u)
        seen |= set(zip(tr[:-1].tolist(), tr[1:].tolist()))
    out = []
    for u in range(data.num_users):
        last = int(data.train_items(u)[-1])
        out.append((last, int(data.val_target[u])) in seen)
    return np.asarray(out, dtype=bool)


def _cfg(processed_dir, name="sasrec", **model_kw) -> Config:
    model = {"name": name, "hidden_size": 32, "num_layers": 2, "num_heads": 4,
             "dropout": 0.0, "max_seq_len": 25, "zero_cold_id": True}
    model.update(model_kw)
    return Config({
        "model": model,
        "data": {"processed_dir": str(processed_dir), "feature_dir": str(processed_dir),
                 "max_sequence_length": 25},
        "loss": {"type": "sampled_softmax", "num_negatives": 32, "temperature": 1.0},
        "negative_sampling": {"mode": "uniform"},
        "training": {"batch_size": 24, "learning_rate": 0.01, "weight_decay": 0.0,
                     "max_grad_norm": 5.0, "amp": False, "epochs": 80,
                     "early_stopping_patience": 200, "scheduler": "none",
                     "monitor": "NDCG@10", "monitor_mode": "max", "seed": 0,
                     "num_workers": 0},
        "evaluation": {"full_ranking": True, "item_chunk_size": 64, "ks": [1, 5, 10, 20]},
    })


def _train(cfg, data, run_dir):
    device = torch.device("cpu")
    model = build_model(cfg, data, device=device)
    trainer = Trainer(model, cfg, data, run_dir, device)
    trainer.fit()
    return model, trainer


def _evaluate_val(trainer, model, data):
    from torch.utils.data import DataLoader

    from src.data.dataset import EvalDataset, collate_eval

    ev = FullRankingEvaluator(num_items=data.num_items, ks=(1, 5, 10, 20), item_chunk_size=64)
    loader = DataLoader(
        EvalDataset(data, trainer.max_seq_len, split="val"),
        batch_size=64, shuffle=False, collate_fn=collate_eval,
    )
    return ev.evaluate(trainer._encode_fn(), model.all_item_embeddings(), loader)


def test_fixture_transitions_are_deterministic(deterministic_data):
    data = deterministic_data
    for u in range(data.num_users):
        items = data.train_items(u)
        assert np.array_equal(items[1:], items[:-1] + 1), f"user {u} is not a chain"
        assert int(data.val_target[u]) == int(items[-1]) + 1
        assert int(data.test_target[u]) == int(items[-1]) + 2
    assert _val_transition_is_learnable(data).mean() > 0.8


def test_fixture_respects_the_real_data_protocol(deterministic_data):
    """The fixture must survive the strict invariants (no escape hatch is used)."""
    data = deterministic_data
    stats = json.loads((Path(data.dir) / "stats.json").read_text())
    assert "relaxed_invariants" not in stats
    for u in range(data.num_users):
        seg = data.full_history(u)
        assert len(set(seg.tolist())) == seg.shape[0], f"user {u} repeats an item"


@pytest.mark.parametrize("name,extra", [
    ("sasrec", {}),
    ("mm_sasrec", {"fusion": "gated", "modalities": {"id": True, "text": True, "image": True}}),
    ("mm_sasrec", {"fusion": "gated", "modalities": {"id": True, "text": True, "image": True},
                   "id_dropout_prob": 0.2}),
])
def test_model_memorises_deterministic_transitions(
    name, extra, deterministic_data, deterministic_dir, tmp_path
):
    set_seed(0)
    cfg = _cfg(deterministic_dir, name=name, **extra)
    model, trainer = _train(cfg, deterministic_data, tmp_path / name)

    losses = [h["train_loss"] for h in trainer.history]
    assert losses[-1] < 0.1, f"{name} could not fit deterministic transitions: {losses[-1]:.4f}"
    assert losses[-1] < losses[0]

    res = _evaluate_val(trainer, model, deterministic_data)
    # for users whose val transition was observable, the successor must rank first
    learnable = _val_transition_is_learnable(deterministic_data)
    rank = res.rank[learnable]
    assert (rank == 1).all(), (
        f"{name}: successor not ranked first for {(rank != 1).sum()} of {rank.size} users "
        f"(mean rank {rank.mean():.2f})"
    )


def test_training_optimises_the_position_wise_objective(deterministic_data, deterministic_dir, tmp_path):
    """The discriminator: a model trained on the right objective scores better
    under the position-wise objective than under the broadcast one."""
    set_seed(0)
    cfg = _cfg(deterministic_dir)
    model, trainer = _train(cfg, deterministic_data, tmp_path / "sasrec")
    model.eval()

    device = torch.device("cpu")
    loader = trainer._train_loader()
    with torch.no_grad():
        per_position, broadcast = [], []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target = batch["target"].to(device)
            neg, _ = model.item_embeddings(
                torch.randint(1, deterministic_data.num_items + 1, (input_ids.shape[0], 16))
            )
            pos, _ = model.item_embeddings(target)
            seq = model.encode_sequence(input_ids)
            per_position.append(sampled_softmax_loss(seq, pos, neg, target).item())
            broadcast.append(
                sampled_softmax_loss(seq[:, -1:].expand_as(seq), pos, neg, target).item()
            )

    assert np.mean(per_position) < np.mean(broadcast), (
        "the trained model is better explained by the broadcast objective "
        f"(position-wise {np.mean(per_position):.4f} vs broadcast {np.mean(broadcast):.4f})"
    )


def test_popularity_baseline_loses_to_sasrec_on_deterministic_data(
    deterministic_data, deterministic_dir, tmp_path
):
    """Sanity: the ordering popular < SASRec must hold on a learnable dataset."""
    set_seed(0)
    device = torch.device("cpu")
    ev = FullRankingEvaluator(num_items=deterministic_data.num_items, ks=(5, 10, 20),
                              item_chunk_size=64)

    pop_cfg = _cfg(deterministic_dir, name="popular")
    pop_cfg.model["modalities"] = None
    pop_model = build_model(pop_cfg, deterministic_data, device=device)
    pop_trainer = Trainer(pop_model, pop_cfg, deterministic_data, tmp_path / "pop", device)
    pop_res = ev.evaluate(pop_trainer._encode_fn(), pop_model.all_item_embeddings(),
                          pop_trainer._eval_loader("val"))

    cfg = _cfg(deterministic_dir)
    model, trainer = _train(cfg, deterministic_data, tmp_path / "sasrec")
    res = ev.evaluate(trainer._encode_fn(), model.all_item_embeddings(),
                      trainer._eval_loader("val"))

    assert pop_res.recall_at(20) < res.recall_at(20)
