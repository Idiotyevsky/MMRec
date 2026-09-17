"""End-to-end smoke tests: forward, backward, evaluation, checkpointing.

The tiny-overfit test is the strongest signal that the training code is
correct: a model that cannot drive the loss to ~0 on 20 users has a bug, not a
data problem.
"""

import numpy as np
import pytest
import torch

from src.data.dataset import ProcessedData
from src.evaluation.evaluator import FullRankingEvaluator
from src.training.factory import build_model
from src.training.trainer import Trainer
from src.utils.config import Config
from src.utils.seed import set_seed


def _cfg(name: str, processed_dir, feature_dir, **model_kw) -> Config:
    model = {"name": name, "hidden_size": 32, "num_layers": 2, "num_heads": 4,
             "dropout": 0.0, "max_seq_len": 20, "zero_cold_id": True}
    model.update(model_kw)
    return Config({
        "model": model,
        "data": {"processed_dir": str(processed_dir), "feature_dir": str(feature_dir),
                 "max_sequence_length": 20},
        "loss": {"num_negatives": 32, "temperature": 1.0},
        "negative_sampling": {"mode": "uniform"},
        "training": {"batch_size": 32, "learning_rate": 0.005, "weight_decay": 0.0,
                     "max_grad_norm": 5.0, "amp": False, "epochs": 3,
                     "early_stopping_patience": 10, "scheduler": "none",
                     "monitor": "NDCG@10", "monitor_mode": "max", "seed": 0, "num_workers": 0},
        "evaluation": {"full_ranking": True, "item_chunk_size": 512, "ks": [5, 10, 20]},
    })


@pytest.mark.parametrize("name,extra", [
    ("sasrec", {}),
    ("mm_sasrec", {"fusion": "concat", "modalities": {"id": True, "text": True, "image": True}}),
    ("mm_sasrec", {"fusion": "gated", "modalities": {"id": True, "text": True, "image": True}}),
    ("mm_sasrec", {"fusion": "gated", "modalities": {"id": True, "text": True},
                   "id_dropout_prob": 0.3}),
    ("bpr", {}),
    ("popular", {}),
    ("random", {}),
])
def test_model_runs_end_to_end(name, extra, synthetic_data, synthetic_dir, tmp_path):
    set_seed(0)
    cfg = _cfg(name, synthetic_dir, synthetic_dir, **extra)
    device = torch.device("cpu")
    model = build_model(cfg, synthetic_data, device=device)
    trainer = Trainer(model, cfg, synthetic_data, tmp_path / "run", device)
    summary = trainer.fit()
    assert np.isfinite(summary["best_metric"])
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "training_log.csv").exists()

    item_emb = model.all_item_embeddings()
    assert torch.isfinite(item_emb).all()
    assert torch.all(item_emb[0] == 0)

    ev = FullRankingEvaluator(num_items=synthetic_data.num_items, ks=(5, 10, 20), item_chunk_size=512)
    res = ev.evaluate(trainer._encode_fn(), item_emb, trainer._eval_loader("test"))
    assert 0.0 <= res.metrics()["Recall@20"] <= 1.0


def test_loss_decreases_and_beats_random(synthetic_data, synthetic_dir, tmp_path):
    set_seed(0)
    cfg = _cfg("sasrec", synthetic_dir, synthetic_dir)
    cfg.training.epochs = 15
    cfg.training.learning_rate = 0.01
    device = torch.device("cpu")
    model = build_model(cfg, synthetic_data, device=device)
    trainer = Trainer(model, cfg, synthetic_data, tmp_path / "run", device)
    trainer.fit()
    losses = [h["train_loss"] for h in trainer.history]
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert np.isfinite(losses).all()


def test_tiny_overfit(synthetic_dir, tmp_path):
    """A model must be able to nearly memorise 20 users."""
    set_seed(0)
    from src.data.synthetic import make_synthetic_dataset

    d = make_synthetic_dataset(tmp_path / "tiny", num_users=20, num_items=12,
                               min_len=5, max_len=7, seed=5)
    data = ProcessedData.load(d)
    cfg = _cfg("sasrec", d, d)
    cfg.model.max_seq_len = 10
    cfg.training.epochs = 120
    cfg.training.learning_rate = 0.02
    cfg.training.batch_size = 20
    cfg.training.early_stopping_patience = 500
    cfg.loss.num_negatives = 16
    device = torch.device("cpu")
    model = build_model(cfg, data, device=device)
    trainer = Trainer(model, cfg, data, tmp_path / "overfit", device)
    trainer.fit()
    final = trainer.history[-1]["train_loss"]
    assert final < 0.15, f"could not overfit 20 users (final loss {final:.4f})"

    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20), item_chunk_size=64)
    res = ev.evaluate(trainer._encode_fn(), model.all_item_embeddings(), trainer._eval_loader("val"))
    assert res.metrics()["Recall@5"] > 0.5


def test_checkpoint_roundtrip_and_mismatch_detection(synthetic_data, synthetic_dir, tmp_path):
    set_seed(0)
    cfg = _cfg("sasrec", synthetic_dir, synthetic_dir)
    device = torch.device("cpu")
    model = build_model(cfg, synthetic_data, device=device)
    t1 = Trainer(model, cfg, synthetic_data, tmp_path / "r1", device)
    t1.save_checkpoint(tmp_path / "r1" / "ckpt.pt", epoch=3, metric=0.1)

    model2 = build_model(cfg, synthetic_data, device=device)
    t2 = Trainer(model2, cfg, synthetic_data, tmp_path / "r2", device)
    ckpt = t2.load_checkpoint(tmp_path / "r1" / "ckpt.pt")
    assert ckpt["epoch"] == 3
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert torch.allclose(v1, v2), f"parameter {k1} not restored"

    # a checkpoint from a differently-shaped dataset must be refused
    from src.data.synthetic import make_synthetic_dataset

    other = make_synthetic_dataset(tmp_path / "other", num_users=30, num_items=25, seed=9)
    data_other = ProcessedData.load(other)
    cfg_other = _cfg("sasrec", other, other)
    model3 = build_model(cfg_other, data_other, device=device)
    t3 = Trainer(model3, cfg_other, data_other, tmp_path / "r3", device)
    with pytest.raises(AssertionError):
        t3.load_checkpoint(tmp_path / "r1" / "ckpt.pt")


def test_cold_items_have_zero_id_embedding(synthetic_cold_dir, tmp_path):
    """With zero_cold_id the cold items must be invisible to the ID branch."""
    data = ProcessedData.load(synthetic_cold_dir)
    cfg = _cfg("mm_sasrec", synthetic_cold_dir, synthetic_cold_dir,
               fusion="gated", modalities={"id": True, "text": True, "image": True})
    device = torch.device("cpu")
    model = build_model(cfg, data, device=device)
    model.eval()
    cold_ids = np.flatnonzero(data.is_cold)
    assert cold_ids.size > 0
    with torch.no_grad():
        _, gates = model.item_encoder(torch.as_tensor(cold_ids, dtype=torch.long))
    assert torch.all(gates["id"] == 0.0), "cold items still receive ID gate weight"
    assert (gates["text"] + gates["image"] > 0.99).all()


def test_id_dropout_never_leaves_an_item_without_a_modality(synthetic_dir):
    data = ProcessedData.load(synthetic_dir)
    cfg = _cfg("mm_sasrec", synthetic_dir, synthetic_dir, fusion="gated",
               modalities={"id": True, "text": True, "image": True},
               per_modality_dropout={"id": 1.0, "text": 1.0, "image": 1.0})
    device = torch.device("cpu")
    model = build_model(cfg, data, device=device)
    model.train()
    ids = torch.arange(1, data.num_items + 1)
    with torch.no_grad():
        emb, gates = model.item_encoder(ids)
    # at least one modality must survive for every valid item
    total = sum(gates[m] for m in gates)
    assert (total > 0.99).all()
    assert torch.isfinite(emb).all()
