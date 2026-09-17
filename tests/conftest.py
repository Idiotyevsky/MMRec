import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory) -> Path:
    from src.data.synthetic import make_synthetic_dataset

    d = tmp_path_factory.mktemp("synthetic")
    return make_synthetic_dataset(d, num_users=60, num_items=40, seed=0)


@pytest.fixture(scope="session")
def synthetic_cold_dir(tmp_path_factory) -> Path:
    from src.data.synthetic import make_synthetic_dataset

    d = tmp_path_factory.mktemp("synthetic_cold")
    return make_synthetic_dataset(d, num_users=80, num_items=50, cold_ratio=0.15, seed=1)


@pytest.fixture(scope="session")
def synthetic_data(synthetic_dir):
    from src.data.dataset import ProcessedData

    return ProcessedData.load(synthetic_dir)


@pytest.fixture(scope="session")
def synthetic_recall_artifacts(synthetic_dir, tmp_path_factory):
    """ItemCF + content-embedding artifacts for the synthetic dataset."""
    import numpy as np

    from src.data.dataset import ProcessedData
    from src.recall.itemcf import build_itemcf_index
    from src.recall.semantic import build_content_embeddings

    data = ProcessedData.load(synthetic_dir)
    index = build_itemcf_index(
        data.flat_items, data.user_offsets, data.train_len, data.num_items,
        top_m=20, chunk_size=64, verbose=False,
    )
    out = tmp_path_factory.mktemp("recall_artifacts")
    np.savez_compressed(out / "itemcf.npz", **index)
    emb = build_content_embeddings(
        feature_dir=synthetic_dir, row_for_item_dir=synthetic_dir,
        num_items=data.num_items, modalities=("text", "image"),
    )
    np.save(out / "content_embeddings.npy", emb)
    return {"itemcf": out / "itemcf.npz", "content": out / "content_embeddings.npy"}


@pytest.fixture(scope="session")
def synthetic_run(synthetic_dir, tmp_path_factory):
    """A minimal but real finished run directory (SASRec, 1 epoch).

    Lets the serving tests exercise the true checkpoint -> config -> model path
    instead of a stub.
    """
    import torch
    import yaml

    from src.data.dataset import ProcessedData
    from src.training.factory import build_model
    from src.training.trainer import Trainer
    from src.utils.config import Config
    from src.utils.seed import set_seed

    set_seed(0)
    data = ProcessedData.load(synthetic_dir)
    cfg = Config({
        "model": {"name": "sasrec", "hidden_size": 32, "num_layers": 2, "num_heads": 4,
                  "dropout": 0.0, "max_seq_len": 20, "zero_cold_id": True},
        "data": {"processed_dir": str(synthetic_dir), "feature_dir": str(synthetic_dir),
                 "max_sequence_length": 20},
        "loss": {"num_negatives": 16, "temperature": 1.0},
        "negative_sampling": {"mode": "uniform"},
        "training": {"batch_size": 32, "learning_rate": 0.005, "weight_decay": 0.0,
                     "max_grad_norm": 5.0, "amp": False, "epochs": 2,
                     "early_stopping_patience": 5, "scheduler": "none",
                     "monitor": "NDCG@10", "monitor_mode": "max", "seed": 0, "num_workers": 0},
        "evaluation": {"full_ranking": True, "item_chunk_size": 256, "ks": [5, 10, 20]},
    })
    run = tmp_path_factory.mktemp("runs") / "sasrec_test_run"
    run.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, data, device=torch.device("cpu"))
    trainer = Trainer(model, cfg, data, run, torch.device("cpu"))
    trainer.fit()
    (run / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    (run / "metrics.json").write_text(
        json.dumps({"run_id": run.name, "model": "sasrec", "seed": 0,
                    "val": {"Recall@20": 0.0}, "test": {"Recall@20": 0.0}}),
        encoding="utf-8",
    )
    return run
