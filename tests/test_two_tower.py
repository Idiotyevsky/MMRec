import numpy as np
import pytest
import torch

from src.training.factory import build_model
from src.utils.config import Config


def _cfg(path):
    return Config({"model": {"name": "two_tower", "hidden_size": 128, "max_history": 20,
        "recency_decay": .9, "dropout": 0., "modalities": {"text": True, "image": True}, "normalize_embeddings": True},
        "data": {"processed_dir": str(path), "feature_dir": str(path), "max_sequence_length": 20}})


def test_shapes_normalization_and_content_only_cold(synthetic_data, synthetic_dir):
    model = build_model(_cfg(synthetic_dir), synthetic_data).eval()
    assert not any("Embedding" in type(m).__name__ for m in model.item_tower.modules())
    with torch.no_grad():
        items = model.encode_items(torch.tensor([0, 1, 2]))
        users = model.encode_history(torch.tensor([[0, 0, 1, 2], [0, 0, 0, 3]]))
    assert items.shape == (3, 128) and users.shape == (2, 128)
    assert torch.all(items[0] == 0)
    assert torch.allclose(items[1:].norm(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.allclose(users.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_batch_and_single_history_agree_and_export_is_deterministic(synthetic_data, synthetic_dir):
    model = build_model(_cfg(synthetic_dir), synthetic_data).eval()
    x = torch.tensor([[0, 0, 1, 2], [0, 0, 3, 4]])
    with torch.no_grad():
        assert torch.allclose(model.encode_history(x)[0], model.encode_history(x[:1])[0], atol=1e-6)
        assert torch.equal(model.all_item_embeddings(), model.all_item_embeddings())


def test_missing_feature_content_is_ignored(synthetic_data, synthetic_dir):
    model = build_model(_cfg(synthetic_dir), synthetic_data).eval()
    # Explicitly mark text missing for one real item; changing its stored feature
    # must then have no effect because the availability bit masks it.
    item = torch.tensor([1])
    model.item_tower.text_available[1] = False
    with torch.no_grad():
        a = model.encode_items(item).clone(); model.item_tower.text_features[1].fill_(999); b = model.encode_items(item)
    assert torch.allclose(a, b)


def test_id_modality_is_rejected(synthetic_data, synthetic_dir):
    cfg = _cfg(synthetic_dir); cfg.model.modalities = {"id": True, "text": True}
    with pytest.raises(ValueError, match="forbids an ID embedding"):
        build_model(cfg, synthetic_data)
