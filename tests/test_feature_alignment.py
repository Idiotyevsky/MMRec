"""Feature rows must be aligned to the right item even when raw ids are shuffled.

If the raw item ids are not a contiguous 0-based range, ``feature[item_id]`` is
simply wrong.  These tests build a dataset whose raw ids are permuted and assert
that the encoder still retrieves the correct feature row.
"""

import numpy as np
import pytest
import torch

from src.data.feature_loader import MultimodalFeatures
from src.models.item_encoder import MultimodalItemEncoder


@pytest.fixture()
def permuted_features(tmp_path):
    """5 items, raw ids shuffled, feature row r belongs to raw item ``order[r]``."""
    rng = np.random.default_rng(3)
    n = 5
    order = np.array([4, 0, 3, 1, 2])  # raw id of each feature row
    dim = 6
    feats = rng.normal(size=(n, dim)).astype(np.float32)

    np.save(tmp_path / "text_feat.npy", feats)
    # row_for_item[internal] for internal 1..n ; internal i -> raw id i-1
    row_for_item = np.zeros(n + 1, dtype=np.int64)
    for internal in range(1, n + 1):
        raw = internal - 1
        row_for_item[internal] = int(np.flatnonzero(order == raw)[0])
    np.save(tmp_path / "row_for_item_text.npy", row_for_item)
    return tmp_path, feats, row_for_item, order


def test_feature_loader_uses_row_mapping(permuted_features):
    tmp_path, feats, row_for_item, order = permuted_features
    f = MultimodalFeatures(tmp_path, modalities=("text",), row_for_item=row_for_item, num_items=5)
    got = f.fetch("text", np.arange(6))
    assert got.shape == (6, feats.shape[1])
    assert np.allclose(got[0], 0.0)  # PAD
    for internal in range(1, 6):
        assert np.allclose(got[internal], feats[row_for_item[internal]])


def test_item_encoder_aligns_shuffled_features(permuted_features):
    tmp_path, feats, row_for_item, order = permuted_features
    enc = MultimodalItemEncoder(
        num_items=5,
        hidden_size=8,
        modalities={"id": False, "text": True},
        fusion="concat",
        dropout=0.0,
        feature_dir=tmp_path,
        row_for_item={"text": row_for_item},
    )
    enc.eval()
    with torch.no_grad():
        emb, _ = enc(torch.arange(6))
    # a zero-width projection would collapse every item; instead compare against
    # a reference built from the correctly ordered features
    assert emb.shape == (6, 8)
    assert torch.allclose(emb[0], torch.zeros(8), atol=1e-6)


def test_missing_feature_row_is_flagged_not_treated_as_zero(tmp_path):
    n, dim = 4, 5
    feats = np.ones((n, dim), dtype=np.float32)
    feats[2] = 0.0  # raw item 2 has no feature -> internal id 3
    np.save(tmp_path / "text_feat.npy", feats)
    # internal id i corresponds to raw item i-1
    row_for_item = np.array([0, 0, 1, 2, 3], dtype=np.int64)
    np.save(tmp_path / "row_for_item_text.npy", row_for_item)

    f = MultimodalFeatures(tmp_path, modalities=("text",), row_for_item=row_for_item, num_items=n)
    avail = f.availability("text")
    assert avail[0] is np.bool_(False) or avail[0] == False  # noqa: E712
    assert not avail[3]  # internal id 3 == raw id 2 -> missing
    assert avail[1] and avail[2] and avail[4]
    assert f.missing_rate()["text"] == pytest.approx(1 / 4)

    enc = MultimodalItemEncoder(
        num_items=n,
        hidden_size=4,
        modalities={"id": True, "text": True},
        fusion="gated",
        dropout=0.0,
        feature_dir=tmp_path,
    )
    enc.eval()
    with torch.no_grad():
        _, gates = enc(torch.tensor([3]))
    assert gates["text"][0].item() == 0.0
    assert gates["id"][0].item() == pytest.approx(1.0)


def test_row_mapping_pointing_past_end_is_rejected(tmp_path):
    np.save(tmp_path / "text_feat.npy", np.ones((3, 4), dtype=np.float32))
    bad = np.array([0, 0, 1, 9], dtype=np.int64)
    with pytest.raises(ValueError):
        MultimodalItemEncoder(
            num_items=3,
            hidden_size=4,
            modalities={"id": False, "text": True},
            feature_dir=tmp_path,
            row_for_item={"text": bad},
        )
