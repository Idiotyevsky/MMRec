"""Fusion correctness: gate weights sum to 1 and missing modalities get zero."""

import numpy as np
import pytest
import torch

from src.models.fusion import ConcatFusion, GatedFusion


def _inputs(batch=4, hidden=8, avail=None):
    torch.manual_seed(0)
    emb = {m: torch.randn(batch, hidden) for m in ("id", "text", "image")}
    if avail is None:
        avail = {m: torch.ones(batch, dtype=torch.bool) for m in emb}
    masks = {m: avail[m].clone() for m in emb}
    return emb, masks


def test_gated_weights_sum_to_one_when_all_available():
    emb, masks = _inputs()
    f = GatedFusion(["id", "text", "image"], hidden_size=8, dropout=0.0)
    f.eval()
    with torch.no_grad():
        _, gates = f(emb, masks)
    total = sum(gates[m] for m in gates)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)
    for m in gates:
        assert (gates[m] >= -1e-6).all()


def test_masked_modality_gets_exactly_zero_weight():
    emb, masks = _inputs()
    masks["image"][:] = False
    f = GatedFusion(["id", "text", "image"], hidden_size=8, dropout=0.0)
    f.eval()
    with torch.no_grad():
        _, gates = f(emb, masks)
    assert torch.all(gates["image"] == 0.0)
    total = gates["id"] + gates["text"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_missing_modality_does_not_change_output_of_available_ones():
    """A missing modality must be *ignored*, not treated as a zero vector."""
    emb, masks = _inputs()
    masks["image"][:] = False
    f = GatedFusion(["id", "text", "image"], hidden_size=8, dropout=0.0)
    f.eval()
    with torch.no_grad():
        out_a, _ = f(emb, masks)

    emb2 = {k: v.clone() for k, v in emb.items()}
    emb2["image"] = torch.randn_like(emb2["image"]) * 100  # junk in the missing slot
    with torch.no_grad():
        out_b, _ = f(emb2, masks)
    assert torch.allclose(out_a, out_b, atol=1e-5)


def test_all_missing_gives_zero_output_and_no_nan():
    emb, masks = _inputs()
    for m in masks:
        masks[m][:] = False
    f = GatedFusion(["id", "text", "image"], hidden_size=8, dropout=0.0)
    f.eval()
    with torch.no_grad():
        out, gates = f(emb, masks)
    assert torch.isfinite(out).all()
    assert torch.all(out == 0.0)
    for m in gates:
        assert torch.all(gates[m] == 0.0)


def test_gate_weights_are_per_item_not_global():
    """Two items with different embeddings must get different gates."""
    torch.manual_seed(0)
    emb = {
        "id": torch.tensor([[3.0, -2.0], [-2.0, 3.0]]),
        "text": torch.tensor([[-1.0, 4.0], [4.0, -1.0]]),
    }
    masks = {m: torch.ones(2, dtype=torch.bool) for m in emb}
    f = GatedFusion(["id", "text"], hidden_size=2, dropout=0.0)
    f.eval()
    with torch.no_grad():
        _, gates = f(emb, masks)
    assert not torch.allclose(gates["id"][0], gates["id"][1], atol=1e-6), (
        "gates are identical for different items -- the gate MLP is not using the input"
    )
    total = gates["id"] + gates["text"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5)


def test_concat_ignores_missing_modality_content():
    emb, masks = _inputs()
    masks["image"][:] = False
    f = ConcatFusion(["id", "text", "image"], hidden_size=8, dropout=0.0)
    f.eval()
    with torch.no_grad():
        out_a, gates_a = f(emb, masks)
    emb2 = {k: v.clone() for k, v in emb.items()}
    emb2["image"] = torch.randn_like(emb2["image"]) * 100
    with torch.no_grad():
        out_b, _ = f(emb2, masks)
    assert torch.allclose(out_a, out_b, atol=1e-5)
    assert torch.all(gates_a["image"] == 0.0)


def test_concat_missing_modality_is_distinguishable_from_zero_vector():
    """'no image' must differ from 'an image whose projection is exactly zero'."""
    torch.manual_seed(0)
    f = ConcatFusion(["id", "text"], hidden_size=4, dropout=0.0)
    f.eval()
    id_e = torch.randn(1, 4)
    zero_img = torch.zeros(1, 4)
    real_img = torch.randn(1, 4)

    with torch.no_grad():
        missing, _ = f({"id": id_e, "text": zero_img}, {"id": torch.ones(1, dtype=torch.bool),
                                                        "text": torch.zeros(1, dtype=torch.bool)})
        present, _ = f({"id": id_e, "text": zero_img}, {"id": torch.ones(1, dtype=torch.bool),
                                                        "text": torch.ones(1, dtype=torch.bool)})
        other, _ = f({"id": id_e, "text": real_img}, {"id": torch.ones(1, dtype=torch.bool),
                                                      "text": torch.ones(1, dtype=torch.bool)})
    assert not torch.allclose(missing, present, atol=1e-6)
    assert not torch.allclose(present, other, atol=1e-6)


def test_gated_gradient_flows_to_available_modalities():
    emb, masks = _inputs()
    for v in emb.values():
        v.requires_grad_(True)
    f = GatedFusion(["id", "text", "image"], hidden_size=8, dropout=0.0)
    out, _ = f(emb, masks)
    out.sum().backward()
    for m in emb:
        assert emb[m].grad is not None and torch.isfinite(emb[m].grad).all()


@pytest.mark.parametrize("kind", ["concat", "gated"])
def test_unknown_fusion_rejected(kind):
    from src.models.fusion import build_fusion

    if kind in ("concat", "gated"):
        build_fusion(kind, ["id", "text"], 8, 0.0)
    with pytest.raises(ValueError):
        build_fusion("nonsense", ["id"], 8, 0.0)
