"""RQVAE, Semantic ID mapping, collision handling and constrained decoding."""

import numpy as np
import pytest
import torch

from src.models.generative_rec import GenerativeRecommender
from src.models.rqvae import RQVAE
from src.models.semantic_id import BOS, PAD, PrefixConstraint, SemanticIDMapper


@pytest.fixture()
def toy_codes():
    # items 1..4 with one deliberate collision (items 2 and 4 share a SID)
    return np.array([
        [0, 1, 2],
        [3, 4, 5],
        [0, 1, 2],
        [6, 7, 8],
    ], dtype=np.int64)


def test_mapper_detects_collisions(toy_codes):
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    s = m.statistics()
    assert s["num_items"] == 4
    assert s["unique_semantic_ids"] == 3
    assert s["collision_rate"] == pytest.approx(0.25)
    assert s["num_colliding_sids"] == 1
    assert m.sid_to_items[(0, 1, 2)] == [1, 3]  # rows 0 and 2 collide
    assert m.item_to_sid[1] == (0, 1, 2)


def test_token_layout_roundtrip(toy_codes):
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    assert m.vocab_size == 3 + 3 * 16
    for item in (1, 2, 3, 4):
        toks = m.item_tokens(item)
        assert m.sid_from_tokens(toks) == m.item_to_sid[item]
    assert m.sid_from_tokens([m.token(0, 1), m.token(1, 2)]) is None  # too short


def test_prefix_constraint_only_allows_real_items(toy_codes):
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    pc = PrefixConstraint(m)
    assert set(pc.allowed([])) == {m.token(0, c) for c in (0, 3, 6)}
    first = m.token(0, 0)
    assert set(pc.allowed([first])) == {m.token(1, 1)}
    assert pc.allowed([m.token(0, 9)]) == []  # illegal prefix


def test_constrained_decoding_always_returns_valid_items(toy_codes):
    """Random logits + constraint must still yield only real Semantic IDs."""
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    torch.manual_seed(0)
    model = GenerativeRecommender(vocab_size=m.vocab_size, num_levels=3,
                                  hidden_size=32, num_layers=2, num_heads=4,
                                  dropout=0.0, max_seq_len=32)
    model.eval()
    histories = [[m.item_tokens(1), m.item_tokens(3)]]
    out = model.generate(histories, m, beam_width=4, batch_size=1)
    assert len(out) == 1
    assert out[0], "decoder produced no valid candidate at all"
    for sid, lp in out[0]:
        assert sid in m.sid_to_items
        assert np.isfinite(lp)


def test_beam_search_is_ranked_by_logprob(toy_codes):
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    torch.manual_seed(1)
    model = GenerativeRecommender(vocab_size=m.vocab_size, num_levels=3,
                                  hidden_size=32, num_layers=2, num_heads=4,
                                  dropout=0.0, max_seq_len=32)
    model.eval()
    out = model.generate([[m.item_tokens(1)]], m, beam_width=3, batch_size=1)[0]
    lps = [lp for _sid, lp in out]
    assert lps == sorted(lps, reverse=True)


def test_rqvae_codes_in_range_and_reconstruction_learns():
    torch.manual_seed(0)
    x = torch.randn(64, 12)
    model = RQVAE(input_dim=12, latent_dim=8, hidden_dim=32,
                  num_codebooks=2, codebook_size=16, commitment=0.25)
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    first = None
    for _ in range(120):
        opt.zero_grad()
        out = model(x)
        out["loss"].backward()
        opt.step()
        if first is None:
            first = float(out["reconstruction"])
    assert float(out["reconstruction"]) < first * 0.5, "RQVAE failed to learn a reconstruction"
    codes = model.encode_codes(x)
    assert codes.shape == (64, 2)
    assert (codes >= 0).all() and (codes < 16).all()


def test_rqvae_usage_tracking_and_reseed():
    torch.manual_seed(0)
    model = RQVAE(input_dim=8, latent_dim=4, hidden_dim=16,
                  num_codebooks=1, codebook_size=64)
    model.train()
    _ = model(torch.randn(32, 8))
    util_before = model.quantizer.utilization()[0]
    assert 0.0 < util_before < 1.0
    model.quantizer.usage[0, :] = 0.0  # pretend every code is dead
    pool = torch.randn(64, 4)
    n = model.quantizer.reseed_dead_codes(pool, threshold=1.0)
    assert n == 64
    assert (model.quantizer.usage == 0).all()
    # re-seeding must not touch codes that are still alive
    model.train()
    _ = model(torch.randn(32, 8))
    before = model.quantizer.codebooks[0].detach().clone()
    model.quantizer.reseed_dead_codes(torch.randn(4, 4), threshold=1.0)
    alive = (before != model.quantizer.codebooks[0]).any(dim=1)
    assert alive.any()


def test_generative_loss_ignores_pad():
    """PAD targets must be excluded from the mean, not counted as a real token."""
    import torch.nn.functional as F

    m = SemanticIDMapper(np.array([[0, 1], [2, 3]]), num_items=2, codebook_size=8)
    torch.manual_seed(0)
    model = GenerativeRecommender(vocab_size=m.vocab_size, num_levels=2,
                                  hidden_size=16, num_layers=2, num_heads=2,
                                  dropout=0.0, max_seq_len=16)
    model.eval()
    ids = torch.tensor([[BOS, 3, 4, 5]])
    tgt = torch.tensor([[3, 4, 5, PAD]])

    loss = model.loss(ids, tgt)
    logits = model.forward(ids)
    per_pos = F.cross_entropy(logits.reshape(-1, model.vocab_size), tgt.reshape(-1),
                              reduction="none", ignore_index=PAD)
    n_real = int((tgt.reshape(-1) != PAD).sum())
    assert torch.allclose(loss, per_pos.sum() / n_real, atol=1e-6)

    tgt_b = tgt.clone()
    tgt_b[0, -1] = 7  # a real token now contributes -> the loss must change
    assert not torch.allclose(loss, model.loss(ids, tgt_b))


def test_generative_model_trains_and_generates(toy_codes):
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    torch.manual_seed(0)
    model = GenerativeRecommender(vocab_size=m.vocab_size, num_levels=3,
                                  hidden_size=32, num_layers=2, num_heads=4,
                                  dropout=0.0, max_seq_len=32)
    seq = [BOS] + m.item_tokens(1) + m.item_tokens(3) + m.item_tokens(2)
    ids = torch.tensor([seq[:-1]])
    tgt = torch.tensor([seq[1:]])
    opt = torch.optim.AdamW(model.parameters(), lr=0.02)
    first = None
    for _ in range(80):
        opt.zero_grad()
        loss = model.loss(ids, tgt)
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss)
    assert float(loss) < first
    out = model.generate([[m.item_tokens(1), m.item_tokens(3)]], m, beam_width=2, batch_size=1)
    assert out[0]
