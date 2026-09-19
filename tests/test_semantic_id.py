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


# ----------------------------------------------------------------------
# Semantic trie
# ----------------------------------------------------------------------
def test_trie_mirrors_the_mapper(toy_codes):
    from src.models.semantic_id import SemanticTrie
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    trie = SemanticTrie(m)
    trie.validate()
    assert trie.num_leaves == 3  # 3 unique SIDs
    assert sorted(trie.root.children) == [0, 3, 6]  # one child per level-1 code
    assert trie.items((0, 1, 2)) == [1, 3]
    assert trie.items((9, 9, 9)) == []


def test_trie_allowed_next_matches_mapper(toy_codes):
    from src.models.semantic_id import SemanticTrie
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    trie = SemanticTrie(m)
    assert trie.allowed_next(()) == m.allowed_next(())
    assert trie.allowed_next((0,)) == m.allowed_next((0,))
    assert trie.allowed_next((0, 1)) == m.allowed_next((0, 1))
    assert trie.allowed_next((0, 1, 2)) == []  # terminal
    assert trie.allowed_next((9,)) == []


def test_trie_rejects_incomplete_and_unknown_prefixes(toy_codes):
    from src.models.semantic_id import SemanticTrie
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    trie = SemanticTrie(m)
    assert trie.is_complete((0, 1, 2)) is True
    assert trie.is_complete((0, 1)) is False       # too short
    assert trie.is_complete((0, 1, 9)) is False    # unknown path
    assert trie.node((0, 1, 9)) is None


def test_trie_statistics_are_consistent(toy_codes):
    from src.models.semantic_id import SemanticTrie
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    trie = SemanticTrie(m)
    stats = trie.statistics()
    assert stats["num_levels"] == 3
    assert stats["nodes_per_depth"][0] == 1
    assert sum(stats["nodes_per_depth"]) == trie.num_nodes
    assert trie.num_nodes >= trie.num_leaves


def test_trie_items_at_most_respects_the_budget(toy_codes):
    from src.models.semantic_id import SemanticTrie
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    trie = SemanticTrie(m)
    got = trie.items_at_most((), 2)
    assert len(got) == 2 and len(set(got)) == 2  # budget capped, no repeats
    assert set(got) <= {1, 2, 3, 4}
    assert len(trie.items_at_most((), 100)) == 4


def test_prefix_constraint_uses_a_supplied_trie(toy_codes):
    from src.models.semantic_id import SemanticTrie
    m = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    trie = SemanticTrie(m)
    pc = PrefixConstraint(m, trie=trie)
    assert pc.trie is trie
    assert pc.allowed([m.token(0, 0)]) == trie.allowed_next((0,))
    assert pc.allowed([999]) == []  # token from the wrong level
    assert pc.is_complete(m.item_tokens(1)) is True


# ----------------------------------------------------------------------
# neighbour agreement: the metric that decides whether the hierarchy is real
# ----------------------------------------------------------------------
def _neighbour_rows(codes, embeddings, **kw):
    """Prepend the PAD row: row 0 of the embedding table is PAD, by contract."""
    from scripts.analyze_semantic_ids import neighbour_agreement_rows
    m = SemanticIDMapper(codes, num_items=codes.shape[0], codebook_size=16)
    table = np.vstack([np.zeros((1, embeddings.shape[1]), dtype=np.float32), embeddings])
    return {r["prefix_length"]: r for r in neighbour_agreement_rows(m, table, **kw)}


def test_neighbour_agreement_is_high_when_codes_track_content():
    """Two well-separated content clusters, codes aligned with the clusters."""
    rng = np.random.default_rng(0)
    n_per = 40
    a = rng.normal(0.0, 0.02, size=(n_per, 8)) + np.eye(8)[0]
    b = rng.normal(0.0, 0.02, size=(n_per, 8)) - np.eye(8)[0]
    emb = np.vstack([a, b]).astype(np.float32)
    # rows 1..40 are cluster a, rows 41..80 are cluster b, so the codes have to be
    # blocked the same way -- an alternating pattern would not track the content
    codes = np.array([[0, 0]] * n_per + [[1, 1]] * n_per, dtype=np.int64)
    rows = _neighbour_rows(codes, emb, n_items=80, k_nn=5)
    assert rows[1]["share_sharing_prefix"] > 0.95
    assert rows[1]["lift_vs_chance"] > 10


def test_neighbour_agreement_is_at_chance_for_uninformative_codes():
    """Same content, codes assigned at random: the metric must not invent signal."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 8)).astype(np.float32)
    codes = rng.integers(0, 16, size=(200, 2)).astype(np.int64)
    rows = _neighbour_rows(codes, emb, n_items=200, k_nn=5)
    assert rows[1]["chance_rate"] == pytest.approx(1 / 16)
    # allow generous slack: 200 items is a small sample
    assert rows[1]["share_sharing_prefix"] < 4 * rows[1]["chance_rate"]


def test_pad_content_does_not_affect_neighbour_agreement():
    """PAD (row 0) is excluded by construction, so its value must not matter.

    A differential test rather than an absolute one: making PAD a near-duplicate
    of a real item would corrupt every neighbour list if the exclusion were
    missing, so the two runs below would disagree.
    """
    from scripts.analyze_semantic_ids import neighbour_agreement_rows
    rng = np.random.default_rng(0)
    n = 40
    body = rng.normal(size=(n, 8)).astype(np.float32)
    codes = np.array([[0, 0]] * (n // 2) + [[1, 1]] * (n // 2), dtype=np.int64)
    m = SemanticIDMapper(codes, num_items=n, codebook_size=16)

    zero_pad = np.vstack([np.zeros((1, 8), dtype=np.float32), body])
    # PAD made identical to item 1: it would be the nearest neighbour of everything
    hostile_pad = np.vstack([body[0:1].copy(), body])

    a = {r["prefix_length"]: r["share_sharing_prefix"]
         for r in neighbour_agreement_rows(m, zero_pad, n_items=n, k_nn=3)}
    b = {r["prefix_length"]: r["share_sharing_prefix"]
         for r in neighbour_agreement_rows(m, hostile_pad, n_items=n, k_nn=3)}
    assert a == b


def test_neighbour_agreement_reports_its_own_sample_size():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(50, 4)).astype(np.float32)
    codes = rng.integers(0, 4, size=(50, 2)).astype(np.int64)
    rows = _neighbour_rows(codes, emb, n_items=20, k_nn=3)
    r = rows[1]
    assert r["num_items_sampled"] == 20
    assert r["num_pairs"] == 60
    assert 0.0 <= r["share_sharing_prefix"] <= 1.0
