"""SASRec must be causal and must ignore padding.

These are the two mistakes that silently destroy sequential recommenders, so
they get explicit tests rather than a comment.
"""

import numpy as np
import torch

from src.models.sasrec import SASRec, SequenceEncoder


def _model(hidden=16, layers=2, heads=4, max_len=10, seed=0) -> SASRec:
    torch.manual_seed(seed)
    m = SASRec(
        num_items=50,
        hidden_size=hidden,
        num_layers=layers,
        num_heads=heads,
        dropout=0.0,
        max_seq_len=max_len,
    )
    m.eval()
    return m


def test_causal_mask_matrix_shape():
    mask = SequenceEncoder.causal_mask(4, torch.device("cpu"))
    assert mask.shape == (4, 4)
    assert mask.dtype == torch.bool
    for i in range(4):
        for j in range(4):
            assert bool(mask[i, j]) == (j > i)


def test_future_tokens_cannot_affect_past_representations():
    m = _model()
    ids = torch.tensor([[3, 7, 11, 13, 17]])
    with torch.no_grad():
        emb = m.item_embeddings(ids)[0]
        h1 = m.encoder(emb, ids, return_sequence=True)

        ids2 = ids.clone()
        ids2[0, 3] = 21  # change the token at position 3
        ids2[0, 4] = 23  # and at position 4
        emb2 = m.item_embeddings(ids2)[0]
        h2 = m.encoder(emb2, ids2, return_sequence=True)

    # positions 0..2 must be bit-identical
    assert torch.allclose(h1[:, :3], h2[:, :3], atol=1e-6), "future tokens leaked into the past"
    # the changed positions themselves must differ
    assert not torch.allclose(h1[:, 3], h2[:, 3], atol=1e-6)


def test_self_attention_still_uses_the_current_token():
    m = _model()
    ids = torch.tensor([[3, 7, 11, 13, 17]])
    with torch.no_grad():
        emb = m.item_embeddings(ids)[0]
        h1 = m.encoder(emb, ids, return_sequence=True)
        ids2 = ids.clone()
        ids2[0, 2] = 29
        h2 = m.encoder(m.item_embeddings(ids2)[0], ids2, return_sequence=True)
    assert not torch.allclose(h1[:, 2], h2[:, 2], atol=1e-6)


def test_last_position_representation_is_the_user_vector():
    m = _model()
    ids = torch.tensor([[0, 0, 5, 9, 12]])
    with torch.no_grad():
        emb = m.item_embeddings(ids)[0]
        full = m.encoder(emb, ids, return_sequence=True)
        last = m.encoder(emb, ids)
    assert torch.allclose(full[:, -1], last, atol=1e-6)


def test_padding_does_not_influence_real_positions():
    """Real-position outputs must not depend on the *content* of padded slots."""
    torch.manual_seed(0)
    enc = SequenceEncoder(hidden_size=16, num_layers=2, num_heads=4, dropout=0.0, max_seq_len=8)
    enc.eval()

    ids = torch.tensor([[0, 0, 5, 9, 12]])
    base = torch.randn(1, 5, 16)
    base[0, 0] = 0.0
    base[0, 1] = 0.0

    junk = base.clone()
    junk[0, 0] = torch.randn(16) * 10
    junk[0, 1] = torch.randn(16) * 10

    with torch.no_grad():
        h_base = enc(base, ids, return_sequence=True)
        h_junk = enc(junk, ids, return_sequence=True)
    # only the real positions matter; padding outputs are never read
    assert torch.allclose(h_base[:, 2:], h_junk[:, 2:], atol=1e-6), "padding content leaked into real positions"


def test_padding_mask_is_not_vacuous():
    """Control: a PAD *between* real items changes outputs only when masked.

    Without padding awareness the encoder would attend to that slot (its content
    is the position embedding, i.e. real information); with the mask it must not.
    """
    torch.manual_seed(0)
    enc = SequenceEncoder(hidden_size=16, num_layers=2, num_heads=4, dropout=0.0, max_seq_len=8)
    enc.eval()

    ids = torch.tensor([[5, 0, 9, 12, 7]])
    base = torch.randn(1, 5, 16)
    base[0, 1] = 0.0
    junk = base.clone()
    junk[0, 1] = torch.randn(16) * 10

    with torch.no_grad():
        masked_a = enc(base, ids, return_sequence=True)
        masked_b = enc(junk, ids, return_sequence=True)
        # pure causal mask, no padding awareness
        pos = torch.arange(5).unsqueeze(0)
        xa = base + enc.position_embedding(pos)
        xb = junk + enc.position_embedding(pos)
        am = enc.causal_mask(5, xa.device)
        naive_a = enc.final_norm(enc.encoder(xa, mask=am))
        naive_b = enc.final_norm(enc.encoder(xb, mask=am))

    assert torch.allclose(masked_a[:, 2:], masked_b[:, 2:], atol=1e-6)
    assert not torch.allclose(naive_a[:, 2:], naive_b[:, 2:], atol=1e-6), (
        "the control experiment is vacuous: padding content did not matter even without a mask"
    )


def test_no_nan_with_left_padding():
    m = _model()
    ids = torch.tensor([[0, 0, 0, 4, 8], [0, 1, 2, 3, 4]])
    with torch.no_grad():
        h = m.encode(ids)
    assert torch.isfinite(h).all(), "left padding produced non-finite activations"


def test_pad_embedding_is_zero():
    m = _model()
    assert torch.all(m.item_embedding.weight[0] == 0)
    with torch.no_grad():
        assert torch.all(m.item_embeddings(torch.tensor([0]))[0] == 0)


def test_sequence_length_guard():
    m = _model(max_len=4)
    with torch.no_grad():
        try:
            m.encode(torch.ones(1, 5, dtype=torch.long))
        except ValueError:
            return
    raise AssertionError("expected a ValueError when the sequence exceeds max_seq_len")
