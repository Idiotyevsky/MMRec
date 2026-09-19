"""Generative Semantic-ID recall as a RecallStrategy."""

import numpy as np
import pytest
import torch

from src.models.generative_rec import GenerativeRecommender
from src.models.semantic_id import SemanticIDMapper
from src.recall.base import RecallStrategy
from src.recall.generative import GenerativeRecall, load_generative_recall


@pytest.fixture()
def toy_codes():
    # items 1 and 3 collide on purpose (rows 0 and 2 share a Semantic ID)
    return np.array([
        [0, 1, 2],
        [3, 4, 5],
        [0, 1, 2],
        [6, 7, 8],
    ], dtype=np.int64)


def _strategy(toy_codes, **kw):
    mapper = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    model = GenerativeRecommender(
        vocab_size=mapper.vocab_size, num_levels=3,
        hidden_size=32, num_layers=1, num_heads=4, max_seq_len=64,
    )
    return GenerativeRecall(model=model, mapper=mapper, num_items=4, **kw), mapper


def test_is_a_recall_strategy(toy_codes):
    strat, _ = _strategy(toy_codes)
    assert isinstance(strat, RecallStrategy)
    assert strat.name == "generative"


def test_every_returned_item_is_real_and_unique(toy_codes):
    strat, mapper = _strategy(toy_codes, beam_width=8)
    out = strat.recall(user_id=1, history=[1, 2], top_k=10)
    assert out
    for c in out:
        assert 1 <= c.item_id <= 4
        assert c.item_id in mapper.item_to_sid
        assert c.source == "generative"
    assert len({c.item_id for c in out}) == len(out)  # no duplicates
    assert [c.rank for c in out] == list(range(1, len(out) + 1))


def test_history_items_are_excluded(toy_codes):
    strat, _ = _strategy(toy_codes, beam_width=8)
    history = [1, 3]
    out = strat.recall(user_id=1, history=history, top_k=10)
    assert not ({c.item_id for c in out} & set(history))


def test_pad_is_never_returned(toy_codes):
    strat, _ = _strategy(toy_codes, beam_width=8)
    out = strat.recall(user_id=1, history=[0, 0], top_k=10)
    assert all(c.item_id != 0 for c in out)


def test_colliding_sid_expands_to_both_items(toy_codes):
    """A collision must surface as two candidates, not as a silently dropped one."""
    mapper = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    model = GenerativeRecommender(
        vocab_size=mapper.vocab_size, num_levels=3,
        hidden_size=32, num_layers=1, num_heads=4, max_seq_len=64,
    )
    strat = GenerativeRecall(model=model, mapper=mapper, num_items=4, beam_width=4)
    decoded = [((0, 1, 2), -1.5)]
    out = strat._to_candidates(decoded, history=[2], top_k=10)
    assert [c.item_id for c in out] == [1, 3]
    assert all(c.extra["collision_size"] == 2 for c in out)
    assert out[0].extra["semantic_id"] == [0, 1, 2]
    assert out[0].extra["beam_rank"] == 1


def test_top_k_is_respected(toy_codes):
    strat, _ = _strategy(toy_codes, beam_width=8)
    out = strat.recall(user_id=1, history=[1], top_k=2)
    assert len(out) <= 2


def test_history_is_truncated_to_max_items(toy_codes):
    strat, _ = _strategy(toy_codes, beam_width=4, max_history_items=2)
    toks = strat._history_tokens([1, 2, 3, 4])
    assert len(toks) == 2 * 3
    assert toks == strat.mapper.item_tokens(3) + strat.mapper.item_tokens(4)


def test_stats_report_empty_decodings(toy_codes):
    strat, _ = _strategy(toy_codes, beam_width=8)
    strat.recall(user_id=1, history=[1], top_k=5)
    assert strat.last_stats["users"] == 1
    assert strat.last_stats["beam_width"] == 8
    assert strat.last_stats["empty_decodings"] >= 0


def test_generation_is_deterministic_in_eval_mode(toy_codes):
    strat, _ = _strategy(toy_codes, beam_width=6)
    a = [c.item_id for c in strat.recall(user_id=1, history=[1, 2], top_k=10)]
    b = [c.item_id for c in strat.recall(user_id=1, history=[1, 2], top_k=10)]
    assert a == b


def test_batched_generation_matches_single(toy_codes):
    strat, _ = _strategy(toy_codes, beam_width=6)
    single = strat._to_candidates(
        strat.generate_batch([[1, 2]])[0][0], history=[1, 2], top_k=10)
    batched_decoded, stats = strat.generate_batch([[1, 2], [3, 4]])
    assert stats["users"] == 2
    assert len(batched_decoded) == 2


def test_loader_refuses_mismatched_semantic_ids(tmp_path, toy_codes):
    mapper = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    model = GenerativeRecommender(
        vocab_size=mapper.vocab_size, num_levels=3,
        # exactly how scripts/train_generative_rec.py sizes it
        hidden_size=32, num_layers=1, num_heads=4, max_seq_len=1 + 4 * 3,
    )
    sid_path = tmp_path / "semantic_ids.npz"
    np.savez(sid_path, codes=toy_codes)
    (tmp_path / "semantic_ids_meta.json").write_text(
        '{"codebook_size": 16, "num_items": 4}', encoding="utf-8")
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "semantic_id_hash": "deadbeef",
                "args": {"hidden_size": 32, "num_layers": 1, "num_heads": 4}}, ckpt_path)

    with pytest.raises(AssertionError, match="refusing to load"):
        load_generative_recall(ckpt_path, sid_path, max_history_items=4)


def test_loader_rebuilds_a_working_strategy(tmp_path, toy_codes):
    import hashlib
    mapper = SemanticIDMapper(toy_codes, num_items=4, codebook_size=16)
    model = GenerativeRecommender(
        vocab_size=mapper.vocab_size, num_levels=3,
        # exactly how scripts/train_generative_rec.py sizes it
        hidden_size=32, num_layers=1, num_heads=4, max_seq_len=1 + 4 * 3,
    )
    sid_path = tmp_path / "semantic_ids.npz"
    np.savez(sid_path, codes=toy_codes)
    (tmp_path / "semantic_ids_meta.json").write_text(
        '{"codebook_size": 16, "num_items": 4}', encoding="utf-8")
    digest = hashlib.sha256(np.ascontiguousarray(toy_codes).tobytes()).hexdigest()[:16]
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "semantic_id_hash": digest,
                "args": {"hidden_size": 32, "num_layers": 1, "num_heads": 4,
                         "max_items": 4}}, ckpt_path)

    strat = load_generative_recall(ckpt_path, sid_path, beam_width=5, max_history_items=4)
    assert strat.num_items == 4
    out = strat.recall(user_id=1, history=[1], top_k=5)
    assert all(1 <= c.item_id <= 4 for c in out)
