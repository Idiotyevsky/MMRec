import numpy as np
import torch
from src.recall.two_tower import TwoTowerRecall
from src.training.factory import build_model
from src.utils.config import Config


def _model(data, path):
    return build_model(Config({"model": {"name":"two_tower","hidden_size":128,"max_history":20,"dropout":0.,"modalities":{"text":True,"image":True}}, "data":{"processed_dir":str(path),"feature_dir":str(path)}}), data).eval()


def test_exact_order_pad_seen_and_topk(synthetic_data, synthetic_dir):
    model = _model(synthetic_data, synthetic_dir)
    with torch.no_grad(): table = model.all_item_embeddings().numpy()
    r = TwoTowerRecall(model, table, max_history=20)
    hist = [int(x) for x in synthetic_data.test_history(0)]
    out = r.recall(0, hist, 10)
    assert len(out) == 10 and 0 not in [c.item_id for c in out] and not (set(hist) & {c.item_id for c in out})
    q = r.query_vector(hist); scores = table @ q; scores[0] = -np.inf; scores[hist] = -np.inf
    assert [c.item_id for c in out] == np.argsort(-scores)[:10].tolist()
    assert [c.rank for c in out] == list(range(1, 11))


def test_empty_history_returns_no_candidates(synthetic_data, synthetic_dir):
    model = _model(synthetic_data, synthetic_dir)
    with torch.no_grad(): table = model.all_item_embeddings().numpy()
    assert TwoTowerRecall(model, table).recall(0, [], 10) == []
