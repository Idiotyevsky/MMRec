"""Exact content-aware learned retrieval channel and verified artifact loader."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import yaml
from ..data.dataset import ProcessedData
from ..models.two_tower import TwoTowerRetriever
from ..retrieval.faiss_index import ItemIndex
from ..training.factory import _row_for_item
from ..utils.config import Config
from ..utils.provenance import dataset_fingerprint
from .base import RecallStrategy, RecallCandidate

class TwoTowerRecall(RecallStrategy):
    name = "two_tower"
    def __init__(self, model: TwoTowerRetriever, embeddings: np.ndarray, device: str | torch.device = "cpu", max_history: int = 50):
        self.model, self.device, self.max_history = model.eval(), torch.device(device), int(max_history)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.num_items = self.embeddings.shape[0] - 1
        self.index = ItemIndex(self.embeddings, normalize=False).build()
    @torch.no_grad()
    def query_vector(self, history: Sequence[int]) -> np.ndarray:
        hist = np.asarray([int(i) for i in history if int(i) > 0][-self.max_history:], dtype=np.int64)
        ids = np.zeros((1, self.max_history), dtype=np.int64)
        ids[0, -len(hist):] = hist if len(hist) else []
        return self.model.encode_history(torch.as_tensor(ids, device=self.device)).cpu().numpy()[0]
    def recall(self, user_id: int, history: Sequence[int], top_k: int) -> list[RecallCandidate]:
        seen = self._seen(history)
        if not seen: return []
        q = self.query_vector(history)[None]
        fetch = min(self.num_items, top_k + len(seen) + 16)
        ids, scores = self.index.search(q, fetch, exclude=np.asarray([list(seen)], dtype=np.int64))
        return self._filter(ids[0], scores[0], seen, top_k)

def load_two_tower_recall(run_dir: str | Path, data: ProcessedData, device: str = "cpu") -> TwoTowerRecall:
    run = Path(run_dir); meta_path = run / "two_tower_item_embeddings.meta.json"; emb_path = run / "two_tower_item_embeddings.npy"
    if not meta_path.exists() or not emb_path.exists(): raise FileNotFoundError("missing Two-Tower embedding artifact or metadata")
    meta = json.loads(meta_path.read_text()); emb = np.load(emb_path)
    if int(meta.get("num_items", -1)) != data.num_items or meta.get("dataset_hash") != dataset_fingerprint(data):
        raise AssertionError("Two-Tower artifact dataset mapping mismatch")
    if emb.shape != (data.num_items + 1, int(meta.get("dim", -1))) or not bool(meta.get("normalized")):
        raise AssertionError("invalid Two-Tower embedding artifact shape/normalization")
    cfg = Config(yaml.safe_load((run / "config.yaml").read_text()))
    model = TwoTowerRetriever(num_items=data.num_items, hidden_size=int(cfg.model.get("hidden_size", 128)),
        max_history=int(cfg.model.get("max_history", 50)), dropout=float(cfg.model.get("dropout", .1)),
        recency_decay=float(cfg.model.get("recency_decay", .9)), feature_dir=cfg.data.feature_dir,
        row_for_item=_row_for_item(data.dir, [m for m,v in dict(cfg.model.modalities).items() if v]),
        modalities=dict(cfg.model.modalities), normalize_embeddings=bool(cfg.model.get("normalize_embeddings", True))).to(device)
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    if ck.get("dataset_hash") != dataset_fingerprint(data): raise AssertionError("Two-Tower checkpoint dataset mismatch")
    model.load_state_dict(ck["model_state"])
    return TwoTowerRecall(model, emb, device=device, max_history=model.max_seq_len)
