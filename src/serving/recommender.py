"""End-to-end recommender: user history -> user vector -> vector index -> top-K.

    rec = Recommender.from_run("results/runs/<run_id>")
    rec.recommend([12, 45, 91, 102], top_k=20)

The served item ids are the *raw* MicroLens ids, while the model works with
internal ids; the conversion happens here and nowhere else.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data.dataset import ProcessedData
from ..retrieval.faiss_index import ItemIndex
from ..training.factory import build_model
from ..utils.config import Config

ROOT = Path(__file__).resolve().parents[2]


class Recommender:
    def __init__(
        self,
        model: torch.nn.Module,
        data: ProcessedData,
        index: ItemIndex,
        device: torch.device,
        max_seq_len: int = 50,
        run_dir: Path | None = None,
    ) -> None:
        self.model = model.eval()
        self.data = data
        self.index = index
        self.device = device
        self.max_seq_len = max_seq_len
        self.run_dir = run_dir
        self._raw_to_internal = {int(r): i + 1 for i, r in enumerate(data.raw_item_ids)}

    # ------------------------------------------------------------------
    @classmethod
    def from_run(cls, run_dir: str | Path, checkpoint: str = "best.pt", device: str = "cpu") -> "Recommender":
        run_dir = Path(run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        cfg = Config(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
        data = ProcessedData.load(ROOT / cfg.data.processed_dir)
        dev = torch.device(device)
        model = build_model(cfg, data, device=dev)
        ckpt = torch.load(run_dir / checkpoint, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        emb_path = run_dir / "item_embeddings.npy"
        if emb_path.exists():
            emb = np.load(emb_path)
        else:
            with torch.no_grad():
                emb = model.all_item_embeddings().cpu().numpy()
        index = ItemIndex(emb, normalize=False).build()
        return cls(model, data, index, dev,
                   max_seq_len=int(cfg.model.get("max_seq_len", 50)), run_dir=run_dir)

    # ------------------------------------------------------------------
    def _to_internal(self, raw_history: list[int]) -> np.ndarray:
        out = [self._raw_to_internal.get(int(r), 0) for r in raw_history]
        return np.asarray(out, dtype=np.int64)

    @torch.no_grad()
    def recommend(
        self,
        history_item_ids: list[int],
        top_k: int = 20,
        exclude_history: bool = True,
        return_scores: bool = True,
    ) -> list[dict]:
        """``history_item_ids`` are **raw** MicroLens item ids, oldest first."""
        internal = self._to_internal(history_item_ids)
        seq = internal[-self.max_seq_len :]
        ids = np.zeros((1, self.max_seq_len), dtype=np.int64)
        ids[0, self.max_seq_len - len(seq) :] = seq

        input_ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        user_repr = self.model.encode(input_ids).cpu().numpy()

        ex = None
        if exclude_history:
            ex = np.concatenate([internal, [0]]).reshape(1, -1)
        item_ids, scores = self.index.search(user_repr, top_k=top_k, exclude=ex)

        out = []
        for i, s in zip(item_ids[0], scores[0]):
            i = int(i)
            if i <= 0:
                continue
            entry = {"item_id": self.data.raw_item(i)}
            if return_scores:
                entry["score"] = float(s)
            out.append(entry)
        return out

    def warm_start(self) -> None:
        """Run one dummy forward pass so the first real request is not slow."""
        if not self.data.raw_item_ids.size:
            return
        self.recommend([int(self.data.raw_item_ids[0])], top_k=1)

    def describe(self) -> dict:
        meta = {}
        if self.run_dir and (self.run_dir / "item_embeddings_meta.json").exists():
            meta = json.loads((self.run_dir / "item_embeddings_meta.json").read_text(encoding="utf-8"))
        return {
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "num_items": self.data.num_items,
            "max_seq_len": self.max_seq_len,
            "model": meta.get("model"),
            "fusion": meta.get("fusion"),
            "modalities": meta.get("modalities"),
        }
