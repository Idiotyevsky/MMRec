"""Ranker registry: load one or more trained models for the ranking stage.

The two-stage pipeline scores only the merged candidate pool (a few hundred
items), but the same objects can also score the full catalogue, which is what
the strict offline evaluation and the "compare models" view use.

Nothing here changes model code.  It wraps the existing factory + checkpoints so
that training and serving cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from ..data.dataset import ProcessedData
from ..training.factory import build_model
from ..utils.config import Config
from ..utils.logging import get_logger

LOG = get_logger("mmrec.rankers")
PAD = 0
ROOT = Path(__file__).resolve().parents[2]


def discover_run(tag: str, root: Path | None = None) -> Path | None:
    """Newest finished run directory for a tag.

    Prefers the exact tag (the seed-42 run the README reports) and only falls
    back to a seeded variant such as ``mm_concat_s2026`` when the primary run is
    absent, so the demo and the documented numbers agree.
    """
    root = root or ROOT
    runs = root / "results" / "runs"

    def finished(d: Path) -> bool:
        return (d / "best.pt").exists() and (d / "metrics.json").exists()

    exact = sorted((d for d in runs.glob(f"{tag}_[0-9]*") if finished(d)), key=lambda d: d.name)
    if exact:
        return exact[-1]
    seeded = sorted((d for d in runs.glob(f"{tag}_s[0-9]*") if finished(d)), key=lambda d: d.name)
    return seeded[0] if seeded else None


def default_ranker_specs(root: Path | None = None) -> dict[str, "RankerSpec"]:
    """The three rankers the demo offers, resolved from the newest finished runs."""
    root = root or ROOT
    out: dict[str, RankerSpec] = {}
    for name, desc in (
        ("sasrec", "SASRec — ID-only sequential baseline"),
        ("mm_concat", "MM-SASRec — ID + text + image, concatenation fusion"),
        ("mm_gated", "MM-SASRec — ID + text + image, gated fusion"),
    ):
        run = discover_run(name, root)
        if run is not None:
            out[name] = RankerSpec(name=name, run_dir=run, description=desc)
    return out


@dataclass
class RankerSpec:
    name: str
    run_dir: Path
    checkpoint: str = "best.pt"
    description: str = ""


class Ranker:
    """One trained model plus its cached item embedding table."""

    def __init__(self, name: str, model: torch.nn.Module, data: ProcessedData,
                 device: torch.device, max_seq_len: int, run_dir: Path | None = None,
                 description: str = "") -> None:
        self.name = name
        self.model = model.eval()
        self.data = data
        self.device = device
        self.max_seq_len = int(max_seq_len)
        self.run_dir = run_dir
        self.description = description
        with torch.no_grad():
            self._item_emb = model.all_item_embeddings().detach()

    # ------------------------------------------------------------------
    def user_vector(self, history_internal) -> torch.Tensor:
        seq = np.asarray(list(history_internal), dtype=np.int64)[-self.max_seq_len :]
        ids = np.zeros((1, self.max_seq_len), dtype=np.int64)
        ids[0, self.max_seq_len - len(seq):] = seq
        with torch.no_grad():
            return self.model.encode(torch.as_tensor(ids, dtype=torch.long, device=self.device))

    def score_candidates(self, history_internal, candidate_ids) -> np.ndarray:
        """Ranking score for a specific candidate list (the serving path)."""
        h = self.user_vector(history_internal)
        idx = torch.as_tensor(np.asarray(candidate_ids, dtype=np.int64), device=self.device)
        with torch.no_grad():
            return (h @ self._item_emb[idx].t()).squeeze(0).float().cpu().numpy()

    def score_all(self, history_internal) -> np.ndarray:
        """Score the whole catalogue (used for comparison views)."""
        h = self.user_vector(history_internal)
        with torch.no_grad():
            out = (h @ self._item_emb.t()).squeeze(0).float().cpu().numpy()
        out[PAD] = -np.inf
        return out

    def rank_of(self, history_internal, target_internal: int) -> int:
        scores = self.score_all(history_internal)
        tgt = scores[int(target_internal)]
        return int((scores > tgt).sum()) + 1

    def num_parameters(self) -> int:
        return int(self.model.num_parameters())

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "params": self.num_parameters(),
            "max_seq_len": self.max_seq_len,
        }


class RankerRegistry:
    """Lazily loads and caches rankers by name."""

    def __init__(self, data: ProcessedData, specs: dict[str, RankerSpec],
                 device: torch.device | str = "cpu") -> None:
        self.data = data
        self.specs = dict(specs)
        self.device = torch.device(device)
        self._cache: dict[str, Ranker] = {}

    # ------------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return list(self.specs.keys())

    def get(self, name: str) -> Ranker:
        if name not in self.specs:
            raise KeyError(f"unknown ranker {name!r}; available: {self.names}")
        if name not in self._cache:
            spec = self.specs[name]
            run_dir = spec.run_dir if spec.run_dir.is_absolute() else ROOT / spec.run_dir
            ckpt_path = run_dir / spec.checkpoint
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"ranker {name!r}: checkpoint {ckpt_path} not found. "
                    "Run scripts/prepare_demo.py first."
                )
            cfg = Config(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
            model = build_model(cfg, self.data, device=self.device)
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            own = model.state_dict()
            state = {k: v for k, v in ckpt["model_state"].items() if k in own}
            model.load_state_dict(state)
            LOG.info(f"loaded ranker {name!r} from {run_dir.name}")
            self._cache[name] = Ranker(
                name=name, model=model, data=self.data, device=self.device,
                max_seq_len=int(cfg.model.get("max_seq_len", 50)),
                run_dir=run_dir, description=spec.description,
            )
        return self._cache[name]

    def warm(self, names: list[str] | None = None) -> None:
        for n in (names or self.names):
            self.get(n)

    def as_dict(self) -> dict:
        return {n: {"description": s.description, "run_dir": str(s.run_dir),
                    "loaded": n in self._cache} for n, s in self.specs.items()}
