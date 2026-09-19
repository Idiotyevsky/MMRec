"""Serving service: assembles the two-stage pipeline and answers API requests.

This is the layer that owns the **raw <-> internal id conversion**, so that the
pipeline, the recall channels and the models can all keep working in internal id
space while the API and the frontend only ever see raw MicroLens ids.

It also serves the Cold Start Explorer, which reads the *cold simulation*
dataset (``data/processed/cold10``) and the offline cold experiment table.  Those
are deliberately kept separate from the base-dataset pipeline: the base dataset
has no cold items, and pretending otherwise would misrepresent the experiment.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..data.dataset import ProcessedData
from ..data.metadata import ItemMetadata
from ..pipeline import RankerRegistry, RankerSpec, TwoStageRecommender
from ..pipeline.rankers import default_ranker_specs, discover_run
from ..recall import (
    CandidateMerger,
    ItemCFRecall,
    PopularRecall,
    SemanticRecall,
    load_itemcf_index,
)
from ..recall.semantic import build_content_embeddings
from ..media.library import MediaLibrary
from ..utils.logging import get_logger

LOG = get_logger("mmrec.service")
ROOT = Path(__file__).resolve().parents[2]


class ShortRecService:
    def __init__(
        self,
        processed_dir: str = "data/processed/base",
        cold_processed_dir: str = "data/processed/cold10",
        device: str = "cpu",
        history_mode: str = "test",
        default_ranker: str = "mm_concat",
        itemcf_index: str = "artifacts/itemcf_neighbors.npz",
        content_embeddings: str = "artifacts/content_embeddings.npy",
        feature_dir: str = "data/raw",
        root: str | Path | None = None,
        ranker_specs: dict[str, RankerSpec] | None = None,
        generative_checkpoint: str | None = None,
        generative_beam: int = 20,
        semantic_ids: str = "artifacts/semantic_ids.npz",
    ) -> None:
        # `root` and `ranker_specs` exist so tests can point the whole service at
        # a synthetic dataset instead of the real MicroLens artifacts.
        self.root = Path(root) if root is not None else ROOT
        self.processed_dir = Path(processed_dir)
        self.device = device
        self.history_mode = history_mode
        self.default_ranker = default_ranker

        self.data = ProcessedData.load(self.root / self.processed_dir)
        self.metadata = ItemMetadata(self.data)
        self._raw_to_internal = {int(r): i + 1 for i, r in enumerate(self.data.raw_item_ids)}

        # ---- recall channels ----
        strategies: list = [PopularRecall(self.data.train_freq, self.data.num_items)]
        self.channel_status: dict[str, tuple[bool, str]] = {}
        ic_path = self.root / itemcf_index
        if ic_path.exists():
            ic = load_itemcf_index(ic_path)
            strategies.append(ItemCFRecall(ic["neighbors"], ic["sims"], self.data.num_items))
            self.itemcf_index = ic
        else:
            LOG.warning(f"ItemCF index missing at {ic_path}; channel disabled")
            self.itemcf_index = None
        emb_path = self.root / content_embeddings
        if emb_path.exists():
            self.content_embeddings = np.load(emb_path)
            strategies.append(SemanticRecall(self.content_embeddings, self.data.num_items))
        else:
            LOG.warning(f"content embeddings missing at {emb_path}; channel disabled")
            self.content_embeddings = None
        self.generative = self._load_generative(
            generative_checkpoint, generative_beam, semantic_ids)
        if self.generative is not None:
            strategies.append(self.generative)
        self.merger = CandidateMerger(strategies)
        self.channel_status = {name: (True, "loaded") for name in self.merger.source_names}

        # ---- rankers ----
        specs: dict[str, RankerSpec] = (ranker_specs if ranker_specs is not None
                                        else default_ranker_specs(self.root))
        if default_ranker not in specs:
            raise RuntimeError(
                f"default ranker {default_ranker!r} has no finished run; "
                f"available: {sorted(specs)}. Run scripts/prepare_demo.py."
            )
        self.registry = RankerRegistry(self.data, specs, device=device)
        self.registry.warm()

        self.recommender = TwoStageRecommender(
            self.data, self.merger, self.registry,
            default_ranker=default_ranker, history_mode=history_mode,
        )

        # ---- cold explorer (separate dataset) ----
        self.cold_data: ProcessedData | None = None
        cold_path = self.root / cold_processed_dir
        if (cold_path / "dataset.npz").exists():
            try:
                self.cold_data = ProcessedData.load(cold_path)
                self.cold_metadata = ItemMetadata(self.cold_data)
            except Exception as exc:  # pragma: no cover - defensive
                LOG.warning(f"could not load cold dataset at {cold_path}: {exc}")
                self.cold_data = None
                self.cold_metadata = None
        else:
            LOG.warning(f"cold dataset missing at {cold_path}; /cold endpoints degraded")
            self.cold_metadata = None
        # ---- demo media (optional; absence must not break serving) ----
        self.media = MediaLibrary(root=self.root)

        self._cold_ids = (np.flatnonzero(self.cold_data.is_simulated_cold)
                          if self.cold_data is not None else np.zeros(0, dtype=np.int64))
        self._feature_dir = self.root / feature_dir

    def _load_generative(self, checkpoint, beam, semantic_ids):
        """Optionally attach the Semantic-ID generative channel.

        Off by default: it is an experimental retrieval route, and the two-stage
        system must behave identically whether or not it is present.  Any failure
        to load is reported and treated as "channel absent" rather than fatal, so
        a missing artifact can never take the service down.
        """
        if not checkpoint:
            return None
        ck = self.root / checkpoint
        if not ck.exists():
            LOG.warning(f"generative checkpoint missing at {ck}; channel disabled")
            return None
        try:
            from src.recall.generative import load_generative_recall

            strat = load_generative_recall(
                ck, self.root / semantic_ids, device=self.device, beam_width=beam)
        except Exception as exc:  # noqa: BLE001 - surfaced, never fatal
            LOG.warning(f"generative channel failed to load ({exc}); channel disabled")
            return None
        ready, detail = strat.is_ready()
        LOG.info(f"generative channel enabled: {detail}")
        return strat if ready else None

    # ------------------------------------------------------------------
    # id conversion
    # ------------------------------------------------------------------
    def to_raw(self, item_internal: int) -> int:
        i = int(item_internal)
        if 1 <= i <= self.data.num_items:
            return int(self.data.raw_item_ids[i - 1])
        return -1

    def to_internal(self, item_raw: int) -> int:
        return int(self._raw_to_internal.get(int(item_raw), 0))

    def _raw_history(self, hist_internal) -> list[int]:
        return [self.to_raw(i) for i in hist_internal]

    def _raw_entry(self, entry: dict) -> dict:
        out = dict(entry)
        out["item_id"] = self.to_raw(entry["item_id"])
        return out

    # ------------------------------------------------------------------
    # endpoints
    # ------------------------------------------------------------------
    def system(self) -> dict:
        info = self.recommender.system_info()
        readiness = {}
        for name, (ok, why) in info.get("recall_readiness", {}).items():
            readiness[name] = [bool(ok), str(why)]
        return {
            "dataset": info["dataset"],
            "users": info["users"],
            "items": info["items"],
            "interactions": info["interactions"],
            "default_ranker": info["default_ranker"],
            "rankers": self.model_infos(),
            "recall_sources": info["recall_sources"],
            "recall_readiness": readiness,
            "item_metadata": info["item_metadata"],
            "exploration": self.exploration_summary(),
            "history_mode": info["history_mode"],
        }

    def user(self, user_id: int) -> dict:
        hist = self.recommender.history(user_id)
        items = [{"item_id": self.to_raw(i), **self.metadata.of(i).as_dict()}
                 for i in hist]
        for it in items:
            it.pop("in_catalogue", None)
        target = self.recommender.target(user_id)
        return {
            "user_id": int(user_id),
            "history_length": len(items),
            "history": items,
            "recent": items[-5:],
            "target_item": self.to_raw(target) if target else None,
        }

    def recall(self, user_id: int, recall_k: int = 200, sources=None) -> dict:
        merged = self.recommender.recall(user_id, recall_k=recall_k, sources=sources)
        cands = []
        for c in merged.candidates:
            meta = self.metadata.of(c.item_id)
            cands.append({
                "item_id": self.to_raw(c.item_id),
                "sources": c.sources,
                "merge_score": float(c.merge_score),
                "popularity_bucket": meta.popularity_bucket,
                "is_cold": meta.is_simulated_cold,
                "is_simulated_cold": meta.is_simulated_cold,
                "is_zero_train_signal": meta.is_zero_train_signal,
                "exploration_candidate": meta.exploration_candidate,
                "train_interactions": meta.train_interactions,
            })
        stats = dict(merged.stats)
        stats["candidates_ranked"] = len(merged.candidates)
        stats["exploration_in_pool"] = sum(
            1 for c in merged.candidates if self.metadata.of(c.item_id).exploration_candidate)
        coverage: dict[str, int] = {}
        for c in merged.candidates:
            for name in c.source_names:
                coverage[name] = coverage.get(name, 0) + 1
        stats["source_coverage_in_pool"] = coverage
        return {"user_id": int(user_id), "recall_k": recall_k,
                "stats": stats, "candidates": cands}

    def recommend(
        self,
        user_id: int,
        recall_k: int = 200,
        top_k: int = 20,
        ranker: str | None = None,
        cold_exploration: bool = False,
        cold_quota: int = 2,
        max_candidates: int | None = None,
        sources=None,
    ) -> dict:
        res = self.recommender.recommend(
            user_id, recall_k=recall_k, final_k=top_k, ranker=ranker,
            cold_exploration=cold_exploration, cold_quota=cold_quota,
            max_candidates=max_candidates, sources=sources,
        )
        out = res.as_dict()
        out["exploration"] = res.rerank
        out["history"] = self._raw_history(res.history)
        out["target_item"] = self.to_raw(res.target_item) if res.target_item else None
        out["recommendations"] = [self._raw_entry(e) for e in res.recommendations]
        out["recall"] = {k: v for k, v in res.recall_stats.items() if k != "errors"}
        return out

    def inspect(
        self,
        user_id: int,
        recall_k: int = 200,
        top_n: int = 30,
        ranker: str | None = None,
        compare: str | None = "sasrec",
        cold_exploration: bool = False,
        max_candidates: int | None = None,
    ) -> dict:
        out = self.recommender.inspect(
            user_id, recall_k=recall_k, top_n=top_n, ranker=ranker,
            compare=compare, cold_exploration=cold_exploration,
            max_candidates=max_candidates,
        )
        out["history"] = self._raw_history(out["history"])
        out["target_item"] = self.to_raw(out["target_item"]) if out["target_item"] else None
        for key in ("recall_candidates", "top_candidates", "baseline_top",
                    "moved_up_by_multimodal", "moved_down_by_multimodal", "final_top_k"):
            out[key] = [self._raw_entry(e) for e in out.get(key, [])]
        out["history_items"] = [
            {"item_id": self.to_raw(i), **self.metadata.of(i).as_dict()}
            for i in self.recommender.history(user_id)
        ]
        return out

    # ------------------------------------------------------------------
    # models / offline metrics
    # ------------------------------------------------------------------
    @staticmethod
    def _run_key(row: dict) -> tuple:
        """Identity of a model configuration, ignoring the seed and the run id."""
        return (row.get("dataset", "base"), row.get("model", ""), row.get("fusion", ""),
                row.get("modalities", ""), row.get("id_dropout", "0.0"),
                row.get("item_dropout", "0.0"))

    def offline_metrics(self) -> dict[tuple, dict]:
        """Mean offline metrics per model configuration, from results/tables/overall.csv.

        Grouping by configuration (not by tag) is what lets a ranker show
        ``mean ± std`` over its seeds, exactly as the README table does.
        """
        path = self.root / "results" / "tables" / "overall.csv"
        if not path.exists():
            return {}
        rows: dict[tuple, list[dict]] = {}
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("dataset") or "base") != "base":
                    continue
                rows.setdefault(self._run_key(r), []).append(r)
        out: dict[tuple, dict] = {}
        for key, rs in rows.items():
            entry: dict = {"n_seeds": len(rs)}
            for metric in ("Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "Coverage@20"):
                vals = [float(r[metric]) for r in rs if r.get(metric) not in (None, "")]
                if vals:
                    entry[metric] = float(np.mean(vals))
                    entry[f"{metric}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            out[key] = entry
        return out

    def _spec_key(self, run_dir) -> tuple | None:
        """Same key for a ranker's own run, read from its config."""
        import yaml

        cfg_path = Path(run_dir) / "config.yaml"
        if not cfg_path.exists():
            return None
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        m = cfg.get("model") or {}
        mods = m.get("modalities")
        if mods:
            order = ["id", "text", "image", "video"]
            modalities = "+".join(k for k in order if dict(mods).get(k))
        else:
            modalities = "id" if m.get("name") == "sasrec" else "-"
        return ("base", m.get("name", ""), m.get("fusion", "-"), modalities,
                f"{float(m.get('id_dropout_prob', 0.0) or 0.0)}",
                f"{float(m.get('item_dropout_prob', 0.0) or 0.0)}")

    def evaluation(self) -> dict:
        """Offline evaluation artifacts, as-is, for the System page charts.

        Nothing is recomputed or hard-coded here: these are the same CSV files
        the README tables are generated from.
        """
        def table(name: str) -> list[dict]:
            path = self.root / "results" / "tables" / name
            if not path.exists():
                return []
            with open(path, newline="", encoding="utf-8") as f:
                rows = []
                for r in csv.DictReader(f):
                    rows.append({k: _f(v) if v not in (None, "") else None for k, v in r.items()})
                return rows

        recall = table("recall_eval.csv")
        ks = sorted({int(k.split("@")[1]) for k in recall[0] if k.startswith("Recall@")}) if recall else []
        return {
            "recall_by_budget": {
                "ks": ks,
                "channels": [
                    {"channel": r["channel"], "values": [r.get(f"Recall@{k}") for k in ks]}
                    for r in recall
                ],
                "num_users": int(recall[0]["num_users"]) if recall and recall[0].get("num_users") else None,
            },
            "pipeline_tradeoff": table("pipeline_tradeoff.csv"),
            "latency": table("latency_benchmark.csv"),
            "source_contribution": table("recall_source_contribution.csv"),
            "notes": {
                "recall": "candidate-generation Recall@K on the test target, history masked",
                "pipeline": "same checkpoint and user sample for both pipeline and full-catalogue",
                "latency": "CPU demo benchmark; median; relative comparison only",
            },
        }

    def model_infos(self) -> list[dict]:
        metrics = self.offline_metrics()
        infos = []
        for name, spec in self.registry.specs.items():
            key = self._spec_key(spec.run_dir) if spec.run_dir else None
            infos.append({
                "name": name,
                "description": spec.description,
                "run_dir": _rel(spec.run_dir, self.root),
                "tag": spec.run_dir.name.rsplit("_", 2)[0] if spec.run_dir else name,
                "offline": metrics.get(key, {}) if key else {},
            })
        return infos

    # ------------------------------------------------------------------
    # demo media
    def media_manifest(self) -> dict:
        return self.media.manifest_summary()

    def item_media(self, item_raw: int) -> dict:
        """Media info for one item, keyed by the raw MicroLens id."""
        raw = int(item_raw)
        if raw not in self._raw_to_internal:
            return {"item_id": raw, "available": False, "verified": False,
                    "reason": "unknown item id"}
        rec = self.media.get(raw)
        if rec is None:
            return {"item_id": raw, "available": False, "verified": False,
                    "reason": "no media prepared for this item"}
        out = rec.as_dict()
        out["is_zero_train_signal"] = self._zero_train(raw)
        return out

    def media_video_path(self, item_raw: int):
        """Resolve a raw item id to a prepared video file, or ``None``.

        The manifest is keyed by raw MicroLens item ids, the same space the API
        exposes, so no conversion happens here.  The caller never builds a path
        from user input.
        """
        return self.media.video_path(int(item_raw))

    def _zero_train(self, raw_item: int) -> bool:
        internal = int(self._raw_to_internal.get(int(raw_item), 0))
        if internal <= 0:
            return False
        return bool(self.metadata.is_zero_train_signal[internal])

    def feed(self, user_id: int | None = None) -> dict:
        """Playable feed in recommendation order, with the raw id mapping applied."""
        entries = []
        for rec in self.media.feed(user_id):
            out = dict(rec)
            out["is_zero_train_signal"] = self._zero_train(rec["item_id"])
            entries.append(out)
        return {
            "user_id": user_id,
            "num_items": len(entries),
            "prepared": self.media.prepared,
            "items": entries,
        }

    # ------------------------------------------------------------------
    # cold start explorer
    # ------------------------------------------------------------------
    def exploration_summary(self) -> dict:
        """Zero-train (serving) vs simulated-cold (benchmark) item counts."""
        base = self.metadata.summary()
        cold = self.cold_metadata.summary() if self.cold_metadata is not None else None
        return {
            "serving_zero_train_items": base["zero_train_frequency_items"],
            "serving_exploration_candidates": base["exploration_candidates"],
            "benchmark_simulated_cold_items": (cold or {}).get("cold_items", 0),
            "note": ("Zero-train items have no observed training interactions in the base "
                     "split and drive the serving exploration quota. Simulated cold items "
                     "are a controlled benchmark subset with all training interactions "
                     "removed, reported in results/tables/cold_start.csv."),
        }

    def cold_summary(self) -> dict:
        path = self.root / "results" / "tables" / "cold_start.csv"
        experiments: list[dict] = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    experiments.append({
                        "model": r.get("tag"),
                        "kind": r.get("model"),
                        "modalities": r.get("modalities"),
                        "cold_recall@10": _f(r.get("Cold Recall@10")),
                        "cold_recall@20": _f(r.get("Cold Recall@20")),
                        "cold_only_recall@10": _f(r.get("ColdOnly Recall@10")),
                        "cold_only_recall@20": _f(r.get("ColdOnly Recall@20")),
                    })
        n_cold = int(self.cold_data.is_simulated_cold.sum()) if self.cold_data is not None else 0
        n_users_cold_target = 0
        if self.cold_data is not None:
            targets = np.concatenate([self.cold_data.val_target, self.cold_data.test_target])
            n_users_cold_target = int(self.cold_data.is_simulated_cold[targets].sum())
        zero_train = int(self.metadata.is_zero_train_signal[1:].sum())
        return {
            "dataset": "MicroLens-100K (simulated cold split)",
            "num_cold_items": n_cold,
            "serving_zero_train_items": zero_train,
            "num_users_with_cold_target": n_users_cold_target,
            "experiments": experiments,
            "note": (
                "ColdOnly Recall ranks within the cold catalogue only; the full-ranking "
                "column ranks against all items. They answer different questions and must "
                "not be compared to each other."
            ),
        }

    def cold_items(self, n: int = 20, seed: int = 0) -> list[dict]:
        if self.cold_data is None or self._cold_ids.size == 0:
            return []
        rng = np.random.default_rng(seed)
        pick = rng.choice(self._cold_ids, size=min(n, self._cold_ids.size), replace=False)
        return [self.cold_item(self.to_raw(int(i))) for i in sorted(pick.tolist())]

    def cold_item(self, item_raw: int) -> dict | None:
        if self.cold_data is None:
            return None
        i = int(self._raw_to_internal.get(int(item_raw), 0))
        if i <= 0 or i > self.cold_data.num_items or not self.cold_data.is_simulated_cold[i]:
            return None
        meta = self.cold_metadata.of(i)
        return {
            "item_id": int(item_raw),
            "train_interactions": meta.train_interactions,
            "is_cold": True,
            "is_simulated_cold": True,
            "is_zero_train_signal": True,
            "popularity_bucket": "simulated_cold",
            "content_available": self.content_availability(i),
            "content_similar": self.content_similar(i, k=6),
        }

    def content_availability(self, item_internal: int) -> dict[str, bool]:
        """Which modalities actually carry a non-zero feature vector."""
        out: dict[str, bool] = {}
        for m in ("text", "image", "video"):
            p = self._feature_dir / f"{m}_feat.npy"
            if not p.exists():
                out[m] = False
                continue
            lut_path = self.processed_dir / f"row_for_item_{m}.npy"
            if not lut_path.exists():
                out[m] = False
                continue
            lut = np.load(lut_path)
            row = int(lut[item_internal]) if item_internal < lut.shape[0] else -1
            if row < 0:
                out[m] = False
                continue
            arr = np.load(p, mmap_mode="r")
            vec = np.asarray(arr[row], dtype=np.float32)
            out[m] = bool(np.isfinite(vec).all() and np.abs(vec).sum() > 0)
        return out

    def content_similar(self, item_internal: int, k: int = 6) -> list[dict]:
        """Nearest items in the raw text+image feature space.

        Labelled as *content-similar* in the UI: MicroLens-100K ships no titles or
        captions, so this is the only honest way to hint at what an item is about.
        """
        if self.content_embeddings is None or item_internal >= self.content_embeddings.shape[0]:
            return []
        emb = self.content_embeddings
        sims = emb @ emb[item_internal]
        sims[item_internal] = -np.inf
        sims[0] = -np.inf
        idx = np.argsort(-sims)[:k]
        out = []
        for j in idx.tolist():
            meta = self.metadata.of(int(j))
            out.append({
                "item_id": self.to_raw(int(j)),
                "similarity": float(sims[j]),
                "popularity_bucket": meta.popularity_bucket,
                "train_interactions": meta.train_interactions,
            })
        return out


def _rel(path, root) -> str | None:
    """Path relative to ``root`` when possible, else absolute (test fixtures)."""
    if path is None:
        return None
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
