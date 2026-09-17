#!/usr/bin/env python
"""Collect every finished run into the paper-ready result tables.

    python analysis/aggregate_results.py

Reads   results/runs/<run_id>/{metrics.json, config.yaml, test_ranking.npz}
Writes  results/tables/{overall,cold_start,long_tail,ablation}.csv
        results/tables/runs_index.csv

Nothing here invents a number: a run only appears once its ``metrics.json``
exists, and per-bucket numbers are recomputed from the stored per-user ranks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.evaluation.evaluator import EvalResult  # noqa: E402
from src.evaluation.slicing import load_ranking, sliced_metrics  # noqa: E402

RUNS_DIR = ROOT / "results" / "runs"
TABLES_DIR = ROOT / "results" / "tables"

_DATASET_CACHE: dict[str, ProcessedData] = {}


def _dataset(path: str) -> ProcessedData | None:
    if path not in _DATASET_CACHE:
        try:
            _DATASET_CACHE[path] = ProcessedData.load(ROOT / path)
        except Exception:
            _DATASET_CACHE[path] = None  # type: ignore[assignment]
    return _DATASET_CACHE[path]


def _modality_label(cfg: dict) -> str:
    m = (cfg.get("model") or {}).get("modalities")
    if not m:
        return "id" if (cfg.get("model") or {}).get("name") == "sasrec" else "-"
    order = ["id", "text", "image", "video"]
    return "+".join(k for k in order if m.get(k))


def load_runs() -> list[dict]:
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        mf, cf = d / "metrics.json", d / "config.yaml"
        sf = d / "train_summary.json"
        # train_summary.json is written by Trainer.fit() and is the marker of a
        # *finished* run; without it the run is still training (or crashed) and
        # must not contribute numbers to a table.
        if not (d.is_dir() and mf.exists() and cf.exists() and sf.exists()):
            continue
        metrics = json.loads(mf.read_text(encoding="utf-8"))
        cfg = yaml.safe_load(cf.read_text(encoding="utf-8")) or {}
        model_cfg = cfg.get("model") or {}
        train = dict(metrics.get("train") or {})
        # older runs stored the parameter count only in train_summary.json
        summary_path = d / "train_summary.json"
        if "num_parameters" not in train and summary_path.exists():
            try:
                train["num_parameters"] = json.loads(
                    summary_path.read_text(encoding="utf-8")
                ).get("num_parameters")
            except Exception:
                pass
        runs.append({
            "run_id": d.name,
            "dir": d,
            "tag": d.name.rsplit("_", 2)[0],
            "model": model_cfg.get("name", "?"),
            "fusion": model_cfg.get("fusion", "-"),
            "modalities": _modality_label(cfg),
            "id_dropout_prob": float(model_cfg.get("id_dropout_prob", 0.0) or 0.0),
            "item_dropout_prob": float(model_cfg.get("item_dropout_prob", 0.0) or 0.0),
            "modality_dropout_prob": float(model_cfg.get("modality_dropout_prob", 0.0) or 0.0),
            "hidden_size": model_cfg.get("hidden_size"),
            "layers": model_cfg.get("num_layers"),
            "processed_dir": (cfg.get("data") or {}).get("processed_dir", ""),
            "dataset": Path((cfg.get("data") or {}).get("processed_dir", "")).name or "-",
            "seed": metrics.get("seed", (cfg.get("training") or {}).get("seed")),
            "params": train.get("num_parameters"),
            "best_epoch": train.get("best_epoch"),
            "train_time_s": train.get("train_time_s"),
            "val": metrics.get("val", {}),
            "test": metrics.get("test", {}),
            "cold": metrics.get("cold"),
            "metrics": metrics,
        })
    return runs


def _row(r: dict) -> dict:
    t = r["test"]
    return {
        "tag": r["tag"],
        "dataset": r["dataset"],
        "model": r["model"],
        "fusion": r["fusion"],
        "modalities": r["modalities"],
        "id_dropout": r["id_dropout_prob"],
        "item_dropout": r["item_dropout_prob"],
        "seed": r["seed"],
        "Recall@5": t.get("Recall@5"),
        "Recall@10": t.get("Recall@10"),
        "Recall@20": t.get("Recall@20"),
        "NDCG@5": t.get("NDCG@5"),
        "NDCG@10": t.get("NDCG@10"),
        "NDCG@20": t.get("NDCG@20"),
        "MRR@20": t.get("MRR@20"),
        "Coverage@20": t.get("Coverage@20"),
        "item_dropout": r["item_dropout_prob"],
        "params": r["params"],
        "best_epoch": r["best_epoch"],
        "train_time_s": r["train_time_s"],
        "run_id": r["run_id"],
    }


def build_overall(runs: list[dict]) -> list[dict]:
    rows = [_row(r) for r in runs]
    order = {"popular": 0, "bpr": 1, "sasrec": 2, "mm_sasrec": 3}
    rows.sort(key=lambda x: (order.get(x["model"], 9), x.get("dataset", ""), x["tag"], str(x["seed"])))
    return rows


def build_ablation(runs: list[dict]) -> list[dict]:
    """Modality / fusion / regularisation matrix.

    Includes the ID-only models because `sasrec_itemdrop` is the control that
    separates "content helps" from "dropout regularises".
    """
    out = []
    for r in runs:
        if r["model"] not in ("sasrec", "mm_sasrec"):
            continue
        t = r["test"]
        mods = r["modalities"].split("+")
        out.append({
            "model": r["model"],
            "dataset": r["dataset"],
            "modalities": r["modalities"],
            "ID": int("id" in mods),
            "Text": int("text" in mods),
            "Image": int("image" in mods),
            "Video": int("video" in mods),
            "Fusion": r["fusion"],
            "item_dropout": r["item_dropout_prob"],
            "id_dropout": r["id_dropout_prob"],
            "Recall@20": t.get("Recall@20"),
            "NDCG@20": t.get("NDCG@20"),
            "Recall@10": t.get("Recall@10"),
            "NDCG@10": t.get("NDCG@10"),
            "params": r["params"],
            "seed": r["seed"],
            "tag": r["tag"],
            "run_id": r["run_id"],
        })
    out.sort(key=lambda x: (x.get("dataset", ""), -(x["ID"] + x["Text"] + x["Image"] + x["Video"]),
                            x["Fusion"], x["tag"]))
    return out


def build_cold(runs: list[dict]) -> list[dict]:
    out = []
    for r in runs:
        c = r["cold"]
        if not c:
            continue
        f = c.get("full_ranking", {})
        ro = c.get("restricted_to_cold", {})
        out.append({
            "tag": r["tag"],
            "model": r["model"],
            "dataset": r["dataset"],
            "fusion": r["fusion"],
            "modalities": r["modalities"],
            "id_dropout": r["id_dropout_prob"],
            "seed": r["seed"],
            "num_cold_items": c.get("num_cold_items"),
            "num_users": c.get("num_users_with_cold_target"),
            "Cold Recall@5": f.get("Recall@5"),
            "Cold Recall@10": f.get("Recall@10"),
            "Cold Recall@20": f.get("Recall@20"),
            "Cold NDCG@10": f.get("NDCG@10"),
            "Cold NDCG@20": f.get("NDCG@20"),
            "ColdOnly Recall@10": ro.get("Recall@10"),
            "ColdOnly Recall@20": ro.get("Recall@20"),
            "ColdOnly NDCG@20": ro.get("NDCG@20"),
            "run_id": r["run_id"],
        })
    out.sort(key=lambda x: (x.get("dataset", ""), x["model"], x["tag"]))
    return out


def build_long_tail(runs: list[dict]) -> list[dict]:
    """Per-popularity-bucket metrics recomputed from the stored per-user ranks."""
    out = []
    for r in runs:
        rk = r["dir"] / "test_ranking.npz"
        data = _dataset(r["processed_dir"]) if r["processed_dir"] else None
        if not rk.exists() or data is None:
            continue
        arr = load_ranking(rk)
        result = EvalResult(
            user_ids=arr["user_ids"],
            targets=arr["targets"],
            rank=arr["rank"],
            topk=arr["topk"],
            num_items=data.num_items,
            ks=(5, 10, 20),
        )
        slices = sliced_metrics(result, data)
        row = {"tag": r["tag"], "model": r["model"], "dataset": r["dataset"],
               "fusion": r["fusion"], "modalities": r["modalities"], "seed": r["seed"],
               "id_dropout": r["id_dropout_prob"],
               "item_dropout": r["item_dropout_prob"],
               "bucket_rule": data.stats["popularity_buckets"]["rule"]}
        for name in ("head", "middle", "tail"):
            row[f"{name}_users"] = slices[name]["num_users"]
            row[f"{name}_Recall@20"] = slices[name]["Recall@20"]
            row[f"{name}_NDCG@20"] = slices[name]["NDCG@20"]
        row["run_id"] = r["run_id"]
        out.append(row)
    out.sort(key=lambda x: (x.get("dataset", ""), x["model"], x["tag"]))
    return out


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        print(f"  {path.name}: (no runs yet)")
        return
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  {path.name}: {len(rows)} rows")


def main() -> None:
    global RUNS_DIR, TABLES_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--tables-dir", default=str(TABLES_DIR))
    args = ap.parse_args()

    RUNS_DIR = Path(args.runs_dir)
    TABLES_DIR = Path(args.tables_dir)

    runs = load_runs()

    # Two runs with the same tag AND the same seed are an orchestration
    # accident, not a second seed.  Keep the one that trained longer and say so
    # loudly rather than silently averaging them as if they were replicates.
    by_key: dict[tuple, dict] = {}
    dropped: list[str] = []
    for r in runs:
        key = (r["tag"], r["seed"])
        prev = by_key.get(key)
        if prev is None:
            by_key[key] = r
        elif (r["best_epoch"] or 0) > (prev["best_epoch"] or 0):
            dropped.append(prev["run_id"])
            by_key[key] = r
        else:
            dropped.append(r["run_id"])
    if dropped:
        print(f"WARNING: dropped {len(dropped)} duplicate (tag, seed) runs: {dropped}")
    runs = sorted(by_key.values(), key=lambda r: r["run_id"])

    print(f"found {len(runs)} finished runs")
    _write_csv(build_overall(runs), TABLES_DIR / "overall.csv")
    _write_csv(build_ablation(runs), TABLES_DIR / "ablation.csv")
    _write_csv(build_cold(runs), TABLES_DIR / "cold_start.csv")
    _write_csv(build_long_tail(runs), TABLES_DIR / "long_tail.csv")
    _write_csv(
        [{k: r[k] for k in ("run_id", "tag", "model", "fusion", "modalities", "seed",
                            "params", "best_epoch", "train_time_s")} for r in runs],
        TABLES_DIR / "runs_index.csv",
    )


if __name__ == "__main__":
    main()
