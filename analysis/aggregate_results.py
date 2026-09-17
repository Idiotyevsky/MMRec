#!/usr/bin/env python
"""Collect every finished run into the paper-ready result tables.

    python analysis/aggregate_results.py

Reads   results/runs/<run_id>/{metrics.json, config.yaml, train_summary.json,
                                test_ranking.npz, run_manifest.json}
Writes  results/tables/{overall,cold_start,long_tail,ablation}.csv
        results/tables/runs_index.csv

Nothing here invents a number: a run only appears once its ``metrics.json``
exists, and per-bucket numbers are recomputed from the stored per-user ranks.

Two gates protect the tables:

* runs without a ``run_manifest.json`` (i.e. produced before provenance was
  recorded, or crashed before finishing) are **legacy** and are skipped unless
  ``--include-legacy`` is given;
* inside one experiment group every surviving run must agree on git SHA, dataset
  fingerprint, config (modulo seed) and parameter count, or the odd ones out are
  excluded -- ``--allow-mixed`` downgrades that to a warning for debugging only.

Legacy runs are not deleted: they were moved to
``results/legacy_pre_autoregressive_fix/`` after the training-objective fix.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.evaluation.evaluator import EvalResult  # noqa: E402
from src.evaluation.slicing import load_ranking, sliced_metrics  # noqa: E402
from src.utils.provenance import config_hash, load_manifest  # noqa: E402

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


def _manifest(run_dir: Path) -> dict:
    return load_manifest(run_dir) or {}


def _config_core_hash(cfg: dict) -> str:
    """Config hash with ``training.seed`` removed.

    Replicates of one experiment are *supposed* to differ in exactly this one
    key; anything else that differs means the runs are not the same experiment.
    """
    core = json.loads(json.dumps(cfg, sort_keys=True, default=str))
    training = dict(core.get("training") or {})
    training.pop("seed", None)
    core["training"] = training
    return config_hash(core)


def load_runs(runs_dir: Path = RUNS_DIR) -> list[dict]:
    runs = []
    for d in sorted(runs_dir.iterdir()):
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
        man = _manifest(d)
        runs.append({
            "run_id": d.name,
            "dir": d,
            "tag": d.name.rsplit("_", 2)[0],
            "manifest": man,
            # No manifest means the code version that produced this run is
            # unknown -- it can never be pooled with a documented run.
            "provenance": bool(man),
            "git_sha": man.get("git_sha"),
            "git_dirty": man.get("git_dirty"),
            "dataset_hash": man.get("dataset_hash"),
            "config_hash": man.get("config_hash"),
            "config_core_hash": _config_core_hash(cfg),
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
        "params": r["params"],
        "best_epoch": r["best_epoch"],
        "train_time_s": r["train_time_s"],
        "run_id": r["run_id"],
    }


def build_runs_index(runs: list[dict]) -> list[dict]:
    """One row per run with the provenance a reviewer needs to trust it."""
    out = []
    for r in runs:
        out.append({
            "run_id": r["run_id"],
            "tag": r["tag"],
            "model": r["model"],
            "fusion": r["fusion"],
            "modalities": r["modalities"],
            "seed": r["seed"],
            "params": r["params"],
            "git_sha": r["git_sha"],
            "git_dirty": r["git_dirty"],
            "dataset_hash": r["dataset_hash"],
            "config_hash": r["config_hash"],
            "best_epoch": r["best_epoch"],
            "train_time_s": r["train_time_s"],
            "provenance": r["provenance"],
        })
    return out


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
            "modality_dropout": r["modality_dropout_prob"],
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


def _group_key(r: dict) -> str:
    """The experiment a run belongs to: its tag without the ``_s<seed>`` suffix.

    ``sasrec``, ``sasrec_s2026`` and ``sasrec_s3407`` are three seeds of one
    experiment, so drift *between* them -- e.g. the 128-parameter difference once
    seen between two SASRec seeds -- is exactly what the compatibility check has
    to catch.  The suffix only counts as a seed when it equals the run's seed, so
    a tag that legitimately ends in ``_s123`` is not split by accident.
    """
    m = re.match(r"^(.*)_s(\d+)$", r["tag"])
    if m and r["seed"] is not None and int(m.group(2)) == int(r["seed"]):
        return m.group(1)
    return r["tag"]


def _signature(r: dict) -> tuple:
    """Everything that must agree for two runs to be comparable at all."""
    return (
        r["git_sha"] if r["provenance"] else "<no-manifest>",
        r["dataset_hash"],
        r["config_core_hash"],
        r["params"],
    )


def _drop_duplicate_seeds(runs: list[dict]) -> tuple[list[dict], list[str]]:
    """One run per (tag, seed): two is an accident, not a replicate."""
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
    return sorted(by_key.values(), key=lambda r: r["run_id"]), dropped


def _enforce_compatibility(runs: list[dict], allow_mixed: bool) -> list[dict]:
    """Never pool runs that were not produced the same way.

    Inside one experiment every run must share the code version (git SHA), the
    dataset fingerprint, the config (modulo the seed) and the parameter count.
    That last one is not theoretical: two SASRec runs once differed by 128
    parameters because one config allocated an optional regulariser, and a seed
    aggregate would have treated them as replicates of one experiment.

    The largest mutually compatible subset of each group is kept; anything that
    disagrees is reported and excluded.  ``--allow-mixed`` downgrades the
    exclusion to a warning (for inspecting old data, never for a result table).
    """
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(_group_key(r), []).append(r)

    keep: list[dict] = []
    for tag, group in sorted(groups.items()):
        buckets: dict[tuple, list[dict]] = {}
        for r in group:
            buckets.setdefault(_signature(r), []).append(r)
        if len(buckets) == 1:
            keep.extend(group)
            continue

        ranked = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
        best_sig, best = ranked[0]
        print(
            f"WARNING: experiment group {tag!r} mixes {len(buckets)} incompatible "
            f"provenances; keeping {len(best)} run(s) with "
            f"git={str(best_sig[0])[:8]} dataset={str(best_sig[1])[:8]} "
            f"params={best_sig[3]}"
        )
        for sig, rs in ranked[1:]:
            for r in rs:
                print(
                    f"    excluded {r['run_id']}: git={str(sig[0])[:8]} "
                    f"dataset={str(sig[1])[:8]} params={sig[3]} "
                    f"(seed {r['seed']}, best_epoch {r['best_epoch']})"
                )
        if allow_mixed:
            print("    --allow-mixed: keeping them anyway (NOT valid for a result table)")
            keep.extend(group)
        else:
            keep.extend(best)
    return keep


def print_group_summary(runs: list[dict], metrics=("Recall@20", "NDCG@20")) -> None:
    """Mean ± std over seeds, printed (not written) and only for real replicates.

    Every run printed here already passed the compatibility gate above, so a
    single seed is shown as a bare value rather than as a std of zero.
    """
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(_group_key(r), []).append(r)
    if not groups:
        return
    print("\nmean ± std over seeds (compatible runs only):")
    for tag, group in sorted(groups.items()):
        seeds = sorted({r["seed"] for r in group})
        line = f"  {tag:<26s} n={len(group)} seeds={seeds}"
        for m in metrics:
            vals = [r["test"][m] for r in group if r["test"].get(m) is not None]
            if not vals:
                continue
            if len(vals) == 1:
                line += f"  {m}={vals[0]:.4f}"
            else:
                line += f"  {m}={np.mean(vals):.4f}±{np.std(vals):.4f}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs-dir", default=str(RUNS_DIR))
    ap.add_argument("--tables-dir", default=str(TABLES_DIR))
    ap.add_argument("--include-legacy", action="store_true",
                    help="also load runs without run_manifest.json (never a valid result table)")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="keep runs whose provenance disagrees (debugging only)")
    args = ap.parse_args()

    runs = load_runs(Path(args.runs_dir))
    print(f"found {len(runs)} finished run(s) in {args.runs_dir}")

    legacy = [r for r in runs if not r["provenance"]]
    if legacy and not args.include_legacy:
        print(f"skipping {len(legacy)} run(s) without run_manifest.json "
              f"(pre-fix code, provenance unknown):")
        for r in legacy:
            print(f"    {r['run_id']}")
        print("    --include-legacy to inspect them; they cannot enter a result table")
        runs = [r for r in runs if r["provenance"]]
    if args.include_legacy and legacy:
        print(f"WARNING: --include-legacy kept {len(legacy)} run(s) of unknown provenance")

    runs, dropped = _drop_duplicate_seeds(runs)
    for rid in dropped:
        print(f"WARNING: dropped duplicate (tag, seed) run {rid}")

    runs = _enforce_compatibility(runs, args.allow_mixed)
    print(f"aggregating {len(runs)} run(s)")

    tables = Path(args.tables_dir)
    _write_csv(build_overall(runs), tables / "overall.csv")
    _write_csv(build_ablation(runs), tables / "ablation.csv")
    _write_csv(build_cold(runs), tables / "cold_start.csv")
    _write_csv(build_long_tail(runs), tables / "long_tail.csv")
    _write_csv(build_runs_index(runs), tables / "runs_index.csv")
    print_group_summary(runs)


if __name__ == "__main__":
    main()
