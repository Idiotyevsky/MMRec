#!/usr/bin/env python
"""Gate interpretability analysis.

    python scripts/export_gates.py --run-dir results/runs/<mm run>   # once per run
    python analysis/analyze_gates.py

Answers, from real exported weights:

* does the ID gate dominate head items while content takes over for the tail?
* how concentrated is the gate distribution?
* how do gates behave for items whose content modality is missing?

Writes ``results/figures/modality_gate_distribution.png`` and
``results/tables/gate_by_bucket.csv``.  The questions above are *tested*, not
assumed: if the data does not show the expected trend it is reported as-is.

Provenance gate (same rule as ``analysis/aggregate_results.py``): a run only
contributes if it carries a ``run_manifest.json``, i.e. it was produced by a
documented code version.  Manifest-less runs (the pre-autoregressive-fix batch)
are listed and skipped, never silently averaged in.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
BUCKET_NAMES = {0: "tail", 1: "middle", 2: "head"}


def _manifest(run_dir: Path) -> dict | None:
    try:
        return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


MODALITY_ORDER = ("id", "text", "image", "video")


def _variant(run_dir: Path) -> str:
    """Grouping key for the gate table, read from the run's own config copy.

    The modality set and the dataset are part of the name: an ablation run that
    drops ``video``, or a cold-split run, gates different items through a
    different model, and merging it into the default rows would average
    incomparable numbers while the table claimed they came from one variant.
    """
    try:
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    except OSError:
        return run_dir.name
    m = cfg.get("model", {}) or {}
    name = str(m.get("name", "?"))
    if name == "sasrec":
        return "sasrec" if float(m.get("id_dropout_prob") or m.get("item_dropout_prob") or 0) == 0 else "sasrec_reg"
    active = "+".join(k for k in MODALITY_ORDER if (m.get("modalities") or {}).get(k))
    variant = f"{name}_iddrop" if float(m.get("id_dropout_prob") or 0) > 0 else name
    if active:
        variant += f"({active})"
    dataset = Path(str((cfg.get("data") or {}).get("processed_dir", ""))).name
    if dataset and dataset != "base":
        variant += f"@{dataset}"
    return variant


def _gate_files() -> tuple[list[Path], list[str]]:
    """Gate exports of documented runs; manifest-less ones are reported, not used."""
    files, skipped = [], []
    for p in sorted(RUNS.glob("*/gate_weights.npz")):
        if _manifest(p.parent) is None:
            skipped.append(p.parent.name)
        else:
            files.append(p)
    return files, skipped


def main() -> None:
    files, skipped = _gate_files()
    if skipped:
        print(f"skipping {len(skipped)} manifest-less run(s): {', '.join(skipped)}")
    if not files:
        print("no gate_weights.npz found -- run scripts/export_gates.py first")
        # a stale table from an older code version must not survive as if current
        TABLES.mkdir(parents=True, exist_ok=True)
        with open(TABLES / "gate_by_bucket.csv", "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["model", "run", "bucket", "modality",
                                          "mean_gate"]).writeheader()
        return

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    by_variant: dict[str, list[Path]] = {}
    for p in files:
        by_variant.setdefault(_variant(p.parent), []).append(p)

    rows: list[dict] = []
    fig, axes = plt.subplots(1, len(by_variant), figsize=(6 * len(by_variant), 4.4), squeeze=False)

    for ax, (variant, paths) in zip(axes[0], sorted(by_variant.items())):
        mods: list[str] = []
        for path in paths:
            d = np.load(path)
            run = path.parent.name
            valid = d["item_id"] > 0
            mods = sorted(k[len("gate_"):] for k in d.files if k.startswith("gate_"))
            bucket = d["bucket"]
            is_cold = d["is_cold"].astype(bool)
            for m in mods:
                g = d[f"gate_{m}"]
                for b_idx, b in zip((2, 1, 0), ("head", "middle", "tail")):
                    rows.append({"model": variant, "run": run, "bucket": b,
                                 "modality": m,
                                 "mean_gate": float(g[valid & (bucket == b_idx)].mean())})
                if is_cold.any():
                    rows.append({"model": variant, "run": run, "bucket": "cold",
                                 "modality": m,
                                 "mean_gate": float(g[is_cold].mean())})
        # mean over the seeds of this variant
        xs, width = np.arange(3), 0.8 / max(len(mods), 1)
        for i, m in enumerate(mods):
            means = [float(np.mean([r["mean_gate"] for r in rows
                                    if r["model"] == variant and r["modality"] == m
                                    and r["bucket"] == b]))
                     for b in ("head", "middle", "tail")]
            ax.bar(xs + i * width - 0.4 + width / 2, means, width, label=m)
        ax.set_xticks(xs)
        ax.set_xticklabels(["Head", "Middle", "Tail"])
        ax.set_ylabel("mean gate weight")
        ax.set_title(f"{variant} (n={len(paths)} run{'s' if len(paths) > 1 else ''})", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Average fusion gate weight by item popularity", fontsize=12)
    fig.tight_layout()
    out = FIGURES / "modality_gate_distribution.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    # ---- gate distribution (first run) ----------------------------------
    d = np.load(files[0])
    item_ids = d["item_id"]
    valid = item_ids > 0
    mods = sorted(k[len("gate_"):] for k in d.files if k.startswith("gate_"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in mods:
        ax.hist(d[f"gate_{m}"][valid], bins=50, alpha=0.55, label=m, density=True)
    ax.set_xlabel("gate weight")
    ax.set_ylabel("density")
    ax.set_title(f"Gate weight distribution — {files[0].parent.name}", fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "gate_weight_histogram.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    fields = ["model", "run", "bucket", "modality", "mean_gate"]
    with open(TABLES / "gate_by_bucket.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "mean_gate": round(r["mean_gate"], 6)})
    print(f"wrote {TABLES / 'gate_by_bucket.csv'} ({len(rows)} rows)")

    # ---- textual check of the tail-vs-head hypothesis --------------------
    print("\nID-vs-content gate by bucket (mean over runs, per variant):")
    for variant in sorted({r["model"] for r in rows}):
        for m in sorted({r["modality"] for r in rows}):
            line = [f"{variant:>14s} {m:>6s}"]
            for b in ("head", "middle", "tail", "cold"):
                vals = [r["mean_gate"] for r in rows
                        if r["model"] == variant and r["modality"] == m and r["bucket"] == b]
                line.append(f"{b}={np.mean(vals):.3f}" if vals else f"{b}=n/a")
            print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
