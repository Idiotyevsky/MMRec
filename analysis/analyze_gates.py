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
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
BUCKET_NAMES = {0: "tail", 1: "middle", 2: "head"}


def _gate_files() -> list[Path]:
    return sorted(RUNS.glob("*/gate_weights.npz"))


def main() -> None:
    files = _gate_files()
    if not files:
        print("no gate_weights.npz found -- run scripts/export_gates.py first")
        return

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    fig, axes = plt.subplots(1, len(files), figsize=(6 * len(files), 4.4), squeeze=False)

    for ax, path in zip(axes[0], files):
        d = np.load(path)
        run = path.parent.name
        item_ids = d["item_id"]
        valid = item_ids > 0
        mods = sorted(k[len("gate_"):] for k in d.files if k.startswith("gate_"))
        bucket = d["bucket"]
        is_cold = d["is_cold"].astype(bool)

        # ---- gate by popularity bucket ----
        xs, width = np.arange(3), 0.8 / max(len(mods), 1)
        for i, m in enumerate(mods):
            g = d[f"gate_{m}"]
            means = [float(g[valid & (bucket == b)].mean()) for b in (2, 1, 0)]
            ax.bar(xs + i * width - 0.4 + width / 2, means, width, label=m)
            for b, mean in zip(("head", "middle", "tail"), means):
                rows.append({"run": run, "bucket": b, "modality": m, "mean_gate": mean})
        if is_cold.any():
            for m in mods:
                rows.append({
                    "run": run, "bucket": "cold", "modality": m,
                    "mean_gate": float(d[f"gate_{m}"][is_cold].mean()),
                })
        ax.set_xticks(xs)
        ax.set_xticklabels(["Head", "Middle", "Tail"])
        ax.set_ylabel("mean gate weight")
        ax.set_title(run, fontsize=8)
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

    with open(TABLES / "gate_by_bucket.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run", "bucket", "modality", "mean_gate"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "mean_gate": round(r["mean_gate"], 6)})
    print(f"wrote {TABLES / 'gate_by_bucket.csv'} ({len(rows)} rows)")

    # ---- textual check of the tail-vs-head hypothesis --------------------
    print("\nID-vs-content gate by bucket (mean over runs):")
    for m in sorted({r["modality"] for r in rows}):
        line = [f"{m:>6s}"]
        for b in ("head", "middle", "tail", "cold"):
            vals = [r["mean_gate"] for r in rows if r["modality"] == m and r["bucket"] == b]
            line.append(f"{b}={np.mean(vals):.3f}" if vals else f"{b}=n/a")
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
