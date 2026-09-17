#!/usr/bin/env python
"""Figure: cold-item performance.

    python analysis/plot_cold_start.py

Reads results/tables/cold_start.csv, writes results/figures/cold_start_performance.png.
Both panels come from real ranking passes: the left panel is the full-catalogue
ranking restricted to users with a cold target, the right panel ranks only the
cold catalogue.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


def _read(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _label(r: dict) -> str:
    name = "SASRec" if r["model"] == "sasrec" else f"MM-SASRec {r['modalities']}"
    if r["model"] == "mm_sasrec":
        name += f" [{r['fusion']}]"
    if float(r.get("id_dropout") or 0) > 0:
        name += f" +IDdrop{float(r['id_dropout']):g}"
    return name


def main() -> None:
    rows = [r for r in _read(TABLES / "cold_start.csv") if r.get("Cold Recall@20")]
    if not rows:
        print("no cold-split runs yet -- skipping cold figure")
        return

    seen: dict[str, dict] = {}
    for r in rows:
        seen.setdefault(r["tag"], r)
    rows = sorted(seen.values(), key=lambda r: float(r["Cold Recall@20"]))

    labels = [_label(r) for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.2, 0.5 * len(rows) + 1.8)))

    panels = [
        (axes[0], [float(r["Cold Recall@20"]) for r in rows], "Cold Recall@20 (full ranking)"),
        (axes[1], [float(r["ColdOnly Recall@20"]) for r in rows], "Cold Recall@20 (cold-only candidates)"),
    ]
    for ax, values, title in panels:
        values = np.array(values)
        bars = ax.barh(y, values, color="#a5563b")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        for b, v in zip(bars, values):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.4f}", va="center", fontsize=8)
        ax.margins(x=0.22)

    n_cold = rows[0].get("num_cold_items")
    fig.suptitle(f"Cold-item evaluation (simulated split, {n_cold} cold items)", fontsize=12)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "cold_start_performance.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
