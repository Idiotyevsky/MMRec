#!/usr/bin/env python
"""Figure: overall Recall@20 / NDCG@20 per model.

    python analysis/plot_overall.py

Reads results/tables/overall.csv, writes results/figures/overall_performance.png.
No number is hard-coded; if the table is empty the figure is skipped.
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

LABELS = {
    "popular": "Popular",
    "bpr": "BPR-MF",
    "sasrec": "SASRec (ID)",
}
FUSION_SUFFIX = {"concat": " (concat)", "gated": " (gated)"}


def _read(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def label_for(row: dict) -> str:
    if row["model"] in LABELS:
        return LABELS[row["model"]]
    base = f"MM-SASRec {row['modalities']}"
    base += FUSION_SUFFIX.get(row["fusion"], "")
    if float(row.get("id_dropout") or 0) > 0:
        base += f" +IDdrop{float(row['id_dropout']):g}"
    return base


#: headline models shown by default (the full matrix lives in the ablation table)
HEADLINE_TAGS = {
    "popular", "bpr", "sasrec", "sasrec_itemdrop",
    "mm_concat", "mm_gated", "mm_gated_iddrop",
}


def main() -> None:
    import sys

    show_all = "--all" in sys.argv
    rows = [r for r in _read(TABLES / "overall.csv") if r.get("Recall@20")]
    if not show_all:
        rows = [r for r in rows if r.get("dataset", "base") == "base" and r["tag"] in HEADLINE_TAGS]
    if not rows:
        print("no finished runs yet -- skipping overall figure")
        return

    # collapse multi-seed runs into one bar each (first seed is representative;
    # the tables carry the mean ± std)
    seen: dict[tuple, dict] = {}
    for r in rows:
        seen.setdefault((r.get("dataset"), r["tag"]), r)
    rows = sorted(seen.values(), key=lambda r: float(r["Recall@20"]))

    labels = [label_for(r) for r in rows]
    recall = np.array([float(r["Recall@20"]) for r in rows])
    ndcg = np.array([float(r["NDCG@20"]) for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.2, 0.42 * len(rows) + 1.6)))
    y = np.arange(len(rows))
    for ax, values, title in ((axes[0], recall, "Recall@20"), (axes[1], ndcg, "NDCG@20")):
        bars = ax.barh(y, values, color="#3b6ea5")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(title)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        for b, v in zip(bars, values):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.4f}", va="center", fontsize=8)
        ax.margins(x=0.18)

    fig.suptitle("MicroLens-100K — full-ranking test performance", fontsize=12)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "overall_performance.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out} ({len(rows)} models)")


if __name__ == "__main__":
    main()
