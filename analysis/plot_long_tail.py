#!/usr/bin/env python
"""Figure: long-tail performance and the popularity-vs-gain curve.

    python analysis/plot_long_tail.py

Reads results/tables/long_tail.csv.

* ``long_tail_performance.png`` — head / middle / tail Recall@20 per model
* ``popularity_vs_gain.png``    — Δ(MM-SASRec − SASRec) per popularity bucket,
  computed from the same runs.  No smoothing, no interpolation.
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
BUCKETS = ("head", "middle", "tail")


def _read(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _label(r: dict) -> str:
    if r["model"] == "sasrec":
        return "SASRec (ID)"
    name = f"MM-SASRec {r['modalities']} [{r['fusion']}]"
    if float(r.get("id_dropout") or 0) > 0:
        name += f" +IDdrop{float(r['id_dropout']):g}"
    return name


def main() -> None:
    rows = [r for r in _read(TABLES / "long_tail.csv") if r.get("tail_Recall@20")]
    if not rows:
        print("no long-tail runs yet -- skipping long-tail figures")
        return

    seen: dict[str, dict] = {}
    for r in rows:
        seen.setdefault(r["tag"], r)
    rows = list(seen.values())
    FIGURES.mkdir(parents=True, exist_ok=True)
    rule = rows[0].get("bucket_rule", "frequency_quantile")

    # ---- per-bucket bar chart -------------------------------------------
    rows_sorted = sorted(rows, key=lambda r: float(r["tail_Recall@20"]))
    labels = [_label(r) for r in rows_sorted]
    x = np.arange(len(BUCKETS))
    width = 0.8 / max(len(rows_sorted), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rows_sorted)))
    for i, (r, c) in enumerate(zip(rows_sorted, colors)):
        vals = [float(r[f"{b}_Recall@20"]) for b in BUCKETS]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=_label(r), color=c)
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in BUCKETS])
    ax.set_ylabel("Recall@20")
    ax.set_title(f"Long-tail performance by item popularity ({rule})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIGURES / "long_tail_performance.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    # ---- popularity vs multimodal gain -----------------------------------
    id_row = next((r for r in rows if r["model"] == "sasrec"), None)
    if id_row is None:
        print("no ID-only SASRec run on the same split -- skipping gain figure")
        return

    mm_rows = [r for r in rows if r["model"] == "mm_sasrec"]
    if not mm_rows:
        print("no MM-SASRec run on the same split -- skipping gain figure")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(BUCKETS))
    for ax, metric in zip(axes, ("Recall@20", "NDCG@20")):
        for r in mm_rows:
            gains = [float(r[f"{b}_{metric}"]) - float(id_row[f"{b}_{metric}"]) for b in BUCKETS]
            ax.plot(x, gains, marker="o", label=_label(r))
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in BUCKETS])
        ax.set_ylabel(f"Δ {metric} (MM − ID)")
        ax.set_title(f"Multimodal gain by popularity — {metric}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Does content help more for the long tail?", fontsize=12)
    fig.tight_layout()
    out = FIGURES / "popularity_vs_gain.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
