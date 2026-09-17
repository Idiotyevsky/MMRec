#!/usr/bin/env python
"""Render the result tables as GitHub-flavoured markdown for the README.

    python analysis/aggregate_results.py
    python analysis/make_readme_tables.py > results/tables/readme_tables.md

Everything is read from results/tables/*.csv.  A metric that has not been
measured yet renders as `TBD`; nothing is ever filled in by hand.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
TBD = "TBD"


def read(name: str) -> list[dict]:
    p = TABLES / name
    if not p.exists() or p.stat().st_size == 0:
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(v, digits: int = 4) -> str:
    if v in (None, "", "None"):
        return TBD
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def label(row: dict) -> str:
    ds = row.get("dataset", "")
    suffix = "" if ds in ("base", "") else f" [{ds}]"
    model = row.get("model", "")
    if model == "popular":
        return "Popular (train-freq)" + suffix
    if model == "bpr":
        return "BPR-MF" + suffix
    if model == "sasrec":
        base = "SASRec (ID-only)"
    else:
        base = f"MM-SASRec ({row.get('modalities', '?')}, {row.get('fusion', '?')})"
    d = row.get("id_dropout") or row.get("id_dropout_prob")
    if d and float(d) > 0:
        base += f" + ID-dropout {float(d):g}"
    it = row.get("item_dropout") or row.get("item_dropout_prob")
    if it and float(it) > 0:
        base += f" + item-dropout {float(it):g}"
    return base + suffix


DEFAULT_KEY = ("dataset", "model", "fusion", "modalities", "id_dropout", "item_dropout")


def group_seeds(rows: list[dict], key_fields=DEFAULT_KEY) -> list[dict]:
    """Collapse multi-seed runs into mean ± std strings."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(r.get(k, "") for k in key_fields), []).append(r)
    out = []
    for key, rs in groups.items():
        merged = dict(rs[0])
        for metric in ("Recall@5", "Recall@10", "Recall@20", "NDCG@5", "NDCG@10",
                       "NDCG@20", "MRR@20", "Coverage@20", "Cold Recall@10",
                       "Cold Recall@20", "Cold NDCG@10", "Cold NDCG@20",
                       "ColdOnly Recall@10", "ColdOnly Recall@20", "ColdOnly NDCG@20"):
            vals = [float(r[metric]) for r in rs if r.get(metric) not in (None, "", "None")]
            if not vals:
                continue
            merged[metric] = (f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"
                              if len(vals) > 1 else f"{vals[0]:.4f}")
            merged[metric + "__n"] = len(vals)
        merged["n_seeds"] = len(rs)
        out.append(merged)
    return out


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def overall_table() -> str:
    rows = group_seeds(read("overall.csv"))
    if not rows:
        return f"_No finished runs yet — {TBD}._"
    order = {"popular": 0, "bpr": 1, "sasrec": 2, "mm_sasrec": 3}
    rows.sort(key=lambda r: (order.get(r["model"], 9), str(r.get("dataset", "")), str(r.get("tag"))))
    body = [[label(r), fmt(r.get("Recall@10")), fmt(r.get("Recall@20")),
             fmt(r.get("NDCG@10")), fmt(r.get("NDCG@20")), fmt(r.get("MRR@20")),
             fmt(r.get("Coverage@20")), r.get("params") or "0", str(r.get("n_seeds", 1))]
            for r in rows]
    return md_table(["Model", "Recall@10", "Recall@20", "NDCG@10", "NDCG@20",
                     "MRR@20", "Coverage@20", "Params", "#seeds"], body)


def ablation_table() -> str:
    rows = group_seeds(read("ablation.csv"))
    if not rows:
        return f"_No ablation runs finished yet — {TBD}._"
    rows.sort(key=lambda r: (str(r.get("dataset", "")), r.get("model") != "sasrec",
                             int(r["ID"]), int(r["Text"]), int(r["Image"]), int(r["Video"]),
                             str(r.get("Fusion")), float(r.get("id_dropout") or 0)))
    def _pct(v):
        v = float(v or 0)
        return f"{v:g}" if v > 0 else "—"

    body = [[("ID-only" if r.get("model") == "sasrec" else "MM"),
             "✓" if int(r["ID"]) else "", "✓" if int(r["Text"]) else "",
             "✓" if int(r["Image"]) else "", "✓" if int(r["Video"]) else "",
             r.get("Fusion", "-"), _pct(r.get("id_dropout")), _pct(r.get("item_dropout")),
             fmt(r.get("Recall@10")), fmt(r.get("Recall@20")),
             fmt(r.get("NDCG@20")), r.get("params") or TBD]
            for r in rows]
    return md_table(["Kind", "ID", "Text", "Image", "Video", "Fusion", "ID-dropout",
                     "Item-dropout", "Recall@10", "Recall@20", "NDCG@20", "Params"], body)


def cold_table() -> str:
    rows = group_seeds(read("cold_start.csv"))
    if not rows:
        return f"_No cold-split runs finished yet — {TBD}._"
    body = [[label(r), fmt(r.get("Cold Recall@10")), fmt(r.get("Cold Recall@20")),
             fmt(r.get("Cold NDCG@10")), fmt(r.get("Cold NDCG@20")),
             fmt(r.get("ColdOnly Recall@10")), fmt(r.get("ColdOnly Recall@20")),
             str(r.get("num_users") or TBD)]
            for r in rows]
    return md_table(["Model", "Cold Recall@10", "Cold Recall@20", "Cold NDCG@10",
                     "Cold NDCG@20", "ColdOnly Recall@10", "ColdOnly Recall@20",
                     "#users with cold target"], body)


def long_tail_table() -> str:
    rows = group_seeds(read("long_tail.csv"))
    if not rows:
        return f"_No long-tail runs finished yet — {TBD}._"
    body = [[label(r), fmt(r.get("head_Recall@20")), fmt(r.get("middle_Recall@20")),
             fmt(r.get("tail_Recall@20")), fmt(r.get("head_NDCG@20")),
             fmt(r.get("middle_NDCG@20")), fmt(r.get("tail_NDCG@20"))]
            for r in rows]
    rule = rows[0].get("bucket_rule", "?")
    return (f"_Popularity rule: `{rule}` (training interactions only)._\n\n"
            + md_table(["Model", "Head Recall@20", "Middle Recall@20", "Tail Recall@20",
                        "Head NDCG@20", "Middle NDCG@20", "Tail NDCG@20"], body))


def _val(row: dict, bucket: str, metric: str = "Recall@20") -> float | None:
    try:
        return float(row[f"{bucket}_{metric}"])
    except (KeyError, TypeError, ValueError):
        return None


def gain_table() -> str:
    """Δ per popularity bucket, against both relevant reference points.

    ``Δ vs SASRec`` answers "does multimodal help?".
    ``Δ vs SASRec+item-dropout`` answers "is it the *content* or just the extra
    regularisation?".  Both are needed to interpret the ID-dropout result.
    """
    rows = group_seeds(read("long_tail.csv"))
    if not rows:
        return f"_No long-tail runs yet — {TBD}._"
    base_rows = [r for r in rows if r.get("dataset", "base") == "base"]
    id_row = next((r for r in base_rows if r.get("model") == "sasrec"
                   and float(r.get("item_dropout") or 0) == 0), None)
    reg_row = next((r for r in base_rows if r.get("model") == "sasrec"
                    and float(r.get("item_dropout") or 0) > 0), None)

    body = []
    for r in base_rows:
        if r.get("model") != "mm_sasrec":
            continue
        if id_row is None:
            body.append([label(r)] + [TBD] * 6)
            continue
        g1 = [(_val(r, b) - _val(id_row, b)) for b in ("head", "middle", "tail")]
        if reg_row is not None:
            g2 = [(_val(r, b) - _val(reg_row, b)) for b in ("head", "middle", "tail")]
        else:
            g2 = [None] * 3
        body.append([label(r)]
                    + [f"{v:+.4f}" if v is not None else TBD for v in g1]
                    + [f"{v:+.4f}" if v is not None else TBD for v in g2])
    if not body:
        return f"_No multimodal run on the base split yet — {TBD}._"

    if reg_row is None:
        body.append([f"_{TBD}: no item-dropout control run finished yet_"]
                    + [TBD] * 6)
    return md_table(
        ["Model", "ΔHead vs SASRec", "ΔMiddle vs SASRec", "ΔTail vs SASRec",
         "ΔHead vs ID+item-drop", "ΔMiddle vs ID+item-drop", "ΔTail vs ID+item-drop"],
        body,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sections = {
        "OVERALL": overall_table(),
        "ABLATION": ablation_table(),
        "COLD": cold_table(),
        "LONGTAIL": long_tail_table(),
        "GAIN": gain_table(),
    }
    text = "\n\n".join(f"<!-- {k} -->\n{v}" for k, v in sections.items())
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
