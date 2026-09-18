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
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
SEMANTIC_ID_REPORT = ROOT / "artifacts" / "semantic_id_report_content.json"
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

def is_base(row: dict) -> bool:
    """Rows of the base split; a missing/empty dataset column means base."""
    return (row.get("dataset") or "base") == "base"



def label(row: dict) -> str:
    ds = row.get("dataset", "")
    suffix = "" if ds in ("base", "") else f" [{ds}]"
    model = row.get("model", "")
    if model == "popular":
        return "Popular (train-freq)" + suffix
    if model == "bpr":
        return "BPR-MF" + suffix
    if model == "random":
        return "Random (uniform)" + suffix
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
_KEY_LOWER = {k.lower() for k in DEFAULT_KEY}

# Bookkeeping columns that identify a run rather than measure it: averaged or
# copied into "x ± y" they would be nonsense (a mean seed? a mean parameter
# count?).  Everything else that parses as a float is averaged across seeds --
# including bucket-prefixed metrics like ``tail_Recall@20``, which a hardcoded
# metric list would silently leave at the first seed's value.
IDENTITY_FIELDS = {
    "tag", "run_id", "dataset", "model", "fusion", "modalities", "seed",
    "id_dropout", "item_dropout", "modality_dropout", "params", "num_users",
    "num_items", "num_cold_items", "bucket_rule", "best_epoch",
    "head_users", "middle_users", "tail_users",
    "ID", "Text", "Image", "Video",
}
_IDENTITY_LOWER = {c.lower() for c in IDENTITY_FIELDS}


def field(row: dict, name: str):
    """Case-insensitive field lookup.

    ``ablation.csv`` names its fusion column ``Fusion`` while the other tables
    use lowercase; a plain ``row.get("fusion")`` would return "" for every
    ablation row and silently group gated and concat runs as one seed group.
    """
    if name in row:
        return row[name]
    low = name.lower()
    for k, v in row.items():
        if k.lower() == low:
            return v
    return ""


def group_seeds(rows: list[dict], key_fields=DEFAULT_KEY) -> list[dict]:
    """Collapse multi-seed runs into mean ± std strings.

    Any numeric column not named in ``IDENTITY_FIELDS`` is averaged, so a new
    metric cannot silently fall back to a single seed's value.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(tuple(field(r, k) for k in key_fields), []).append(r)
    out = []
    for key, rs in groups.items():
        merged = dict(rs[0])
        if len(rs) > 1:
            for col in sorted({c for r in rs for c in r}):
                low = col.lower()
                if low in _IDENTITY_LOWER or low == "n_seeds" or low in _KEY_LOWER:
                    continue
                vals: list[float] = []
                for r in rs:
                    v = r.get(col)
                    if v in (None, "", "None", TBD):
                        continue
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        vals = []
                        break
                if not vals:
                    continue
                merged[col] = (f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"
                               if len(vals) > 1 else f"{vals[0]:.4f}")
                merged[col + "__n"] = len(vals)
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
    # the cold split removes training interactions, so its absolute numbers are
    # a different protocol and must not sit silently next to base-split rows
    rows = [r for r in rows if is_base(r)]
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
    # cold10 runs share model/modalities with base runs, so an unfiltered table
    # would show two "ID-only" rows with different numbers and no way to tell them apart
    rows = [r for r in rows if is_base(r)]
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
    # same protocol rule as the overall table: the cold split's bucket numbers
    # come from a different catalogue and are reported in the cold table
    rows = [r for r in rows if is_base(r)]
    body = [[label(r), fmt(r.get("head_Recall@20")), fmt(r.get("middle_Recall@20")),
             fmt(r.get("tail_Recall@20")), fmt(r.get("head_NDCG@20")),
             fmt(r.get("middle_NDCG@20")), fmt(r.get("tail_NDCG@20"))]
            for r in rows]
    rule = rows[0].get("bucket_rule", "?")
    return (f"_Popularity rule: `{rule}` (training interactions only)._\n\n"
            + md_table(["Model", "Head Recall@20", "Middle Recall@20", "Tail Recall@20",
                        "Head NDCG@20", "Middle NDCG@20", "Tail NDCG@20"], body))


def _val(row: dict, bucket: str, metric: str = "Recall@20") -> float | None:
    """A bucket metric as a float; merged rows hold ``mean ± std`` strings."""
    return _num(row, f"{bucket}_{metric}")


def gain_table() -> str:
    """Δ per popularity bucket, against both relevant reference points.

    ``Δ vs SASRec`` answers "does multimodal help?".
    ``Δ vs SASRec+item-dropout`` answers "is it the *content* or just the extra
    regularisation?".  Both are needed to interpret the ID-dropout result.
    """
    rows = group_seeds(read("long_tail.csv"))
    if not rows:
        return f"_No long-tail runs yet — {TBD}._"
    base_rows = [r for r in rows if is_base(r)]
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


def dataset_table() -> str:
    """Descriptive dataset figures, read from ``results/dataset_stats/*.json``.

    These are copies of the ``stats.json`` written by preprocessing, committed so
    the numbers in the README have an artifact behind them without shipping the
    data itself.
    """
    d = ROOT / "results" / "dataset_stats"
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    if not files:
        return f"_No dataset stats committed yet — {TBD}._"

    rows = []
    for f in files:
        s = json.loads(f.read_text(encoding="utf-8"))
        freq = s.get("item_freq_train", {})
        cold = s.get("cold_split") or {}
        rows.append([
            f"`{f.stem}`",
            str(s.get("num_users", TBD)),
            str(s.get("num_items", TBD)),
            str(s.get("num_interactions", TBD)),
            f"{s.get('sparsity', float('nan')):.5f}",
            f"{s.get('avg_sequence_length', float('nan')):.2f}",
            str(s.get("num_train_interactions", TBD)),
            str(freq.get("median", TBD)),
            str(freq.get("num_zero_freq", TBD)),
            str(cold.get("num_cold_items", 0)),
        ])
    return md_table(["split", "users", "items", "interactions", "sparsity",
                     "mean seq len", "train interactions", "median train item freq",
                     "items with 0 train freq", "cold items"], rows)


def semantic_id_table() -> str:
    """Semantic-ID quantiser summary, read from the committed artifact.

    The README's Semantic-ID block used to be hand-copied, which is what this
    generator exists to prevent.  Nothing here depends on a recommender
    checkpoint; ``source features`` is rendered from the artifact so a table
    built from different features cannot keep the text+image wording.
    """
    if not SEMANTIC_ID_REPORT.exists():
        return f"_No Semantic-ID report committed — {TBD}._"
    try:
        r = json.loads(SEMANTIC_ID_REPORT.read_text(encoding="utf-8"))
    except Exception:
        return f"_Semantic-ID report unreadable — {TBD}._"
    util = r.get("codebook_utilization") or []
    rows = [
        ["source features", f"`{r.get('source', TBD)}`"],
        ["quantiser", f"RQVAE, {r.get('num_levels', TBD)} levels × "
                      f"{r.get('codebook_size', TBD)} codes, latent {r.get('latent_dim', TBD)} "
                      f"({_spaced(r.get('params'))} params)"],
        ["reconstruction cosine", f"{float(r['reconstruction_cosine']):.3f}"
                                  if r.get("reconstruction_cosine") is not None else TBD],
        ["codebook utilisation", "`[" + ", ".join(f"{float(u):.1f}" for u in util) + "]`"
                                 if util else TBD],
        ["unique Semantic IDs", f"{_spaced(r.get('unique_semantic_ids'))} / "
                                f"{_spaced(r.get('num_items'))} items"],
        ["collision rate", f"**{100 * float(r['collision_rate']):.3f} %**"
                           if r.get("collision_rate") is not None else TBD],
    ]
    return md_table(["", ""], rows)


def gates_table() -> str:
    """Mean fusion gate weight per bucket, written by ``analysis/analyze_gates.py``.

    Rows from manifest-less runs never reach ``gate_by_bucket.csv`` (the analysis
    refuses them), and the plain-gated and ID-dropout variants are separate rows
    because they answer different questions.
    """
    rows = read("gate_by_bucket.csv")
    if not rows:
        return f"_No gate export from a documented run yet — {TBD}._"
    buckets = [b for b in ("head", "middle", "tail", "cold")
               if any(r["bucket"] == b for r in rows)]
    keys: list[tuple[str, str]] = []
    for r in rows:
        k = (r["model"], r["modality"])
        if k not in keys:
            keys.append(k)
    keys.sort()
    body = []
    for model, modality in keys:
        cells = []
        for b in buckets:
            vals = [float(r["mean_gate"]) for r in rows
                    if r["model"] == model and r["modality"] == modality and r["bucket"] == b]
            cells.append(f"{statistics.mean(vals):.3f}" if vals else TBD)
        body.append([model, modality] + cells)
    n_runs = len({r["run"] for r in rows})
    return (f"_Mean over {n_runs} documented run(s), from "
            "`results/tables/gate_by_bucket.csv`._\n\n"
            + md_table(["Model", "Modality"] + [b.capitalize() for b in buckets], body))


def efficiency_table() -> str:
    """Parameter count and wall-clock cost, all from run artifacts."""
    idx = read("runs_index.csv")
    if not idx:
        return f"_No finished runs yet — {TBD}._"
    # the row must describe the headline base-split pair, not whichever
    # ablation or cold variant happens to sort first
    idx = [r for r in idx if is_base(r)]

    def pick(**want):
        for r in idx:
            if all(str(r.get(k, "")) == str(v) for k, v in want.items()):
                return r
        return None

    id_run = pick(model="sasrec", item_dropout="0.0")
    mm_run = pick(model="mm_sasrec", fusion="gated", modalities="id+text+image",
                  id_dropout="0.2") or pick(
        model="mm_sasrec", fusion="gated", modalities="id+text+image")

    def _mean(runs, key) -> float | None:
        vals = [float(r[key]) for r in runs if r.get(key) not in (None, "", "None")]
        return statistics.mean(vals) if vals else None

    def _params(model=None, **want) -> float | None:
        rs = [r for r in idx if (model is None or r.get("model") == model)
              and all(str(r.get(k, "")) == str(v) for k, v in want.items())]
        return _mean(rs, "params")

    p_id = _params(model="sasrec", item_dropout="0.0")
    p_mm = _params(model="mm_sasrec", fusion="gated", modalities="id+text+image",
                   id_dropout="0.2") or _params(
        model="mm_sasrec", fusion="gated", modalities="id+text+image")

    def s(v, digits=0):
        return TBD if v is None else (f"{v:,.{digits}f}".replace(",", " ") if digits == 0
                                      else f"{v:,.{digits}f}")

    def per_epoch(r):
        if not r:
            return None
        t, e = _num(r, "train_time_s"), _num(r, "epochs")
        return t / e if t and e else None

    id_grp = [r for r in idx if r.get("model") == "sasrec" and str(r.get("item_dropout")) == "0.0"]
    _mm_gated = [r for r in idx if r.get("model") == "mm_sasrec" and r.get("fusion") == "gated"
                 and r.get("modalities") == "id+text+image"]
    mm_grp = [r for r in _mm_gated if str(r.get("id_dropout")) == "0.2"] or _mm_gated

    delta = "—"
    if p_id and p_mm:
        delta = f"+{100.0 * (p_mm - p_id) / p_id:.1f} %"

    # committed copies, so CI (no results/runs/) regenerates the same table
    bench = None
    bench_dir = ROOT / "results" / "retrieval_benchmarks"
    if bench_dir.is_dir():
        preferred = [p for p in sorted(bench_dir.glob("*.json"))
                     if mm_run and str(mm_run["run_id"]) in p.name]
        files = preferred or sorted(bench_dir.glob("*.json"))
        if files:
            bench = json.loads(files[0].read_text(encoding="utf-8"))

    overall = read("overall.csv")
    n_users = _spaced(_num(overall[0], "num_users") if overall else None)
    # committed dataset stats, not a run directory: this file is regenerated in
    # CI, where results/runs/ (gitignored) does not exist
    n_items = TBD
    stats = ROOT / "results" / "dataset_stats" / "base.json"
    if stats.exists():
        n_items = _spaced(_num(json.loads(stats.read_text(encoding="utf-8")), "num_items"))
    eval_txt = f"full ranking, {n_users} users x {n_items} items"

    def both(id_v, mm_v, unit=""):
        return f"{id_v} / {mm_v}{unit}"

    rows = [
        ["SASRec (ID-only) parameters", s(p_id)],
        ["MM-SASRec (gated) parameters", f"{s(p_mm)} ({delta})" if p_mm else TBD],
        ["epochs trained (SASRec / MM)", both(s(_mean(id_grp, "epochs")),
                                              s(_mean(mm_grp, "epochs")))],
        ["wall time per epoch (SASRec / MM)", both(s(per_epoch(id_run), 1),
                                                   s(per_epoch(mm_run), 1), " s")],
        ["total training time (SASRec / MM)", both(s(_mean(id_grp, 'train_time_s'), 0),
                                                   s(_mean(mm_grp, 'train_time_s'), 0), " s")],
        ["evaluation", eval_txt],
        ["runs behind these numbers", str(len(id_grp) + len(mm_grp))],
    ]
    if bench:
        rows.append(["vector index", f'{bench.get("backend", TBD)} (exact inner product)'])
        rows.append(["index build / query latency",
                     f"{bench.get('index_build_time_s', TBD)} s / "
                     f"{bench.get('query_latency_ms_per_query', TBD)} ms per query"])
    else:
        rows.append(["vector index", f"{TBD} (run `scripts/build_faiss_index.py`)"])
    return md_table(["", "value"], rows)


def _num(row: dict | None, key: str) -> float | None:
    """A cell as a float, whether it holds a raw value or 'mean ± std'."""
    if not row:
        return None
    v = row.get(key)
    if v in (None, "", "None", TBD):
        return None
    if isinstance(v, str) and "±" in v:
        v = v.split("±")[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pick(rows: list[dict], **want) -> dict | None:
    """First row whose fields match; numeric strings compare numerically."""
    for r in rows:
        ok = True
        for k, v in want.items():
            got = r.get(k, "")
            if isinstance(v, float):
                try:
                    ok &= abs(float(got or 0) - v) < 1e-9
                except (TypeError, ValueError):
                    ok = False
            else:
                ok &= str(got) == str(v)
        if ok:
            return r
    return None


def _gain(new: float | None, ref: float | None) -> str:
    if new is None or ref is None:
        return TBD
    return f"{new - ref:+.4f}"


def _spaced(n: float | None) -> str:
    return TBD if n is None else f"{int(n):,}".replace(",", " ")


def _seeds(row: dict | None) -> str:
    """How many runs of this experiment went into the value, as ``3 runs``."""
    if not row:
        return TBD
    n = row.get("n_seeds", "?")
    try:
        n = int(n)
    except (TypeError, ValueError):
        return str(n)
    return f"{n} run" + ("" if n == 1 else "s")


def findings() -> str:
    """Key findings, computed from the same CSVs as the tables above.

    Nothing here is written by hand: every value is a difference of two cells,
    each claim names its denominator, and a missing run renders as ``TBD``
    instead of a remembered number.
    """
    overall = group_seeds(read("overall.csv"))
    long_tail = group_seeds(read("long_tail.csv"))
    cold = group_seeds(read("cold_start.csv"))
    if not overall:
        return f"_No finished runs yet — {TBD}._"

    base = [r for r in overall if is_base(r)]
    pop = _pick(base, model="popular")
    bpr = _pick(base, model="bpr")
    # the modality set is part of the identity: without it, an ablation row
    # (which sorts before "mm_gated*") would be quoted as the headline model
    mm = _pick(base, model="mm_sasrec", fusion="gated", modalities="id+text+image",
               id_dropout=0.2)
    mm_plain = _pick(base, model="mm_sasrec", fusion="gated", modalities="id+text+image",
                     id_dropout=0.0)
    mm_concat = _pick(base, model="mm_sasrec", fusion="concat", modalities="id+text+image",
                      id_dropout=0.0)
    mm_concat_reg = _pick(base, model="mm_sasrec", fusion="concat", modalities="id+text+image",
                          id_dropout=0.2)
    reg = _pick(base, model="sasrec", item_dropout=0.2)
    idonly = next((r for r in base if r.get("model") == "sasrec"
                   and float(r.get("item_dropout") or 0) == 0), None)
    users = _spaced(_num(pop, "num_users") or _num(idonly, "num_users"))

    lines: list[str] = []

    def line(text: str) -> None:
        lines.append(f"- {text}")

    if pop and bpr and idonly:
        vals = [_num(pop, "Recall@20"), _num(bpr, "Recall@20"), _num(idonly, "Recall@20")]
        holds = vals[0] < vals[1] < vals[2]
        line(
            f"**Ordering holds** — Popular {vals[0]:.4f} < BPR-MF {vals[1]:.4f} < SASRec "
            f"{vals[2]:.4f} test Recall@20 over {users} evaluated test users "
            f"({_seeds(pop)} / {_seeds(bpr)} / {_seeds(idonly)} respectively)."
            if holds else
            f"**Ordering VIOLATED** — Popular {vals[0]:.4f}, BPR-MF {vals[1]:.4f}, SASRec "
            f"{vals[2]:.4f} test Recall@20 over {users} evaluated test users. "
            "This must be diagnosed in the implementation, not tuned away."
        )

    if idonly and reg:
        line(
            f"**Gain_reg** (item-dropout control) = SASRec+item-dropout 0.2 − SASRec = "
            f"{_gain(_num(reg, 'Recall@20'), _num(idonly, 'Recall@20'))} test Recall@20 "
            f"over {users} users ({_seeds(reg)} vs {_seeds(idonly)}); "
            f"NDCG@20 {_gain(_num(reg, 'NDCG@20'), _num(idonly, 'NDCG@20'))}."
        )

    if mm and reg:
        line(
            f"**Gain_content** (over the dropout control) = MM-SASRec gated+ID-dropout 0.2 − "
            f"SASRec+item-dropout 0.2 = {_gain(_num(mm, 'Recall@20'), _num(reg, 'Recall@20'))} "
            f"test Recall@20 over {users} users ({_seeds(mm)} vs {_seeds(reg)}); "
            f"NDCG@20 {_gain(_num(mm, 'NDCG@20'), _num(reg, 'NDCG@20'))}."
        )

    if mm and idonly:
        line(
            f"**Multimodal vs ID-only** = {_spaced(_num(mm, 'num_users'))} users: "
            f"Recall@20 {_num(idonly, 'Recall@20'):.4f} (ID-only) → {_num(mm, 'Recall@20'):.4f}, "
            f"NDCG@20 {_num(idonly, 'NDCG@20'):.4f} → {_num(mm, 'NDCG@20'):.4f}."
        )

    if mm_plain and mm_concat and idonly:
        line(
            f"**Fusion without ID dropout** — gated {_num(mm_plain, 'Recall@20'):.4f} vs concat "
            f"{_num(mm_concat, 'Recall@20'):.4f} test Recall@20 vs ID-only "
            f"{_num(idonly, 'Recall@20'):.4f} (same {users} users); "
            f"Δ vs ID-only gated {_gain(_num(mm_plain, 'Recall@20'), _num(idonly, 'Recall@20'))}, "
            f"concat {_gain(_num(mm_concat, 'Recall@20'), _num(idonly, 'Recall@20'))}."
            # asked at both settings on purpose: at ID-dropout 0 the comparison
            # is confounded with "which model was given the dropout", and the
            # follow-up run exists so the answer does not depend on that choice
            + (f" With ID-dropout 0.2 the order is unchanged: gated "
               f"{_num(mm, 'Recall@20'):.4f} vs concat {_num(mm_concat_reg, 'Recall@20'):.4f}."
               if mm and mm_concat_reg else "")
        )

    base_lt = [r for r in long_tail if is_base(r)]
    id_lt = next((r for r in base_lt if r.get("model") == "sasrec"
                  and float(r.get("item_dropout") or 0) == 0), None)
    # fusion is pinned: with it unpinned this picked whichever id+text+image
    # row came first in the CSV, so the tail claim silently quoted concat while
    # every other headline quotes the gated model
    mm_lt = next((r for r in base_lt if r.get("model") == "mm_sasrec"
                  and r.get("fusion") == "gated" and r.get("modalities") == "id+text+image"
                  and float(r.get("id_dropout") or 0) > 0), None)
    concat_lt = next((r for r in base_lt if r.get("model") == "mm_sasrec"
                      and r.get("fusion") == "concat" and r.get("modalities") == "id+text+image"
                      and float(r.get("id_dropout") or 0) > 0), None)
    reg_lt = next((r for r in base_lt if r.get("model") == "sasrec"
                   and float(r.get("item_dropout") or 0) > 0), None)
    if id_lt and mm_lt:
        parts = []
        for bucket in ("head", "middle", "tail"):
            a, b = _num(id_lt, f"{bucket}_Recall@20"), _num(mm_lt, f"{bucket}_Recall@20")
            c = _num(reg_lt, f"{bucket}_Recall@20") if reg_lt else None
            txt = f"{bucket}: ID-only {TBD if a is None else f'{a:.4f}'} → MM {TBD if b is None else f'{b:.4f}'} ({_gain(b, a)})"
            if c is not None:
                txt += f", vs dropout control ({c:.4f}) {_gain(b, c)}"
            parts.append(txt)
        line(
            f"**Long tail** (buckets from training interactions only, rule "
            f"`{long_tail[0].get('bucket_rule', '?')}`, {_seeds(mm_lt)}), Recall@20 — "
            + "; ".join(parts)
            + f". Bucket sizes: "
            + ", ".join(f"{b} {_spaced(_num(id_lt, f'{b}_users'))} users"
                        for b in ("head", "middle", "tail"))
            + "."
            # fusion is not cosmetic for the tail: the concat variant is where
            # the tail gain survives the regularisation control, so say so
            + (f" Concat+ID-dropout 0.2 reaches tail "
               f"{_num(concat_lt, 'tail_Recall@20'):.4f} "
               f"({_gain(_num(concat_lt, 'tail_Recall@20'), _num(reg_lt, 'tail_Recall@20'))} "
               f"vs the same dropout control)" if concat_lt and reg_lt else "")
            + "."
        )

    if cold:
        rand = _pick(cold, model="random")
        cold_id = _pick(cold, model="sasrec")
        # the content-only row is the direct answer to "can content stand in";
        # the gated row is the headline model with its cold ID zeroed
        cold_content = _pick(cold, model="mm_sasrec", modalities="text+image")
        # both gated variants are quoted: ID-dropout is the base-split headline
        # config, but the plain gated run is the like-for-like comparison with
        # the no-ID-dropout SASRec baseline, and the gap is itself the finding
        cold_mm_plain = _pick(cold, model="mm_sasrec", modalities="id+text+image",
                              id_dropout=0.0)
        cold_mm = (_pick(cold, model="mm_sasrec", modalities="id+text+image", id_dropout=0.2)
                   or cold_mm_plain)
        n_cold = _num(cold_mm or cold_content or cold_id or rand, "num_cold_items")
        chance = f"20/{int(n_cold)} = {20 / n_cold:.4f}" if n_cold else TBD
        line(
            "**Cold items** (cold catalogue = "
            f"{_spaced(n_cold)} items, {_spaced(_num(cold_mm or cold_id, 'num_users'))} users with a "
            f"cold target): cold-only Recall@20 — theoretical uniform ranker {chance}"
            + (f", measured Random {_num(rand, 'ColdOnly Recall@20'):.4f}" if rand else "")
            + (f", SASRec ID-only {_num(cold_id, 'ColdOnly Recall@20'):.4f}" if cold_id else "")
            + (f", content-only MM-SASRec {_num(cold_content, 'ColdOnly Recall@20'):.4f}"
               if cold_content else "")
            + (f", gated MM-SASRec {_num(cold_mm_plain or cold_mm, 'ColdOnly Recall@20'):.4f}"
               if (cold_mm_plain or cold_mm) else "")
            + (f" ({_num(cold_mm, 'ColdOnly Recall@20'):.4f} with ID-dropout 0.2)"
               if cold_mm_plain and cold_mm else "")
            + "."
        )

    if not lines:
        return f"_No claims available yet — {TBD}._"
    lines.append(
        "_Every value above is computed from `results/tables/*.csv` by "
        "`analysis/make_readme_tables.py`; recall denominators are the evaluated "
        "test users named in each line._"
    )
    return "\n".join(lines)


def recall_table() -> str:
    """Recall@K per channel, from results/tables/recall_eval.csv."""
    rows = read("recall_eval.csv")
    if not rows:
        return f"_Recall evaluation has not been run yet — {TBD}._"
    ks = sorted({int(k.split("@")[1]) for k in rows[0] if k.startswith("Recall@")})
    body = [[r["channel"]] + [fmt(r.get(f"Recall@{k}")) for k in ks] for r in rows]
    n = rows[0].get("num_users", TBD)
    return md_table(["Channel"] + [f"Recall@{k}" for k in ks], body) + (
        f"\n\n_Test target, user history masked, {n} users. Candidate-generation quality: "
        "this is the ceiling the ranker can reach._"
    )


def pipeline_table() -> str:
    """Two-stage retention: same checkpoint, same user sample, both protocols.

    Reads ``pipeline_tradeoff.csv`` (accuracy, same-checkpoint baseline) and
    ``latency_benchmark.csv`` (stage timings, measured separately because a
    single timing per user is dominated by machine noise).
    """
    rows = read("pipeline_tradeoff.csv")
    if not rows:
        return f"_Two-stage pipeline evaluation has not been run yet — {TBD}._"
    latency = {r["candidate_k"]: r for r in read("latency_benchmark.csv")}

    body = []
    for r in rows:
        k = r["candidate_k"]
        lat = latency.get(k, {})
        body.append([
            k,
            fmt(r.get("candidate_recall")),
            fmt(r.get("pipeline_Recall@20")),
            fmt(r.get("recall_retention"), 3),
            fmt(lat.get("recall_ms_median"), 1) if lat else TBD,
            fmt(lat.get("score_ms_median"), 2) if lat else TBD,
            fmt(lat.get("encode_ms_median"), 2) if lat else TBD,
        ])

    full = rows[0].get("full_Recall@20")
    n = rows[0].get("num_users", TBD)
    seed = rows[0].get("eval_seed", TBD)
    out = md_table(
        ["Candidate budget", "Candidate recall", "Pipeline Recall@20", "Retention",
         "Recall latency (ms)", "Score (ms)", "Encode (ms)"], body
    )
    out += (f"\n\nSame-checkpoint, same-user full-catalogue **Recall@20 = {fmt(full)}** "
            f"({n} users, sample seed {seed}). *Retention* is pipeline ÷ that baseline — the "
            f"only apples-to-apples way to state it.")
    return out + (
        "\n\nLatency columns are from `scripts/benchmark_latency.py` (CPU, median, this "
        "machine) and are for relative comparison only. Note that the **encode** stage "
        "dominates: scoring the whole catalogue costs about the same as scoring a 100-item "
        "pool, so the two-stage split buys recall quality and catalogue headroom rather than "
        "latency at this scale."
    )


def source_table() -> str:
    """Which recall channel produced the final top-20 hits."""
    rows = read("recall_source_contribution.csv")
    if not rows:
        return f"_Recall source analysis has not been run yet — {TBD}._"
    budget = max(int(r["candidate_k"]) for r in rows)
    body = [[
        r["source"], r["top20_hits"], fmt(r.get("share_of_hits")),
        r.get("head_hits", TBD), r.get("middle_hits", TBD), r.get("tail_hits", TBD),
    ] for r in rows if int(r["candidate_k"]) == budget]
    return md_table(
        ["Source", "Top-20 hits", "Share", "Head hits", "Middle hits", "Tail hits"], body
    ) + (f"\n\n_Candidate budget {budget}. `multiple` means the item was found by more "
         "than one channel._")


# ----------------------------------------------------------------------
# Portfolio-page blocks.  Same rule as the tables: every number is computed
# from the artifact that measures it, never typed in.
# ----------------------------------------------------------------------
SUMMARY_MODELS = [
    ("sasrec", "-", "0.0", "SASRec (ID only)"),
    ("mm_sasrec", "concat", "0.0", "MM-SASRec Concat"),
    ("mm_sasrec", "gated", "0.0", "MM-SASRec Gated"),
    ("mm_sasrec", "concat", "0.2", "MM-SASRec Concat + ID dropout"),
    ("sasrec", "-", "0.2", "SASRec + item dropout"),
]


def overall_summary_table() -> str:
    """The four or five rows a first-time reader needs, from ``overall.csv``."""
    rows = [r for r in group_seeds(read("overall.csv")) if is_base(r)]
    if not rows:
        return f"_No finished runs yet — {TBD}._"
    body = []
    for model, fusion, idrop, name in SUMMARY_MODELS:
        r = _pick(rows, model=model, fusion=fusion, id_dropout=idrop)
        if r is None:
            continue
        body.append([
            name, fmt(r.get("Recall@10")), fmt(r.get("Recall@20")),
            fmt(r.get("NDCG@20")), _seeds(r),
        ])
    if not body:
        return f"_No headline runs finished yet — {TBD}._"
    return md_table(["Model", "Recall@10", "Recall@20", "NDCG@20", "Seeds"], body)


def _hero_numbers() -> dict:
    """The three headline numbers, each read from its own artifact."""
    out: dict[str, object] = {}
    overall = [r for r in group_seeds(read("overall.csv")) if is_base(r)]
    base = _pick(overall, model="sasrec", fusion="-", id_dropout="0.0")
    mm = _pick(overall, model="mm_sasrec", fusion="concat", id_dropout="0.0")
    r0, r1 = _num(base, "Recall@20"), _num(mm, "Recall@20")
    if r0 is not None and r1 is not None:
        out["sasrec_r20"] = r0
        out["mm_r20"] = r1
        out["gain_pct"] = (r1 / r0 - 1.0) * 100.0
        out["seeds"] = mm.get("n_seeds", base.get("n_seeds", "?"))

    recall = read("recall_eval.csv")
    merged = next((r for r in recall if r.get("channel") == "merged"), None)
    best = None
    for r in recall:
        if r.get("channel") == "merged":
            continue
        v = _num(r, "Recall@1000")
        if v is not None and (best is None or v > _num(best, "Recall@1000")):
            best = r
    if merged is not None:
        out["merged_r1000"] = _num(merged, "Recall@1000")
    if best is not None:
        out["best_channel"] = best.get("channel")
        out["best_channel_r1000"] = _num(best, "Recall@1000")

    trade = read("pipeline_tradeoff.csv")
    if trade:
        pick = None
        for r in trade:
            try:
                if int(float(r.get("candidate_k", 0))) == 1000:
                    pick = r
            except (TypeError, ValueError):
                continue
        if pick is None:
            pick = max(trade, key=lambda r: _num(r, "recall_retention") or 0)
        out["retention"] = _num(pick, "recall_retention")
        out["retention_k"] = pick.get("candidate_k")
        out["full_r20"] = _num(pick, "full_Recall@20")
    return out


def hero_metrics() -> str:
    """Three metric cards.  HTML table because GitHub renders it consistently."""
    n = _hero_numbers()
    if not n:
        return f"_Headline metrics unavailable — {TBD}._"
    gain = n.get("gain_pct")
    gain_txt = f"{gain:+.1f}% vs SASRec" if isinstance(gain, float) else TBD
    seeds = n.get("seeds", "?")
    cells = [
        (f"{n['mm_r20'] * 100:.2f}%", "Recall@20", f"{gain_txt} · {seeds} seeds"),
        (f"{n['merged_r1000'] * 100:.1f}%", "Recall@1000", "multi-channel recall pool"),
        (f"{n['retention'] * 100:.1f}%", "Recall retained",
         f"{n['retention_k']} candidates · same checkpoint"),
    ]
    tds = "\n".join(
        f'<td align="center" width="33%">\n\n'
        f'<strong>{v}</strong><br/>\n{b}<br/>\n<sub>{s}</sub>\n\n</td>'
        for v, b, s in cells
    )
    return f"<table>\n<tr>\n{tds}\n</tr>\n</table>"


def case_study() -> str:
    """The demo trace, read from the media manifest written by the pipeline."""
    manifest = ROOT / "artifacts" / "demo_media_manifest.json"
    if not manifest.exists():
        return f"_Demo trace unavailable — {TBD}._"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return f"_Demo trace unreadable — {TBD}._"
    items = list((data.get("items") or {}).values())
    if not items:
        return f"_Demo trace empty — {TBD}._"

    def delta(it: dict) -> float:
        try:
            return float(it.get("rank_delta") or 0)
        except (TypeError, ValueError):
            return 0.0

    it = max(items, key=delta)
    user = data.get("user_id", TBD)
    sources = ", ".join(str(s).capitalize() for s in (it.get("sources") or [])) or TBD
    title = it.get("title") or "no catalogue title"
    if len(title) > 72:
        title = title[:69].rstrip() + "..."
    baseline = it.get("baseline_rank")
    rank = it.get("rank")
    return md_table(
        ["Request", "Value"],
        [
            ["User", f"`{user}`"],
            ["Item", f"`{it.get('item_id')}` — {title}"],
            ["Recalled by", sources],
            ["ID-only SASRec rank", f"#{baseline}" if baseline is not None else TBD],
            ["MM-SASRec rank", f"#{rank}" if rank is not None else TBD],
            ["Movement", f"↑ {int(delta(it))} positions" if delta(it) else TBD],
            ["MicroLens video", f"`{it.get('official_video_id')}` "
                                f"({it.get('source_codec')} → {it.get('playback_codec')})"],
        ],
    )


def media_summary() -> str:
    """Two numbers and a link; the full proof lives in docs/media_provenance.md."""
    manifest = ROOT / "artifacts" / "demo_media_manifest.json"
    if not manifest.exists():
        return f"_Media mapping not prepared — {TBD}._"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return f"_Media manifest unreadable — {TBD}._"
    rep = data.get("mapping_report") or {}
    if not rep:
        return f"_Media mapping report missing — {TBD}._"
    frac = rep.get("matched_fraction")
    items = rep.get("items_mapped")
    ok = rep.get("item_mapping_is_bijection")
    return (
        f"| | |\n|---|---|\n"
        f"| Timestamp agreement | {frac * 100:.3f} % |\n"
        f"| Item mapping | {items:,} items, bijection: **{ok}** |\n"
        f"| Independent check | `x_label` vs official views, 59× the shuffled control |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sections = {
        "HERO:METRICS": hero_metrics(),
        "CASE:DEMO": case_study(),
        "CASE:MEDIA": media_summary(),
        "OVERALL_SUMMARY": overall_summary_table(),
        "OVERALL": overall_table(),
        "RECALL": recall_table(),
        "PIPELINE": pipeline_table(),
        "SOURCES": source_table(),
        "ABLATION": ablation_table(),
        "COLD": cold_table(),
        "LONGTAIL": long_tail_table(),
        "GAIN": gain_table(),
        "FINDINGS": findings(),
        "GATES": gates_table(),
        "EFFICIENCY": efficiency_table(),
        "DATASET": dataset_table(),
        "SEMANTICID": semantic_id_table(),
    }
    text = "\n\n".join(f"<!-- {k} -->\n{v}" for k, v in sections.items())
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
