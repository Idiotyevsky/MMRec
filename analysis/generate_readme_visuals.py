#!/usr/bin/env python
"""Generate the README's vector assets.

    python analysis/generate_readme_visuals.py

Writes ``assets/icons/*.svg``, ``assets/system_architecture.svg`` and
``assets/cold_start_summary.svg``.

Design constraints (kept deliberately narrow so the page does not turn into a
landing page):

* stroke-only, single colour, 24x24 icons on a shared grid;
* one accent colour (``#2563EB``) plus neutral greys;
* system sans-serif, no embedded fonts;
* **no number is typed in** — the cold-start summary is computed from
  ``results/tables/cold_start.csv``, and the architecture diagram is pure
  structure.  Anything that could go stale is generated, not drawn by hand.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ICONS = ASSETS / "icons"
TABLES = ROOT / "results" / "tables"

INK = "#111827"
MUTED = "#6B7280"
ACCENT = "#2563EB"
LINE = "#D1D5DB"
SURFACE = "#F8FAFC"

FONT = ("-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
        "Helvetica, Arial, sans-serif")


# ----------------------------------------------------------------------
# icons — 24x24 grid, stroke only, round caps, 1.8 stroke width
# ----------------------------------------------------------------------
ICON_PATHS: dict[str, str] = {
    # retention: funnel keeping a subset of candidates
    "retention": (
        '<path d="M3.5 4.5h17l-6.4 7.6v6.1l-4.2 2.3v-8.4z"/>'
    ),
    # popular: trending-up bars
    "popular": (
        '<path d="M4 19.5V14"/><path d="M9.5 19.5V10"/>'
        '<path d="M15 19.5v-7"/><path d="M20.5 19.5V5.5"/>'
        '<path d="M4 11l5-4 4 3 7-6"/>'
    ),
    # itemcf: co-occurrence graph
    "itemcf": (
        '<circle cx="6" cy="7" r="2.4"/><circle cx="18" cy="7" r="2.4"/>'
        '<circle cx="12" cy="18" r="2.4"/>'
        '<path d="M8.2 8.2 10.6 16"/><path d="M15.8 8.2 13.4 16"/>'
        '<path d="M8.4 7h7.2"/>'
    ),
    # semantic: image + text layers
    "semantic": (
        '<rect x="3" y="4" width="18" height="12" rx="2"/>'
        '<path d="M3 12.5 7.5 9l4 3.2L15 9l6 4.5"/>'
        '<circle cx="8.5" cy="7.8" r="1.2"/>'
        '<path d="M7 20h10"/>'
    ),
    # merge: channels converging
    "merge": (
        '<path d="M4 5v3a4 4 0 0 0 4 4h8a4 4 0 0 1 4 4v3"/>'
        '<path d="M12 5v6"/><path d="M20 5v3a4 4 0 0 1-4 4H8a4 4 0 0 0-4 4v3"/>'
        '<path d="M9.6 16.5 12 19l2.4-2.5"/>'
    ),
    # ranking: ordered bars with a leading arrow
    "ranking": (
        '<path d="M4 6h9"/><path d="M4 12h6"/><path d="M4 18h3"/>'
        '<path d="M20.5 8.5 17 12l-2.2-2.2"/>'
        '<path d="M17 12V5.5"/>'
    ),
    # feed: play inside a phone-ish frame
    "feed": (
        '<rect x="6.5" y="2.5" width="11" height="19" rx="2.6"/>'
        '<path d="M11 10.2v5.6l4.4-2.8z"/>'
    ),
}


def icon_svg(name: str, colour: str = INK) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" '
        f'fill="none" stroke="{colour}" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" role="img" aria-label="{name}">\n'
        f"  {ICON_PATHS[name]}\n"
        f"</svg>\n"
    )


# ----------------------------------------------------------------------
# architecture — offline build above, online request path below
# ----------------------------------------------------------------------
def _box(x, y, w, h, label, sub=None, accent=False):
    fill = "#EFF6FF" if accent else "#FFFFFF"
    stroke = ACCENT if accent else LINE
    out = [f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
           f'stroke="{stroke}" stroke-width="1.2"/>']
    # centre the label (and optional subtitle) as a block inside the box
    cy = y + h / 2 - (4 if sub else 0)
    out.append(f'  <text x="{x + w / 2}" y="{cy + 5}" text-anchor="middle" font-size="13.5" '
               f'font-weight="600" fill="{INK}">{label}</text>')
    if sub:
        out.append(f'  <text x="{x + w / 2}" y="{cy + 21}" text-anchor="middle" '
                   f'font-size="11.5" fill="{MUTED}">{sub}</text>')
    return out


def _arrow(x1, y1, x2, y2):
    return (f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" '
            f'stroke-width="1.4" marker-end="url(#a)"/>')


def _label(x, y, text, size=11, colour=MUTED, weight="700", anchor="start"):
    return (f'  <text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
            f'font-weight="{weight}" fill="{colour}" letter-spacing="0.08em">{text}</text>')


def architecture_svg() -> str:
    W, H = 900, 470
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="{FONT}" role="img" '
        f'aria-label="ShortRec offline build and online serving architecture">',
        '  <defs>',
        f'    <marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" '
        f'orient="auto"><polygon points="0 0, 7 3.5, 0 7" fill="{LINE}"/></marker>',
        '  </defs>',
        f'  <rect width="{W}" height="{H}" fill="#FFFFFF"/>',
    ]

    # ---- offline ----
    p.append(_label(28, 30, "OFFLINE · BATCH"))
    p += _box(28, 44, 150, 52, "Interactions", "MicroLens-100K")
    p.append(_arrow(178, 70, 214, 70))
    p += _box(216, 44, 156, 52, "Preprocess", "chronological split")
    p.append(_arrow(372, 70, 408, 70))
    p += _box(410, 26, 190, 44, "Train rankers", "SASRec / MM-SASRec", accent=True)
    p += _box(410, 78, 190, 40, "ItemCF index", "cosine co-occurrence")
    p += _box(410, 126, 190, 40, "Content index", "text + image, exact IP")
    p += _box(654, 78, 150, 44, "artifacts/", "checkpoints, indices")
    for y in (48, 98, 146):
        p.append(f'  <line x1="600" y1="{y}" x2="648" y2="{y}" stroke="{LINE}" stroke-width="1.4"/>')
    p.append(f'  <line x1="28" y1="196" x2="872" y2="196" stroke="#E5E7EB" stroke-width="1"/>')

    # ---- online ----
    p.append(_label(28, 226, "ONLINE · SERVING"))
    p += _box(28, 242, 150, 52, "User request", "history + user id")
    p.append(_arrow(103, 294, 103, 322))

    p += _box(28, 324, 150, 34, "Popular recall")
    p += _box(28, 364, 150, 34, "ItemCF recall")
    p += _box(28, 404, 150, 34, "Semantic recall")

    for y1, y2 in ((341, 372), (381, 372), (421, 372)):
        p.append(f'  <line x1="178" y1="{y1}" x2="252" y2="{y2}" stroke="{LINE}" stroke-width="1.4"/>')
    p += _box(254, 350, 158, 46, "Candidate merge", "dedup + RRF", accent=True)
    p.append(_arrow(412, 373, 446, 373))
    p += _box(448, 350, 158, 46, "MM-SASRec", "multimodal ranking", accent=True)
    p.append(_arrow(606, 373, 640, 373))
    p += _box(642, 350, 138, 46, "Rerank", "exploration quota")
    p.append(_arrow(711, 396, 711, 424))
    p += _box(642, 426, 138, 32, "Video feed")

    p.append("</svg>")
    return "\n".join(p) + "\n"


# ----------------------------------------------------------------------
# cold-start summary — every number read from cold_start.csv
# ----------------------------------------------------------------------
def _cold_numbers() -> list[dict]:
    path = TABLES / "cold_start.csv"
    if not path.exists():
        return []
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))

    def pick(tag: str) -> dict | None:
        for r in rows:
            if r.get("tag") == tag:
                return r
        return None

    def val(r: dict | None, key: str) -> float | None:
        if not r or r.get(key) in (None, "", "None"):
            return None
        try:
            return float(r[key])
        except (TypeError, ValueError):
            return None

    out = []
    for tag, label in (("cold_random", "Random"),
                       ("cold_sasrec", "ID-only SASRec"),
                       ("cold_content_only", "Content only (text + image)"),
                       ("cold_mm_gated", "MM-SASRec Gated")):
        r = pick(tag)
        if r is None:
            continue
        out.append({"label": label, "cold_only": val(r, "ColdOnly Recall@20"),
                    "full": val(r, "Cold Recall@20")})
    return out


def cold_start_summary_svg() -> str:
    data = _cold_numbers()
    W = 820
    row_h = 34
    top = 92
    H = top + len(data) * row_h + 128
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="{FONT}" role="img" '
        f'aria-label="Cold-start recall by model">',
        f'  <rect width="{W}" height="{H}" fill="#FFFFFF"/>',
    ]
    x0, bar_x, bar_w = 32, 260, 430
    p.append(_label(x0, 34, "COLD-ONLY RANKING · RECALL@20"))
    p.append(f'  <text x="{x0}" y="56" font-size="12" fill="{MUTED}">'
             f'ranking restricted to the cold catalogue, for users whose next item is cold</text>')

    vmax = max((d["cold_only"] or 0) for d in data) if data else 1.0
    for i, d in enumerate(data):
        y = top + i * row_h
        v = d["cold_only"] or 0.0
        w = max(int(bar_w * (v / vmax)), 2) if vmax else 2
        colour = ACCENT if v > 0.02 else "#9CA3AF"
        p.append(f'  <text x="{x0}" y="{y + 14}" font-size="13" fill="{INK}">{d["label"]}</text>')
        p.append(f'  <rect x="{bar_x}" y="{y}" width="{bar_w}" height="18" rx="3" fill="{SURFACE}"/>')
        p.append(f'  <rect x="{bar_x}" y="{y}" width="{w}" height="18" rx="3" fill="{colour}"/>')
        p.append(f'  <text x="{bar_x + bar_w + 14}" y="{y + 14}" font-size="13" '
                 f'font-weight="600" fill="{INK}">{v * 100:.2f}%</text>')

    y = top + len(data) * row_h + 20
    p.append(f'  <line x1="{x0}" y1="{y}" x2="{W - 32}" y2="{y}" stroke="#E5E7EB" stroke-width="1"/>')
    full = [d["full"] for d in data if d["full"] is not None]
    fmax = max(full) if full else None
    p.append(_label(x0, y + 30, "BUT: SAME MODELS, FULL CATALOGUE"))
    p.append(f'  <text x="{x0}" y="{y + 52}" font-size="12" fill="{MUTED}">'
             f'content ranks cold items well against each other, not against warm items</text>')
    if fmax is not None:
        p.append(f'  <text x="{x0}" y="{y + 78}" font-size="13" fill="{INK}">'
                 f'best full-catalogue cold Recall@20</text>')
        p.append(f'  <text x="{bar_x + bar_w + 14}" y="{y + 78}" text-anchor="end" '
                 f'font-size="13" font-weight="600" fill="{MUTED}">{fmax * 100:.3f}%</text>')
    p.append("</svg>")
    return "\n".join(p) + "\n"


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ASSETS))
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "icons").mkdir(parents=True, exist_ok=True)

    for name in ICON_PATHS:
        (out / "icons" / f"{name}.svg").write_text(icon_svg(name), encoding="utf-8")
    print(f"wrote {len(ICON_PATHS)} icons to {out / 'icons'}")

    (out / "system_architecture.svg").write_text(architecture_svg(), encoding="utf-8")
    print(f"wrote {out / 'system_architecture.svg'}")

    data = _cold_numbers()
    if data:
        (out / "cold_start_summary.svg").write_text(cold_start_summary_svg(), encoding="utf-8")
        print(f"wrote {out / 'cold_start_summary.svg'} "
              f"({len(data)} models from results/tables/cold_start.csv)")
    else:
        print("cold_start.csv not found; skipped cold-start summary")


if __name__ == "__main__":
    main()
