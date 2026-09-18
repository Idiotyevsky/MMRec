#!/usr/bin/env python
"""Verify the ShortRec item id -> official MicroLens video id mapping.

    python scripts/verify_media_mapping.py

Prints the evidence behind the mapping used by the playable feed, and exits
non-zero if any check fails.  Run it whenever the official metadata changes.

Checks
------
1. the mapping is a **function**: every HF item's timestamp matches agree on one
   official video id;
2. it is **injective**: no two HF items share an official video id (i.e. it is a
   bijection over all 19 738 items);
3. the induced **user** mapping is also a bijection;
4. an independent cross-check: per-item mean ``x_label`` correlates with the
   official ``likes`` far more strongly under the correct mapping than under a
   shuffled control.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.media.mapping import build_mapping, load_likes_views  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shuffled-control", action="store_true", default=True)
    args = ap.parse_args()

    lut, report = build_mapping()

    print("=== id mapping evidence ===")
    print(f"  method                     : {report['method']}")
    print(f"  HF interactions            : {report['hf_interactions']}")
    print(f"  official interactions      : {report['official_interactions']}")
    print(f"  exact timestamp matches    : {report['unambiguous_timestamp_matches']} "
          f"({report['matched_fraction']:.4%})")
    print(f"  items mapped               : {report['items_mapped']} / 19738")
    print(f"  official video id range    : {report['official_video_id_range']}")
    print(f"  item mapping is bijection  : {report['item_mapping_is_bijection']}")
    print(f"  user mapping is bijection  : {report['user_mapping_is_bijection']}")

    ok = (report["items_mapped"] == 19738 and report["item_mapping_is_bijection"]
          and report["user_mapping_is_bijection"] and report["matched_fraction"] > 0.999)

    # ---- independent cross-check: x_label vs official likes -------------
    likes_views = load_likes_views()
    if likes_views:
        import pandas as pd

        inter = pd.read_csv(ROOT / "data" / "raw" / "microlens_100k.inter", sep="\t")
        views = np.zeros(19739)
        likes = np.zeros(19739)
        for vid, (lk, vw) in likes_views.items():
            if 0 < vid < len(views):
                views[vid], likes[vid] = vw, lk

        g = inter.groupby("itemID")["x_label"].agg(["mean", "count"])
        items, mean_x, cnt = g.index.values, g["mean"].values, g["count"].values
        w = cnt / cnt.sum()

        def wcorr(a: np.ndarray, b: np.ndarray) -> float:
            ma, mb = (w * a).sum(), (w * b).sum()
            cov = (w * (a - ma) * (b - mb)).sum()
            sa = np.sqrt((w * (a - ma) ** 2).sum())
            sb = np.sqrt((w * (b - mb) ** 2).sum())
            return float(cov / (sa * sb))

        # apply the mapping: HF item id -> official video id
        correct = wcorr(mean_x, np.log1p(views[lut[items]]))
        rng = np.random.default_rng(0)
        shuffled = wcorr(mean_x, np.log1p(views[rng.permutation(lut[items])]))
        ratio = correct / max(abs(shuffled), 1e-9)
        print("\n=== independent cross-check (x_label vs official views) ===")
        print(f"  correlation, correct mapping : {correct:+.4f}")
        print(f"  correlation, shuffled control: {shuffled:+.4f}")
        print(f"  ratio                        : {ratio:.1f}x")
        ok &= ratio > 5.0

    print("\nRESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
