#!/usr/bin/env python
"""Report long-tail metrics for every finished run.

    python scripts/run_long_tail.py
    python scripts/run_long_tail.py --rule interaction_mass   # robustness check

The per-bucket numbers are recomputed from the stored per-user ranks, so this
works for any finished run without retraining.  Two bucketing rules are
supported so the main result can be robustness-checked with a different
definition of "tail":

* ``frequency_quantile`` (default, used in the paper table)
* ``interaction_mass``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.data.popularity import BUCKET_NAMES, popularity_buckets  # noqa: E402
from src.evaluation.evaluator import EvalResult  # noqa: E402
from src.evaluation.slicing import load_ranking  # noqa: E402

RUNS = ROOT / "results" / "runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="frequency_quantile",
                    choices=["frequency_quantile", "interaction_mass"])
    ap.add_argument("--head-frac", type=float, default=0.2)
    ap.add_argument("--tail-frac", type=float, default=0.6)
    args = ap.parse_args()

    rows = []
    for d in sorted(RUNS.iterdir()):
        rk, cf = d / "test_ranking.npz", d / "config.yaml"
        if not (d.is_dir() and rk.exists() and cf.exists()):
            continue
        import yaml

        cfg = yaml.safe_load(cf.read_text(encoding="utf-8")) or {}
        pdir = (cfg.get("data") or {}).get("processed_dir")
        if not pdir:
            continue
        data = ProcessedData.load(ROOT / pdir)
        buckets = popularity_buckets(
            data.train_freq, rule=args.rule, head_frac=args.head_frac,
            tail_frac=args.tail_frac, num_items=data.num_items,
        )["bucket"]
        arr = load_ranking(rk)
        res = EvalResult(arr["user_ids"], arr["targets"], arr["rank"], arr["topk"],
                         data.num_items, (5, 10, 20))
        row = {"tag": d.name.rsplit("_", 2)[0], "model": (cfg.get("model") or {}).get("name")}
        for bid, name in BUCKET_NAMES.items():
            m = buckets[res.targets] == bid
            sub = res.subset(m)
            row[f"{name}_n"] = int(m.sum())
            row[f"{name}_Recall@20"] = round(sub.recall_at(20), 4)
            row[f"{name}_NDCG@20"] = round(sub.ndcg_at(20), 4)
        rows.append(row)

    if not rows:
        print("no finished runs with stored rankings yet")
        return

    print(f"popularity rule: {args.rule} "
          f"(head={args.head_frac:.0%}, tail={args.tail_frac:.0%})")
    header = f"{'tag':<24s}{'model':<12s}" + "".join(
        f"{n + '_R@20':>12s}{n + '_N@20':>12s}" for n in ("head", "middle", "tail"))
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['tag']:<24s}{str(r['model']):<12s}"
        for n in ("head", "middle", "tail"):
            line += f"{r[f'{n}_Recall@20']:>12.4f}{r[f'{n}_NDCG@20']:>12.4f}"
        print(line)

    # multimodal gain over the ID-only baseline, per bucket
    id_row = next((r for r in rows if r["model"] == "sasrec"), None)
    if id_row is None:
        print("\n(no ID-only SASRec run on this split -> gain column not available)")
        return
    print("\nΔ Recall@20 (multimodal − ID-only):")
    for r in rows:
        if r["model"] != "mm_sasrec":
            continue
        gains = "".join(
            f"{r[f'{n}_Recall@20'] - id_row[f'{n}_Recall@20']:>+12.4f}" for n in ("head", "middle", "tail")
        )
        print(f"{r['tag']:<24s}{gains}")


if __name__ == "__main__":
    main()
