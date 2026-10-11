#!/usr/bin/env python
"""Compare completed RRF/snake pipeline artifacts; never launches evaluation.

    python analysis/compare_merge_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compare(rrf_path: Path, snake_path: Path) -> tuple[list[dict], dict]:
    rrf = json.loads(rrf_path.read_text())
    snake = json.loads(snake_path.read_text())
    for key in ("processed_dir", "ranker", "num_users", "eval_seed", "user_subset_hash",
                "recall_k_per_channel", "recall_sources", "final_k"):
        if rrf[key] != snake[key]:
            raise ValueError(f"incompatible comparison: {key}")
    if Path(rrf["ranker_run"]).resolve() != Path(snake["ranker_run"]).resolve():
        raise ValueError("incompatible ranker runs")
    if rrf["merge_method"] != "rrf" or snake["merge_method"] != "snake":
        raise ValueError("expected RRF and snake artifacts, in that order")
    if rrf["full_catalogue"] != snake["full_catalogue"]:
        raise ValueError("full-catalogue baselines differ; investigate before comparison")
    a = {r["candidate_k"]: r for r in rrf["tradeoff"]}
    b = {r["candidate_k"]: r for r in snake["tradeoff"]}
    if a.keys() != b.keys():
        raise ValueError("candidate budgets differ")
    rows = []
    for k in sorted(a):
        row = {"candidate_k": k, "num_users": rrf["num_users"],
               "user_subset_hash": rrf["user_subset_hash"]}
        for metric in ("candidate_recall", "pipeline_Recall@20", "pipeline_NDCG@20",
                       "recall_retention", "mean_pool_size"):
            row[f"rrf_{metric}"] = a[k][metric]
            row[f"snake_{metric}"] = b[k][metric]
            row[f"delta_{metric}"] = b[k][metric] - a[k][metric]
        rows.append(row)
    return rows, rrf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rrf", default="results/tables/two_tower_pipeline_rrf_control.json")
    ap.add_argument("--snake", default="results/tables/two_tower_pipeline_snake.json")
    ap.add_argument("--out", default="results/tables/two_tower_merge_comparison.csv")
    ap.add_argument("--report", default="docs/merge_comparison.md")
    args = ap.parse_args()
    rows, meta = compare(ROOT / args.rrf, ROOT / args.snake)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Four-channel RRF vs snake", "",
        f"Completed test evaluation: {meta['num_users']:,} users, seed {meta['eval_seed']}, "
        f"user sample hash `{meta['user_subset_hash']}`.", "",
        f"Ranker run: `{Path(meta['ranker_run']).name}`. Both methods use the same "
        f"checkpoint, histories, targets and per-channel top-{meta['recall_k_per_channel']} "
        "inputs. Final candidate budgets are identical. Exploration reranking is off.", "",
        "Snake is the ping-pong variant `Popular → ItemCF → Semantic → Two-Tower → "
        "Semantic → ItemCF → …`. Each visit consumes one list entry; a duplicate "
        "consumes its visit but is emitted only once. Interior channels receive twice "
        "the visits of the endpoints while all lists remain active. This is not an "
        "equal-quota interleave. All contributing source traces are retained.", "",
        "| Budget | Candidate R (RRF) | Candidate R (snake) | R@20 (RRF) | R@20 (snake) | NDCG@20 (RRF) | NDCG@20 (snake) |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        cols = [str(r["candidate_k"])]
        for metric in ("candidate_recall", "pipeline_Recall@20", "pipeline_NDCG@20"):
            cols.extend(f"{r[f'{method}_{metric}']:.4f}" for method in ("rrf", "snake"))
        lines.append("| " + " | ".join(cols) + " |")
    lower_candidate = sum(r["delta_candidate_recall"] < 0 for r in rows)
    lines += ["", "## Observations", "",
        f"Snake has lower candidate recall at {lower_candidate}/{len(rows)} tested "
        "budgets. Candidate recall and final ranking quality are different metrics; "
        "a smaller candidate hit rate need not imply a lower final NDCG."]
    for r in rows:
        if r["candidate_k"] == 1000:
            lines += ["", "At 1,000 candidates, snake minus RRF is "
                f"{100 * r['delta_candidate_recall']:+.2f} percentage points in "
                f"candidate recall, {100 * r['delta_pipeline_Recall@20']:+.2f} "
                "percentage points in final Recall@20, and "
                f"{r['delta_pipeline_NDCG@20']:+.6f} in NDCG@20. "
                f"Recall retention is {r['rrf_recall_retention']:.4f} (RRF) vs "
                f"{r['snake_recall_retention']:.4f} (snake)."]
    lines += ["", f"Same-checkpoint full-catalogue Recall@20 = {meta['full_catalogue']['Recall@20']:.4f}; "
        f"NDCG@20 = {meta['full_catalogue']['NDCG@20']:.4f}.", "",
        "These are descriptive results for one user sample and checkpoint, not a "
        "multi-seed significance claim. Retention can exceed 1 when candidate filtering "
        "removes items ranked above the target. RRF remains the serving default.", "",
        f"Source artifacts: [{Path(args.rrf).name}](../{args.rrf}), "
        f"[{Path(args.snake).name}](../{args.snake}).", "",
        "## Reproduction", "",
        "Run from the repository root with the processed base dataset, recall "
        "indexes and trained checkpoints available. These commands evaluate "
        "existing checkpoints; they do not train a model.", "",
        "```bash",
        "for method in rrf snake; do",
        "  python scripts/evaluate_pipeline.py \\",
        "    --two-tower-run results/runs/two_tower_base \\",
        f"    --mm-run results/runs/{Path(meta['ranker_run']).name} \\",
        "    --device cuda --two-tower-device cuda \\",
        "    --max-users 10000 --eval-seed 42 --recall-k 1000 \\",
        "    --candidate-sizes 100 200 500 1000 2000 --final-k 20 \\",
        '    --merge-method "$method" \\',
        '    --out "results/tables/recheck_${method}.csv" \\',
        '    --source-out "results/tables/recheck_${method}_sources.csv" \\',
        '    --json-out "results/tables/recheck_${method}.json"',
        "done",
        "```", "",
        "Regenerate this report from the completed original artifacts with "
        "`python analysis/compare_merge_results.py`."]
    report = ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
