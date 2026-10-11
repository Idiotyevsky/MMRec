#!/usr/bin/env python
"""Generate the Swing findings strictly from a completed paired test artifact."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(data):
    protocol = data["protocol"]
    params = data["provenance"]["swing_parameters"]
    lines = ["# ItemCF vs Swing", "",
        "Swing is an optional collaborative I2I channel, not a new neural model. "
        "The default serving channel remains ItemCF; existing artifacts are unchanged.", "",
        "## Implementation", "",
        "The graph uses binary **training-prefix interactions only**. For every "
        "distinct user pair `u < v`, each distinct item pair in their shared history "
        "receives the symmetric contribution:", "",
        "```text", "w(u) = (alpha1 + |I(u)|)^(-beta)",
        "contribution(u,v) = w(u) * w(v) / (alpha2 + |I(u) ∩ I(v)|)", "```", "",
        f"Fixed parameters: alpha1={params['alpha1']:g}, alpha2={params['alpha2']:g}, "
        f"beta={params['beta']:g}. Each item retains top-{data['provenance']['top_m']} "
        "neighbors, matching ItemCF. There is no sampling, history cap during index "
        "construction, or item-popularity normalization. Distinct user pairs are "
        "counted once (rather than twice in an ordered-pair sum).", "",
        "This is the weighted Swing variant described in the "
        "[Alibaba Cloud algorithm documentation]"
        "(https://www.alibabacloud.com/help/en/airec/what-is-pai-rec/user-guide/swing-algorithm-tools), "
        "with the item-normalization factor fixed to 1. Online aggregation reuses "
        "ItemCF's recency decay 0.9 and maximum history 50, excluding PAD and all "
        "seen items. It does not fill unsupported items with zero-score candidates. "
        "Zero-train items have no collaborative edges in either algorithm.", "",
        "## Evaluation protocol", "",
        f"Base test split, {protocol['num_users']:,} uniformly sampled users, "
        f"seed {protocol['eval_seed']}, user hash `{protocol['user_subset_hash']}`. "
        "Both methods run inside the same paired loop on identical histories and targets. "
        "This is a sample evaluation, not a full 100,000-user test or multi-seed result.", "",
        "Four-channel comparisons replace only ItemCF with Swing: "
        "`Popular + {ItemCF or Swing} + Semantic + Two-Tower`. Shared channels are "
        "computed once per user. RRF uses k=60; each channel fetches at most 1,000 "
        "items, followed by the same final candidate budget and MM-SASRec checkpoint. "
        "Serving exploration reranking is off. No parameters are tuned on this test.", "",
        f"Ranker: `{protocol['mm_run']}`. Dataset, mapping, index and checkpoint "
        "fingerprints are recorded in the JSON artifact.", "",
        "## Single-channel recall", "",
        "| Channel | Recall@100 | Recall@200 | Recall@500 | Recall@1000 | Mean candidates |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in data["retrieval"]:
        lines.append(f"| {r['channel']} | " + " | ".join(f"{r[f'Recall@{k}']:.4f}" for k in (100,200,500,1000))
                     + f" | {r['mean_candidates']:.1f} |")
    retrieval = {r["channel"]:r for r in data["retrieval"]}
    if retrieval["swing"]["Recall@1000"] < retrieval["itemcf"]["Recall@1000"]:
        lines += ["", "In this run, Swing loses single-channel coverage at K=1000. "
            "Its smaller returned candidate set should not be interpreted as "
            "a quality gain by itself."]
    o = data["overlap"]
    lines += ["", f"At K=1000: both hit the target for {o['both_hits']} users; "
        f"ItemCF-only hits = {o['itemcf_only_hits']}; Swing-only hits = {o['swing_only_hits']}. "
        "These are pairwise unique hits, not uniqueness against all four channels.", "",
        "## Four-channel RRF pipeline", "",
        "| Budget | Candidate R (ItemCF) | Candidate R (Swing) | R@20 (ItemCF) | R@20 (Swing) | NDCG@20 (ItemCF) | NDCG@20 (Swing) |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    rows = {(r["collaborative_channel"], r["candidate_k"]):r for r in data["pipeline"]}
    for k in protocol["candidate_sizes"]:
        vals = [f"{rows[c,k][m]:.4f}" for m in ("candidate_recall", "Recall@20", "NDCG@20")
                for c in ("itemcf", "swing")]
        lines.append(f"| {k} | " + " | ".join(vals) + " |")
    full = data["full_catalogue"]
    lines += ["", f"Same-checkpoint full-catalogue Recall@20={full['Recall@20']:.4f}, "
        f"NDCG@20={full['NDCG@20']:.6f}.", "",
        "## Target popularity", "",
        "Buckets use train frequency. Warm-tail excludes zero-train items, which "
        "are reported separately. No simulated-cold experiment was run here.", "",
        "| Bucket | Channel | Users | Recall@1000 |", "|---|---|---:|---:|"]
    for r in data["popularity"]:
        value = f"{r['Recall@1000']:.4f}" if r['Recall@1000'] is not None else "N/A"
        lines.append(f"| {r['bucket']} | {r['channel']} | {r['num_users']} | {value} |")
    groups = {(r["bucket"],r["channel"]):r["Recall@1000"] for r in data["popularity"]}
    if (all(groups.get((b,c)) is not None for b in ("head", "tail_warm") for c in ("itemcf", "swing"))
            and groups["head","swing"] > groups["head","itemcf"]
            and groups["tail_warm","swing"] < groups["tail_warm","itemcf"]):
        lines += ["", "Swing is stronger on head targets but weaker on warm-tail targets "
            "at K=1000 in this sample. The overall loss is therefore not uniform "
            "across popularity buckets."]
    lines += ["", "## Interpretation at 1,000 candidates", ""]
    for r in data["paired_differences"]:
        if r["candidate_k"] == 1000:
            lines.append(f"- Swing minus ItemCF, {r['metric']}: {r['delta']:+.6f}; "
                         f"paired approximate 95% CI [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}].")
    final_deltas = [r for r in data["paired_differences"] if r["candidate_k"] == 1000
                    and r["metric"] in ("Recall@20", "NDCG@20")]
    if len(final_deltas) == 2 and all(r["ci95_low"] <= 0 for r in final_deltas):
        lines += ["", "This test does not establish an improvement in final Recall@20 "
            "or NDCG@20 at 1,000 candidates. Keep ItemCF as the default and Swing "
            "as an experimental alternative; these results do not justify a replacement."]
    lines += ["", "Intervals use the sample standard error of per-user paired differences "
        "with a normal approximation. They describe this checkpoint/user sample, "
        "are not corrected for multiple budgets, and are not a multi-seed or online test. "
        "No serving default is changed based on this one comparison.", "",
        "## Reproduction", "", "```bash", "python scripts/build_swing.py",
        "python scripts/evaluate_swing.py --device cuda", "python analysis/report_swing.py", "```", "",
        "The paired evaluator saves progress every 1,000 users. Add `--resume` "
        "with otherwise identical arguments to resume; mismatching data, code, "
        "configuration or artifact fingerprints are rejected.", "",
        "The generic evaluators also support `--collaborative-channel swing "
        "--swing-index artifacts/swing_neighbors.npz`; choose new output paths "
        "to preserve historical results.", "",
        "Artifacts: [paired test](../results/tables/swing_comparison.json), "
        "[single-channel recall](../results/tables/swing_recall.csv), "
        "[pipeline](../results/tables/swing_pipeline.csv), "
        "[paired deltas](../results/tables/swing_paired_deltas.csv), "
        "[popularity](../results/tables/swing_popularity.csv), "
        "[index report](../artifacts/swing_report.json)."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default="results/tables/swing_comparison.json")
    ap.add_argument("--out", default="docs/swing_findings.md")
    args = ap.parse_args()
    report = render(json.loads((ROOT / args.artifact).read_text()))
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
