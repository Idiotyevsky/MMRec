#!/usr/bin/env python
"""Generate MMR findings from the completed evaluation artifact."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(data):
    protocol = data["protocol"]
    rows = data["results"]
    k = protocol["final_k"]
    lines = ["# MMR serving rerank findings", "",
        "MMR adds a configurable relevance/diversity tradeoff after MM-SASRec ranking. "
        "It is opt-in. The existing recall merge and score-order baseline remain available.", "",
        "## Implementation", "",
        "```text", "MMR(i) = λ * minmax(ranking_score(i))",
        "         - (1-λ) * max cosine(content(i), content(selected))", "```", "",
        "The first selection is the highest-ranked item. The relevance scale is "
        "computed once over the ranking window; ties preserve input order. "
        "Lambda 1 returns exactly the original score order. Similarity uses fixed "
        "text+image content features, normalized per item. Greedy updates require "
        "O(window*K*D), with no full catalogue pairwise matrix.", "",
        "The implementation follows the relevance/redundancy criterion in "
        "[Carbonell and Goldstein (SIGIR 1998)]"
        "(https://www.cs.cmu.edu/afs/cs/Web/People/jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf). "
        "Missing table artifacts trigger an explicit score-order fallback. "
        "Unavailable item vectors get conservative redundancy 1 after the first "
        "selection, rather than an artificial novelty advantage. If all vectors "
        "are unavailable the ranking is preserved.", "",
        "Seen filtering and deduplication precede MMR. Exploration, when enabled, "
        "selects its quota-constrained set first; MMR only reorders that set. "
        "Original ranking scores and all source traces remain intact. API and "
        "Inspector expose original position, final rank, MMR relevance, maximum "
        "redundancy and selection-time objective.", "",
        "## Protocol", "",
        f"Base test split, {protocol['num_users']:,} uniformly sampled users; "
        f"seed {protocol['eval_seed']}, user hash `{protocol['user_subset_hash']}`. "
        "Histories are train prefix plus the observed validation item; test targets "
        "are used only for evaluation.", "",
        f"Four recall channels (Popular, ItemCF, Semantic, Two-Tower), top-1000 "
        f"per channel, RRF k=60, final candidate budget {protocol['candidate_k']}. "
        f"Same MM-SASRec checkpoint `{protocol['mm_run']}` for all methods. "
        f"MMR window={protocol['window']}, final K={k}, exploration off. "
        "A fingerprinted ranked-candidate cache fixes the inputs across lambdas. "
        "This is a 10K-user sample by default, not a full 100K-user test.", "",
        "Lambda 0.8 is the predeclared opt-in default. The fixed test lambda grid "
        "is descriptive and does not select or promote a serving default.", "",
        "## Relevance and content diversity", "",
        f"| Method | Recall@{k} | NDCG@{k} | Content ILD | Near-duplicate pairs | Catalogue coverage | CPU rerank mean ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value, digits=4):
        return f"{value:.{digits}f}" if value is not None else "N/A"
    for r in rows:
        vals = [r[f"Recall@{k}"], r[f"NDCG@{k}"], r["ILD"], r["near_duplicate_pair_share"],
                r["catalogue_coverage"], r["rerank_cpu_mean_ms"], r["rerank_cpu_p95_ms"]]
        lines.append("| " + r["method"] + " | " + " | ".join(fmt(v) for v in vals) + " |")
    baseline = rows[0]
    default = next((r for r in rows if r["method"] == "mmr_0.8"), None)
    if default:
        lines += ["", "At the predeclared lambda 0.8, relative to score ordering: "
            f"Recall@{k} changes by {100*(default[f'Recall@{k}']-baseline[f'Recall@{k}']):+.2f} "
            f"percentage points; NDCG@{k} changes by "
            f"{default[f'NDCG@{k}']-baseline[f'NDCG@{k}']:+.6f}; content ILD "
            f"changes from {fmt(baseline['ILD'])} to {fmt(default['ILD'])}. "
            "The diversity result alone does not establish better relevance or "
            "online satisfaction."]
    lines += ["", f"Candidate recall is identical for all methods: {baseline['candidate_recall']:.4f}. "
        "Reranking cannot recover a target absent from this fixed recall pool.", "",
        "ILD = 1 - mean pairwise content cosine. Near-duplicate pair share uses "
        "cosine >= 0.9. Both metrics exclude unavailable feature pairs; valid-pair "
        "fraction and fallback-user counts are recorded in the CSV. Content similarity "
        "is a proxy and does not measure explicit category or creator diversity.", "",
        "CPU latency measures only reranking, including trace diagnostics, with "
        "one-thread BLAS during this run. It excludes recall, model inference and "
        "HTTP overhead. The shared candidate cache may be produced on GPU; "
        "MMR itself runs on CPU.", "",
        "## Exposure", "",
        "| Method | Unique items | Head share | Tail share | Zero-train share | Mean changed positions |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['unique_items']} | {r['head_share']:.4f} | "
                     f"{r['tail_share']:.4f} | {r['zero_train_share']:.4f} | {r['mean_changed_positions']:.2f} |")
    lines += ["", "Shares are over served items, using train-frequency buckets. "
        "Tail includes zero-train items; zero-train exposure is also listed separately. "
        "These are exposure shares, not bucket-specific target recall.", "",
        "## Paired changes", "",
        "| Method | Metric | Delta vs score ordering | Approximate 95% CI |",
        "|---|---|---:|---|"]
    for r in data["paired_differences"]:
        label = {"hit":f"Recall@{k}", "ndcg":f"NDCG@{k}", "ild":"ILD"}[r["metric"]]
        lines.append(f"| {r['method']} | {label} | {r['delta']:+.6f} | "
                     f"[{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] |")
    lines += ["", "Intervals use normal approximations to per-user paired differences. "
        "They are descriptive for this checkpoint and sample, not multi-seed or "
        "online results, and are not adjusted for the lambda grid.", "",
        "## Use and reproduction", "", "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\",
        "  python scripts/evaluate_mmr.py --device cuda",
        "# Resume a partial cache or replay completed ranking with a new lambda grid:",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\",
        "  python scripts/evaluate_mmr.py --device cuda --reuse-cache",
        "python analysis/report_mmr.py", "```", "",
        "Serving: `GET /users/7/recommend?mmr=true&mmr_lambda=0.8&mmr_candidate_k=100`. "
        "The same parameters work on `/users/7/inspect`; Live Demo and Inspector "
        "offer MMR controls. MMR remains off by default.", "",
        "Artifacts: [tradeoff CSV](../results/tables/mmr_tradeoff.csv), "
        "[evaluation and fingerprints](../results/tables/mmr_evaluation.json). "
        "Regenerate this document with `python analysis/report_mmr.py`."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default="results/tables/mmr_evaluation.json")
    ap.add_argument("--out", default="docs/mmr_findings.md")
    args = ap.parse_args()
    report = render(json.loads((ROOT / args.artifact).read_text()))
    (ROOT / args.out).write_text(report)
    print(report)


if __name__ == "__main__":
    main()
