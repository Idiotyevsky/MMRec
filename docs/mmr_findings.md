# MMR serving rerank findings

MMR adds a configurable relevance/diversity tradeoff after MM-SASRec ranking. It is opt-in. The existing recall merge and score-order baseline remain available.

## Implementation

```text
MMR(i) = λ * minmax(ranking_score(i))
         - (1-λ) * max cosine(content(i), content(selected))
```

The first selection is the highest-ranked item. The relevance scale is computed once over the ranking window; ties preserve input order. Lambda 1 returns exactly the original score order. Similarity uses fixed text+image content features, normalized per item. Greedy updates require O(window*K*D), with no full catalogue pairwise matrix.

The implementation follows the relevance/redundancy criterion in [Carbonell and Goldstein (SIGIR 1998)](https://www.cs.cmu.edu/afs/cs/Web/People/jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf). Missing table artifacts trigger an explicit score-order fallback. Unavailable item vectors get conservative redundancy 1 after the first selection, rather than an artificial novelty advantage. If all vectors are unavailable the ranking is preserved.

Seen filtering and deduplication precede MMR. Exploration, when enabled, selects its quota-constrained set first; MMR only reorders that set. Original ranking scores and all source traces remain intact. API and Inspector expose original position, final rank, MMR relevance, maximum redundancy and selection-time objective.

## Protocol

Base test split, 10,000 uniformly sampled users; seed 42, user hash `3dd297000dca95e5`. Histories are train prefix plus the observed validation item; test targets are used only for evaluation.

Four recall channels (Popular, ItemCF, Semantic, Two-Tower), top-1000 per channel, RRF k=60, final candidate budget 1000. Same MM-SASRec checkpoint `results/runs/mm_concat_20260917-075552_b84b95` for all methods. MMR window=100, final K=20, exploration off. A fingerprinted ranked-candidate cache fixes the inputs across lambdas. This is a 10K-user sample by default, not a full 100K-user test.

Lambda 0.8 is the predeclared opt-in default. The fixed test lambda grid is descriptive and does not select or promote a serving default.

## Relevance and content diversity

| Method | Recall@20 | NDCG@20 | Content ILD | Near-duplicate pairs | Catalogue coverage | CPU rerank mean ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| score_only | 0.1424 | 0.0655 | 0.7181 | 0.0004 | 0.5385 | 0.1243 | 0.1523 |
| mmr_0.7 | 0.1413 | 0.0647 | 0.7378 | 0.0002 | 0.5409 | 4.5323 | 5.2606 |
| mmr_0.8 | 0.1417 | 0.0652 | 0.7295 | 0.0003 | 0.5412 | 4.5116 | 5.2468 |
| mmr_0.9 | 0.1418 | 0.0653 | 0.7231 | 0.0003 | 0.5399 | 4.4910 | 5.2261 |
| mmr_0.95 | 0.1423 | 0.0654 | 0.7205 | 0.0004 | 0.5394 | 4.4946 | 5.2263 |

At the predeclared lambda 0.8, relative to score ordering: Recall@20 changes by -0.07 percentage points; NDCG@20 changes by -0.000316; content ILD changes from 0.7181 to 0.7295. The diversity result alone does not establish better relevance or online satisfaction.

Candidate recall is identical for all methods: 0.4658. Reranking cannot recover a target absent from this fixed recall pool.

ILD = 1 - mean pairwise content cosine. Near-duplicate pair share uses cosine >= 0.9. Both metrics exclude unavailable feature pairs; valid-pair fraction and fallback-user counts are recorded in the CSV. Content similarity is a proxy and does not measure explicit category or creator diversity.

CPU latency measures only reranking, including trace diagnostics, with one-thread BLAS during this run. It excludes recall, model inference and HTTP overhead. The shared candidate cache may be produced on GPU; MMR itself runs on CPU.

## Exposure

| Method | Unique items | Head share | Tail share | Zero-train share | Mean changed positions |
|---|---:|---:|---:|---:|---:|
| score_only | 10629 | 0.5309 | 0.2324 | 0.0000 | 0.00 |
| mmr_0.7 | 10676 | 0.5278 | 0.2326 | 0.0000 | 14.34 |
| mmr_0.8 | 10682 | 0.5291 | 0.2325 | 0.0000 | 12.06 |
| mmr_0.9 | 10656 | 0.5299 | 0.2326 | 0.0000 | 7.94 |
| mmr_0.95 | 10647 | 0.5304 | 0.2326 | 0.0000 | 4.66 |

Shares are over served items, using train-frequency buckets. Tail includes zero-train items; zero-train exposure is also listed separately. These are exposure shares, not bucket-specific target recall.

## Paired changes

| Method | Metric | Delta vs score ordering | Approximate 95% CI |
|---|---|---:|---|
| mmr_0.7 | Recall@20 | -0.001100 | [-0.003184, +0.000984] |
| mmr_0.7 | NDCG@20 | -0.000804 | [-0.001408, -0.000201] |
| mmr_0.7 | ILD | +0.019739 | [+0.019371, +0.020106] |
| mmr_0.8 | Recall@20 | -0.000700 | [-0.002304, +0.000904] |
| mmr_0.8 | NDCG@20 | -0.000316 | [-0.000771, +0.000138] |
| mmr_0.8 | ILD | +0.011487 | [+0.011227, +0.011747] |
| mmr_0.9 | Recall@20 | -0.000600 | [-0.001743, +0.000543] |
| mmr_0.9 | NDCG@20 | -0.000230 | [-0.000541, +0.000082] |
| mmr_0.9 | ILD | +0.005087 | [+0.004923, +0.005251] |
| mmr_0.95 | Recall@20 | -0.000100 | [-0.000908, +0.000708] |
| mmr_0.95 | NDCG@20 | -0.000087 | [-0.000311, +0.000136] |
| mmr_0.95 | ILD | +0.002442 | [+0.002325, +0.002558] |

Intervals use normal approximations to per-user paired differences. They are descriptive for this checkpoint and sample, not multi-seed or online results, and are not adjusted for the lambda grid.

## Use and reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/evaluate_mmr.py --device cuda
# Resume a partial cache or replay completed ranking with a new lambda grid:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/evaluate_mmr.py --device cuda --reuse-cache
python analysis/report_mmr.py
```

Serving: `GET /users/7/recommend?mmr=true&mmr_lambda=0.8&mmr_candidate_k=100`. The same parameters work on `/users/7/inspect`; Live Demo and Inspector offer MMR controls. MMR remains off by default.

Artifacts: [tradeoff CSV](../results/tables/mmr_tradeoff.csv), [evaluation and fingerprints](../results/tables/mmr_evaluation.json). Regenerate this document with `python analysis/report_mmr.py`.
