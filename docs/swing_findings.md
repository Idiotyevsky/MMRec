# ItemCF vs Swing

Swing is an optional collaborative I2I channel, not a new neural model. The default serving channel remains ItemCF; existing artifacts are unchanged.

## Implementation

The graph uses binary **training-prefix interactions only**. For every distinct user pair `u < v`, each distinct item pair in their shared history receives the symmetric contribution:

```text
w(u) = (alpha1 + |I(u)|)^(-beta)
contribution(u,v) = w(u) * w(v) / (alpha2 + |I(u) ∩ I(v)|)
```

Fixed parameters: alpha1=5, alpha2=1, beta=0.3. Each item retains top-100 neighbors, matching ItemCF. There is no sampling, history cap during index construction, or item-popularity normalization. Distinct user pairs are counted once (rather than twice in an ordered-pair sum).

This is the weighted Swing variant described in the [Alibaba Cloud algorithm documentation](https://www.alibabacloud.com/help/en/airec/what-is-pai-rec/user-guide/swing-algorithm-tools), with the item-normalization factor fixed to 1. Online aggregation reuses ItemCF's recency decay 0.9 and maximum history 50, excluding PAD and all seen items. It does not fill unsupported items with zero-score candidates. Zero-train items have no collaborative edges in either algorithm.

## Evaluation protocol

Base test split, 10,000 uniformly sampled users, seed 42, user hash `3dd297000dca95e5`. Both methods run inside the same paired loop on identical histories and targets. This is a sample evaluation, not a full 100,000-user test or multi-seed result.

Four-channel comparisons replace only ItemCF with Swing: `Popular + {ItemCF or Swing} + Semantic + Two-Tower`. Shared channels are computed once per user. RRF uses k=60; each channel fetches at most 1,000 items, followed by the same final candidate budget and MM-SASRec checkpoint. Serving exploration reranking is off. No parameters are tuned on this test.

Ranker: `results/runs/mm_concat_20260917-075552_b84b95`. Dataset, mapping, index and checkpoint fingerprints are recorded in the JSON artifact.

## Single-channel recall

| Channel | Recall@100 | Recall@200 | Recall@500 | Recall@1000 | Mean candidates |
|---|---:|---:|---:|---:|---:|
| itemcf | 0.1465 | 0.1852 | 0.2361 | 0.2455 | 474.9 |
| swing | 0.1448 | 0.1689 | 0.1782 | 0.1787 | 162.3 |

In this run, Swing loses single-channel coverage at K=1000. Its smaller returned candidate set should not be interpreted as a quality gain by itself.

At K=1000: both hit the target for 1597 users; ItemCF-only hits = 858; Swing-only hits = 190. These are pairwise unique hits, not uniqueness against all four channels.

## Four-channel RRF pipeline

| Budget | Candidate R (ItemCF) | Candidate R (Swing) | R@20 (ItemCF) | R@20 (Swing) | NDCG@20 (ItemCF) | NDCG@20 (Swing) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.2061 | 0.1944 | 0.1409 | 0.1372 | 0.0663 | 0.0652 |
| 200 | 0.2611 | 0.2495 | 0.1459 | 0.1441 | 0.0673 | 0.0668 |
| 500 | 0.3662 | 0.3573 | 0.1439 | 0.1433 | 0.0662 | 0.0660 |
| 1000 | 0.4658 | 0.4671 | 0.1424 | 0.1415 | 0.0655 | 0.0651 |
| 2000 | 0.5880 | 0.5957 | 0.1406 | 0.1405 | 0.0648 | 0.0649 |

Same-checkpoint full-catalogue Recall@20=0.1396, NDCG@20=0.064395.

## Target popularity

Buckets use train frequency. Warm-tail excludes zero-train items, which are reported separately. No simulated-cold experiment was run here.

| Bucket | Channel | Users | Recall@1000 |
|---|---|---:|---:|
| head | itemcf | 3329 | 0.2986 |
| head | swing | 3329 | 0.3253 |
| middle | itemcf | 2186 | 0.2360 |
| middle | swing | 2186 | 0.1551 |
| tail_warm | itemcf | 4265 | 0.2216 |
| tail_warm | swing | 4265 | 0.0856 |
| zero_train | itemcf | 220 | 0.0000 |
| zero_train | swing | 220 | 0.0000 |

Swing is stronger on head targets but weaker on warm-tail targets at K=1000 in this sample. The overall loss is therefore not uniform across popularity buckets.

## Interpretation at 1,000 candidates

- Swing minus ItemCF, candidate_recall: +0.001300; paired approximate 95% CI [-0.003301, +0.005901].
- Swing minus ItemCF, Recall@20: -0.000900; paired approximate 95% CI [-0.002185, +0.000385].
- Swing minus ItemCF, NDCG@20: -0.000340; paired approximate 95% CI [-0.000785, +0.000106].

This test does not establish an improvement in final Recall@20 or NDCG@20 at 1,000 candidates. Keep ItemCF as the default and Swing as an experimental alternative; these results do not justify a replacement.

Intervals use the sample standard error of per-user paired differences with a normal approximation. They describe this checkpoint/user sample, are not corrected for multiple budgets, and are not a multi-seed or online test. No serving default is changed based on this one comparison.

## Reproduction

```bash
python scripts/build_swing.py
python scripts/evaluate_swing.py --device cuda
python analysis/report_swing.py
```

The paired evaluator saves progress every 1,000 users. Add `--resume` with otherwise identical arguments to resume; mismatching data, code, configuration or artifact fingerprints are rejected.

The generic evaluators also support `--collaborative-channel swing --swing-index artifacts/swing_neighbors.npz`; choose new output paths to preserve historical results.

Artifacts: [paired test](../results/tables/swing_comparison.json), [single-channel recall](../results/tables/swing_recall.csv), [pipeline](../results/tables/swing_pipeline.csv), [paired deltas](../results/tables/swing_paired_deltas.csv), [popularity](../results/tables/swing_popularity.csv), [index report](../artifacts/swing_report.json).
