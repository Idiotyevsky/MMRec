# Four-channel RRF vs snake

Completed test evaluation: 10,000 users, seed 42, user sample hash `3dd297000dca95e5`.

Ranker run: `mm_concat_20260917-075552_b84b95`. Both methods use the same checkpoint, histories, targets and per-channel top-1000 inputs. Final candidate budgets are identical. Exploration reranking is off.

Snake is the ping-pong variant `Popular → ItemCF → Semantic → Two-Tower → Semantic → ItemCF → …`. Each visit consumes one list entry; a duplicate consumes its visit but is emitted only once. Interior channels receive twice the visits of the endpoints while all lists remain active. This is not an equal-quota interleave. All contributing source traces are retained.

| Budget | Candidate R (RRF) | Candidate R (snake) | R@20 (RRF) | R@20 (snake) | NDCG@20 (RRF) | NDCG@20 (snake) |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.2061 | 0.1646 | 0.1409 | 0.1332 | 0.0663 | 0.0641 |
| 200 | 0.2611 | 0.2246 | 0.1459 | 0.1449 | 0.0673 | 0.0671 |
| 500 | 0.3662 | 0.3230 | 0.1439 | 0.1439 | 0.0662 | 0.0664 |
| 1000 | 0.4658 | 0.4281 | 0.1424 | 0.1430 | 0.0655 | 0.0656 |
| 2000 | 0.5880 | 0.5606 | 0.1406 | 0.1408 | 0.0648 | 0.0649 |

## Observations

Snake has lower candidate recall at 5/5 tested budgets. Candidate recall and final ranking quality are different metrics; a smaller candidate hit rate need not imply a lower final NDCG.

At 1,000 candidates, snake minus RRF is -3.77 percentage points in candidate recall, +0.06 percentage points in final Recall@20, and +0.000166 in NDCG@20. Recall retention is 1.0201 (RRF) vs 1.0244 (snake).

Same-checkpoint full-catalogue Recall@20 = 0.1396; NDCG@20 = 0.0644.

These are descriptive results for one user sample and checkpoint, not a multi-seed significance claim. Retention can exceed 1 when candidate filtering removes items ranked above the target. RRF remains the serving default.

Source artifacts: [two_tower_pipeline_rrf_control.json](../results/tables/two_tower_pipeline_rrf_control.json), [two_tower_pipeline_snake.json](../results/tables/two_tower_pipeline_snake.json).

## Reproduction

Run from the repository root with the processed base dataset, recall indexes and trained checkpoints available. These commands evaluate existing checkpoints; they do not train a model.

```bash
for method in rrf snake; do
  python scripts/evaluate_pipeline.py \
    --two-tower-run results/runs/two_tower_base \
    --mm-run results/runs/mm_concat_20260917-075552_b84b95 \
    --device cuda --two-tower-device cuda \
    --max-users 10000 --eval-seed 42 --recall-k 1000 \
    --candidate-sizes 100 200 500 1000 2000 --final-k 20 \
    --merge-method "$method" \
    --out "results/tables/recheck_${method}.csv" \
    --source-out "results/tables/recheck_${method}_sources.csv" \
    --json-out "results/tables/recheck_${method}.json"
done
```

Regenerate this report from the completed original artifacts with `python analysis/compare_merge_results.py`.
