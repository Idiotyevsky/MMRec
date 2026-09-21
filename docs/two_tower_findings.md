# Content-aware Two-Tower findings

This document is intentionally artifact-gated.  No base-data Two-Tower run is
present in this checkout, so it makes no numerical claim and does not recommend
changing the serving default.

The implemented route is retrieval: a 128-D user query is matched against
precomputed, L2-normalized 128-D item vectors with exact inner product.  The
item tower has text/image branches and availability bits, but no trainable
item-ID embedding, so zero-train and simulated-cold items remain encodable.
Semantic Recall is training-free recency-weighted content similarity; Two-Tower
uses train-prefix next-item supervision to adapt both content projections and
the user projection.

To generate evidence, train and export first:

```bash
python scripts/train.py --config configs/two_tower.yaml --run-dir results/runs/two_tower_base
python scripts/export_two_tower_embeddings.py --run-dir results/runs/two_tower_base
python scripts/evaluate_recall.py --two-tower-run results/runs/two_tower_base \
  --out results/tables/two_tower_recall.csv
```

The reported comparison must use the same users, targets, seen masking and
fixed total candidate budget.  Record Semantic-vs-Two-Tower and
ItemCF-vs-Two-Tower hit overlap, Jaccard, mean candidate overlap and unique
target hits; slice by train-frequency Head/Middle/Tail/Zero-train and run the
separate `cold10` split for simulated cold.  Until those artifacts exist, the
channel remains opt-in (`two_tower_checkpoint`) in serving.

`IndexFlatIP` is exact, not ANN.  At 100M items retain these tower outputs but
publish versioned vectors to a sharded HNSW or IVF-PQ index, apply filtering/
over-fetching before ranking, and validate approximate-retrieval recall against
an exact held-out baseline.
