# Design notes

Why the code is shaped the way it is. Read this before changing a module.

## 1. One item representation, one scoring path

`SASRec` and `MM-SASRec` share everything except how the item embedding table is
produced:

```
SASRec     : nn.Embedding(num_items+1, H)
MM-SASRec  : MultimodalItemEncoder(...)  ->  (num_items+1, H)
                    │
                    ▼
            SequenceEncoder (position embedding, causal Transformer)
                    │
                    ▼
            h_u = out[:, -1]        # left padding => last slot is the newest item
                    │
                    ▼
            score(u, i) = h_u · e_i
```

Both expose the same three methods — `encode(input_ids)`, `item_embeddings(ids)`,
`all_item_embeddings()` — so the trainer, the evaluator, the embedding exporter
and the serving layer are written once and cannot diverge between models.

This is also why the ID-only vs multimodal comparison is clean: the *only*
difference between the two runs is the embedding table. Position embeddings,
attention, loss, negatives, optimiser and evaluation are byte-for-byte the same
code path.

## 2. Left padding and the attention mask

Sequences are left-padded so that `out[:, -1]` is always the most recent item
regardless of history length. That creates a subtle problem: a padding *query*
attends to nothing (all its keys are padding), and `softmax` over an all-masked
row is `NaN`. PyTorch's fused attention propagates that `NaN` into the real
positions.

The fix is in `SequenceEncoder.attention_mask`: we build a per-batch
`(B*num_heads, L, L)` boolean mask where

* the causal triangle blocks `j > i`, and
* padding keys are blocked for every real query, and
* each padding query may attend only to **itself**.

Every row therefore has at least one legal key, no `NaN` is produced, and no real
position ever sees a padding slot. The cost is ~30 % more attention time than the
2D path, which is irrelevant at `L = 50`.

`tests/test_attention_mask.py` proves causality, proves padding isolation, and
includes a *control* experiment showing the padding mask is not vacuous.

## 3. Missing modalities are not zeros

A zero feature vector and an absent feature are different things. The encoder
carries a boolean availability mask per modality (`{text,image,video}_available`
buffers, built from the feature file at load time). Downstream:

* `GatedFusion` adds `finfo.min` to unavailable logits before the softmax, so the
  available weights still sum to 1 and a missing modality gets **exactly** 0;
* `ConcatFusion` zeroes the missing embedding *and* concatenates the availability
  bit, so the MLP can tell "no image" from "an image that projects to zero";
* a PAD position is fully masked and its fused embedding is exactly zero.

`tests/test_fusion.py` checks each of these, including that changing the content
of a missing modality does not change the output.

## 4. Cold items

Under the simulated cold protocol a cold item has **zero training interactions**.
Its ID embedding is therefore untrained. Rather than letting a random vector act
as a hidden prior, `zero_cold_id: true` (default) zeroes the ID component for
cold items at inference in every model that owns an ID branch:

* `SASRec` / `BPR-MF`: the ID embedding is masked to zero;
* `MM-SASRec`: the ID gate weight is forced to zero.

So the cold experiment measures exactly one thing: **can content features stand
in for missing collaborative signal?**

## 5. Regularisation that targets the research question

`id_dropout_prob` and `modality_dropout_prob` are training-only. On each step a
Bernoulli draw removes modalities from the item representation. If every modality
of a *valid* item would be removed, one is restored (ID if present, otherwise the
first content branch), so an item is never represented by an all-zero vector.

The hypothesis under test — "a strong ID branch suppresses the content branches"
— is answered by comparing runs that differ only in `id_dropout_prob`, and by
reading the exported gate weights (`scripts/export_gates.py`), not by assertion.

## 6. Negative sampling

`NegativeSampler` draws from a uniform or `freq^0.75` distribution and then
rejects anything that is:

* PAD (item 0),
* the positive target of the current batch row,
* any interaction of that user in the whole dataset (train + val + test).

Membership is an `O(log n)` `searchsorted` over a sorted
`user * num_items + item` key array, so exclusion is exact rather than
probabilistic. `tests/test_negative_sampling.py` asserts each guarantee.

## 7. Evaluation

One ranking pass per split produces a per-user `rank` array, which is saved to
`test_ranking.npz`. Overall, cold and per-popularity-bucket tables are all slices
of that same array, so they cannot disagree with each other. Chunked item scoring
keeps memory flat at any catalogue size.

Ranks use the optimistic tie policy (`1 + #{strictly higher}`) — deterministic and
independent of sort implementation. A target outside a restricted candidate set
is assigned the `NOT_RETRIEVED` sentinel rather than inheriting a small rank from
a short candidate list.

## 8. Reproducibility

`set_seed` covers `random`, `numpy`, `torch` (CPU + CUDA) and `PYTHONHASHSEED`.
Data is fully in memory with `num_workers = 0`, so no worker seeding ambiguity
exists. Every run writes its config, environment fingerprint, training log,
metrics and raw ranking arrays. Checkpoints embed a dataset hash (train
frequency + cold mask + offsets) and refuse to load into a different dataset.

## 9. Deliberate non-goals

* No DDP / distributed training — a single L40S trains every model in the project
  in minutes; adding distributed machinery would obscure the recommendation logic.
* No Hydra / registries / factories-of-factories. A 60-line YAML loader with
  dotted overrides is enough and is readable in one sitting.
* No training of large visual or language backbones. Pre-extracted features in,
  lightweight projections out.
* No LLM. The generative recommender is a ~4-layer causal Transformer over
  semantic code tokens.

## 10. Module map

| path | responsibility |
|---|---|
| `src/data/preprocess.py` | raw → processed, mappings, splits, cold split |
| `src/data/dataset.py` | `ProcessedData`, train/eval torch datasets |
| `src/data/negative_sampler.py` | exclusion-safe negative sampling |
| `src/data/popularity.py` | training-only frequency and buckets |
| `src/models/item_encoder.py` | multimodal item encoder + modality dropout |
| `src/models/fusion.py` | concat / gated fusion with masking |
| `src/models/sasrec.py` | causal sequence encoder + ID-only model |
| `src/models/mm_sasrec.py` | multimodal model |
| `src/models/bpr.py`, `popular.py` | baselines |
| `src/models/rqvae.py`, `semantic_id.py`, `generative_rec.py` | semantic-ID extension |
| `src/evaluation/metrics.py` | rank-based metrics |
| `src/evaluation/evaluator.py` | full-ranking evaluator |
| `src/evaluation/slicing.py` | cold / popularity slices of one ranking pass |
| `src/training/trainer.py` | training loop, AMP, checkpointing, early stopping |
| `src/retrieval/faiss_index.py` | ANN index with exact fallback |
| `src/serving/recommender.py` | history → user vector → ANN → top-K |
