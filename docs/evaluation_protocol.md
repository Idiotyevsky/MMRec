# Evaluation protocol

This document is the contract every reported number has to satisfy. If a result
in the README cannot be traced to a command in this document and an artifact in
`results/`, it does not belong in the README.

## 1. Split

Strictly chronological, leave-one-out, per user. The raw timestamps are used to
order each user's history; no random split is ever performed.

```
user history (sorted by timestamp):  i1  i2  i3  i4  i5  i6  i7
train items                        :  i1  i2  i3  i4  i5
validation target                  :  i6
test target                        :  i7
```

* Validation history = the train items.
* Test history = train items **+** the validation target.
* Training uses the standard autoregressive shift over the train items
  (`input = [i1..i4]`, `target = [i2..i5]`), with the loss computed only at
  positions whose *input* token is not PAD.

`ProcessedData.assert_no_target_leak()` runs on **every** dataset load and raises
if a validation target appears in the train history, or a test target appears in
the train+val history.

## 2. Candidate set and seen-item masking

Evaluation is **full ranking over the entire catalogue** (`num_items = 19738`).

For each user the evaluator:

1. scores every item `1..num_items` (PAD, id 0, is forced to `-inf`);
2. sets the score of every item in the history fed to the model to `-inf`;
3. **never** masks the ground-truth target — `_mask_seen` re-enables it and the
   evaluator asserts the target score is finite afterwards.

Test history therefore masks `i1..i6`, and `i7` remains rankable.

Candidate dot-products are computed in item chunks (`evaluation.item_chunk_size`,
default 4096), while the batch-level full score matrix is retained for exact
ranking: the `batch × chunk` product is the largest temporary, and ranking runs
over the entire catalogue — no candidate pre-filtering and no approximate top-k.

**Rank definition.** `rank = 1 + #{items scoring strictly higher} +
(#{items scoring equal} − 1) / 2` — the tie-neutral (midpoint) rank. Ties are
split evenly, so the metric does not depend on any sort implementation's
tie-breaking, and it is the only tie policy that neither rewards nor punishes a
model. This matters concretely for cold items: an ID-only model scores every
cold item exactly 0, and an optimistic rank would report a perfect hit for all
of them. Every metric is a function of this single rank array, so overall /
cold / per-bucket numbers are guaranteed to come from one and the same ranking
pass.

## 3. Metrics

Primary (reported for every run):

```
Recall@5,  Recall@10,  Recall@20
NDCG@5,    NDCG@10,    NDCG@20
```

Secondary: `MRR@k`, `HitRate@k`, `Coverage@20` (fraction of the catalogue that
appears in at least one user's top-20).

* Single ground-truth item, so `HitRate@k == Recall@k`; both are reported for
  convenience and the equality is asserted by construction.
* `NDCG@k = mean( 1 / log2(rank+1) )` over users with `rank <= k`.
* Full ranking only. **No sampled evaluation is reported anywhere.** The
  evaluator has no sampled mode; if one is ever added it must be tagged
  `sampled` in both the artifact and any table that shows it.

## 4. Validation vs test

Validation is used for early stopping and checkpoint selection, so validation
numbers are **optimistically biased**. Test numbers are reported from the best
checkpoint and are the only ones used for conclusions. Both are written to
`metrics.json` so the gap is visible rather than hidden.

## 5. Simulated cold-item protocol

Implemented in `src/data/preprocess.py` with `--cold-ratio 0.1 --cold-seed 42`,
producing `data/processed/cold10/`.

1. Perform the ordinary chronological split first.
2. Candidate cold items = items appearing as a validation **or** test target at
   least `cold_min_eval_occurrences` times (default 1) — otherwise they could not
   be evaluated at all. From those, a fixed-seed random sample of
   `round(0.10 × num_items)` items is designated cold.
3. **Every training interaction of a cold item is removed** — both from input
   positions and from target positions. Cold items therefore have exactly zero
   training frequency, which `tests/test_cold_split.py` asserts.
4. Validation and test targets are untouched. Users whose train part becomes
   empty are dropped (58 of 100 000).
5. Cold items keep their **content features**. Their ID representation is zeroed
   at inference in *every* model that owns an ID branch (`zero_cold_id: true`,
   default), so no model is credited for a randomly-initialised embedding. This
   makes the comparison "only content can help" for all models.

Cold results are reported two ways, both from real ranking passes:

| metric | candidate set | target subset |
|---|---|---|
| `cold_full_ranking` | all 19 738 items | users whose test target is cold |
| `cold_restricted_to_cold` | only the 1 974 cold items | users whose test target is cold |

`cold_restricted_to_cold` is the sensitive measure of *content quality*; the
full-ranking number is the honest end-to-end one. They are never mixed in the
same table without the candidate-set column being visible.

### The uniform-random floor

A uniform ranker over the cold catalogue is the null model for this protocol, and
its expected score is known in closed form: with `N` cold candidates the
expected restricted Recall@`k` is `k/N` and the mean rank is `(N+1)/2`.
`RandomRecommender` (`src/models/random_model.py`) implements exactly that —
frozen seeded item noise, zero parameters, PAD scored 0 — and it runs through
the same `scripts/train.py`, evaluator and aggregation path as every other
model (a parameter-free model skips the training loop and is evaluated once).
With `N = 1974` the prediction is `20/1974 = 0.01013`; the measured cold-only
Recall@20 is `0.00967` over 12 717 users (`se = 0.00089`), i.e. within one
standard error. The cold evaluation path is therefore measuring the models, not
an artefact of the candidate set. `tests/test_random_baseline.py` pins the
closed form through the real evaluator on synthetic data.

### Current cold split (`data/processed/cold10`)

```
cold items                1974  (10.0 % of the catalogue)
cold val+test targets    25345  (12.7 % of all targets)
train interactions removed 59889
users dropped                58
```

## 6. Long-tail protocol

Popularity is computed on **training interactions only** (`src/data/popularity.py`).

Two bucketing rules are implemented so the main result can be robustness-checked:

* `frequency_quantile` (**default**): head = top 20 % of items by training
  frequency, tail = bottom 60 %, middle = the rest. This treats the tail as what
  it is on a content platform: a large part of the catalogue with little traffic.
* `interaction_mass`: head = the smallest item set covering 20 % of training
  interactions, tail = the item set covering 60 % of interactions counted from
  the least popular end.

Items with zero training frequency are always tail. The rule actually used is
stored in `stats.json` and copied into every run's `metrics.json`, so a table can
never be read without knowing which rule produced it.

Reported per bucket: `Recall@20`, `NDCG@20`, and the gain
`Δ = Metric(MM-SASRec) − Metric(SASRec)`.

## 7. Reproducibility

* `set_seed` seeds `random`, `numpy`, `torch` (CPU + all CUDA devices) and
  `PYTHONHASHSEED`.
* Negative sampling, cold-item selection and popularity bucketing all use
  explicitly seeded generators.
* Training data is fully in memory with `num_workers = 0`, so no worker seeding
  ambiguity exists.
* GPU kernels are not guaranteed to be bitwise deterministic. Runs are
  reproducible in distribution; the intent is that two runs with the same config
  and seed agree to within normal floating-point noise, and that the *data*,
  *negative samples* and *initialisation* are identical.
* Every run writes `config.yaml`, `environment.txt` (python/torch/CUDA/GPU/git
  commit), `training_log.csv`, `metrics.json` and the raw per-user ranking arrays
  to `results/runs/<run_id>/`.
* `--resume` refuses to load a checkpoint whose `num_items`, `num_users` or
  dataset hash does not match the current dataset.

## 8. Sanity checks that must hold

| check | expectation |
|---|---|
| Popular vs random | Popular must beat a uniform-random scorer |
| Random vs closed form | measured cold-only Recall@20 within sampling error of `k/N` |
| BPR vs Popular | a trained MF must beat the popularity prior |
| SASRec vs BPR | sequential modelling must beat static MF |
| SASRec vs Popular | must beat the prior |
| train loss | must decrease monotonically over epochs (checked on tiny data) |
| tiny overfit | SASRec must reach < 0.15 loss on 20 synthetic users |
| gate weights | must sum to 1 over available modalities, exactly 0 for missing |
| cold ID | cold items must have zero ID gate weight |
