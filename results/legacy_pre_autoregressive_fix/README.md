# Legacy runs — invalidated by the training-objective fix

Everything in this directory was produced **before** commit `503aa40`
("Fix the autoregressive training objective and the negative-sampling leak").
It is kept as evidence, and it must never be quoted as a result.

## Why these numbers are wrong

Two correctness bugs were present in the code that produced them:

1. **Training objective.** `model.encode()` returned the *last* hidden state for
   every position, so the loss asked the model to predict `i2, i3, …` from the
   single representation of the whole prefix (`h_last → i2`, `h_last → i3`, …)
   instead of the position-wise `h1 → i2, h2 → i3, …`. Training therefore
   optimised an easier, non-autoregressive objective. The reported test metric
   came from the correct "predict the next item from the last state" path, which
   is *why the bug was invisible*: evaluation was fine, training was not.
2. **Negative-sampling leak.** Negative sampling excluded every item in the
   user's full history, including the validation and test targets. The targets
   were thus never sampled as negatives during training, so training knew where
   the future positive was.

## What is here

* `runs/` — the 26 pre-fix run directories (configs, logs, checkpoints, ranking
  dumps). `cold_mm_concat_20260917-044712_9321ae` is incomplete (crashed before
  writing `metrics.json`); the other 25 finished.
* `tables/` — the per-run tables those runs produced, regenerated with
  `python analysis/aggregate_results.py --include-legacy --allow-mixed
  --runs-dir results/legacy_pre_autoregressive_fix/runs
  --tables-dir results/legacy_pre_autoregressive_fix/tables`.
  They are *not* in `results/tables/`, which now only ever receives runs that
  carry a `run_manifest.json` and pass the provenance check.

## Still valid / still invalid

* Valid: the code paths these runs exercised (evaluation protocol, full-ranking
  ranking, cold-mask construction, feature alignment) — those never depended on
  the training objective.
* Invalid: every test metric in `tables/`, the ablation and long-tail
  comparisons, the cold-start numbers, `results/case_study.{md,json}` (both
  quoted runs live here) and any README/PROGRESS number that cites them.

Legacy runs have no `run_manifest.json`, so `analysis/aggregate_results.py`
skips them by default — the guarantee that they cannot re-enter a result table
is structural, not a convention.
