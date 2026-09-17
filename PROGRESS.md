# PROGRESS

Living status log. Every entry is something that was actually executed and
verified in this repository — no aspirational items.

## Current repository state (2026-09-17)

The project is in a **correctness-audit / re-run phase**. Two training-side bugs
were found and fixed, every earlier run was quarantined as legacy, a provenance
and aggregation guard layer was added, and the **main experiment matrix has been
re-run to completion on the fixed code** (commit `cc06c20`, 14 runs, all three
seeds per model). The modality/fusion ablation and the cold-item matrix are
staged (`results/queue_rerun_ab*.txt`, `results/queue_rerun_cold.txt`) but not
launched yet.

Numbers in the README are generated from `results/tables/*.csv`; numbers in
`results/` are produced only by runs that carry a `run_manifest.json`.

Headline result on the fixed code (test Recall@20, 100 000 evaluated users,
3 seeds each): Popular `0.0036` < BPR-MF `0.0334` < SASRec `0.1224 ± 0.0020` <
SASRec+item-dropout 0.2 `0.1283 ± 0.0009` < MM-SASRec gated+ID-dropout 0.2
`0.1314 ± 0.0006`.

## Completed

- **Repo scaffold** — `src/{data,models,evaluation,retrieval,serving,training,utils}`,
  `configs/`, `scripts/`, `analysis/`, `tests/`, `docs/`. Installed as an editable
  package (`pyproject.toml`), YAML config with dotted CLI overrides.
- **Data pipeline** (`src/data/preprocess.py`) — explicit `raw → internal` id
  mapping (0 = PAD), chronological sort with stable tie-break, leave-one-out
  split, training-only item frequency, popularity buckets, simulated cold split.
- **Metrics + full-ranking evaluator** — exact ranking over all 19 738 items,
  seen-item masking that provably never masks the ground truth, per-user ranks
  stored so cold/tail slices come from the same ranking pass.
- **Models** — Popular, BPR-MF, SASRec, MM-SASRec (ID/text/image/video with
  per-modality projection, concat and gated fusion, modality + ID dropout).
- **Correctness fixes (this phase)**
  - *Autoregressive objective*: training previously broadcast the **last**
    hidden state over the whole sequence, so every position shared one
    prediction. Now the model returns `[B, L, H]` for training and position-wise
    sampled softmax runs over `[B, L, 1 + N]` logits; evaluation still uses the
    last non-pad position. Covered by `tests/test_autoregressive_training.py`
    (different positions → different hidden states, future suffix cannot change
    earlier predictions, handcrafted alignment).
  - *Negative-sampling leakage*: negatives were excluded using **all** known
    interactions, including the validation and test targets, which leaks the
    evaluation label into training. Now only PAD, the current positive and the
    user's *training* interactions are excluded
    (`build_training_interaction_keys`); covered by
    `tests/test_negative_sampling.py`.
- **Legacy quarantine** — every pre-fix run, table and case study was moved to
  `results/legacy_pre_autoregressive_fix/` (evidence kept, excluded from
  aggregation, documented in its README).
- **Provenance** (`src/utils/provenance.py`) — each run writes `git_sha`,
  `git_dirty` (restricted to result-determining paths), `dataset_hash`,
  `config_hash`, `seed`, `num_parameters` and a copy of `config.yaml` into
  `results/runs/<id>/run_manifest.json`, plus a committed copy under
  `results/manifests/`.
- **Main matrix re-run (Phase 6–8) — finished.** 14 runs on commit `cc06c20`,
  3 seeds (42/2026/3407) for each of SASRec, SASRec+item-dropout 0.2,
  MM-SASRec gated, MM-SASRec gated+ID-dropout 0.2, plus Popular and BPR-MF.
  The ordering `Popular < BPR-MF < SASRec` holds on real data, and every run
  passed the aggregation guard (same SHA, dataset hash, core config and
  parameter count within each group; no seed-3407 parameter anomaly recurred).
- **Long-tail table averaging bug (found by the generated table itself)** —
  `group_seeds()` merged only a hard-coded metric list, so `head_*`/`tail_*`
  columns were copied from the *first* seed while the table said "3 runs".
  Now every numeric column not in an identity list is averaged, and
  `tests/test_readme_tables.py::test_bucket_metrics_are_averaged_across_seeds`
  pins it. The corrected 3-seed long tail changes the conclusion: against the
  item-dropout control the content contribution is ΔHead `+0.0078`,
  ΔMiddle `+0.0014`, ΔTail `+0.0003` — i.e. a head effect, not a tail effect.
- **Uniform-Random baseline** (`src/models/random_model.py`) — frozen seeded
  item noise, 0 parameters, registered as `random` in the factory. It measures
  the cold-catalogue chance floor (expected `k/N`) instead of asserting it;
  `tests/test_random_baseline.py` checks the closed form through the real
  full-ranking evaluator.
- **Provenance sampled at run start** — `scripts/train.py` now calls
  `git_state()` before training and passes it into `build_manifest(git=...)`,
  so an edit made while a run is in flight can no longer retroactively mark it
  dirty. `provenance._status_paths` returns `None` (unknown) instead of `[]`
  when the `git status` call fails, so a timeout is never reported as "clean".
- **`run_suite.sh` skip guard** — the guard globbed `results/runs/${tag}_*`,
  which made tag `sasrec` match `sasrec_itemdrop_*` and silently skipped a job
  whose run did not exist. It now matches a full run id
  (`<tag>_<8 digits>-<time>_<hash>/metrics.json`).
- **Aggregation guards** (`analysis/aggregate_results.py`) — the aggregator
  refuses to pool runs across different git SHAs, dataset hashes, core configs
  (seed stripped) or parameter counts, keeps the largest mutually compatible
  subset, and never touches manifest-less runs. This is what caught the earlier
  incident in which a seed-3407 SASRec run with 128 extra parameters was being
  averaged in with the seed-42 run. Covered by `tests/test_aggregation_guard.py`.
- **Gate analysis** (`analysis/analyze_gates.py`) — only documented runs
  contribute to `results/tables/gate_by_bucket.csv`; plain-gated and ID-dropout
  variants are separate rows.
- **Generated README** — `analysis/make_readme_tables.py` renders every results
  table *and* the key-findings text from `results/tables/*.csv`;
  `analysis/update_readme.py` injects them between markers and has a `--check`
  mode. No result number in the README is hand-written.
- **CI** (`.github/workflows/tests.yml`) — Python 3.10/3.11, CPU-only torch,
  `pytest -q`, plus the README/tables consistency check. Tests use synthetic
  fixtures only, no real data or GPU. (The `ruff check` step was dropped: the
  repo has ~59 pre-existing lint findings, and a red CI on day one would hide
  the tests that do guard the results.)

## Verified

- `pytest -q` → 153 passed (unit + regression tests for both training bugs,
  provenance and aggregation guards, the README generator, and the Random
  baseline's closed-form cold floor).
- `python scripts/smoke_test.py` → 14 checks pass on synthetic data: ordering
  `popular < BPR-MF < SASRec`, losses decrease, tiny-overfit reaches < 0.15
  loss, position-wise objective beats the broadcast-objective reference.
- Real-data preprocessing: base split 100 000 users / 19 738 items / 719 405
  interactions, avg length 7.19, sparsity 0.99964; cold split 1 974 cold items
  with **zero** training interactions and 25 345 cold eval targets
  (`tests/test_cold_split.py`).
- Popularity buckets are computed from training interactions only
  (`sum(train_freq) == sum(train_len) == 519 405`); the rule actually used is
  copied into every run's `metrics.json` and `stats.json`.
- Provenance chain on a live run: `git_sha`/`dataset_hash`/`config_hash` in
  `run_manifest.json` reproduce exactly from the run's own `config.yaml`.
- All 14 main-matrix manifests carry `git_sha = cc06c20` with
  `git_dirty = False` — the batch ran on one committed code version.
- Gate analysis on all 6 MM runs (`gate_weights.npz` exported per run and read
  back from the checkpoint, never recomputed): the ID gate dominates
  (~0.96) and is lowest on tail items, while text/image weights rise toward the
  tail (e.g. gated+ID-dropout: image 0.016 head → 0.029 tail, text 0.017 → 0.033).
- `analysis/update_readme.py --check` passes, i.e. every generated README number
  matches `results/tables/*.csv` byte for byte.

## Known issues / deferred (in priority order)

- The pre-fix numbers in `results/legacy_pre_autoregressive_fix/` are invalid for
  any conclusion; only the evaluation protocol, the cold-item mask and the
  feature-alignment path from that period remain valid.
- GPUs on this host are shared with other users; wall-clock times vary a lot
  (the efficiency table reports them as measured, not as a benchmark).
- The HuggingFace 100K subset ships **no item titles**, so the case study uses
  content-space nearest neighbours instead of captions.
- `git_dirty = None` (status call failed) is recorded but nothing currently
  *acts* on it; the aggregator only requires equal `git_sha`. Low risk, noted.

## Experiment status

| group | runs | status |
|---|---|---|
| A: Popular, BPR-MF, SASRec (seeds 42/2026/3407) | 5 | **finished** |
| B: SASRec + item-dropout 0.2 (3 seeds) | 3 | **finished** |
| C: MM-SASRec gated id+text+image (3 seeds) | 3 | **finished** |
| D: MM-SASRec gated + ID-dropout 0.2 (3 seeds) | 3 | **finished** |
| modality / fusion ablation (base) | 8 | queued (`queue_rerun_ab1/2.txt`) |
| cold split (Random, SASRec, content-only, gated, gated+ID-dropout) | 5 | queued (`queue_rerun_cold.txt`) |
| Semantic-ID extension | — | paused until the discriminative results stabilise |

Queues: `results/queue_main_{a,b,c,d}.txt` (GPU 1/2/3/5) — finished.

## Next highest-priority tasks

1. Launch the staged ablation and cold queues on the fixed code; re-export
   gates for the ablation runs.
2. Check the measured Random cold-only Recall@20 against the closed form
   20/1974 and record the comparison in the generated findings.
3. Re-run the retrieval benchmark on the fixed code with the new MM checkpoint
   (the committed one is from the seed-42 run; the table already names it).
4. Update the README's modality/fusion and cold sections once the runs land —
   the numbers come from `results/tables/*.csv`, so only the queues need care.
5. Semantic-ID extension (RQVAE + constrained generative decoding).
