# PROGRESS

Living status log. Every entry is something that was actually executed and
verified in this repository — no aspirational items.

## Current repository state (2026-09-17)

The project is in a **correctness-audit / re-run phase**. Two training-side bugs
were found and fixed, every earlier run was quarantined as legacy, a provenance
and aggregation guard layer was added, and the **main experiment matrix, the
modality/fusion ablation and the cold-item matrix have all been re-run to
completion on the fixed code** (33 finished runs spanning ten commits whose
result-determining code a human has verified to be equivalent, declared in
`analysis/code_equivalence.json`; see below).

Numbers in the README are generated from `results/tables/*.csv`; numbers in
`results/` are produced only by runs that carry a `run_manifest.json`.

Headline result on the fixed code (test Recall@20, 100 000 evaluated users,
3 seeds each): Popular `0.0036` < BPR-MF `0.0334` < SASRec `0.1224 ± 0.0020` <
SASRec+item-dropout 0.2 `0.1283 ± 0.0009` < MM-SASRec gated+ID-dropout 0.2
`0.1314 ± 0.0006` < MM-SASRec concat+ID-dropout 0.2 `0.1438` (1 seed).

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
- **Table-hygiene guards for the queued runs** — two ways the incoming
  ablation/cold runs could have silently corrupted the generated README were
  closed before they landed: the OVERALL/ABLATION tables now filter to
  `dataset == "base"` (a cold10 row would otherwise appear as a second,
  unexplained "ID-only" row), and the gate table keys runs by
  `model(modalities)@dataset` instead of model name alone (an ablation or
  cold export would otherwise be averaged into the base gated rows). Pinned by
  `tests/test_readme_tables.py::test_cold_split_runs_stay_out_of_the_headline_tables`
  and `tests/test_gate_analysis.py`.
- **Aggregation guards** (`analysis/aggregate_results.py`) — the aggregator
  refuses to pool runs across different code versions, dataset hashes, core
  configs (seed stripped) or parameter counts, keeps the largest mutually
  compatible subset, and never touches manifest-less runs. This is what caught
  the earlier incident in which a seed-3407 SASRec run with 128 extra
  parameters was being averaged in with the seed-42 run. Covered by
  `tests/test_aggregation_guard.py`.
- **"Same code" is a checked claim, not a string comparison** — the first gate
  was equality of the commit SHA, which is both too strict and too weak: a
  commit that only touches the README cannot change a number, while the runs of
  one matrix legitimately carry different SHAs because a later batch's change
  touched an inert path. A run's code identity is now decided in three steps,
  most specific first:
  1. `git_dirty = true` ⇒ the run is compared only with itself: its code is not
     the commit's code, so no commit-level claim — not even a declared class —
     applies to it;
  2. a commit listed in `analysis/code_equivalence.json` takes that class's
     identity. The file is data a human wrote and re-checked (per class: a
     reference commit, the commits, and why the diff is inert), not an
     inference the aggregator makes; the class id and its reason are printed
     whenever it actually pools more than one commit;
  3. otherwise the SHA-256 of `git ls-tree -r <sha> -- src scripts configs
     pyproject.toml` compares the code itself, falling back to the SHA only
     when git cannot answer. The class index is closed over that tree hash, so
     an undeclared commit whose result-determining tree equals a class
     member's joins the class — a declaration can only widen comparability,
     never narrow it below what the hash already proves.
  `tests/test_aggregation_guard.py` pins this against the real repository: the
  declared commits exist, the declaration is self-consistent, two class commits
  pool end-to-end into `runs_index.csv` with the class named in the output, two
  undeclared commits are still refused, equal trees always compare equal, the
  tree closure actually fires, and a dirty run never pools with a clean one.
  Each guard was mutation-checked (removing it makes its test fail). The
  practical effect: the Popular/BPR/SASRec/MM seed groups pool to `n=3` again
  instead of silently dropping every seed that ran after the last table
  commit — the pre-fix behaviour printed "keeping 1 run(s) with git=cc06c207"
  and computed single-seed means.
- **Gate analysis** (`analysis/analyze_gates.py`) — only documented runs
  contribute to `results/tables/gate_by_bucket.csv`; plain-gated and ID-dropout
  variants are separate rows.
- **Modality / fusion ablation (Phase 9) — finished.** 9 base-split runs at the
  regularised setting plus the fusion follow-up (`results/queue_rerun_ab3.txt`).
  Content-only modality sets (test Recall@20, 1 seed): text `0.0702`, image
  `0.0874`, video `0.0905`, text+image `0.1126`; paired with ID: id+text
  `0.1176`, id+image `0.1237`; the full gated set `0.1267 ± 0.0008` (3 seeds);
  adding video to id+text+image `0.1271` (1 seed, within the multi-seed std).
  Fusion at the same modality set and regularisation: concat `0.1390` /
  concat+ID-dropout 0.2 `0.1438` vs gated `0.1267` / `0.1314 ± 0.0006` — the
  fusion choice matters more than the modality set, and this is reported as
  measured, not folded into the gated headline.
- **Cold-item matrix (Phase 10) — finished.** All five cold10 runs carry a
  manifest. Cold-only Recall@20 (ranking restricted to the 1 974 cold items,
  12 717 users with a cold target): Random `0.0097` (closed form `0.01013`),
  SASRec ID-only `0.0000`, content-only text+image `0.1092`, gated
  id+text+image `0.0983`, gated+ID-dropout 0.2 `0.0789`. In the *full*-catalogue
  ranking the same runs score `0.0000–0.0008`: a cold target has to beat all
  19 738 items, which is where the honest end-to-end number lives. Gate export
  on the cold runs shows the designed zeroing directly — cold items get ID gate
  weight exactly `0.000`, and the content weight splits image/text `0.480/0.520`
  (plain gated) and `0.797/0.203` (ID-dropout).
- **Gate re-export for the new runs (Phase 11)** — `analyze_gates.py` now keys
  each row by `model(modalities)@dataset`, so ablation and cold exports can
  never be averaged into the base gated rows (13 documented runs).
- **Result-determining-code equivalence across batches** — the table entries
  span ten commits (`cc06c20` … `5f84798`), so "one code version" was checked
  hunk by hunk rather than asserted: `git diff cc06c20 <sha> -- src scripts` is
  non-empty, but every hunk is one of (a) the `random` model branch in the
  factory/trainer, (b) provenance and git-state recording, (c) an evaluator
  docstring, (d) the queue runner's skip guard. None of them can change a
  non-Random model's training or ranking. That review is now recorded as data
  in `analysis/code_equivalence.json` (class `ar-fix-2026-09-17`, reference
  `cc06c20`, 14 commits, the same reason), which is what the aggregator reads
  instead of trusting or refusing the SHA, and what the no-run-dropped pooling
  of the 33 runs is validated against.
- **Three table-integrity bugs found and fixed (all caught by the generated
  tables, all now pinned by tests)**
  - `group_seeds()` keyed on lowercase `fusion` while `ablation.csv` writes
    `Fusion`, so gated and concat runs at the same modality set were averaged
    into one `n_seeds: 4` row and the gated ID-dropout rows dropped out of the
    ABLATION table. The lookup is now case-insensitive
    (`analysis/make_readme_tables.py::field`).
  - The long-tail headline picked the first `id+text+image` row with ID-dropout
    > 0, which was the **concat** run, while every other headline quotes the
    gated model; fusion is now pinned and the concat tail is reported as its own
    clause.
  - The cold claim silently quoted whichever gated variant was picked; it now
    names both (plain gated and ID-dropout 0.2), since the gap between them is
    itself the finding.
- **Case study regenerated on the fixed code** — the committed
  `results/case_study.{md,json}` quoted two pre-fix runs (which the legacy
  README had already flagged as invalid). It is now measured from the seed-42
  `sasrec` and `mm_gated_iddrop` checkpoints, and its Recall@20 values match
  those runs' own `metrics.json`.
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

- `pytest -q` → 170 passed (unit + regression tests for both training bugs,
  provenance and aggregation guards, the declared code-equivalence mechanism,
  the README generator, the table-integrity guards, and the Random baseline's
  closed-form cold floor).
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
- All 33 run manifests carry `git_dirty = False`, and the aggregator now
  *requires* that exact value before a run may be pooled under a commit-level
  identity, so a run marked dirty (or one whose status check failed, `None`)
  cannot ride on its SHA — or on a declared class — into an average. The 33
  runs behind the tables span ten commits, and the `src`/`scripts` diff between
  them was read rather than assumed empty (see above).
- Gate analysis on all 6 MM runs (`gate_weights.npz` exported per run and read
  back from the checkpoint, never recomputed): the ID gate dominates
  (~0.96) and is lowest on tail items, while text/image weights rise toward the
  tail (e.g. gated+ID-dropout: image 0.016 head → 0.029 tail, text 0.017 → 0.033).
- `analysis/update_readme.py --check` passes, i.e. every generated README number
  matches `results/tables/*.csv` byte for byte.
- **The uniform-Random cold floor agrees with its closed form**: with 1 974 cold
  candidates the predicted cold-only Recall@20 is `20/1974 = 0.01013`; measured
  on the cold10 split over 12 717 users with a cold target: `0.00967` (0.5
  standard errors away, `se = 0.00089`). The cold evaluation path is unbiased,
  which is what makes the cold *model* numbers interpretable.

## Known issues / deferred (in priority order)

- The pre-fix numbers in `results/legacy_pre_autoregressive_fix/` are invalid for
  any conclusion; only the evaluation protocol, the cold-item mask and the
  feature-alignment path from that period remain valid.
- GPUs on this host are shared with other users; wall-clock times vary a lot
  (the efficiency table reports them as measured, not as a benchmark).
- The HuggingFace 100K subset ships **no item titles**, so the case study uses
  content-space nearest neighbours instead of captions.
- The 14-run main matrix was produced on `cc06c20`; the batches after it carry
  newer commits. The aggregator now pools them through the declared class, but
  re-running the whole matrix on one final commit would still be the strongest
  possible provenance — deferred because the class's diff was read hunk by hunk
  and the ordering it reproduces is stable across all seeds.

## Experiment status

| group | runs | status |
|---|---|---|
| A: Popular, BPR-MF, SASRec (seeds 42/2026/3407) | 9 | **finished** (all three models now have all three seeds) |
| B: SASRec + item-dropout 0.2 (3 seeds) | 3 | **finished** |
| C: MM-SASRec gated id+text+image (3 seeds) | 3 | **finished** |
| D: MM-SASRec gated + ID-dropout 0.2 (3 seeds) | 3 | **finished** |
| modality / fusion ablation (base) | 9 | **finished** (`queue_rerun_ab1/2/3.txt`) |
| cold split (Random, SASRec, content-only, gated, gated+ID-dropout) | 5 | **finished** (`queue_rerun_cold.txt`) |
| concat fusion at seeds 2026/3407 (± ID-dropout 0.2) | 4 | **running** (`queue_rerun_ab5.txt`, GPU 0) |
| Semantic-ID extension | — | paused until the discriminative results stabilise |

Queues: `results/queue_main_{a,b,c,d}.txt` (GPU 1/2/3/5) — finished;
`results/queue_rerun_ab4.txt` (Popular + BPR-MF seeds 2026/3407) — finished.
Total finished runs entering the tables: 33 (all with `run_manifest.json`),
spanning ten commits of one declared-equivalence class.

## Next highest-priority tasks

1. ~~Launch the staged ablation and cold queues on the fixed code; re-export
   gates for the ablation runs~~ — done, 15 runs finished, gates re-exported.
2. ~~Check the measured Random cold-only Recall@20 against the closed form
   20/1974 and record the comparison in the generated findings~~ — done,
   `0.0097` vs `0.0101`, documented in `docs/evaluation_protocol.md` and the
   generated cold finding.
3. ~~Re-run the retrieval benchmark on the fixed code with the new MM
   checkpoint~~ — already satisfied: the committed
   `results/retrieval_benchmarks/mm_gated_20260917-053644_dfe900.json` names a
   `cc06c20` run (`git_dirty = False`), i.e. it was measured on the fixed code.
4. ~~Update the README's modality/fusion and cold sections once the runs land~~ —
   done; every number is generated from `results/tables/*.csv` and
   `update_readme.py --check` passes.
5. ~~Multi-seed the concat fusion~~ — in flight: `queue_rerun_ab5.txt` adds
   seeds 2026/3407 to concat and concat+ID-dropout 0.2 (seed 42 and 2026
   already landed; 3407 for both settings still running). The cold
   content-only row is still 1 seed — the cold matrix is a different table and
   its single-seed status is stated in the generated finding rather than
   implied away.
6. Semantic-ID extension (RQVAE + constrained generative decoding).
