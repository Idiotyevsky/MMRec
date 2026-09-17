# PROGRESS

Living status log. Every entry is something that was actually executed and
verified in this repository — no aspirational items.

## Current repository state

MicroLens-100K is downloaded and inspected. The full pipeline exists and runs:
raw data → mappings → chronological leave-one-out split → cold split → training
(Popular / BPR-MF / SASRec / MM-SASRec with concat or gated fusion) →
full-ranking evaluation → result tables and figures. 82 unit tests pass.

## Completed

- **Repo scaffold** — `src/{data,models,evaluation,retrieval,serving,training,utils}`,
  `configs/`, `scripts/`, `analysis/`, `tests/`, `docs/`. Installed as an editable
  package (`pyproject.toml`), YAML config with dotted CLI overrides.
- **Real data inspection** (`scripts/inspect_data.py`) — TSV with header
  (`userID, itemID, timestamp, x_label`), 719 405 interactions, 100 000 users,
  19 738 items, feature files with exactly `num_items` rows. Written up from
  evidence in `docs/data_schema.md`.
- **Data pipeline** (`src/data/preprocess.py`) — explicit `raw → internal` id
  mapping (0 = PAD), chronological sort with stable tie-break, leave-one-out
  split, config-driven filtering, training-only item frequency, popularity
  buckets, simulated cold split, `npz`/`npy`/`json` artifacts (no pickle).
- **Feature alignment** — `row_for_item_<modality>.npy` built and asserted;
  missing modalities are flagged, never treated as a zero vector.
- **Negative sampler** — uniform and popularity modes; never samples PAD, the
  positive target, or any known interaction of the user.
- **Metrics + full-ranking evaluator** — exact ranks, chunked item scoring,
  seen-item masking that provably never masks the ground truth, per-user ranks
  stored so cold/tail slices come from the same ranking pass.
- **Models** — Popular, BPR-MF, SASRec, MM-SASRec (ID/text/image/video with
  per-modality projection, concat and gated fusion, modality + ID dropout).
- **Training** — sampled softmax (or BPR), AMP, grad clipping, cosine/plateau/step
  schedulers, early stopping, checkpoint/resume with dataset-hash mismatch
  refusal, per-run `config.yaml` / `environment.txt` / `training_log.csv` /
  `metrics.json`.
- **Tests** (82 passing) — metrics vs hand-computed values, split ordering,
  target-leak detection, feature alignment under permuted ids, negative-sampling
  guarantees, causal masking, padding isolation, fusion masking, cold-split
  leakage, evaluator masking, retrieval, smoke training and tiny-overfit.
- **Analysis** — `analysis/aggregate_results.py` (overall / ablation / cold /
  long-tail tables), `plot_overall.py`, `plot_cold_start.py`, `plot_long_tail.py`,
  `plot_popularity_gain.py` (in `plot_long_tail.py`), `analyze_gates.py`.
- **Retrieval + serving** — embedding export, Faiss `IndexFlatIP` with an exact
  fallback, `scripts/recommend.py` CLI, `src/serving/app.py` FastAPI wrapper,
  `scripts/case_study.py`.

## Verified

- `pytest -q` → 82 passed.
- `python scripts/smoke_test.py` → ordering `popular < BPR-MF < SASRec` holds on
  synthetic data, losses decrease, tiny-overfit reaches < 0.15 loss.
- Real-data preprocessing: base split 100 000 users / 19 738 items / 719 405
  interactions, avg length 7.19, sparsity 0.99964; cold split 1 974 cold items
  with 0 training interactions and 25 345 cold eval targets.
- SASRec trains ~12 s/epoch on one L40S; MM-SASRec ~15 s/epoch.
- Full-ranking evaluation on all 100 000 test users takes ~4 s.

## Known issues

- Raw feature inspection initially assumed `feature[item_id]`; a unit test on a
  permuted id space exposed that `MultimodalFeatures` computed availability from
  row index rather than through the mapping. Fixed.
- Left-padded attention produced NaN under `src_key_padding_mask` (softmax over a
  fully-masked row). Fixed by building a per-batch 3D mask where padding queries
  attend only to themselves; `tests/test_attention_mask.py` covers both the
  causality and the padding-isolation properties.
- `GatedFusion` initially let the *content* of a missing modality leak into the
  other modalities' weights through the shared gate MLP. Fixed by zeroing before
  the gate; covered by `tests/test_fusion.py`.
- `-1e9` masking overflowed float16 under AMP. Now derived from `torch.finfo`.
- The HuggingFace 100K subset ships **no item titles**, so the case study uses
  content-space nearest neighbours instead of captions.
- GPUs on this host are shared with other users; wall-clock times vary a lot.

## Experiment status

| group | runs | status |
|---|---|---|
| baselines (Popular, BPR-MF, SASRec) | 3 | running / done |
| main models (concat, gated, gated+ID-dropout) | 3 | running |
| modality ablation (text/image/video/id combos) | 7 | queued |
| cold split (SASRec, gated, gated+ID-dropout) | 3 | not started |
| multi-seed (42 / 2026 / 3407) | 8 | not started |

All numbers currently in `results/tables/` come from real artifacts. Nothing has
been written into the README as a result yet.

## Next highest-priority tasks

1. Finish the base-dataset runs; check `SASRec > BPR-MF > Popular` on real data.
2. Run the modality ablation and the cold-split experiment.
3. Export gate weights and run the popularity-vs-gain analysis.
4. Multi-seed runs for the four headline models; report mean ± std.
5. Fill the README results section strictly from `results/tables/*.csv`.
6. Semantic-ID extension (RQVAE + constrained generative decoding).
