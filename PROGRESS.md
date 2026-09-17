# PROGRESS

Living status log. Every entry is something that was actually executed and
verified in this repository — no aspirational items.

## Current repository state

ShortRec is a two-stage multimodal recommendation prototype on MicroLens-100K:

```
multi-channel recall → candidate merge → MM-SASRec ranker → rerank → top-K
```

It has an offline experiment stack (full-catalogue ranking, cold/long-tail
analysis, modality ablations), a serving stack (recall channels, merge, ranker
registry, rerank policies, FastAPI), a React demo (Feed / Inspector / Cold Start
/ System), and a Semantic-ID extension. 246 tests pass.

## Completed

### Offline stack (earlier phases, unchanged)

- **Data pipeline** — explicit `raw → internal` id mapping, chronological
  leave-one-out split, config-driven filtering, training-only statistics,
  simulated cold split, `npz`/`npy`/`json` artifacts.
- **Models** — Popular, BPR-MF, SASRec, MM-SASRec (concat / gated fusion, ID and
  modality dropout, missing-modality masks).
- **Training** — sampled softmax, AMP, early stopping, checkpointing with a
  dataset-hash mismatch guard, per-run provenance.
- **Evaluation** — full-catalogue ranking, chunked scoring, seen-item masking
  that never masks the ground truth, tie-neutral average ranks, cold and
  long-tail slices from the same ranking pass.
- **Semantic-ID extension** — RQVAE with residual-aware re-seeding, 0.23 %
  collision rate, constrained generative decoding.

### Two-stage serving stack (this phase)

- **`src/recall/`** — one `RecallStrategy` interface with a shared guarantee
  (PAD never returned, history always excluded, ranked output). Channels:
  `popular`, `itemcf`, `semantic`. `CandidateMerger` does dedup + reciprocal
  rank fusion and keeps the full source trace on every candidate.
- **`scripts/build_itemcf.py`** — cosine co-occurrence over training
  interactions, top-100 neighbours, 1.47 M non-zero pairs, 22 s.
- **`scripts/build_semantic_index.py`** — `L2([L2(text) ; L2(image)])` content
  embeddings (19739 × 1152) plus a `faiss.IndexFlatIP` **exact** inner-product
  index. Per-modality normalisation is required; without it the image block
  dominates.
- **`src/pipeline/`** — `RankerRegistry` (wraps the existing factory and
  checkpoints, caches item embedding tables) and `TwoStageRecommender`
  (recall → merge → rank → rerank, with per-stage latency and full metadata).
- **`src/rerank/simple.py`** — seen filter, dedup, cold exploration quota.
- **`src/data/metadata.py`** — training-derived cold / bucket / frequency
  metadata exposed by the API and the UI.
- **`src/serving/`** — service layer owning the raw ↔ internal id conversion,
  Pydantic schemas, and the FastAPI app (`/system`, `/models`, `/users/{u}`,
  `/recall`, `/recommend`, `/inspect`, `/cold/*`, legacy `POST /recommend`).
- **Offline evaluation of the serving path** — `scripts/evaluate_recall.py`
  (Recall@100/200/500/1000 per channel) and `scripts/evaluate_pipeline.py`
  (candidate budget vs candidate recall vs final Recall@20 vs latency, plus
  recall-source attribution).
- **`frontend/`** — React + Vite + TypeScript dashboard: Feed, Recommendation
  Inspector, Cold Start Explorer, System. Clean light theme, no fake item
  titles.
- **Demo tooling** — `scripts/prepare_demo.py` (readiness check + index build),
  `scripts/start_demo.sh` (API + UI), `scripts/capture_screenshots.py`.

## Verified

- `pytest -q` → **246 passed** (was 175 before this phase).
- `python analysis/update_readme.py --check` → README tables in sync.
- Two-stage pipeline on CPU, 500-candidate budget: recall ~19 ms, rank ~6 ms,
  rerank ~5 ms, ~30 ms end to end (150 users × 20 repetitions, minimum).
- Ranking the entire 19 738-item catalogue costs 6.6 ms — the same as a
  100-item pool, because the ranker's cost is dominated by encoding the user
  sequence. The two-stage split buys recall quality and catalogue headroom, not
  latency, at this scale; the README says so explicitly.
- `mm_concat` ranker reproduces the official evaluator exactly
  (`tests/test_pipeline.py` and a manual 500-user cross-check: 0.114 both ways).
- Demo runs end to end: API on :8000, Vite on :5173, `/api/*` proxied.
- Screenshots captured from the running app into `assets/`.

## Bugs found and fixed this phase

- **Off-by-one between serving user ids and `ProcessedData`.** The serving layer
  uses 1-based internal user ids while `ProcessedData` indexes users 0-based.
  `TwoStageRecommender.history()` was serving each user the *next* user's
  history, which silently capped pipeline Recall@20 at ~1 % instead of ~12 %.
  Fixed, and pinned by `test_history_and_target_belong_to_the_same_user`.
- **Exact-search exclusion was a Python loop** over ~4 000 over-fetched rows per
  query, dominating the recall layer. Vectorised.
- **`PopularRecall` walked the global order in Python.** Vectorised.
- **`from __future__ import annotations` in the FastAPI app** turned a JSON body
  into a required query parameter (422) because the locally-imported Pydantic
  model could not be resolved at runtime. Removed with a comment.
- **PyTorch thread oversubscription in serving.** `torch` defaults to one thread
  per core; inside FastAPI a sync endpoint runs in a threadpool, so each request
  spawned ~20 OpenMP threads on a busy 40-core machine. The same request took
  ~190 ms. `configure_threads()` in `src/serving/app.py` now caps intra-op
  threads (default 1), taking it to ~9 ms rank / ~30 ms end to end.
- **Pipeline latency was being reported from a single contended timing per
  user.** Accuracy and latency are now measured separately: accuracy over 10 000
  users with one timing each, latency over 150 users × 20 repetitions in
  `scripts/benchmark_latency.py`.
- **Merge truncation dropped single-channel candidates.** The ranker now scores
  the full merged union by default; `max_candidates` is an explicit opt-in.
- **Optimistic tie-breaking made the cold-only metric meaningless** (all 1 974
  tied cold items would rank first). Switched to average ranks.
- **`analysis/aggregate_results.py` counted unfinished runs** and merged
  cold-split runs with base runs. A run now counts only once
  `train_summary.json` exists, and the dataset is part of the grouping key.
- **Efficiency table called `IndexFlatIP` an "ANN index".** Corrected to
  "exact inner-product" throughout the docs and the UI.

## Known issues / limitations

- MicroLens-100K ships **no item titles or captions**, so the UI identifies items
  by raw id and content-space neighbours. Nothing is invented.
- The base dataset has no cold items; the Cold Start Explorer reads the separate
  `cold10` dataset and the offline cold table.
- Content retrieval is exact (`IndexFlatIP`) and therefore O(catalogue) per
  query. Approximate indexing (IVF/HNSW) is deliberately not claimed.
- Cold items remain poorly calibrated against warm items in the full catalogue;
  the exploration quota mitigates exposure but does not fix ranking.
- The API is single-process, unauthenticated, uncached, and CORS-open: it is a
  local demo, not a deployment.

## Experiment status

| group | status |
|---|---|
| full-catalogue model matrix (baselines, modalities, fusion, seeds) | complete |
| cold split (random, ID-only, content-only, gated, gated+ID-dropout) | complete |
| gate analysis export | complete |
| recall evaluation (Recall@100/200/500/1000) | complete |
| two-stage pipeline trade-off (candidate budget vs accuracy vs latency) | complete |
| Semantic-ID quantiser + generative prototype | complete |

## Next highest-priority tasks

1. Approximate indexing (IVF/HNSW) as an explicit accuracy/latency comparison.
2. A learned reranker stage on top of the current pool, measured against the
   current score-only ordering.
3. Hard-negative mining for the ranking stage, evaluated on the tail bucket.
4. An A/B-style offline replay harness for serving policy changes (e.g. the cold
   quota) reporting exposure and coverage deltas.
