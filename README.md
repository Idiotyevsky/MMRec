# ShortRec

**Multimodal two-stage recommendation for short-video feeds.**

ShortRec is an offline + serving prototype that studies one practical problem:

> How can a short-video platform recommend **new and low-frequency content** when
> collaborative interaction signals are sparse or absent?

It combines **Popular, ItemCF and semantic content recall** with **SASRec-based
multimodal ranking**, and ships an interactive inspector for understanding how
recall channels, content features and sparse collaborative signals shape the
final feed. It is an offline + serving prototype on a public dataset — not a
production deployment.

![ShortRec demo](assets/shortrec_demo.gif)

<p align="center"><em>One real request: history → three recall channels → candidate merge → multimodal ranking → served feed → the trace behind a single recommendation.</em></p>

## Architecture

```
                        OFFLINE · BATCH
  raw interactions ─► preprocess ─► train SASRec / MM-SASRec
                                 ─► ItemCF neighbour index
                                 ─► content embeddings + exact IP index
                                 ─► artifacts/

                        ONLINE · SERVING
  user request
       │
       ▼
  ┌──────────── multi-channel recall ────────────┐
  │  Popular        ItemCF         Semantic      │
  │  (train freq) (co-occurrence) (text+image)   │
  └───────────────────────┬──────────────────────┘
                          ▼
                  merge / dedup / RRF
                  (every source kept on the item)
                          ▼
                  ~600 candidates
                          ▼
                  MM-SASRec ranking
             user sequence + ID + text + image
                          ▼
                  rerank: seen filter, dedup,
                  zero-train exploration quota
                          ▼
                       Top-K feed
```

## Three results

| | |
|---|---|
| **Multimodal ranking** | MM-SASRec **12.24 % → 13.93 %** Recall@20 vs the ID-only baseline (**+13.8 %**, 3 seeds) |
| **Multi-channel recall** | merged pool **Recall@1000 = 30.0 %**, above the best single channel (ItemCF 24.8 %) |
| **Two-stage retention** | at 1 000 candidates the pipeline keeps **95.8 %** of the same checkpoint's full-catalogue Recall@20 |

## Why two stages?

**Full-catalogue evaluation** ranks all 19 738 items for every user. It is the
strict model comparison.

**Two-stage serving** recalls a few hundred candidates and ranks only those. It
simulates how a real recommender is deployed.

The two answer different questions, use different protocols, and are reported in
different tables. A pipeline number can legitimately be lower than the
full-catalogue number for the same model; neither is wrong.

## Why multimodal?

```
warm video        ID embedding + behaviour + content
new / sparse      little or no collaborative signal
                        │
                        ▼
                  content provides the only semantic signal
```

Multimodal content improves overall ranking, and content recall is the only
channel that can reach an item with **zero** interactions. It does **not** solve
cold start: content separates cold items from each other, but their scores are
still not calibrated against warm items in a shared ranking. Both halves of that
sentence are measured results — see the Cold-start section.

## Product view

The recommendation pipeline ultimately serves a ranked short-video feed. This is
that feed: the served top-K rendered as playable videos, with the same
recommendation trace the Inspector shows overlaid on each item.

![ShortRec video feed](assets/feed_playback_demo.gif)

<p align="center"><em>Playable feed — real MicroLens videos for the recommended items, swipe/Next, and "Why this video?" showing recall source, baseline rank and multimodal rank movement.</em></p>

Videos are resolved from the **official MicroLens media source** through a
verified id mapping, and the overlay is generated from the live
`/users/{u}/inspect` response — recall source, baseline rank, multimodal rank and
popularity metadata are not mocked.

### How the media is matched to a recommendation

The modelling dataset is the HuggingFace re-upload `sisuo/Microlens_100k`, which
**re-indexed both users and items** — its `itemID` is not the official MicroLens
`videoID`. Binding a video by assuming `item_id == filename` would be wrong.

`scripts/prepare_media_demo.py` therefore recovers the permutation from the
official `MicroLens-100k_pairs.csv` by joining on the **exact millisecond
timestamp**, and `scripts/verify_media_mapping.py` checks it:

| check | result |
|---|---|
| exact millisecond timestamp matches | 719 299 / 719 405 (99.985 %) |
| item mapping is a bijection | 19 738 HF items → 19 738 distinct video ids, zero collisions |
| user mapping is a bijection | 100 000 HF users → 100 000 distinct official users |
| independent check: mean `x_label` vs official views | correlation 59× the shuffled control |

Video files in the official archive are `MicroLens-100k_videos/<videoID>.mp4`, so
the recovered id is directly the filename. Only the needed videos are fetched,
by HTTP range request against the split archive (the index alone is ~2 MB rather
than 477 GB for the whole archive).

> **Playback media is a local artefact.** The official videos are HEVC/H.265,
> which browsers cannot decode, so `scripts/prepare_media_demo.py` transcodes the
> first 15 s of each needed video to H.264 and stores it under `data/demo_media/`.
> That directory is **not** in git: the media is downloaded from the official
> source and is not redistributed here. The ranking model itself uses ID + text +
> image features, **not** the video stream.

## Demo

![Recommendation Inspector](assets/inspector_demo.png)

<p align="center"><em>Recommendation Inspector — pick a served item and follow it from each recall channel through the merge and both rankers to its final position.</em></p>

![Cold Start Explorer](assets/cold_start_demo.png)

<p align="center"><em>Cold Start Explorer — an item with zero training interactions, its content availability, and the cold-only vs full-catalogue gap.</em></p>

The demo has four pages:

| route | page | answers |
|---|---|---|
| `/` | Live Demo | what does the system do with this user's history? |
| `/watch` | Feed View | what does the user actually see? |
| `/inspect` | Recommendation Inspector | why is *this* item recommended? |
| `/cold` | Cold Start Explorer | why do new videos need content features? |
| `/system` | System | is this a real system or a model demo? |

## Motivation

A short-video platform produces new videos every day. Those videos have no
clicks, no watches, no co-occurrence — no collaborative signal at all. A model
that represents an item by a learned ID embedding has nothing to learn for them.
On MicroLens-100K this is not a corner case:

* 99.96 % of the user × item matrix is empty;
* the median item has a handful of training interactions;
* 376 items have **zero** training interactions in the base split;
* under the simulated cold split, 1 974 items have **every** training interaction
  removed.

ShortRec attacks this from two directions: a **content-based recall channel** that
never looks at collaborative signal, and a **multimodal ranker** whose item
representation mixes ID with text and cover-image features.

## Dataset

MicroLens-100K, downloaded from `huggingface.co/datasets/sisuo/Microlens_100k`.
Full schema evidence in [`docs/data_schema.md`](docs/data_schema.md).

<!-- TABLE:DATASET -->
| split | users | items | interactions | sparsity | mean seq len | train interactions | median train item freq | items with 0 train freq | cold items |
|---|---|---|---|---|---|---|---|---|---|
| `base` | 100000 | 19738 | 719405 | 0.99964 | 7.19 | 519405 | 15.0 | 376 | 0 |
| `cold10` | 99942 | 19738 | 659400 | 0.99967 | 6.60 | 459516 | 12.0 | 2300 | 1974 |
<!-- /TABLE:DATASET -->

### Data leakage policy

| signal | source |
|---|---|
| item frequency, popularity buckets, recall scores | **training interactions only** |
| ItemCF similarities, negative sampling | training interactions only |
| cold-item selection | training protocol + fixed seed |
| `x_label` / likes / views | **not used at all** |
| evaluation | user history masked, ground truth protected |

`ProcessedData.assert_no_target_leak()` re-checks on every load that no
validation target appears in the train history and no test target appears in the
train+val history. Details in [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md).

## Quick start

```bash
pip install -e ".[dev]"

# 1. data (190 MB, one time)
mkdir -p data/raw && (cd data/raw && \
  for f in microlens_100k.inter text_feat.npy image_feat.npy video_feat.npy; do
    curl -L -o "$f" "https://huggingface.co/datasets/sisuo/Microlens_100k/resolve/main/$f"
  done)

# 2. processed datasets
python -m src.data.preprocess --out data/processed/base
python -m src.data.preprocess --out data/processed/cold10 --cold-ratio 0.1 --cold-seed 42

# 3. train the two headline rankers (~25 min on one L40S) — checkpoints are NOT in git
python scripts/train.py --config configs/sasrec.yaml
python scripts/train.py --config configs/mm_sasrec_concat.yaml

# 4. build the recall indices
python scripts/prepare_demo.py

# 5. (optional) download + transcode the demo videos for the playable feed
python scripts/prepare_media_demo.py --fetch-official

# 6. run the demo
bash scripts/start_demo.sh          # API :8000 + UI :5173
```

`/watch` needs step 5; everything else works without it.

Open <http://127.0.0.1:5173>. **A fresh clone needs trained checkpoints** — step 3
is required, the demo is not one command from zero.

The serving path is CPU-only by default; no GPU is needed to run the demo once
the checkpoints exist.

Sanity checks:

```bash
pytest -q
python scripts/smoke_test.py
python analysis/update_readme.py --check
```

## Two-stage pipeline

### 1. Candidate recall (`src/recall`)

Every channel implements one interface and returns at most `top_k` candidates,
with **PAD never returned** and **the user's own history always excluded**.

| channel | what it uses | what it is for |
|---|---|---|
| `popular` | training interaction counts | the always-available fallback; works for a brand-new user with no history |
| `itemcf` | cosine-normalised item co-occurrence, `sim(i,j) = |U_i∩U_j| / sqrt(|U_i||U_j|)`, top-100 neighbours precomputed offline | finds items that co-occur with what the user watched — the workhorse of industrial recall |
| `semantic` | `L2([L2(text) ; L2(image)])` content embedding, recency-weighted mean over the history, exact inner-product search | the only channel that can retrieve an item with **zero** interactions |

Per-modality normalisation in the semantic channel is not cosmetic: raw text
features are unit-norm while raw image features have norm ≈ 26, so concatenating
them un-normalised would silently turn it into an image-only channel.

> **Terminology.** Content retrieval uses `faiss.IndexFlatIP`, which is an
> **exact** inner-product index. It is not approximate nearest neighbour; that
> would need IVF/HNSW. Latencies reported here are exact-search latencies.

### 2. Candidate merge (`src/recall/pipeline.py`)

Channels produce incomparable scores — popularity counts (0…546), ItemCF
similarity sums (0…10), cosine similarities (−1…1). The merge therefore ranks by
**reciprocal rank fusion**:

```
merge_score(j) = Σ_channels 1 / (60 + rank_channel(j))
```

so no channel can dominate the pool just because its numbers are larger.
Duplicates are **merged, not dropped**: the candidate keeps every contributing
channel with its rank inside that channel, which is what the Inspector displays.

### 3. Ranking (`src/pipeline`)

The default ranker is **MM-SASRec with concatenation fusion** — the best 3-seed
result in the offline table. `sasrec` (ID-only) and `mm_gated` are selectable in
the UI for comparison. Only the merged candidate pool is scored.

### 4. Reranking (`src/rerank/simple.py`)

Three policies and nothing more: seen filter, dedup, and an optional
**zero-train exploration quota** that guarantees at least *N* items with no
training signal reach the final list.

The quota is an **exposure policy, not a model improvement**. Offline metrics are
measured with it off. It exists because the cold-item experiment shows content
alone cannot out-score warm items in the full catalogue, so without a quota those
items would receive no impressions at all.

## Evaluation

### Recall layer

<!-- TABLE:RECALL -->
| Channel | Recall@100 | Recall@200 | Recall@500 | Recall@1000 |
|---|---|---|---|---|
| popular | 0.0170 | 0.0299 | 0.0625 | 0.1071 |
| itemcf | 0.1487 | 0.1852 | 0.2376 | 0.2479 |
| semantic | 0.0732 | 0.1004 | 0.1555 | 0.2225 |
| merged | 0.1269 | 0.1652 | 0.2327 | 0.3004 |

_Test target, user history masked, 100000 users. Candidate-generation quality: this is the ceiling the ranker can reach._
<!-- /TABLE:RECALL -->

### Which channel actually produced the recommendations?

<!-- TABLE:SOURCES -->
| Source | Top-20 hits | Share | Head hits | Middle hits | Tail hits |
|---|---|---|---|---|---|
| popular | 62 | 0.0446 | 62 | 0 | 0 |
| itemcf | 382 | 0.2750 | 126 | 103 | 153 |
| semantic | 131 | 0.0943 | 43 | 42 | 46 |
| multiple | 814 | 0.5860 | 468 | 126 | 220 |

_Candidate budget 2000. `multiple` means the item was found by more than one channel._
<!-- /TABLE:SOURCES -->

### Two-stage retention and serving latency

<!-- TABLE:PIPELINE -->
| Candidate budget | Candidate recall | Pipeline Recall@20 | Retention | Recall latency (ms) | Score (ms) | Encode (ms) |
|---|---|---|---|---|---|---|
| 100 | 0.1237 | 0.1028 | 0.738 | 23.6 | 0.12 | 8.37 |
| 200 | 0.1621 | 0.1168 | 0.838 | 23.8 | 0.16 | 8.47 |
| 500 | 0.2320 | 0.1305 | 0.937 | 23.9 | 0.31 | 8.37 |
| 1000 | 0.2967 | 0.1335 | 0.958 | 23.6 | 0.51 | 8.38 |
| 2000 | 0.3791 | 0.1389 | 0.997 | 23.9 | 0.66 | 8.41 |

Same-checkpoint, same-user full-catalogue **Recall@20 = 0.1393** (10000 users, sample seed 42). *Retention* is pipeline ÷ that baseline — the only apples-to-apples way to state it.

Latency columns are from `scripts/benchmark_latency.py` (CPU, median, this machine) and are for relative comparison only. Note that the **encode** stage dominates: scoring the whole catalogue costs about the same as scoring a 100-item pool, so the two-stage split buys recall quality and catalogue headroom rather than latency at this scale.
<!-- /TABLE:PIPELINE -->

### Model results (full-catalogue ranking)

Strict offline evaluation: every user, every item, the user's history masked, the
ground truth never masked. **This is not the same measurement as the pipeline
table above.**

<!-- TABLE:OVERALL -->
| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | MRR@20 | Coverage@20 | Params | #seeds |
|---|---|---|---|---|---|---|---|---|
| Popular (train-freq) | 0.0023 ± 0.0000 | 0.0036 ± 0.0000 | 0.0011 ± 0.0000 | 0.0014 ± 0.0000 | 0.0008 ± 0.0000 | 0.0025 ± 0.0000 | 0 | 3 |
| BPR-MF | 0.0199 ± 0.0001 | 0.0331 ± 0.0003 | 0.0096 ± 0.0001 | 0.0129 ± 0.0001 | 0.0074 ± 0.0001 | 0.7275 ± 0.0121 | 15326720 | 3 |
| SASRec (ID-only) | 0.0852 ± 0.0012 | 0.1224 ± 0.0025 | 0.0463 ± 0.0006 | 0.0557 ± 0.0009 | 0.0371 ± 0.0005 | 0.8017 ± 0.0212 | 2929792 | 3 |
| SASRec (ID-only) + item-dropout 0.2 | 0.0886 ± 0.0008 | 0.1283 ± 0.0011 | 0.0479 ± 0.0002 | 0.0579 ± 0.0001 | 0.0382 ± 0.0005 | 0.7426 ± 0.0155 | 2929920 | 3 |
| MM-SASRec (id+image, gated) | 0.0851 | 0.1237 | 0.0463 | 0.0560 | 0.0371 | 0.8438 | 3096322 | 1 |
| MM-SASRec (id+text, gated) | 0.0821 | 0.1176 | 0.0448 | 0.0538 | 0.0360 | 0.8524 | 3046402 | 1 |
| MM-SASRec (id+text+image+video, gated) | 0.0879 | 0.1271 | 0.0475 | 0.0574 | 0.0380 | 0.7668 | 3345668 | 1 |
| MM-SASRec (image, gated) | 0.0541 | 0.0874 | 0.0267 | 0.0351 | 0.0207 | 0.2746 | 536705 | 1 |
| MM-SASRec (text+image, gated) | 0.0704 | 0.1126 | 0.0355 | 0.0461 | 0.0278 | 0.3555 | 619778 | 1 |
| MM-SASRec (text, gated) | 0.0413 | 0.0702 | 0.0196 | 0.0269 | 0.0151 | 0.2455 | 486785 | 1 |
| MM-SASRec (video, gated) | 0.0570 | 0.0905 | 0.0281 | 0.0365 | 0.0217 | 0.2872 | 569985 | 1 |
| MM-SASRec (id+text+image, concat) | 0.0960 ± 0.0002 | 0.1393 ± 0.0003 | 0.0516 ± 0.0001 | 0.0625 ± 0.0001 | 0.0412 ± 0.0001 | 0.7708 ± 0.0107 | 3212288 | 3 |
| MM-SASRec (id+text+image, concat) + ID-dropout 0.2 | 0.0982 ± 0.0015 | 0.1429 ± 0.0013 | 0.0524 ± 0.0011 | 0.0637 ± 0.0010 | 0.0416 ± 0.0009 | 0.7531 ± 0.0039 | 3212288 | 2 |
| MM-SASRec (id+text+image, gated) | 0.0868 ± 0.0006 | 0.1267 ± 0.0008 | 0.0466 ± 0.0006 | 0.0567 ± 0.0006 | 0.0372 ± 0.0006 | 0.7388 ± 0.0210 | 3179395 | 3 |
| MM-SASRec (id+text+image, gated) + ID-dropout 0.2 | 0.0902 ± 0.0012 | 0.1314 ± 0.0007 | 0.0487 ± 0.0007 | 0.0591 ± 0.0005 | 0.0390 ± 0.0005 | 0.7891 ± 0.0183 | 3179395 | 3 |
<!-- /TABLE:OVERALL -->

<!-- TABLE:FINDINGS -->
- **Ordering holds** — Popular 0.0036 < BPR-MF 0.0331 < SASRec 0.1224 test Recall@20 over 100 000 evaluated test users (3 runs / 3 runs / 3 runs respectively).
- **Gain_reg** (item-dropout control) = SASRec+item-dropout 0.2 − SASRec = +0.0059 test Recall@20 over 100 000 users (3 runs vs 3 runs); NDCG@20 +0.0022.
- **Gain_content** (over the dropout control) = MM-SASRec gated+ID-dropout 0.2 − SASRec+item-dropout 0.2 = +0.0031 test Recall@20 over 100 000 users (3 runs vs 3 runs); NDCG@20 +0.0012.
- **Multimodal vs ID-only** = 100 000 users: Recall@20 0.1224 (ID-only) → 0.1314, NDCG@20 0.0557 → 0.0591.
- **Fusion without ID dropout** — gated 0.1267 vs concat 0.1393 test Recall@20 vs ID-only 0.1224 (same 100 000 users); Δ vs ID-only gated +0.0043, concat +0.0169. With ID-dropout 0.2 the order is unchanged: gated 0.1314 vs concat 0.1429.
- **Long tail** (buckets from training interactions only, rule `frequency_quantile`, 3 runs), Recall@20 — head: ID-only 0.1878 → MM 0.2031 (+0.0153), vs dropout control (0.1953) +0.0078; middle: ID-only 0.1185 → MM 0.1270 (+0.0085), vs dropout control (0.1256) +0.0014; tail: ID-only 0.0752 → MM 0.0797 (+0.0045), vs dropout control (0.0794) +0.0003. Bucket sizes: head 33 521 users, middle 21 904 users, tail 44 575 users. Concat+ID-dropout 0.2 reaches tail 0.0881 (+0.0087 vs the same dropout control).
- **Cold items** (cold catalogue = 1 974 items, 12 717 users with a cold target): cold-only Recall@20 — theoretical uniform ranker 20/1974 = 0.0101, measured Random 0.0097, SASRec ID-only 0.0000, content-only MM-SASRec 0.1092, gated MM-SASRec 0.0983 (0.0789 with ID-dropout 0.2).
_Every value above is computed from `results/tables/*.csv` by `analysis/make_readme_tables.py`; recall denominators are the evaluated test users named in each line._
<!-- /TABLE:FINDINGS -->

### Modality / fusion ablation

<!-- TABLE:ABLATION -->
| Kind | ID | Text | Image | Video | Fusion | ID-dropout | Item-dropout | Recall@10 | Recall@20 | NDCG@20 | Params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ID-only | ✓ |  |  |  | - | — | — | 0.0852 ± 0.0012 | 0.1224 ± 0.0025 | 0.0557 ± 0.0009 | 2929792 |
| ID-only | ✓ |  |  |  | - | — | 0.2 | 0.0886 ± 0.0008 | 0.1283 ± 0.0011 | 0.0579 ± 0.0001 | 2929920 |
| MM |  |  |  | ✓ | gated | — | — | 0.0570 | 0.0905 | 0.0365 | 569985 |
| MM |  |  | ✓ |  | gated | — | — | 0.0541 | 0.0874 | 0.0351 | 536705 |
| MM |  | ✓ |  |  | gated | — | — | 0.0413 | 0.0702 | 0.0269 | 486785 |
| MM |  | ✓ | ✓ |  | gated | — | — | 0.0704 | 0.1126 | 0.0461 | 619778 |
| MM | ✓ |  | ✓ |  | gated | — | — | 0.0851 | 0.1237 | 0.0560 | 3096322 |
| MM | ✓ | ✓ |  |  | gated | — | — | 0.0821 | 0.1176 | 0.0538 | 3046402 |
| MM | ✓ | ✓ | ✓ |  | concat | — | — | 0.0960 ± 0.0002 | 0.1393 ± 0.0003 | 0.0625 ± 0.0001 | 3212288 |
| MM | ✓ | ✓ | ✓ |  | concat | 0.2 | — | 0.0982 ± 0.0015 | 0.1429 ± 0.0013 | 0.0637 ± 0.0010 | 3212288 |
| MM | ✓ | ✓ | ✓ |  | gated | — | — | 0.0868 ± 0.0006 | 0.1267 ± 0.0008 | 0.0567 ± 0.0006 | 3179395 |
| MM | ✓ | ✓ | ✓ |  | gated | 0.2 | — | 0.0902 ± 0.0012 | 0.1314 ± 0.0007 | 0.0591 ± 0.0005 | 3179395 |
| MM | ✓ | ✓ | ✓ | ✓ | gated | — | — | 0.0879 | 0.1271 | 0.0574 | 3345668 |
<!-- /TABLE:ABLATION -->

### Long-tail analysis

<!-- TABLE:LONGTAIL -->
_Popularity rule: `frequency_quantile` (training interactions only)._

| Model | Head Recall@20 | Middle Recall@20 | Tail Recall@20 | Head NDCG@20 | Middle NDCG@20 | Tail NDCG@20 |
|---|---|---|---|---|---|---|
| BPR-MF | 0.0728 ± 0.0009 | 0.0232 ± 0.0007 | 0.0082 ± 0.0010 | 0.0290 ± 0.0002 | 0.0091 ± 0.0003 | 0.0027 ± 0.0003 |
| MM-SASRec (id+image, gated) | 0.2006 | 0.1182 | 0.0686 | 0.0948 | 0.0530 | 0.0283 |
| MM-SASRec (id+text, gated) | 0.1936 | 0.1118 | 0.0632 | 0.0934 | 0.0497 | 0.0260 |
| MM-SASRec (id+text+image+video, gated) | 0.2064 | 0.1233 | 0.0692 | 0.0985 | 0.0545 | 0.0279 |
| MM-SASRec (image, gated) | 0.1989 | 0.0553 | 0.0194 | 0.0848 | 0.0184 | 0.0058 |
| MM-SASRec (text+image, gated) | 0.2256 | 0.0999 | 0.0339 | 0.0990 | 0.0366 | 0.0109 |
| MM-SASRec (text, gated) | 0.1736 | 0.0282 | 0.0130 | 0.0692 | 0.0089 | 0.0039 |
| MM-SASRec (video, gated) | 0.2039 | 0.0715 | 0.0145 | 0.0856 | 0.0262 | 0.0047 |
| MM-SASRec (id+text+image, concat) | 0.2145 ± 0.0021 | 0.1330 ± 0.0016 | 0.0858 ± 0.0017 | 0.1008 ± 0.0007 | 0.0590 ± 0.0005 | 0.0355 ± 0.0008 |
| MM-SASRec (id+text+image, concat) + ID-dropout 0.2 | 0.2189 ± 0.0011 | 0.1380 ± 0.0007 | 0.0881 ± 0.0017 | 0.1027 ± 0.0003 | 0.0608 ± 0.0010 | 0.0357 ± 0.0015 |
| MM-SASRec (id+text+image, gated) | 0.2103 ± 0.0045 | 0.1204 ± 0.0018 | 0.0670 ± 0.0036 | 0.0996 ± 0.0013 | 0.0527 ± 0.0007 | 0.0264 ± 0.0018 |
| MM-SASRec (id+text+image, gated) + ID-dropout 0.2 | 0.2031 ± 0.0020 | 0.1270 ± 0.0045 | 0.0797 ± 0.0024 | 0.0968 ± 0.0008 | 0.0565 ± 0.0011 | 0.0321 ± 0.0013 |
| Popular (train-freq) | 0.0106 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0043 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| SASRec (ID-only) | 0.1878 ± 0.0040 | 0.1185 ± 0.0042 | 0.0752 ± 0.0006 | 0.0885 ± 0.0025 | 0.0538 ± 0.0022 | 0.0320 ± 0.0008 |
| SASRec (ID-only) + item-dropout 0.2 | 0.1953 ± 0.0025 | 0.1256 ± 0.0021 | 0.0794 ± 0.0004 | 0.0923 ± 0.0005 | 0.0564 ± 0.0005 | 0.0327 ± 0.0004 |
<!-- /TABLE:LONGTAIL -->

### Where does the multimodal gain come from?

`Δ vs SASRec` answers "does multimodal help?". `Δ vs ID+item-dropout` answers
"is it the *content* or just the extra regularisation?". Both are needed.

<!-- TABLE:GAIN -->
| Model | ΔHead vs SASRec | ΔMiddle vs SASRec | ΔTail vs SASRec | ΔHead vs ID+item-drop | ΔMiddle vs ID+item-drop | ΔTail vs ID+item-drop |
|---|---|---|---|---|---|---|
| MM-SASRec (id+image, gated) | +0.0129 | -0.0003 | -0.0066 | +0.0053 | -0.0074 | -0.0108 |
| MM-SASRec (id+text, gated) | +0.0058 | -0.0067 | -0.0120 | -0.0017 | -0.0138 | -0.0162 |
| MM-SASRec (id+text+image+video, gated) | +0.0186 | +0.0048 | -0.0060 | +0.0111 | -0.0023 | -0.0102 |
| MM-SASRec (image, gated) | +0.0111 | -0.0632 | -0.0558 | +0.0036 | -0.0703 | -0.0600 |
| MM-SASRec (text+image, gated) | +0.0378 | -0.0186 | -0.0413 | +0.0303 | -0.0257 | -0.0455 |
| MM-SASRec (text, gated) | -0.0142 | -0.0903 | -0.0622 | -0.0217 | -0.0974 | -0.0664 |
| MM-SASRec (video, gated) | +0.0161 | -0.0470 | -0.0607 | +0.0086 | -0.0541 | -0.0649 |
| MM-SASRec (id+text+image, concat) | +0.0267 | +0.0145 | +0.0106 | +0.0192 | +0.0074 | +0.0064 |
| MM-SASRec (id+text+image, concat) + ID-dropout 0.2 | +0.0311 | +0.0195 | +0.0129 | +0.0236 | +0.0124 | +0.0087 |
| MM-SASRec (id+text+image, gated) | +0.0225 | +0.0019 | -0.0082 | +0.0150 | -0.0052 | -0.0124 |
| MM-SASRec (id+text+image, gated) + ID-dropout 0.2 | +0.0153 | +0.0085 | +0.0045 | +0.0078 | +0.0014 | +0.0003 |
<!-- /TABLE:GAIN -->

### Gate analysis

`python scripts/export_gates.py --run-dir <mm run>` then
`python analysis/analyze_gates.py` produce
`results/figures/modality_gate_distribution.png`.

<!-- TABLE:GATES -->
_Mean over 13 documented run(s), from `results/tables/gate_by_bucket.csv`._

| Model | Modality | Head | Middle | Tail | Cold |
|---|---|---|---|---|---|
| mm_sasrec(id+image) | id | 0.988 | 0.989 | 0.984 | TBD |
| mm_sasrec(id+image) | image | 0.012 | 0.011 | 0.016 | TBD |
| mm_sasrec(id+text) | id | 0.977 | 0.981 | 0.974 | TBD |
| mm_sasrec(id+text) | text | 0.023 | 0.019 | 0.026 | TBD |
| mm_sasrec(id+text+image) | id | 0.964 | 0.965 | 0.954 | TBD |
| mm_sasrec(id+text+image) | image | 0.013 | 0.013 | 0.019 | TBD |
| mm_sasrec(id+text+image) | text | 0.024 | 0.022 | 0.027 | TBD |
| mm_sasrec(id+text+image)@cold10 | id | 0.984 | 0.984 | 0.812 | 0.000 |
| mm_sasrec(id+text+image)@cold10 | image | 0.006 | 0.006 | 0.087 | 0.480 |
| mm_sasrec(id+text+image)@cold10 | text | 0.009 | 0.009 | 0.100 | 0.520 |
| mm_sasrec(id+text+image+video) | id | 0.960 | 0.959 | 0.944 | TBD |
| mm_sasrec(id+text+image+video) | image | 0.012 | 0.012 | 0.017 | TBD |
| mm_sasrec(id+text+image+video) | text | 0.021 | 0.020 | 0.024 | TBD |
| mm_sasrec(id+text+image+video) | video | 0.007 | 0.009 | 0.015 | TBD |
| mm_sasrec(text+image) | image | 0.613 | 0.635 | 0.623 | TBD |
| mm_sasrec(text+image) | text | 0.387 | 0.365 | 0.377 | TBD |
| mm_sasrec(text+image)@cold10 | image | 0.652 | 0.665 | 0.651 | 0.641 |
| mm_sasrec(text+image)@cold10 | text | 0.348 | 0.335 | 0.349 | 0.359 |
| mm_sasrec_iddrop(id+text+image) | id | 0.967 | 0.961 | 0.939 | TBD |
| mm_sasrec_iddrop(id+text+image) | image | 0.016 | 0.019 | 0.029 | TBD |
| mm_sasrec_iddrop(id+text+image) | text | 0.017 | 0.020 | 0.033 | TBD |
| mm_sasrec_iddrop(id+text+image)@cold10 | id | 0.969 | 0.963 | 0.785 | 0.000 |
| mm_sasrec_iddrop(id+text+image)@cold10 | image | 0.013 | 0.015 | 0.150 | 0.797 |
| mm_sasrec_iddrop(id+text+image)@cold10 | text | 0.018 | 0.022 | 0.065 | 0.203 |
<!-- /TABLE:GATES -->

The expected trend is present — the ID weight falls and the content weights rise
monotonically from head to tail — but the magnitude is small. The model keeps
most of its weight on the ID branch, which is exactly why naive gated fusion
underperforms the ID-only baseline and why ID dropout is needed.

## Cold start

Two different notions of "cold" are kept strictly apart, in the code, the API and
the UI:

| | simulated cold | zero-train signal |
|---|---|---|
| what | controlled benchmark subset with **all** training interactions removed | items with zero observed training interactions in the base split |
| where | `data/processed/cold10` (1 974 items) | base split (376 items) |
| used for | the scientific cold-start experiment below | the serving exploration quota |

Simulated cold split: 10 % of items have every training interaction removed; they
keep their content features and their ID representation is zeroed at inference
for every model. 12 717 users have a cold test target.

`Cold *` = full ranking over all items for users whose target is cold.
`ColdOnly *` = ranking restricted to the cold catalogue.

<!-- TABLE:COLD -->
| Model | Cold Recall@10 | Cold Recall@20 | Cold NDCG@10 | Cold NDCG@20 | ColdOnly Recall@10 | ColdOnly Recall@20 | #users with cold target |
|---|---|---|---|---|---|---|---|
| MM-SASRec (text+image, gated) [cold10] | 0.0001 | 0.0001 | 0.0000 | 0.0000 | 0.0687 | 0.1092 | 12717 |
| MM-SASRec (id+text+image, gated) [cold10] | 0.0002 | 0.0007 | 0.0001 | 0.0002 | 0.0646 | 0.0983 | 12717 |
| MM-SASRec (id+text+image, gated) + ID-dropout 0.2 [cold10] | 0.0001 | 0.0008 | 0.0000 | 0.0002 | 0.0456 | 0.0789 | 12717 |
| Random (uniform) [cold10] | 0.0013 | 0.0016 | 0.0006 | 0.0007 | 0.0036 | 0.0097 | 12717 |
| SASRec (ID-only) [cold10] | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 12717 |
<!-- /TABLE:COLD -->

The two columns answer different questions and must not be compared to each
other. The gap between them is the **cold/warm score calibration problem**: a
content model can tell cold items apart from each other, but their scores are not
comparable to warm items' scores in a shared ranking. That is what the serving
layer's exploration quota exists to mitigate.

Ranks use the **average-rank tie policy**. With an optimistic policy, the 1 974
cold items all tied at score 0 would each be reported as a perfect hit — a
measurement artefact rather than a result.

## Efficiency

<!-- TABLE:EFFICIENCY -->
|  | value |
|---|---|
| SASRec (ID-only) parameters | 2 929 792 |
| MM-SASRec (gated) parameters | 3 179 395 (+8.5 %) |
| epochs trained (SASRec / MM) | 80 / 70 |
| wall time per epoch (SASRec / MM) | 14.4 / 20.8 s |
| total training time (SASRec / MM) | 1 164 / 1 457 s |
| evaluation | full ranking, 100 000 users x 19 738 items |
| runs behind these numbers | 6 |
| vector index | faiss.IndexFlatIP (exact inner product) |
| index build / query latency | 0.0417 s / 0.5017 ms per query |
<!-- /TABLE:EFFICIENCY -->

## Advanced extension: Semantic IDs and generative retrieval

`src/models/rqvae.py`, `src/models/semantic_id.py`, `src/models/generative_rec.py`

Content embeddings are quantised into Semantic IDs with an RQVAE, and a small
causal Transformer generates the next item's ID instead of scoring vectors.
This is an **extension**, not part of the main architecture.

<!-- TABLE:SEMANTICID -->
|  |  |
|---|---|
| source features | `text+image(L2-normalised per modality)` |
| quantiser | RQVAE, 4 levels × 256 codes, latent 64 (1 471 168 params) |
| reconstruction cosine | 0.681 |
| codebook utilisation | `[1.0, 1.0, 1.0, 1.0]` |
| unique Semantic IDs | 19 692 / 19 738 items |
| collision rate | **0.233 %** |
<!-- /TABLE:SEMANTICID -->

Three things had to be fixed before the codes were usable, all documented in the
code: residual-aware code re-seeding, the commitment term, and per-modality
normalisation. With random codebook initialisation and a codebook-only loss the
collision rate was **49 %**, which would have made generative retrieval
meaningless.

Decoding is constrained by a prefix trie, so the generator can only emit a code
tuple that corresponds to a real item — never a hallucinated one, and never a
random fallback.

## API

```
GET /system                      dataset, rankers, recall channels, readiness
GET /models                      rankers + offline metrics from results/tables
GET /evaluation                  recall / pipeline / latency artifacts
GET /users/{u}                   history with cold / bucket metadata
GET /users/{u}/recall            candidate generation with per-source trace
GET /users/{u}/recommend         the served top-K
GET /users/{u}/inspect           full request trace (Inspector page)
GET /cold/summary                cold-start experiment results
GET /cold/items[/{id}]           cold items + content availability
GET /media/manifest              prepared demo media + the id-mapping evidence
GET /media/video/{item_id}       the prepared mp4 (integer id, manifest lookup only)
GET /feed                        playable feed in recommendation order
POST /recommend                  legacy history-based endpoint
```

```bash
curl -s --noproxy '*' localhost:8000/users/7/recommend?top_k=5
```

Item ids in the API are **raw MicroLens ids**. The conversion from the internal
`1..N` id space happens in exactly one place (`src/serving/service.py`).

![System page](assets/system_demo.png)

<p align="center"><em>System — recall quality by candidate budget, two-stage retention, and serving latency split by stage.</em></p>

## Repository structure

```
configs/          one YAML per experiment
docs/             data_schema.md, evaluation_protocol.md, design.md
src/data/         preprocessing, datasets, negative sampling, popularity, metadata
src/models/       bpr, sasrec, mm_sasrec, item_encoder, fusion, rqvae, generative_rec
src/recall/       popular, itemcf, semantic channels + candidate merge
src/rerank/       seen filter, dedup, zero-train exploration quota
src/pipeline/     ranker registry + two-stage recommender
src/evaluation/   metrics, full-ranking evaluator, cold/tail slicing
src/training/     model factory, trainer (AMP, early stopping, checkpointing)
src/retrieval/    exact inner-product index (Faiss, with numpy fallback)
src/serving/      service, schemas, FastAPI app
scripts/          CLI entry points (train, prepare_demo, evaluate_recall, ...)
analysis/         aggregation, plotting, gate analysis, README table generation
frontend/         React + Vite + TypeScript demo (Live Demo / Inspector / Cold / System)
tests/            unit + integration tests
results/          runs/, tables/, figures/   (generated)
artifacts/        indices, semantic IDs, demo manifest   (generated)
```

## Reproducibility

```python
set_seed(seed)   # random, numpy, torch (CPU + CUDA), PYTHONHASHSEED
```

Negative sampling, cold-item selection and popularity bucketing use explicitly
seeded generators. The pipeline evaluation draws its user sample with a fixed
seed and records the sample hash in the artifact. Training data is fully in
memory with `num_workers = 0`. Checkpoints store a dataset hash and refuse to
load into a dataset with a different item mapping or cold split.

GPU kernels are not guaranteed to be bitwise deterministic. Runs are reproducible
in distribution: same config + same seed ⇒ identical data order, negative samples
and initialisation.

## Limitations

* **This is not a production system.** It is an offline + serving prototype on a
  public dataset with a 19.7 K item catalogue. There is no streaming ingestion,
  no feature store, no A/B harness.
* MicroLens-100K has short histories (median 6 interactions), which caps what any
  sequential model can learn.
* The subset ships **no item titles or captions**, so the demo identifies items by
  raw id and content-space neighbours only.
* The cold protocol is **simulated** (interactions removed by a fixed seed), not a
  naturally occurring cold-start stream.
* Validation is used for early stopping, so validation numbers are optimistic;
  test numbers are reported from the best checkpoint.
* Cold items remain poorly calibrated against warm items in the full catalogue.
  The exploration quota is a mitigation, not a solution.
* Content retrieval is **exact** (`IndexFlatIP`) and therefore O(catalogue) per
  query. Approximate indexing (IVF/HNSW) is deliberately not claimed.
* Serving latency numbers are CPU demo benchmarks on a shared machine, for
  relative comparison only — not a production SLA.
* Video features are wired through the architecture but the headline tables use
  ID + text + image; the video ablation is reported separately.
* Serving is single-process and CPU-only by default; the API has no
  authentication, rate limiting or caching.

## Reproducing the numbers

```bash
python analysis/aggregate_results.py     # results/tables/*.csv from results/runs/
python analysis/update_readme.py         # inject the tables into this README
python analysis/plot_overall.py
python analysis/plot_long_tail.py
python analysis/plot_cold_start.py
python scripts/export_gates.py --run-dir results/runs/<mm run>
python analysis/analyze_gates.py
python scripts/evaluate_recall.py
python scripts/evaluate_pipeline.py
python scripts/benchmark_latency.py
python scripts/find_demo_user.py
python scripts/capture_demo.py
```

`python analysis/update_readme.py --check` fails if this README and
`results/tables/*.csv` disagree.
