# Data schema — MicroLens-100K

Everything below was verified by running `python scripts/inspect_data.py --data-dir data/raw`
against the actual files. The raw report is written to `artifacts/raw_data_report.json`.
No field in this document is inferred from a filename.

## 1. Source

| | |
|---|---|
| Dataset | MicroLens-100K (subset of MicroLens, Westlake / Kuaishou) |
| Downloaded from | `https://huggingface.co/datasets/sisuo/Microlens_100k` |
| Paper | https://arxiv.org/abs/2409.09638 |
| Local path | `data/raw/` (git-ignored) |
| Size on disk | 190 MB |

| file | shape | dtype | size |
|---|---|---|---|
| `microlens_100k.inter` | 719 405 rows × 4 cols | text (TSV) | 19 MB |
| `text_feat.npy` | (19738, 384) | float32 | 29 MB |
| `image_feat.npy` | (19738, 768) | float32 | 58 MB |
| `video_feat.npy` | (19738, 1024) | float32 | 77 MB |

## 2. Interactions — `microlens_100k.inter`

Tab-separated, **with a header row**:

```
userID	itemID	timestamp	x_label
0	10209	1658197898237	0
0	16123	1661660511750	0
...
99999	12050	1662964670949	2
```

| column | type | range | notes |
|---|---|---|---|
| `userID` | int64 | `[0, 99999]` | 100 000 distinct values, contiguous 0-based |
| `itemID` | int64 | `[0, 19737]` | 19 738 distinct values, contiguous 0-based |
| `timestamp` | int64 | `[1583378629552, 1662984152842]` | milliseconds, spans 921 days |
| `x_label` | int64 | `{0, 1, 2}` | engagement bucket (70.9 % / 14.3 % / 14.7 %) |

Verified properties:

* **719 405 interactions**, 100 000 users, 19 738 items.
* `(userID, itemID)` is **unique** — 0 duplicates. There is no repeat-interaction
  semantics to model.
* No missing values in any column.
* Timestamps are **not globally sorted**, and rows are **not** grouped per user.
  The pipeline sorts by `(userID, timestamp)` with a stable sort; equal timestamps
  keep their original file order.
* Every user has **≥ 5** interactions, so the default `min_user_interactions: 5`
  filter drops nothing. Sequence lengths: min 5, median 6, mean 7.19, max 218.
* Every item has **≥ 1** interaction.
* `x_label` is **not used as a model feature**. It is an aggregate engagement
  signal that is not guaranteed to be available (or causally clean) at serving
  time; using it would weaken the leakage story for no modelling benefit.

## 3. Multimodal features

All three feature files have **exactly `num_items = 19738` rows**, and row `r`
belongs to raw item id `r`. The pipeline still builds an explicit
`raw_item_id -> feature_row` array (`row_for_item_<modality>.npy`) and asserts
`row_for_item[internal_id] < num_rows` at load time, so a differently-indexed
feature dump fails loudly instead of silently mis-aligning.

| modality | dim | row norm (mean ± std) | zero rows | NaN | Inf |
|---|---|---|---|---|---|
| text | 384 | 1.000 ± 3e-8 | 0 | 0 | 0 |
| image | 768 | 25.85 ± 0.39 | 0 | 0 | 0 |
| video | 1024 | 29.42 ± 2.89 | 0 | 0 | 0 |

* **text** is already L2-normalised (row norm exactly 1).
* **image** and **video** are unnormalised encoder outputs with norms around 26–30.
* The pipeline therefore normalises **inside the model** (`LayerNorm` as the first
  layer of every modality projection). Raw files are never modified.
* Missing modalities are represented by an **all-zero row**, which is flagged as
  *unavailable* and receives exactly zero fusion weight. A zero row is never fed
  to the model as if it were a real semantic vector. In this dataset the missing
  rate is 0 % for all three modalities, but the machinery is exercised by
  `tests/test_feature_alignment.py`.

## 4. Internal id convention

Downstream code uses a single convention everywhere:

```
0                = PAD
1 .. num_items   = valid items        (num_items = 19738)
1 .. num_users   = valid users        (num_users = 100000)
```

`data/processed/<name>/mappings.json` stores the `raw_item_ids` and
`raw_user_ids` arrays, so `internal_id -> raw_id` is `raw_item_ids[internal_id - 1]`.

Because the raw ids are already contiguous and 0-based, the mapping happens to be
the identity plus one. This is **asserted**, not assumed: `build_mapping` in
`src/data/preprocess.py` raises if any raw id is left unmapped, and
`tests/test_feature_alignment.py` runs the whole encoder on a **permuted** id
space to prove the mapping path works when the identity assumption is false.

## 5. Processed artifacts

`python -m src.data.preprocess --out data/processed/base` writes:

| file | contents |
|---|---|
| `dataset.npz` | `flat_items`, `user_offsets`, `train_len`, `val_target`, `test_target`, `train_freq`, `popularity_bucket`, `is_cold` |
| `mappings.json` | `raw_user_ids`, `raw_item_ids`, sizes |
| `row_for_item_{text,image,video}.npy` | feature row per internal item id |
| `stats.json` / `meta.json` | filter settings, distributions, bucket rule, cold-set description |

`flat_items` + `user_offsets` is a CSR-style ragged array: user `u`'s
chronological history is `flat_items[user_offsets[u] : user_offsets[u+1]]`.
Format is `npz`/`npy`/`json`; **no pickle** is used for any artifact.

## 6. Base dataset statistics (`data/processed/base`)

```
num_users                 100000
num_items                  19738
num_interactions          719405
avg_sequence_length         7.19
median_sequence_length      6
min / max sequence length   5 / 218
avg_train_length            5.19
sparsity                    0.99964
train item freq  min 0 | median 15 | max 546 | mean 26.3 | zero-freq items 376
popularity buckets (frequency_quantile, head 20% / tail 60% of the catalogue)
  head    3948 items -> 58.8 % of training interactions
  middle  3947 items -> 22.4 %
  tail   11843 items -> 18.8 %
```

The 376 items with **zero training frequency** are items that only ever appear as
a validation or test target. They are a natural, unmodified cold-start set and
are reported separately from the simulated cold split.

## 7. Leakage policy

| signal | source | rationale |
|---|---|---|
| item frequency / popularity buckets | **training interactions only** | full-lifetime counts would encode the existence of val/test interactions |
| negative sampling distribution | training interactions only | same |
| cold-item selection | training protocol + fixed seed | see `docs/evaluation_protocol.md` |
| `x_label`, likes, views | **not used** | aggregate over the full lifetime, cannot be shown to be causally clean |
| user/item history masks | the exact history fed to the model | the ground-truth target is explicitly protected |

`src/data/popularity.py` is the only module that computes frequency, and it is
only ever called on the training slice. `src/data/dataset.py::assert_no_target_leak`
re-verifies on every load that no evaluation target appears in the history used
to predict it.
