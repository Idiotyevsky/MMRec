# Generative retrieval over Semantic IDs

This document covers the third recall route: instead of embedding items into a
vector space and looking up nearest neighbours, the catalogue is mapped to
discrete **Semantic IDs** and the next item is produced by an autoregressive
decoder. It is an **extension**, not the main system — the two-stage pipeline
described in `docs/design.md` is unchanged, and the generative channel is off by
default in serving (`generative_checkpoint=None`).

Everything below is measured. Where a result was negative or ambiguous it is
reported as such; the point of the exercise is to find out whether this route
earns its place, not to argue that it does.

## 1. Why this is a separate route

The embedding channels (`itemcf`, `semantic`) share a property that makes them
cheap and predictable: retrieval is an index lookup, so the candidate budget is
independent of the computation. Generative retrieval breaks that property. The
candidate set *is* the decoder's output, so the budget is set by the beam width,
and widening the beam costs latency roughly linearly. Section 6 quantifies that
trade; it is the main practical finding.

## 2. Pipeline

```
item content (text + image)
        │  per-modality L2 normalisation, then concatenation
        ▼
   RQ-VAE encoder ──► residual quantiser (4 levels × 256 codes)
        │
        ▼
   Semantic ID  (c1, c2, c3, c4)          one per item
        │
        ▼
   token vocabulary:  0=PAD, 1=BOS, 2=EOS, code token = 3 + level*256 + code
        │
        ▼
   autoregressive decoder over history tokens ──► constrained beam search
        │
        ▼
   Semantic ID ──► item(s)   (collisions expanded, seen items removed)
```

Source: `src/models/rqvae.py`, `src/models/semantic_id.py`,
`src/models/generative_rec.py`, `src/recall/generative.py`.
Training: `scripts/train_rqvae.py`, `scripts/train_generative_rec.py`, driven by
`configs/rqvae.yaml` and `configs/generative_rec.yaml`.

## 3. Semantic ID quality

Measured by `scripts/analyze_semantic_ids.py`; artifacts in
`artifacts/semantic_ids.npz` and `artifacts/semantic_ids_meta.json`, tables in
`results/tables/semantic_id_quality.csv`.

| level | codebook utilisation | dead codes | perplexity |
|---|---|---|---|
| 1 | 1.000 | 0 | 222.7 |
| 2 | 1.000 | 0 | 230.3 |
| 3 | 1.000 | 0 | 243.1 |
| 4 | 1.000 | 0 | 244.4 |

19 692 unique Semantic IDs over 19 738 items → **0.233 % collision rate**,
38 colliding IDs covering 84 items, largest collision group 5 items.
Reconstruction cosine 0.681.

The meta file records `semantic_id_hash`, `dataset_hash` and
`rqvae_checkpoint_sha16`. `load_generative_recall` recomputes the Semantic-ID
hash and **refuses to load** a checkpoint trained against a different table —
otherwise token ids would silently refer to different items.

## 4. Does the hierarchy carry meaning?

This question has two plausible tests and they disagree, which is worth
recording.

**Mean content cosine between items sharing a prefix** (insensitive):

| pairs | mean cosine | lift over random |
|---|---|---|
| same level-1 code | +0.1747 | +0.0007 |
| same level-1..2 | +0.1728 | −0.0012 |
| same level-1..3 | +0.1885 | +0.0145 |
| same full SID | +0.2029 | +0.0289 |
| random pairs | +0.1740 | — |

Read alone, this says the coarse levels are indistinguishable from chance. That
reading is **wrong**, and the reason is that the content space is concentrated:
two random items already share a cosine of +0.174, so the metric has almost no
dynamic range and cannot separate signal from the global geometry.

**Do an item's content nearest neighbours share its codes?** (decisive), from
`results/tables/semantic_id_neighbour_agreement.csv`, 3 000 items × 10 nearest
neighbours:

| prefix | share of neighbours sharing it | chance | lift |
|---|---|---|---|
| level-1 | 0.3819 | 0.003906 | 97.8× |
| level-1..2 | 0.0337 | 1.5e-5 | 2210× |
| level-1..3 | 0.0027 | 6.0e-8 | 45 299× |
| full SID | 0.0006 | 2.3e-10 | 2.7e6× |

38 % of an item's 10 nearest content neighbours share its first code, against a
0.39 % chance rate. The hierarchy **is** meaningful; the cosine test was simply
the wrong instrument. Both tests are kept in the code so the disagreement is
visible rather than resolved by deleting the inconvenient one.

Note the sampling detail: pairs are drawn by picking a random *item* and then a
partner in its prefix group, not by picking a random group. Sampling groups
proportionally to size over-weights the largest groups and makes short prefixes
dominate; the item-based draw is the comparison the table above claims to make.

## 5. Constrained decoding

`SemanticTrie` (`src/models/semantic_id.py`) is an explicit prefix tree over the
catalogue. At each step `allowed_next` returns only the codes that exist under
the current prefix, so the decoder **cannot** emit a Semantic ID that
corresponds to no item. `SemanticTrie.validate()` asserts the trie and the mapper
describe the same catalogue, and `PrefixConstraint` is now a thin wrapper over
it.

Two failure modes are counted rather than hidden:

* `empty_decodings` — the constrained beam died and produced nothing. Reported
  per evaluation run; it is 0 across every measurement in this document.
* **collisions** — one Semantic ID maps to several items. These are expanded,
  not dropped: `sid_to_items` returns all of them and each candidate carries
  `extra["collision_size"]`.

## 6. Cost of decoding

`scripts/benchmark_generative.py`, single-threaded CPU (how serving runs),
medians of 3 repeats after a warm-up pass, 50 users:

| beam width | ms/user | users/s | candidates/user | empty decodings |
|---|---|---|---|---|
| 1 | 42.7 | 23.4 | 0.96 | 0 |
| 5 | 193.1 | 5.2 | 4.88 | 0 |
| 10 | 363.2 | 2.8 | 9.80 | 0 |
| 20 | 741.9 | 1.3 | 19.68 | 0 |
| 50 | 1 843.5 | 0.5 | 49.66 | 0 |

For scale, the three embedding channels together cost **23.6 ms/user** and
produce a merged pool of ~2 355 candidates (`results/tables/latency_benchmark.csv`).
At beam 20 the generative channel costs **31× that** for 19.7 candidates.

Latency grows roughly linearly in the beam width, and so does the candidate
count, so there is no beam width at which this channel is both cheap and
useful on CPU. At beam 1 it costs 42.7 ms/user — still more than every embedding
channel combined — and yields under one candidate. This is the single strongest
argument against putting the channel in the serving path, and it is why
`generative_checkpoint` defaults to `None`.

A GPU decode is roughly 15 ms/user at beam 10, but the service runs on CPU with
one thread, so the CPU column is what governs the deployment decision. The GPU
numbers were **not** used in the table above because the shared machine made them
unstable — beam 50 measured *faster* than beam 20 under contention, which cannot
be a property of the algorithm — and quoting the flattering half of a noisy
measurement would be exactly the kind of thing this document is supposed to
prevent. The raw GPU run is kept in `results/tables/generative_latency_gpu.json` with that caveat attached.

## 7. Recall at a fixed budget

The generative channel can only return as many candidates as its beam produces,
so it is compared at the **same K** as every other channel, on the **same 3 000
test users**, with the beam set wide enough to fill the budget (beam 100 →
99.3 candidates/user on average, 0 users with an empty pool).

Reproduce with `scripts/evaluate_recall.py --max-users 3000 --ks 20 50 100
--generative-checkpoint results/runs/genrec_sid/best.pt --generative-beam 100`;
table in `results/tables/genrec_comparison.csv`.

| channel | Recall@20 | Recall@50 | Recall@100 |
|---|---|---|---|
| popular | 0.0020 | 0.0077 | 0.0130 |
| itemcf | 0.1020 | 0.1413 | 0.1827 |
| semantic (content kNN) | 0.0423 | 0.0607 | 0.0887 |
| **generative (Semantic ID)** | **0.0643** | **0.0980** | **0.1333** |
| merged, generative off | 0.0853 | 0.1313 | 0.1633 |
| merged, generative on | 0.0913 | 0.1403 | 0.1847 |

Two things are worth stating plainly.

**It beats the embedding channel it is most comparable to.** Generative recall
is above `semantic` (content kNN over the same content features) at every K —
0.1333 vs 0.0887 at K=100, a 50 % relative improvement. The two channels use
identical inputs; the difference is that RQ-VAE quantisation plus a sequence
model captures co-occurrence structure that raw cosine similarity over content
does not.

**It is still below item-based collaborative filtering**, by a wide margin
(0.1333 vs 0.1827 at K=100). A 1.5 M-parameter decoder trained on 100 K users
does not overturn a well-populated co-occurrence index.

The 3 000-user sample is easier than the full test set — `itemcf` scores 0.1827
at K=100 here versus 0.1487 over all 100 000 users in
`results/tables/recall_eval.csv` — so these rows must only be compared **within**
this table, never against the full-set table.

On the full 100 000-user test set at beam 20 the trained model reaches
Recall@20 = 0.0590 (val 0.0817, early stopping on val). The val/test gap is
reported rather than hidden: the checkpoint was selected on val.

## 8. Complementarity

Same 3 000 users, widest K. The only difference between the two merged rows is
whether the generative channel is in the pool.

| | generative off | generative on | delta |
|---|---|---|---|
| targets found in the pool | 1 359 | 1 412 | **+53** |
| merged Recall@20 | 0.0853 | 0.0913 | +0.0060 |
| merged Recall@50 | 0.1313 | 0.1403 | +0.0090 |
| merged Recall@100 | 0.1633 | 0.1847 | +0.0213 |

The generative channel finds 400 targets at K ≤ 100, of which **53 are found by
no other channel** — and 1 412 − 1 359 = 53 exactly, so the pool gains precisely
the targets that only this channel can reach. Overlap is moderate rather than
redundant (Jaccard 0.282 against `itemcf`, 0.253 against `semantic`, versus 0.390
between `itemcf` and `semantic`).

The Recall@100 gain (+0.0213) is larger than 53/3000 = +0.0177 because adding a
fourth channel also perturbs the RRF ordering and lifts some candidates that
were already in the pool. Both numbers are reported; the +53 is the part that is
unambiguously attributable to this channel.

So the honest summary is: **generative retrieval is complementary, and the
complement is small.** It buys ~1.8 points of pool coverage on this sample, at
741.9 ms/user on CPU against 23.6 ms/user for everything else.

## 9. What this route is not

* It is not faster than the embedding channels, and no claim is made that it is.
  On CPU it is one to two orders of magnitude slower per user for a far smaller
  candidate pool.
* It does not "solve" cold start. A cold item still needs content features to
  receive a Semantic ID, and the generative model still has to have learned to
  emit that ID — the same cold-item problem, moved one stage earlier.
* A higher `Recall@K` for this channel would not by itself justify it. The
  question that matters is whether it finds targets the other channels miss,
  which is what section 8 measures.
