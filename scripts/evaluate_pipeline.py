#!/usr/bin/env python
"""Offline evaluation of the two-stage serving pipeline.

    python scripts/evaluate_pipeline.py
    python scripts/evaluate_pipeline.py --candidate-sizes 100 200 500 1000 --max-users 20000

For each candidate budget this measures the trade-off that decides how a real
recommender is deployed:

    candidate pool size  ->  how much recall headroom the ranker gets
                         ->  final Recall@20 / NDCG@20
                         ->  latency

Two things are reported separately and must never be conflated:

* **candidate recall** — does the pool even contain the ground-truth target?
  This is the hard ceiling of the whole system.
* **final Recall@20 / NDCG@20** — what the ranker achieves inside that pool.

The strict full-catalogue numbers in ``results/tables/overall.csv`` come from a
different evaluation and are not comparable to the pool numbers here.

Also reports which recall channels contributed to the final Top-20, split by
popularity bucket — that is the evidence for whether semantic recall actually
finds items the collaborative channels miss.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.data.popularity import BUCKET_NAMES  # noqa: E402
from src.pipeline import RankerRegistry, RankerSpec, TwoStageRecommender  # noqa: E402
from src.pipeline.rankers import discover_run  # noqa: E402
from src.recall import (  # noqa: E402
    CandidateMerger,
    ItemCFRecall,
    PopularRecall,
    SemanticRecall,
    load_itemcf_index,
)
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

LOG = get_logger("mmrec.evaluate_pipeline")


def build(args, data: ProcessedData):
    strategies = [PopularRecall(data.train_freq, data.num_items)]
    ic_path = ROOT / args.itemcf_index
    if ic_path.exists():
        ic = load_itemcf_index(ic_path)
        strategies.append(ItemCFRecall(ic["neighbors"], ic["sims"], data.num_items))
    else:
        LOG.warning(f"ItemCF index missing at {ic_path}")
    emb_path = ROOT / args.content_embeddings
    if emb_path.exists():
        strategies.append(SemanticRecall(np.load(emb_path), data.num_items))
    else:
        LOG.warning(f"content embeddings missing at {emb_path}")
    merger = CandidateMerger(strategies)

    specs = {}
    for name in ("sasrec", "mm_concat"):
        run = Path(getattr(args, f"{name}_run")) if getattr(args, f"{name}_run", None) else None
        if run is None or not (ROOT / run).exists():
            run = discover_run(name)
        if run is None:
            raise SystemExit(f"no finished run found for ranker {name!r}; "
                             "run scripts/train.py first")
        specs[name] = RankerSpec(name, run)
    registry = RankerRegistry(data, specs, device=args.device)
    rec = TwoStageRecommender(data, merger, registry, default_ranker=args.ranker)
    return rec, merger


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--itemcf-index", default="artifacts/itemcf_neighbors.npz")
    ap.add_argument("--content-embeddings", default="artifacts/content_embeddings.npy")
    ap.add_argument("--sasrec-run", default=None, help="default: newest finished 'sasrec' run")
    ap.add_argument("--mm-run", default=None, help="default: newest finished 'mm_concat' run")
    ap.add_argument("--ranker", default="mm_concat")
    ap.add_argument("--candidate-sizes", nargs="*", type=int, default=[100, 200, 500, 1000, 2000])
    ap.add_argument("--recall-k", type=int, default=1000, help="per-channel budget")
    ap.add_argument("--final-k", type=int, default=20)
    ap.add_argument("--max-users", type=int, default=20000)
    ap.add_argument("--latency-users", type=int, default=150,
                    help="users in the dedicated latency benchmark")
    ap.add_argument("--latency-repeats", type=int, default=20,
                    help="ranking repetitions per user in the latency benchmark")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="results/tables/pipeline_tradeoff.csv")
    ap.add_argument("--source-out", default="results/tables/recall_source_contribution.csv")
    args = ap.parse_args()

    data = ProcessedData.load(ROOT / args.processed_dir)
    rec, merger = build(args, data)
    rec.registry.warm()
    LOG.info(f"dataset: {data.summary()} | ranker={args.ranker}")

    users = np.arange(data.num_users, dtype=np.int64)[: args.max_users]
    targets = data.test_target[users]

    rows: list[dict] = []
    source_rows: list[dict] = []
    model = rec.registry.get(args.ranker)

    # Recall is the expensive part, so it runs once per user at the largest
    # per-channel budget and every candidate size is evaluated by truncating the
    # same merged pool.  That also guarantees the sizes are directly comparable.
    per_size = {
        k: {"cand_hit": np.zeros(users.shape[0], dtype=bool),
            "final_hit": np.zeros(users.shape[0], dtype=bool),
            "ndcg": np.zeros(users.shape[0], dtype=np.float64),
            "pool": np.zeros(users.shape[0], dtype=np.int64),
            "recall_latency": np.zeros(users.shape[0], dtype=np.float64),
            "rank_latency": np.zeros(users.shape[0], dtype=np.float64),
            "contrib": {**{name: 0 for name in merger.source_names}, "multiple": 0},
            "contrib_by_bucket": {b: {**{n: 0 for n in merger.source_names}, "multiple": 0}
                                  for b in BUCKET_NAMES.values()}}
        for k in args.candidate_sizes
    }
    bucket_n = {b: 0 for b in BUCKET_NAMES.values()}
    t0 = time.perf_counter()

    for i, u in enumerate(users.tolist()):
        hist = [int(x) for x in data.test_history(u)]
        t_req = time.perf_counter()
        merged = merger.recall_and_merge(u, hist, per_source_k=args.recall_k, total_k=None)
        tgt = int(targets[i])
        bname = BUCKET_NAMES.get(int(data.popularity_bucket[tgt]), "unknown")
        if bname in bucket_n:
            bucket_n[bname] += 1

        full_ids = merged.item_ids()
        recall_ms = (time.perf_counter() - t_req) * 1000

        for cand_k in args.candidate_sizes:
            st = per_size[cand_k]
            ids = full_ids[:cand_k]
            st["pool"][i] = len(ids)
            st["recall_latency"][i] = recall_ms
            if not ids:
                continue
            # Timings here are indicative only; the dedicated latency benchmark
            # below produces the numbers that go into the table.
            t_rank = time.perf_counter()
            scores = model.score_candidates(hist, ids)
            order = np.argsort(-scores)
            st["rank_latency"][i] = (time.perf_counter() - t_rank) * 1000
            st["cand_hit"][i] = tgt in set(ids)
            keep = order[: args.final_k]
            top = [ids[j] for j in keep.tolist()]
            if tgt in top:
                pos = top.index(tgt) + 1
                st["final_hit"][i] = True
                st["ndcg"][i] = 1.0 / np.log2(pos + 1)
                srcs = {s["name"] for s in merged.candidates[keep[pos - 1]].sources}
                key = next(iter(srcs)) if len(srcs) == 1 else "multiple"
                st["contrib"][key] += 1
                if bname in st["contrib_by_bucket"]:
                    st["contrib_by_bucket"][bname][key] += 1

        if (i + 1) % 2000 == 0:
            LOG.info(f"  {i+1}/{users.shape[0]} users ({time.perf_counter()-t0:.0f}s)")

    n = users.shape[0]
    for cand_k in args.candidate_sizes:
        st = per_size[cand_k]
        rows.append({
            "candidate_k": cand_k,
            "ranker": args.ranker,
            "num_users": n,
            "candidate_recall": float(st["cand_hit"].mean()),
            "final_Recall@20": float(st["final_hit"].mean()),
            "final_NDCG@20": float(st["ndcg"].mean()),
            "mean_pool_size": float(st["pool"].mean()),
            "recall_latency_ms": float(np.median(st["recall_latency"])),
            "rank_latency_ms": float(np.median(st["rank_latency"])),
            "latency_ms_mean": float(np.median(st["recall_latency"] + st["rank_latency"])),
            "latency_ms_p95": float(np.percentile(st["recall_latency"] + st["rank_latency"], 95)),
        })
        LOG.info(f"candidate_k={cand_k}: pool={st['pool'].mean():.0f} "
                 f"cand_recall={st['cand_hit'].mean():.4f} R@20={st['final_hit'].mean():.4f}")

        total_hits = max(sum(st["contrib"].values()), 1)
        for name, c in st["contrib"].items():
            row = {"candidate_k": cand_k, "source": name,
                   "top20_hits": c, "share_of_hits": c / total_hits}
            for b in BUCKET_NAMES.values():
                row[f"{b}_hits"] = st["contrib_by_bucket"][b].get(name, 0)
                row[f"{b}_users"] = bucket_n[b]
            source_rows.append(row)

    # ---- dedicated latency benchmark -----------------------------------
    # Accuracy needs many users but only one timing each; latency needs few
    # users but many repetitions.  Mixing the two either makes the run
    # prohibitively slow or leaves the latency column dominated by machine
    # noise, so they are measured separately.
    latency: dict[int, dict] = {}
    bench_users = np.arange(min(args.latency_users, users.shape[0]), dtype=np.int64)
    for cand_k in args.candidate_sizes:
        rec_ms: list[float] = []
        rank_ms: list[float] = []
        for u in bench_users.tolist():
            hist = [int(x) for x in data.test_history(u)]
            t0 = time.perf_counter()
            merged = merger.recall_and_merge(u, hist, per_source_k=args.recall_k, total_k=None)
            rec_ms.append((time.perf_counter() - t0) * 1000)
            ids = merged.item_ids()[:cand_k]
            if not ids:
                continue
            for _ in range(args.latency_repeats):
                t1 = time.perf_counter()
                _ = model.score_candidates(hist, ids)
                rank_ms.append((time.perf_counter() - t1) * 1000)
        latency[cand_k] = {
            "recall_ms_median": float(np.median(rec_ms)) if rec_ms else None,
            "rank_ms_median": float(np.median(rank_ms)) if rank_ms else None,
            "rank_ms_p95": float(np.percentile(rank_ms, 95)) if rank_ms else None,
            "samples": len(rank_ms),
        }
        LOG.info(f"latency cand_k={cand_k}: recall {latency[cand_k]['recall_ms_median']:.1f} ms "
                 f"| rank {latency[cand_k]['rank_ms_median']:.1f} ms "
                 f"(p95 {latency[cand_k]['rank_ms_p95']:.1f}, n={len(rank_ms)})")
    for row in rows:
        lat = latency.get(row["candidate_k"], {})
        row["recall_latency_ms"] = lat.get("recall_ms_median")
        row["rank_latency_ms"] = lat.get("rank_ms_median")
        row["rank_latency_p95_ms"] = lat.get("rank_ms_p95")
        row["latency_samples"] = lat.get("samples", 0)
        if row["recall_latency_ms"] is not None and row["rank_latency_ms"] is not None:
            row["latency_ms_mean"] = row["recall_latency_ms"] + row["rank_latency_ms"]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    sout = ROOT / args.source_out
    with open(sout, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(source_rows[0].keys()))
        w.writeheader()
        w.writerows(source_rows)

    print(f"\n=== candidate size vs accuracy vs latency "
          f"({args.ranker}, {users.shape[0]} users) ===")
    header = (f"{'cand_k':>8s}{'pool':>8s}{'cand_recall':>13s}"
              f"{'Recall@20':>11s}{'NDCG@20':>10s}{'lat_mean':>10s}{'lat_p95':>9s}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['candidate_k']:>8d}{r['mean_pool_size']:>8.0f}"
              f"{r['candidate_recall']:>13.4f}{r['final_Recall@20']:>11.4f}"
              f"{r['final_NDCG@20']:>10.4f}{r['latency_ms_mean']:>10.1f}"
              f"{r['latency_ms_p95']:>9.1f}")

    print("\n=== which recall channel produced the final top-20 hits? ===")
    for r in source_rows:
        if r["candidate_k"] != args.candidate_sizes[-1]:
            continue
        print(f"  {r['source']:<10s} hits={r['top20_hits']:<6d} share={r['share_of_hits']:.3f}")

    save_json({
        "processed_dir": args.processed_dir,
        "ranker": args.ranker,
        "num_users": int(users.shape[0]),
        "recall_k_per_channel": args.recall_k,
        "final_k": args.final_k,
        "tradeoff": rows,
        "source_contribution": source_rows,
        "note": ("two-stage serving simulation; NOT comparable to the "
                 "full-catalogue evaluation in results/tables/overall.csv"),
    }, ROOT / "results" / "tables" / "pipeline_tradeoff.json")
    print(f"\nwrote {out} and {sout}")


if __name__ == "__main__":
    main()
