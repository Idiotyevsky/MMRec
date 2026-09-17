#!/usr/bin/env python
"""Qualitative comparison of an ID-only model against a multimodal model.

    python scripts/case_study.py \
        --mm-run results/runs/mm_gated_xxx \
        --id-run results/runs/sasrec_xxx \
        --out results/case_study.md

For real test users it prints the history, the ground-truth next item and the
top-K of both models, and it deliberately reports **both** directions:

* ``rescued``   -- ID-only misses the target, the multimodal model retrieves it
* ``broken``    -- the multimodal model misses a target the ID-only model finds
* ``both_fail`` -- hard cases neither model solves

The 100K subset shipped on HuggingFace has no item titles, so semantics are
illustrated with nearest neighbours in the *content* embedding space rather than
with captions.  Nothing is cherry-picked away: counts for every category are
printed, and the number of examples shown per category is a CLI argument.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import EvalDataset, ProcessedData, collate_eval  # noqa: E402
from src.evaluation.evaluator import FullRankingEvaluator  # noqa: E402
from src.training.factory import build_model  # noqa: E402
from src.utils.config import Config  # noqa: E402


def _load(run_dir: Path, device: torch.device):
    cfg = Config(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
    data = ProcessedData.load(ROOT / cfg.data.processed_dir)
    model = build_model(cfg, data, device=device)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return cfg, data, model


def _encode_fn(model, cfg, device):
    name = str(cfg.model.get("name", "")).lower()
    if name in ("bpr", "popular"):
        return lambda b: model.encode(b["user_id"].to(device))
    return lambda b: model.encode(b["input_ids"].to(device))


@torch.no_grad()
def _rank(run_dir: Path, device: torch.device, max_len: int, ks=(5, 10, 20)):
    cfg, data, model = _load(run_dir, device)
    loader = DataLoader(
        EvalDataset(data, max_len, split="test"), batch_size=1024, shuffle=False, collate_fn=collate_eval
    )
    ev = FullRankingEvaluator(num_items=data.num_items, ks=ks, item_chunk_size=4096, device=device)
    res = ev.evaluate(_encode_fn(model, cfg, device), model.all_item_embeddings(), loader)
    return cfg, data, model, res


def _content_neighbours(data: ProcessedData, feature_dir: Path, internal_id: int, k: int = 5) -> list[int]:
    """Nearest items in the raw text+image feature space (semantic proxy)."""
    try:
        feats = []
        for m in ("text", "image"):
            p = feature_dir / f"{m}_feat.npy"
            lut_p = feature_dir / f"row_for_item_{m}.npy"
            if not p.exists():
                continue
            arr = np.load(p, mmap_mode="r")
            lut = np.load(lut_p) if lut_p.exists() else np.arange(arr.shape[0])
            feats.append(np.asarray(arr[lut], dtype=np.float32))
        if not feats:
            return []
        F = np.concatenate(feats, axis=1)
        F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-8)
        sims = F @ F[internal_id]
        sims[internal_id] = -np.inf
        sims[0] = -np.inf
        return [int(i) for i in np.argsort(-sims)[:k]]
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mm-run", required=True)
    ap.add_argument("--id-run", required=True)
    ap.add_argument("--out", default="results/case_study.md")
    ap.add_argument("--json-out", default="results/case_study.json")
    ap.add_argument("--per-category", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    mm_cfg, data, mm_model, mm_res = _rank(ROOT / args.mm_run, device, 50)
    id_cfg, _, _, id_res = _rank(ROOT / args.id_run, device, 50)

    assert np.array_equal(mm_res.user_ids, id_res.user_ids), "runs evaluated different users"
    assert np.array_equal(mm_res.targets, id_res.targets), "runs evaluated different targets"

    feature_dir = ROOT / mm_cfg.data.get("feature_dir", "data/raw")
    rank_mm, rank_id = mm_res.rank, id_res.rank
    K = 20
    rescued = np.flatnonzero((rank_id > K) & (rank_mm <= K))
    broken = np.flatnonzero((rank_id <= K) & (rank_mm > K))
    both_fail = np.flatnonzero((rank_id > K) & (rank_mm > K))
    both_ok = np.flatnonzero((rank_id <= K) & (rank_mm <= K))

    summary = {
        "num_users": int(mm_res.user_ids.shape[0]),
        "both_ok": int(both_ok.size),
        "rescued_by_mm": int(rescued.size),
        "broken_by_mm": int(broken.size),
        "both_fail": int(both_fail.size),
        "recall20_mm": float((rank_mm <= K).mean()),
        "recall20_id": float((rank_id <= K).mean()),
        "mm_run": args.mm_run,
        "id_run": args.id_run,
    }
    print(json.dumps(summary, indent=2))

    def describe(idx: int) -> dict:
        u = int(mm_res.user_ids[idx])
        tgt = int(mm_res.targets[idx])
        hist = [data.raw_item(int(i)) for i in data.test_history(u)]
        return {
            "user_id": u,
            "raw_history": hist,
            "raw_target": data.raw_item(tgt),
            "target_internal": tgt,
            "rank_mm": int(rank_mm[idx]),
            "rank_id": int(rank_id[idx]),
            "topk_mm": [data.raw_item(int(i)) for i in mm_res.topk[idx][: args.top_k] if i > 0],
            "topk_id": [data.raw_item(int(i)) for i in id_res.topk[idx][: args.top_k] if i > 0],
            "content_neighbours_of_target": [
                data.raw_item(i) for i in _content_neighbours(data, feature_dir, tgt)
            ],
            "target_popularity_bucket": int(data.popularity_bucket[tgt]),
            "target_train_freq": int(data.train_freq[tgt]),
            "target_is_cold": bool(data.is_cold[tgt]),
        }

    cases = {
        "rescued": [describe(int(i)) for i in rescued[: args.per_category]],
        "broken": [describe(int(i)) for i in broken[: args.per_category]],
        "both_fail": [describe(int(i)) for i in both_fail[: args.per_category]],
    }

    bucket_names = {0: "tail", 1: "middle", 2: "head"}
    lines = [
        "# Case study — ID-only SASRec vs multimodal MM-SASRec",
        "",
        f"* ID-only run: `{args.id_run}`",
        f"* Multimodal run: `{args.mm_run}`",
        f"* Test users: {summary['num_users']:,}",
        f"* Recall@20 — ID-only **{summary['recall20_id']:.4f}**, "
        f"multimodal **{summary['recall20_mm']:.4f}**",
        "",
        "| category | users | share |",
        "|---|---|---|",
        f"| both retrieve | {summary['both_ok']:,} | {summary['both_ok']/summary['num_users']:.1%} |",
        f"| **rescued by multimodal** | {summary['rescued_by_mm']:,} | {summary['rescued_by_mm']/summary['num_users']:.2%} |",
        f"| **broken by multimodal** | {summary['broken_by_mm']:,} | {summary['broken_by_mm']/summary['num_users']:.2%} |",
        f"| both fail | {summary['both_fail']:,} | {summary['both_fail']/summary['num_users']:.1%} |",
        "",
        "Item ids are the original MicroLens ids. The 100K subset has no titles, so",
        "`content nbrs` lists the target's nearest neighbours in the raw text+image",
        "feature space — a semantic proxy for what the item is about.",
        "",
    ]
    for title, key in (("Rescued by multimodal", "rescued"),
                       ("Broken by multimodal (counter-examples)", "broken"),
                       ("Hard cases neither model solves", "both_fail")):
        lines += [f"## {title}", ""]
        if not cases[key]:
            lines += ["_none in the first users scanned_", ""]
            continue
        for c in cases[key]:
            lines += [
                f"### user {c['user_id']}",
                "",
                f"* history (raw ids, oldest first): `{c['raw_history']}`",
                f"* ground truth: **{c['raw_target']}** "
                f"(popularity bucket `{bucket_names.get(c['target_popularity_bucket'], '?')}`, "
                f"train freq {c['target_train_freq']}, cold={c['target_is_cold']})",
                f"* rank: ID-only `{c['rank_id']}` | multimodal `{c['rank_mm']}`",
                f"* ID-only top-{args.top_k}: `{c['topk_id']}`",
                f"* multimodal top-{args.top_k}: `{c['topk_mm']}`",
                f"* content nbrs of target: `{c['content_neighbours_of_target']}`",
                "",
            ]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    (ROOT / args.json_out).write_text(json.dumps({"summary": summary, "cases": cases}, indent=2), encoding="utf-8")
    print(f"wrote {out} and {ROOT / args.json_out}")


if __name__ == "__main__":
    main()
