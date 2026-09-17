#!/usr/bin/env python
"""Fast end-to-end sanity check on synthetic data.

    python scripts/smoke_test.py

Builds a tiny dataset, trains every model family for a few epochs, evaluates with
full ranking and checks the ordering constraints that must hold:

    random < Popular <= BPR-MF < SASRec

If this script fails, the implementation is broken; no amount of real data will
fix it.  It does **not** claim anything about MicroLens results.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.data.synthetic import make_synthetic_dataset  # noqa: E402
from src.evaluation.evaluator import FullRankingEvaluator  # noqa: E402
from src.training.factory import build_model  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def make_cfg(name: str, d: Path, **model_kw) -> Config:
    model = {"name": name, "hidden_size": 64, "num_layers": 2, "num_heads": 4,
             "dropout": 0.1, "max_seq_len": 20, "zero_cold_id": True}
    model.update(model_kw)
    return Config({
        "model": model,
        "data": {"processed_dir": str(d), "feature_dir": str(d), "max_sequence_length": 20},
        "loss": {"num_negatives": 64, "temperature": 1.0},
        "negative_sampling": {"mode": "uniform"},
        "training": {"batch_size": 128, "learning_rate": 0.005, "weight_decay": 0.0,
                     "max_grad_norm": 5.0, "amp": False, "epochs": 30,
                     "early_stopping_patience": 30, "scheduler": "none",
                     "monitor": "NDCG@10", "monitor_mode": "max", "seed": 0, "num_workers": 0},
        "evaluation": {"full_ranking": True, "item_chunk_size": 1024, "ks": [5, 10, 20]},
    })


def run_model(name: str, d: Path, data: ProcessedData, device, run_dir: Path, **kw) -> dict:
    cfg = make_cfg(name, d, **kw)
    model = build_model(cfg, data, device=device)
    trainer = Trainer(model, cfg, data, run_dir, device)
    trainer.fit()
    with torch.no_grad():
        item_emb = model.all_item_embeddings()
    ev = FullRankingEvaluator(num_items=data.num_items, ks=(5, 10, 20), item_chunk_size=1024)
    res = ev.evaluate(trainer._encode_fn(), item_emb, trainer._eval_loader("test"))
    m = res.metrics()
    m["train_loss_last"] = trainer.history[-1].get("train_loss")
    m["train_loss_first"] = trainer.history[0].get("train_loss")

    # Which objective did training actually optimise?  A model trained with the
    # position-wise SASRec objective must score better under it than under the
    # "broadcast the last hidden state" objective, and vice versa for the bug.
    if name not in ("popular", "bpr"):
        from src.models.loss import sampled_softmax_loss

        model.eval()
        pos_l, bcast_l = [], []
        with torch.no_grad():
            for batch in trainer._train_loader():
                ids = batch["input_ids"].to(device)
                tgt = batch["target"].to(device)
                neg, _ = model.item_embeddings(
                    torch.randint(1, data.num_items + 1, (ids.shape[0], 32), device=device)
                )
                pe, _ = model.item_embeddings(tgt)
                seq = model.encode_sequence(ids)
                pos_l.append(sampled_softmax_loss(seq, pe, neg, tgt).item())
                bcast_l.append(
                    sampled_softmax_loss(seq[:, -1:].expand_as(seq), pe, neg, tgt).item()
                )
        m["poswise_loss"] = float(np.mean(pos_l))
        m["broadcast_loss"] = float(np.mean(bcast_l))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="artifacts/smoke_test_results.json")
    args = ap.parse_args()

    set_seed(0)
    device = torch.device(args.device)
    tmp = Path(tempfile.mkdtemp(prefix="mmrec_smoke_"))
    # sequential structure is required for "SASRec > Popular" to be a meaningful
    # check: on i.i.d. data the popularity marginal *is* the Bayes-optimal model
    d = make_synthetic_dataset(tmp / "data", num_users=300, num_items=120,
                               min_len=5, max_len=15, cold_ratio=0.0,
                               structure="sequential", seed=0)
    data = ProcessedData.load(d)
    print(f"synthetic dataset: {data.summary()}")

    results: dict[str, dict] = {}
    plan = [
        ("popular", {}),
        ("bpr", {}),
        ("sasrec", {}),
        ("mm_sasrec", {"fusion": "concat",
                       "modalities": {"id": True, "text": True, "image": True}}),
        ("mm_sasrec", {"fusion": "gated",
                       "modalities": {"id": True, "text": True, "image": True}}),
        ("mm_sasrec", {"fusion": "gated",
                       "modalities": {"id": True, "text": True, "image": True},
                       "id_dropout_prob": 0.2}),
    ]
    for name, kw in plan:
        key = name if not kw else f"{name}_{kw.get('fusion', '')}_{'iddrop' if kw.get('id_dropout_prob') else ''}".strip("_")
        print(f"\n>>> {key}")
        m = run_model(name, d, data, device, tmp / key, **kw)
        results[key] = m
        loss_txt = (
            "loss n/a (parameter-free)"
            if m["train_loss_first"] is None
            else f"loss {m['train_loss_first']:.3f} -> {m['train_loss_last']:.3f}"
        )
        print(f"    Recall@20={m['Recall@20']:.4f}  NDCG@20={m['NDCG@20']:.4f}  {loss_txt}")

    # ---- ordering constraints ------------------------------------------
    checks: list[tuple[str, bool, str]] = []
    r_pop = results["popular"]["Recall@20"]
    r_sas = results["sasrec"]["Recall@20"]
    r_bpr = results["bpr"]["Recall@20"]
    chance = 20 / data.num_items

    checks.append(("popular > chance", r_pop > chance,
                   f"{r_pop:.4f} > {chance:.4f}"))
    checks.append(("bpr >= popular", r_bpr >= r_pop * 0.9,
                   f"{r_bpr:.4f} vs {r_pop:.4f}"))
    checks.append(("sasrec > popular", r_sas > r_pop,
                   f"{r_sas:.4f} > {r_pop:.4f}"))
    checks.append(("sasrec > bpr", r_sas > r_bpr,
                   f"{r_sas:.4f} > {r_bpr:.4f}"))
    for key in ("mm_sasrec_concat", "mm_sasrec_gated", "mm_sasrec_gated_iddrop"):
        if key in results:
            checks.append((f"{key} > popular", results[key]["Recall@20"] > r_pop,
                           f"{results[key]['Recall@20']:.4f} > {r_pop:.4f}"))
            checks.append((f"{key} loss decreased",
                           results[key]["train_loss_last"] < results[key]["train_loss_first"],
                           f"{results[key]['train_loss_first']:.3f} -> "
                           f"{results[key]['train_loss_last']:.3f}"))

    # the objective the model was trained on must be the position-wise one
    for key, m in results.items():
        if "poswise_loss" in m:
            checks.append((f"{key} position-wise objective",
                           m["poswise_loss"] < m["broadcast_loss"],
                           f"{m['poswise_loss']:.4f} < {m['broadcast_loss']:.4f}"))

    print("\n=== sanity checks ===")
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:<34s} {detail}")
        ok &= passed

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results,
                               "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks]},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    if not ok:
        print("\nSMOKE TEST FAILED -- the implementation is not sane yet")
        raise SystemExit(1)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
