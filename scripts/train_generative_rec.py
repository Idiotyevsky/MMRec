#!/usr/bin/env python
"""Train and evaluate a Semantic-ID generative recommender.

    python scripts/train_rqvae.py --source text_image --out-tag content
    python scripts/train_generative_rec.py --semantic-ids artifacts/semantic_ids_content.npz

Evaluation is *generative retrieval*: the model decodes a Semantic ID under a
prefix constraint and the resulting item(s) are the recommendation.  This is a
different candidate mechanism from vector search, so the numbers are reported in
their own table and are never mixed into the discriminative full-ranking table
without saying so.

Also reported: model size, decoding latency, and the share of decoded SIDs that
map to a real item (should be 100 % because of the prefix constraint).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.models.generative_rec import GenerativeRecommender  # noqa: E402
from src.models.semantic_id import BOS, PAD, SemanticIDMapper  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


class TokenSeqDataset(Dataset):
    """One autoregressive token sequence per user."""

    def __init__(self, data: ProcessedData, mapper: SemanticIDMapper, max_items: int, split: str):
        self.data = data
        self.mapper = mapper
        self.max_items = max_items
        self.split = split
        self.users = np.arange(data.num_users, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.users.shape[0])

    def _items(self, u: int) -> np.ndarray:
        if self.split == "train":
            return self.data.train_items(u)
        if self.split == "val":
            return self.data.val_history(u)
        return self.data.test_history(u)

    def __getitem__(self, idx: int) -> dict:
        u = int(self.users[idx])
        items = self._items(u)[-self.max_items :]
        tokens = [BOS]
        for it in items:
            tokens.extend(self.mapper.item_tokens(int(it)))
        seq = np.asarray(tokens, dtype=np.int64)
        return {
            "input_ids": seq[:-1],
            "target": seq[1:],
            "user_id": u,
            "length": seq.shape[0] - 1,
        }


def collate(batch: list[dict], max_len: int) -> dict:
    L = min(max_len, max(b["length"] for b in batch))
    B = len(batch)
    inp = np.zeros((B, L), dtype=np.int64)
    tgt = np.zeros((B, L), dtype=np.int64)
    for i, b in enumerate(batch):
        n = min(b["length"], L)
        inp[i, L - n :] = b["input_ids"][-n:]
        tgt[i, L - n :] = b["target"][-n:]
    return {
        "input_ids": torch.from_numpy(inp),
        "target": torch.from_numpy(tgt),
        "user_id": torch.tensor([b["user_id"] for b in batch], dtype=torch.long),
    }


@torch.no_grad()
def evaluate_generative(
    model: GenerativeRecommender,
    mapper: SemanticIDMapper,
    data: ProcessedData,
    split: str = "test",
    ks=(5, 10, 20),
    beam_width: int = 20,
    batch_size: int = 64,
    max_items: int = 50,
    device: torch.device = torch.device("cpu"),
    max_users: int | None = None,
) -> dict:
    model.eval()
    users = np.arange(data.num_users, dtype=np.int64)
    if max_users is not None:
        users = users[:max_users]

    histories = []
    for u in users:
        items = (data.val_history(u) if split == "val" else data.test_history(u))[-max_items:]
        toks: list[int] = []
        for it in items:
            toks.extend(mapper.item_tokens(int(it)))
        histories.append(toks)

    t0 = time.perf_counter()
    decoded = model.generate(histories, mapper, beam_width=beam_width, batch_size=batch_size)
    latency = (time.perf_counter() - t0) / max(len(users), 1) * 1000

    targets = data.val_target[users] if split == "val" else data.test_target[users]
    max_k = max(ks)
    ranks = np.full(len(users), np.iinfo(np.int32).max, dtype=np.int64)
    n_collision_expansions = 0

    for i, cands in enumerate(decoded):
        seen: list[int] = []
        for sid, _lp in cands:
            items = mapper.sid_to_items.get(sid, [])
            if len(items) > 1:
                n_collision_expansions += 1
            for it in items:
                if it not in seen:
                    seen.append(it)
            if len(seen) >= max_k:
                break
        tgt = int(targets[i])
        for pos, it in enumerate(seen[:max_k], start=1):
            if it == tgt:
                ranks[i] = pos
                break

    out: dict = {"num_users": int(len(users)), "beam_width": beam_width,
                 "latency_ms_per_user": round(latency, 3),
                 "collision_expansions": int(n_collision_expansions)}
    for k in ks:
        out[f"Recall@{k}"] = float((ranks <= k).mean())
        r = ranks.astype(np.float64)
        out[f"NDCG@{k}"] = float(np.where(r <= k, 1.0 / np.log2(r + 1.0), 0.0).mean())
    out["decoded_sids"] = len(decoded)
    out["empty_decodings"] = int(sum(1 for d in decoded if not d))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semantic-ids", default="artifacts/semantic_ids_content.npz")
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max-items", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--beam-width", type=int, default=20)
    ap.add_argument("--eval-users", type=int, default=None)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    data = ProcessedData.load(ROOT / args.processed_dir)

    npz = np.load(ROOT / args.semantic_ids)
    codes = npz["codes"].astype(np.int64)
    codebook_size = int(npz["codebook_size"])
    num_items = int(npz["num_items"])
    if num_items != data.num_items:
        raise SystemExit(f"semantic IDs built for {num_items} items but dataset has {data.num_items}")
    mapper = SemanticIDMapper(codes, num_items=num_items, codebook_size=codebook_size)
    print(f"semantic IDs: {mapper.statistics()}")

    max_seq_len = 1 + args.max_items * mapper.num_levels
    model = GenerativeRecommender(
        vocab_size=mapper.vocab_size,
        num_levels=mapper.num_levels,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_seq_len=max_seq_len,
    ).to(device)
    print(f"generative model params: {model.num_parameters():,} | max_seq_len={max_seq_len}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    from functools import partial

    train_ds = TokenSeqDataset(data, mapper, args.max_items, "train")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=partial(collate, max_len=max_seq_len),
    )

    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "results" / "runs" / f"genrec_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best = -1.0
    best_epoch = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        tot, n = 0.0, 0
        t0 = time.time()
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            tgt = batch["target"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss = model.loss(ids, tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * ids.shape[0]
            n += ids.shape[0]

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val = evaluate_generative(
                model, mapper, data, split="val", beam_width=args.beam_width,
                batch_size=64, max_items=args.max_items, device=device,
                max_users=min(5000, data.num_users),
            )
            metric = val["Recall@20"]
            history.append({"epoch": epoch, "train_loss": tot / max(n, 1), **val})
            print(f"epoch {epoch:3d} | loss {tot/max(n,1):.4f} | val Recall@20 {metric:.4f} "
                  f"NDCG@20 {val['NDCG@20']:.4f} | {time.time()-t0:.1f}s")
            if metric > best:
                best, best_epoch = metric, epoch
                torch.save({"model_state": model.state_dict(), "args": vars(args),
                            "mapper_stats": mapper.statistics()}, run_dir / "best.pt")
            elif epoch - best_epoch >= args.patience:
                print("early stopping")
                break

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    results = {"run_dir": str(run_dir), "params": model.num_parameters(),
               "semantic_id_stats": mapper.statistics(), "history": history}
    for split in ("val", "test"):
        res = evaluate_generative(
            model, mapper, data, split=split, beam_width=args.beam_width,
            batch_size=64, max_items=args.max_items, device=device,
            max_users=args.eval_users,
        )
        results[split] = res
        print(f"{split}: " + " ".join(f"{k}={v:.4f}" for k, v in res.items()
                                      if isinstance(v, float) and ("Recall" in k or "NDCG" in k)))
        print(f"  latency {res['latency_ms_per_user']} ms/user | "
              f"empty decodings {res['empty_decodings']}")

    # cold / tail slices
    if data.is_cold.any():
        users = np.arange(data.num_users)
        is_cold = data.is_cold[data.test_target[users]]
        results["cold_targets"] = int(is_cold.sum())

    save_json(results, run_dir / "metrics.json")
    print(f"wrote {run_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
