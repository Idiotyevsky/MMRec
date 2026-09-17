#!/usr/bin/env python
"""Train an RQVAE to turn item content embeddings into Semantic IDs.

    # content-only semantic IDs from the raw text+image features
    python scripts/train_rqvae.py --source text_image --out-tag content

    # semantic IDs from a trained ID-free multimodal item encoder
    python scripts/train_rqvae.py --source run --run-dir results/runs/mm_text_image_xxx

Reports the quality metrics that actually decide whether the codes are usable:
reconstruction error, per-level codebook utilisation, unique Semantic IDs and
the collision rate.  Writes ``artifacts/semantic_ids_<tag>.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import ProcessedData  # noqa: E402
from src.models.rqvae import RQVAE  # noqa: E402
from src.models.semantic_id import SemanticIDMapper  # noqa: E402
from src.training.factory import build_model  # noqa: E402
from src.utils.config import Config  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def load_item_embeddings(args, data: ProcessedData) -> tuple[np.ndarray, str]:
    if args.source == "text_image":
        feats = []
        for m in ("text", "image"):
            arr = np.load(ROOT / args.feature_dir / f"{m}_feat.npy", mmap_mode="r")
            lut_path = ROOT / args.processed_dir / f"row_for_item_{m}.npy"
            if not lut_path.exists():
                lut_path = ROOT / args.feature_dir / f"row_for_item_{m}.npy"
            lut = np.load(lut_path)
            f = np.zeros((data.num_items + 1, arr.shape[1]), dtype=np.float32)
            f[1:] = np.asarray(arr[lut[1:]], dtype=np.float32)
            # put every modality on the same footing: raw text is unit-norm while
            # raw image has norm ~26, so an un-normalised concatenation lets the
            # image block dominate the reconstruction loss and the codebooks
            norms = np.linalg.norm(f, axis=1, keepdims=True)
            f = f / np.maximum(norms, 1e-8)
            f[0] = 0.0
            feats.append(f)
        return np.concatenate(feats, axis=1), "text+image(L2-normalised per modality)"

    if args.source == "run":
        if not args.run_dir:
            raise SystemExit("--source run requires --run-dir")
        run_dir = ROOT / args.run_dir
        cfg = Config(yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")))
        p = run_dir / "item_embeddings.npy"
        if p.exists():
            return np.load(p), f"fused({run_dir.name})"
        model = build_model(cfg, data, device=torch.device("cpu"))
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        with torch.no_grad():
            emb = model.all_item_embeddings().numpy()
        return emb, f"fused({run_dir.name})"

    raise SystemExit(f"unknown --source {args.source!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="text_image", choices=["text_image", "run"])
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--processed-dir", default="data/processed/base")
    ap.add_argument("--feature-dir", default="data/raw")
    ap.add_argument("--out-tag", default="content")
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--num-codebooks", type=int, default=3)
    ap.add_argument("--codebook-size", type=int, default=256)
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--commitment", type=float, default=0.25)
    ap.add_argument("--reseed-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    data = ProcessedData.load(ROOT / args.processed_dir)

    emb, source_name = load_item_embeddings(args, data)
    x = torch.from_numpy(emb[1:]).to(device)  # items only, no PAD
    print(f"source: {source_name} | {x.shape[0]} items x {x.shape[1]} dims")

    model = RQVAE(
        input_dim=x.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        commitment=args.commitment,
    ).to(device)
    # seed codebooks from real encoder outputs -- random init leaves most codes
    # far from any residual, which is the usual cause of dead codes + collisions
    with torch.no_grad():
        z0 = model.encoder(x[: min(16384, x.shape[0])])
    model.quantizer.init_from_data(z0)
    print(f"codebooks seeded from data | util after init "
          f"{[round(u,3) for u in model.quantizer.measure_utilization(z0)]}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    print(f"RQVAE params: {model.num_parameters():,}")

    N = x.shape[0]
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = rng.permutation(N)
        totals = {"loss": 0.0, "recon": 0.0, "commit": 0.0, "cb": 0.0, "n": 0}
        model.train()
        for i in range(0, N, args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb = x[idx]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            totals["loss"] += float(out["loss"]) * xb.shape[0]
            totals["recon"] += float(out["reconstruction"]) * xb.shape[0]
            totals["commit"] += float(out["commitment"]) * xb.shape[0]
            totals["cb"] += float(out["codebook"]) * xb.shape[0]
            totals["n"] += xb.shape[0]
        sched.step()
        if (epoch + 1) % args.reseed_every == 0:
            with torch.no_grad():
                z_pool = model.encoder(x[rng.choice(N, size=min(N, 8192), replace=False)])
            n_new = model.quantizer.reseed_dead_codes(z_pool, threshold=1.0)
            util = model.quantizer.measure_utilization(z_pool)
            print(f"  epoch {epoch+1}: re-seeded {n_new} codes | "
                  f"util {[round(u,3) for u in util]}")
        if (epoch + 1) % max(args.epochs // 10, 1) == 0 or epoch == 0:
            n = max(totals["n"], 1)
            print(f"epoch {epoch+1:4d} | loss {totals['loss']/n:.5f} "
                  f"| recon {totals['recon']/n:.5f} | commit {totals['commit']/n:.5f} "
                  f"| cb {totals.get('cb', 0.0)/n:.5f}")

    train_time = time.time() - t0

    # ---- quality metrics -------------------------------------------------
    model.eval()
    with torch.no_grad():
        codes = model.encode_codes(x).cpu().numpy()
        x_hat = model.reconstruct(x)
        mse = float(torch.nn.functional.mse_loss(x_hat, x))
        cos = float(
            torch.nn.functional.cosine_similarity(x_hat, x, dim=-1).mean()
        )
        util = model.utilization(x)

    mapper = SemanticIDMapper(codes, num_items=data.num_items, codebook_size=args.codebook_size)
    stats = mapper.statistics()
    report = {
        "source": source_name,
        "input_dim": int(x.shape[1]),
        "latent_dim": args.latent_dim,
        "num_codebooks": args.num_codebooks,
        "codebook_size": args.codebook_size,
        "params": int(model.num_parameters()),
        "train_time_s": round(train_time, 1),
        "epochs": args.epochs,
        "reconstruction_mse": mse,
        "reconstruction_cosine": cos,
        "codebook_utilization": util,
        "mean_codebook_utilization": float(np.mean(util)),
        **stats,
    }
    print("\n=== Semantic ID quality ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"semantic_ids_{args.out_tag}.npz",
        codes=codes.astype(np.int16),
        codebook_size=np.int64(args.codebook_size),
        num_items=np.int64(data.num_items),
        source=np.array([source_name]),
    )
    torch.save(
        {"model_state": model.state_dict(), "config": vars(args), "report": report},
        out_dir / f"rqvae_{args.out_tag}.pt",
    )
    save_json(report, out_dir / f"semantic_id_report_{args.out_tag}.json")
    print(f"\nwrote {out_dir}/semantic_ids_{args.out_tag}.npz and rqvae_{args.out_tag}.pt")


if __name__ == "__main__":
    main()
