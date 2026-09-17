#!/usr/bin/env python
"""Small RQVAE configuration sweep driven by the Semantic-ID quality metrics.

    python scripts/sweep_rqvae.py --source text_image --epochs 200

Trains a handful of configurations and ranks them by collision rate, then by
reconstruction cosine.  This is a search over the *quantiser*, not over the
recommender, so it cannot leak into the main results.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = [
    # latent_dim, num_codebooks, codebook_size, commitment, lr
    (32, 3, 256, 0.05, 1e-3),
    (64, 3, 256, 0.05, 1e-3),
    (128, 3, 256, 0.05, 1e-3),
    (64, 4, 256, 0.05, 1e-3),
    (64, 3, 512, 0.05, 1e-3),
    (64, 3, 256, 0.25, 1e-3),
    (128, 4, 512, 0.05, 5e-4),
    (256, 3, 512, 0.05, 5e-4),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="text_image")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-configs", type=int, default=len(CONFIGS))
    args = ap.parse_args()

    results = []
    for i, (latent, levels, size, commit, lr) in enumerate(CONFIGS[: args.max_configs]):
        tag = f"sweep_l{latent}_k{levels}_c{size}_b{str(commit).replace('.','')}"
        print(f"\n=== [{i+1}] latent={latent} levels={levels} codebook={size} "
              f"commit={commit} lr={lr} ===", flush=True)
        cmd = [
            sys.executable, "scripts/train_rqvae.py",
            "--source", args.source, "--out-tag", tag,
            "--latent-dim", str(latent), "--num-codebooks", str(levels),
            "--codebook-size", str(size), "--commitment", str(commit),
            "--lr", str(lr), "--epochs", str(args.epochs),
            "--reseed-every", "25", "--device", args.device,
        ]
        rc = subprocess.call(cmd, cwd=str(ROOT))
        report_path = ROOT / "artifacts" / f"semantic_id_report_{tag}.json"
        if rc != 0 or not report_path.exists():
            print(f"  FAILED rc={rc}", flush=True)
            continue
        rep = json.loads(report_path.read_text(encoding="utf-8"))
        results.append({"tag": tag, "config": {
            "latent_dim": latent, "num_codebooks": levels, "codebook_size": size,
            "commitment": commit, "lr": lr}, **rep})
        print(f"  collision={rep['collision_rate']:.4f} "
              f"unique={rep['unique_semantic_ids']} "
              f"cos={rep['reconstruction_cosine']:.4f} "
              f"util={[round(u,3) for u in rep['codebook_utilization']]}", flush=True)

    results.sort(key=lambda r: (r["collision_rate"], -r["reconstruction_cosine"]))
    out = ROOT / "artifacts" / f"rqvae_sweep_{args.source}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== ranking (best first) ===")
    for r in results:
        c = r["config"]
        print(f"  collision={r['collision_rate']:.4f} cos={r['reconstruction_cosine']:.4f} "
              f"unique={r['unique_semantic_ids']:6d}  latent={c['latent_dim']} "
              f"levels={c['num_codebooks']} codebook={c['codebook_size']} "
              f"commit={c['commitment']} lr={c['lr']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
