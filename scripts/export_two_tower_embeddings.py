#!/usr/bin/env python
"""Export deterministic, dataset-verified Two-Tower item vectors."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch, yaml
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.data.dataset import ProcessedData
from src.training.factory import build_model
from src.utils.config import Config
from src.utils.provenance import dataset_fingerprint

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--run-dir', required=True); ap.add_argument('--checkpoint', default='best.pt'); ap.add_argument('--device', default='cpu'); args = ap.parse_args()
    run = Path(args.run_dir); cfg = Config(yaml.safe_load((run / 'config.yaml').read_text())); data = ProcessedData.load(ROOT / cfg.data.processed_dir)
    model = build_model(cfg, data, args.device); ck = torch.load(run / args.checkpoint, map_location=args.device, weights_only=False)
    if ck.get('dataset_hash') != dataset_fingerprint(data): raise AssertionError('checkpoint/dataset mismatch')
    model.load_state_dict(ck['model_state']); model.eval()
    with torch.no_grad(): table = model.all_item_embeddings().cpu().numpy().astype(np.float32)
    if table.shape != (data.num_items + 1, int(cfg.model.hidden_size)): raise AssertionError('unexpected item embedding shape')
    if not np.allclose(np.linalg.norm(table[1:], axis=1), 1, atol=2e-5): raise AssertionError('item embeddings must be L2 normalized')
    np.save(run / 'two_tower_item_embeddings.npy', table)
    (run / 'two_tower_item_embeddings.meta.json').write_text(json.dumps({
        'dim': int(table.shape[1]), 'normalized': True, 'num_items': data.num_items,
        'dataset_hash': dataset_fingerprint(data), 'checkpoint_hash': hashlib.sha256((run / args.checkpoint).read_bytes()).hexdigest(),
        'modalities': ['text', 'image']}, indent=2))
    print(f'wrote {run / "two_tower_item_embeddings.npy"} {table.shape}')
if __name__ == '__main__': main()
