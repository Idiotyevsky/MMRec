"""Provenance for every run: git state, dataset and config fingerprints.

A reported number is only meaningful together with the code and data that
produced it.  Every run writes a ``run_manifest.json`` built here, and the
aggregator refuses to average runs whose fingerprints disagree.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_state() -> dict:
    """``git_sha`` plus whether the working tree was dirty when the run started."""
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return {"git_sha": None, "git_dirty": None, "git_branch": None, "git_available": False}
    status = _git("status", "--porcelain")
    return {
        "git_sha": sha,
        "git_dirty": bool(status),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_available": True,
    }


def config_hash(cfg: dict) -> str:
    """Stable hash of a config dict (key order does not matter)."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def dataset_fingerprint(data) -> str:
    """Hash of the parts of a processed dataset that change the task.

    Covers the split, the item mapping and the cold mask -- everything a model
    could otherwise be silently trained and evaluated on inconsistently.
    """
    h = hashlib.md5()
    h.update(np.ascontiguousarray(np.asarray(data.flat_items, dtype=np.int64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(data.user_offsets, dtype=np.int64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(data.train_len, dtype=np.int64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(data.val_target, dtype=np.int64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(data.test_target, dtype=np.int64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(data.train_freq, dtype=np.int64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(data.is_cold, dtype=np.int8)).tobytes())
    return h.hexdigest()


def build_manifest(
    *,
    cfg,
    data,
    model,
    seed: int,
    run_id: str,
    tag: str | None = None,
    best_epoch: int | None = None,
    best_metric: float | None = None,
    train_time_s: float | None = None,
    metrics: dict | None = None,
) -> dict:
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    model_cfg = cfg_dict.get("model", {}) or {}
    manifest = {
        "run_id": run_id,
        "tag": tag or run_id.rsplit("_", 2)[0],
        "model": model_cfg.get("name"),
        "fusion": model_cfg.get("fusion"),
        "modalities": model_cfg.get("modalities"),
        "seed": int(seed),
        "num_parameters": int(model.num_parameters()) if model is not None else None,
        "dataset_dir": str(getattr(data, "dir", "")),
        "dataset_hash": dataset_fingerprint(data) if data is not None else None,
        "num_users": int(data.num_users) if data is not None else None,
        "num_items": int(data.num_items) if data is not None else None,
        "config_hash": config_hash(cfg_dict),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "train_time_s": train_time_s,
    }
    manifest.update(git_state())
    if metrics:
        manifest["test"] = metrics.get("test")
        manifest["val"] = metrics.get("val")
        manifest["cold"] = metrics.get("cold")
    return manifest


def save_manifest(manifest: dict, run_dir: Path, manifests_dir: Path | None = None) -> None:
    """Write the manifest into the run dir and (optionally) a committed copy."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    if manifests_dir is not None:
        manifests_dir = Path(manifests_dir)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        (manifests_dir / f"{manifest['run_id']}.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )


def load_manifest(run_dir: Path) -> dict | None:
    p = Path(run_dir) / "run_manifest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
