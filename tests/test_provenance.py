"""The provenance record: what a run must be able to prove about itself."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.dataset import ProcessedData
from src.utils import provenance


def test_config_hash_is_order_insensitive_and_content_sensitive():
    a = {"model": {"hidden_size": 128}, "training": {"seed": 42}}
    b = {"training": {"seed": 42}, "model": {"hidden_size": 128}}
    assert provenance.config_hash(a) == provenance.config_hash(b)
    assert provenance.config_hash(a) != provenance.config_hash(
        {"model": {"hidden_size": 256}, "training": {"seed": 42}}
    )


def test_dataset_fingerprint_changes_with_the_split(synthetic_dir, tmp_path):
    import shutil

    data = ProcessedData.load(synthetic_dir)
    before = provenance.dataset_fingerprint(data)

    other = tmp_path / "copy"
    shutil.copytree(synthetic_dir, other)
    path = other / "dataset.npz"
    payload = dict(np.load(path))
    payload["test_target"] = payload["test_target"].copy()
    payload["test_target"][0] = payload["test_target"][1]
    np.savez(path, **payload)

    assert provenance.dataset_fingerprint(ProcessedData.load(other)) != before


def test_dirty_flag_ignores_generated_artifacts():
    """A queue that writes manifests or tables must not dirty the next run."""
    scratch = provenance.ROOT / "results" / "logs" / "__provenance_probe__"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    before = provenance.git_state()
    scratch.write_text("x", encoding="utf-8")
    try:
        after = provenance.git_state()
    finally:
        scratch.unlink()
    assert after["git_dirty"] == before["git_dirty"]


def test_dirty_flag_catches_a_modified_source_file():
    probe = provenance.ROOT / "scripts" / "__provenance_probe__.py"
    probe.write_text("# x\n", encoding="utf-8")
    try:
        state = provenance.git_state()
    finally:
        probe.unlink()
    assert state["git_dirty"] is True
    assert any("__provenance_probe__" in f for f in state["git_dirty_files"])


class _Model:
    def num_parameters(self) -> int:
        return 1234


def test_manifest_carries_the_required_provenance(synthetic_data):
    cfg = {"model": {"name": "sasrec", "modalities": {"id": True}},
           "data": {"processed_dir": "data/processed/base"},
           "training": {"seed": 3407}}
    manifest = provenance.build_manifest(
        cfg=cfg, data=synthetic_data, model=_Model(), seed=3407,
        run_id="sasrec_20260101-000000_aaaaaa", tag="sasrec", best_epoch=7,
        metrics={"test": {"Recall@20": 0.1}},
    )
    for key in ("git_sha", "git_dirty", "dataset_hash", "config_hash",
                "seed", "num_parameters"):
        assert key in manifest, f"manifest must record {key}"
    assert manifest["seed"] == 3407
    assert manifest["num_parameters"] == 1234
    assert manifest["dataset_hash"] == provenance.dataset_fingerprint(synthetic_data)
    assert manifest["config_hash"] == provenance.config_hash(cfg)
    assert manifest["test"] == {"Recall@20": 0.1}


def test_manifest_round_trips_through_the_run_directory(tmp_path):
    manifest = {"run_id": "x_20260101-000000_aaaaaa", "git_sha": "a" * 40}
    provenance.save_manifest(manifest, tmp_path, tmp_path / "manifests")
    assert provenance.load_manifest(tmp_path) == manifest
    assert (tmp_path / "manifests" / "x_20260101-000000_aaaaaa.json").exists()


def test_load_manifest_is_none_when_absent(tmp_path):
    assert provenance.load_manifest(tmp_path) is None
