"""YAML configs must actually drive the scripts they document.

A config file that nothing reads is documentation that can silently drift out of
sync with the code, so these tests pin the wiring: the shipped configs parse, the
dotted mapping finds the right values, and explicit CLI flags still win.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.utils.config import add_config_flag, config_defaults, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["rqvae.yaml", "generative_rec.yaml"])
def test_shipped_configs_load(name):
    cfg = load_config(ROOT / "configs" / name)
    assert cfg.model.name
    assert cfg.training.epochs > 0
    assert cfg.training.seed == 42


def test_config_defaults_extracts_dotted_keys(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "model:\n  latent_size: 64\n  num_levels: 4\ntraining:\n  epochs: 300\n",
        encoding="utf-8",
    )
    out = config_defaults(cfg, {"latent_dim": "model.latent_size",
                                "num_codebooks": "model.num_levels",
                                "epochs": "training.epochs",
                                "missing": "model.nope"})
    assert out == {"latent_dim": 64, "num_codebooks": 4, "epochs": 300}


def test_config_defaults_ignores_absent_keys(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("model:\n  latent_size: 64\n", encoding="utf-8")
    assert config_defaults(cfg, {"batch_size": "training.batch_size"}) == {}


def _parser(tmp_path, monkeypatch):
    from src.utils.config import add_config_flag as add

    cfg = tmp_path / "c.yaml"
    cfg.write_text("model:\n  hidden_size: 192\ntraining:\n  epochs: 120\n", encoding="utf-8")

    def build(argv):
        monkeypatch.setattr("sys.argv", ["prog", *argv])
        ap = argparse.ArgumentParser()
        ap.add_argument("--hidden-size", type=int, default=256)
        ap.add_argument("--epochs", type=int, default=100)
        add(ap, {"hidden_size": "model.hidden_size", "epochs": "training.epochs"})
        return ap.parse_args()

    return build, cfg


def test_cli_flag_overrides_config(tmp_path, monkeypatch):
    build, cfg = _parser(tmp_path, monkeypatch)
    args = build(["--config", str(cfg), "--epochs", "7"])
    assert args.epochs == 7
    assert args.hidden_size == 192  # untouched by the CLI, still from the config


def test_script_defaults_apply_without_config(tmp_path, monkeypatch):
    build, _ = _parser(tmp_path, monkeypatch)
    args = build([])
    assert (args.hidden_size, args.epochs) == (256, 100)


def test_rqvae_and_generative_mappings_match_their_configs():
    from scripts.train_generative_rec import GENERATIVE_CONFIG_MAP
    from scripts.train_rqvae import RQVAE_CONFIG_MAP

    rq = config_defaults(ROOT / "configs" / "rqvae.yaml", RQVAE_CONFIG_MAP)
    assert rq["latent_dim"] == 64
    assert rq["num_codebooks"] == 4
    assert rq["codebook_size"] == 256
    assert rq["commitment"] == pytest.approx(0.05)

    gen = config_defaults(ROOT / "configs" / "generative_rec.yaml", GENERATIVE_CONFIG_MAP)
    assert gen["hidden_size"] == 192
    assert gen["num_layers"] == 3
    assert gen["max_items"] == 20
    assert gen["semantic_ids"] == "artifacts/semantic_ids.npz"


def test_shipped_config_points_at_the_canonical_semantic_ids():
    """The generative config must reference the artifact the loader validates."""
    gen = load_config(ROOT / "configs" / "generative_rec.yaml")
    assert gen.data.semantic_ids == "artifacts/semantic_ids.npz"
    assert (ROOT / gen.data.semantic_ids).exists()
