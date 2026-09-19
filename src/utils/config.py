"""Lightweight YAML config with dotted CLI overrides.

Deliberately *not* Hydra: we want a config system a reviewer can read in one
minute.  Supports::

    cfg = load_config("configs/sasrec.yaml", overrides=["training.batch_size=256", "seed=7"])
    cfg.training.batch_size  # 256
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def _coerce(value: str) -> Any:
    """Parse a CLI string into a python scalar using YAML rules."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


class Config(dict):
    """Dict with attribute access (nested dicts are wrapped lazily)."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict:
        out: dict = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, dict):
                out[k] = Config(v).to_dict()
            else:
                out[k] = v
        return out

    def copy(self) -> "Config":  # type: ignore[override]
        return Config(copy.deepcopy(self.to_dict()))


def _set_in(d: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw)}")

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_in(raw, k.strip(), _coerce(v.strip()))

    return Config(raw)


def save_config(cfg: dict | Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, Config) else dict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def config_defaults(config_path: str | Path, mapping: dict[str, str]) -> dict:
    """Read dotted keys out of a YAML config for use as argparse defaults.

    Lets a script keep its flat ``--flag`` interface while still being driven by
    a structured config file::

        mapping = {"latent_dim": "model.latent_size", "epochs": "training.epochs"}
        parser.set_defaults(**config_defaults("configs/rqvae.yaml", mapping))

    Only keys actually present in the file are returned, so unspecified values
    keep the script's own defaults.
    """
    cfg = load_config(config_path)
    out: dict = {}
    for dest, dotted in mapping.items():
        cur: Any = cfg
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                cur = None
                break
        if cur is not None:
            out[dest] = cur
    return out


def add_config_flag(parser, mapping: dict[str, str]) -> None:
    """Add ``--config`` to ``parser`` and apply it as defaults.

    Must be called *before* ``parse_args``.  Values given explicitly on the
    command line still win, because argparse only falls back to defaults.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()
    # the real parser must also accept the flag, or argparse rejects it as unknown
    parser.add_argument("--config", default=known.config,
                        help="YAML config supplying defaults for the flags below")
    if known.config:
        parser.set_defaults(**config_defaults(known.config, mapping))
