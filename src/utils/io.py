"""Small IO helpers: json, run ids, directory creation."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    def _default(o: Any):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_default, ensure_ascii=False)


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_id(prefix: str = "run") -> str:
    """Unique, sortable run identifier: <prefix>_<utc timestamp>_<short uuid>."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:6]}"
