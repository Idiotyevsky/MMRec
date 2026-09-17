"""Device selection and environment fingerprinting."""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

import torch


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def env_fingerprint() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        try:
            info["gpu_total_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            )
        except Exception:  # pragma: no cover
            pass
    try:
        info["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        info["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        info["git_commit"] = None
    return info
