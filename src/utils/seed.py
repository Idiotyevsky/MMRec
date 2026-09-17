"""Reproducibility helpers.

GPU kernels used by recommendation models are not guaranteed to be bitwise
deterministic, but we seed every RNG we control so that runs are *reproducible
in distribution*: same config + same seed => same data order, same negative
samples, same initialisation.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed python / numpy / torch (CPU + all CUDA devices)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding: derive a distinct, stable seed per worker."""
    base = torch.initial_seed() % (2**31)
    np.random.seed((base + worker_id) % (2**31))
    random.seed((base + worker_id) % (2**31))
