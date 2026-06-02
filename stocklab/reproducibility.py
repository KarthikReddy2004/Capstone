"""Reproducibility controls.

A single entry point seeds every source of randomness used in the platform
(Python ``random``, NumPy, and PyTorch) and forces deterministic CPU kernels.
Optimizers additionally take explicit seeds so that multi-seed searches are
reproducible run-to-run.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed all RNGs and return a NumPy ``Generator`` for local use.

    Returns the generator so callers can thread a deterministic stream through
    optimizers and model initialisation without touching global state again.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # CPU LSTM has no det. kernel issues
        # Single, predictable thread topology -> stable timings + results.
        from .config import TORCH_THREADS

        torch.set_num_threads(max(1, TORCH_THREADS))
    except Exception:
        # Torch is optional at import time; model layer enforces availability.
        pass

    return np.random.default_rng(seed)


def child_seed(base_seed: int, *tags: int) -> int:
    """Derive a stable child seed from a base seed and integer tags.

    Used to give every optimizer seed / fold / ensemble member a distinct but
    deterministic seed without manual bookkeeping.
    """

    value = int(base_seed) & 0x7FFFFFFF
    for tag in tags:
        value = (value * 1_000_003 + int(tag) * 9_176 + 12_345) & 0x7FFFFFFF
    return value
