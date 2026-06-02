"""Hyper-parameter search-space bounds and vector -> param-dict decoders.

Optimizers operate on continuous boxed vectors; these decoders translate a
vector into a validated, typed hyper-parameter dict for each learner family.
Clipping inside the decoder guarantees every candidate the optimizer proposes is
feasible, even at the box edges.
"""

from __future__ import annotations

import numpy as np

from .elm import _ACTIVATIONS

# Each bound is (low, high). Dimensions are documented inline.
LSTM_BOUNDS = [
    (32, 160),    # 0 hidden units
    (1, 3),       # 1 stacked layers
    (0.0, 0.45),  # 2 dropout
    (1e-4, 5e-3), # 3 learning rate
    (16, 96),     # 4 batch size
    (16, 64),     # 5 sequence length
    (1e-6, 5e-3), # 6 weight decay (L2)
]
ELM_BOUNDS = [
    (40, 400),    # 0 hidden neurons
    (1e-5, 2.0),  # 1 ridge coefficient
    (0.05, 1.0),  # 2 input weight scale
    (0, 2),       # 3 activation index (tanh/sigmoid/relu)
]
GBDT_BOUNDS = [
    (80, 400),    # 0 estimators (boosting iterations)
    (0.01, 0.25), # 1 learning rate
    (2, 8),       # 2 max depth
    (8, 60),      # 3 min samples per leaf
    (0.0, 0.3),   # 4 L2 regularisation
]
EXTRA_TREE_BOUNDS = [
    (160, 520),   # 0 estimators
    (4, 18),      # 1 max depth
    (1, 16),      # 2 min samples per leaf
    (0.35, 1.0),  # 3 max features fraction
]


def decode_lstm(values, base_seq_len: int | None = None) -> dict:
    v = np.asarray(values, float)
    seq_len = int(np.clip(round(v[5]), 16, 64))
    if base_seq_len is not None and len(v) <= 5:
        seq_len = int(base_seq_len)
    return {
        "n_hidden": int(np.clip(round(v[0]), 32, 160)),
        "n_layers": int(np.clip(round(v[1]), 1, 3)),
        "dropout": float(np.clip(v[2], 0.0, 0.45)),
        "lr": float(np.clip(v[3], 1e-4, 5e-3)),
        "batch_size": int(np.clip(round(v[4] / 8) * 8, 16, 96)),
        "seq_len": seq_len,
        "alpha": float(np.clip(v[6], 1e-6, 5e-3)) if len(v) > 6 else 3e-4,
    }


def decode_elm(values) -> dict:
    v = np.asarray(values, float)
    return {
        "n_hidden": int(np.clip(round(v[0]), 40, 400)),
        "alpha": float(np.clip(v[1], 1e-5, 2.0)),
        "scale": float(np.clip(v[2], 0.05, 1.0)),
        "activation": _ACTIVATIONS[int(np.clip(round(v[3]), 0, 2))],
    }


def decode_gbdt(values) -> dict:
    v = np.asarray(values, float)
    return {
        "n_estimators": int(np.clip(round(v[0]), 80, 400)),
        "lr": float(np.clip(v[1], 0.01, 0.25)),
        "max_depth": int(np.clip(round(v[2]), 2, 8)),
        "min_samples_leaf": int(np.clip(round(v[3]), 8, 60)),
        "l2": float(np.clip(v[4], 0.0, 0.3)),
    }


def decode_extra_trees(values) -> dict:
    v = np.asarray(values, float)
    return {
        "n_estimators": int(np.clip(round(v[0]), 160, 520)),
        "max_depth": int(np.clip(round(v[1]), 4, 18)),
        "min_samples_leaf": int(np.clip(round(v[2]), 1, 16)),
        "max_features": float(np.clip(v[3], 0.35, 1.0)),
    }
