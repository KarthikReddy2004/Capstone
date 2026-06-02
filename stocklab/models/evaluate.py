"""Walk-forward CV objectives and validation collectors.

* ``eval_*``    -> scalar RMSE (price units) used as the optimizer objective.
* ``collect_*`` -> aligned ``{actual, pred, rmse, tune_idx}`` traces used to fit
                   the ensemble fusion weights with temporal cross-validation.

All evaluation happens on the trailing *tune* slice of the training data with
walk-forward folds, so model selection never touches the held-out test set.
"""

from __future__ import annotations

import numpy as np

from ..data.dataset import build_sequences
from ..validation.metrics import rmse as _rmse
from ..validation.splitters import walk_forward_splits
from .lstm import TorchLSTM, predict_ensemble
from .elm import ELM
from .trees import GBDT, ExtraTrees

_BIG = 1e9


def _idx(dataset, feature_idx):
    return dataset.feature_subset(feature_idx)


# --------------------------------------------------------------------------- #
# LSTM                                                                         #
# --------------------------------------------------------------------------- #
def eval_lstm(dataset, params, feature_idx=None, n_splits=3):
    fi = _idx(dataset, feature_idx)
    x = dataset.x_tune[:, fi]
    y = dataset.y_tune_scaled
    seq_len = int(params["seq_len"])
    x_seq, y_seq = build_sequences(x, y, seq_len)
    folds = walk_forward_splits(len(x_seq), n_splits=n_splits, min_train=max(40, seq_len), min_val=12)
    if not folds:
        return _BIG
    ens = max(1, int(params.get("ensemble", 1)))
    scores = []
    for fold_i, (_, tr_end, v0, v1) in enumerate(folds):
        models = [TorchLSTM({**params, "epochs": params.get("tune_epochs", 28)},
                            seed=200 + fold_i * 17 + k).fit(x_seq[:tr_end], y_seq[:tr_end])
                  for k in range(ens)]
        pred = predict_ensemble(models, x_seq[v0:v1])
        scores.append(_rmse(y_seq[v0:v1], pred))
    return float(np.mean(scores)) * dataset.target_scale


def collect_lstm_validation(dataset, params, feature_idx=None, n_splits=3):
    fi = _idx(dataset, feature_idx)
    x = dataset.x_tune[:, fi]
    y = dataset.y_tune_scaled
    seq_len = int(params["seq_len"])
    x_seq, y_seq = build_sequences(x, y, seq_len)
    folds = walk_forward_splits(len(x_seq), n_splits=n_splits, min_train=max(40, seq_len), min_val=12)
    if not folds:
        return _empty_validation()
    ens = max(1, int(params.get("ensemble", 1)))
    tune_start = int(dataset.extras.get("tune_start", 0))
    a_parts, p_parts, idx_parts = [], [], []
    for fold_i, (_, tr_end, v0, v1) in enumerate(folds):
        models = [TorchLSTM({**params, "epochs": params.get("tune_epochs", 28)},
                            seed=300 + fold_i * 17 + k).fit(x_seq[:tr_end], y_seq[:tr_end])
                  for k in range(ens)]
        local = np.arange(seq_len + v0, seq_len + v1)
        prev = dataset.prev_train[tune_start + local]
        p_parts.append(dataset.reconstruct(prev, predict_ensemble(models, x_seq[v0:v1])))
        a_parts.append(dataset.reconstruct(prev, y_seq[v0:v1]))
        idx_parts.append(local)
    return _pack(a_parts, p_parts, idx_parts)


# --------------------------------------------------------------------------- #
# Flat learners (ELM / GBDT / ExtraTrees)                                      #
# --------------------------------------------------------------------------- #
def _eval_flat(dataset, build, feature_idx, n_splits):
    fi = _idx(dataset, feature_idx)
    x = dataset.x_tune[:, fi]
    y = dataset.y_tune_scaled
    folds = walk_forward_splits(len(x), n_splits=n_splits, min_train=50, min_val=16)
    if not folds:
        return _BIG
    scores = []
    for fold_i, (_, tr_end, v0, v1) in enumerate(folds):
        model = build(42 + fold_i * 13).fit(x[:tr_end], y[:tr_end])
        pred = model.predict(x[v0:v1])
        scores.append(_rmse(y[v0:v1], pred))
    return float(np.mean(scores)) * dataset.target_scale


def _collect_flat(dataset, build, feature_idx, n_splits):
    fi = _idx(dataset, feature_idx)
    x = dataset.x_tune[:, fi]
    y = dataset.y_tune_scaled
    folds = walk_forward_splits(len(x), n_splits=n_splits, min_train=50, min_val=16)
    if not folds:
        return _empty_validation()
    tune_start = int(dataset.extras.get("tune_start", 0))
    a_parts, p_parts, idx_parts = [], [], []
    for fold_i, (_, tr_end, v0, v1) in enumerate(folds):
        model = build(142 + fold_i * 13).fit(x[:tr_end], y[:tr_end])
        local = np.arange(v0, v1)
        prev = dataset.prev_train[tune_start + local]
        p_parts.append(dataset.reconstruct(prev, model.predict(x[v0:v1])))
        a_parts.append(dataset.reconstruct(prev, y[v0:v1]))
        idx_parts.append(local)
    return _pack(a_parts, p_parts, idx_parts)


def eval_elm(dataset, params, feature_idx=None, n_splits=3):
    return _eval_flat(dataset, lambda s: ELM(params, seed=s), feature_idx, n_splits)


def collect_elm_validation(dataset, params, feature_idx=None, n_splits=3):
    return _collect_flat(dataset, lambda s: ELM(params, seed=s), feature_idx, n_splits)


def eval_gbdt(dataset, params, feature_idx=None, n_splits=3):
    return _eval_flat(dataset, lambda s: GBDT(params, seed=s), feature_idx, n_splits)


def collect_gbdt_validation(dataset, params, feature_idx=None, n_splits=3):
    return _collect_flat(dataset, lambda s: GBDT(params, seed=s), feature_idx, n_splits)


def eval_extra_trees(dataset, params, feature_idx=None, n_splits=3):
    return _eval_flat(dataset, lambda s: ExtraTrees(params, seed=s), feature_idx, n_splits)


def collect_extra_trees_validation(dataset, params, feature_idx=None, n_splits=3):
    return _collect_flat(dataset, lambda s: ExtraTrees(params, seed=s), feature_idx, n_splits)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _pack(a_parts, p_parts, idx_parts):
    actual = np.concatenate(a_parts)
    pred = np.concatenate(p_parts)
    return {
        "actual": actual,
        "pred": pred,
        "rmse": _rmse(actual, pred),
        "tune_idx": np.concatenate(idx_parts),
    }


def _empty_validation():
    return {"actual": np.empty(0), "pred": np.empty(0), "rmse": _BIG, "tune_idx": np.empty(0, dtype=int)}
