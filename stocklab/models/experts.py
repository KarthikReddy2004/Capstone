"""Model runners: fit, predict train/val/test, forecast, and package a result.

Each ``run_*`` returns the canonical result dict from :func:`build_result`,
including a validation curve (walk-forward), a recursive forecast with prediction
intervals, a learning curve (LSTM), and feature importances (tree learners).
``feature_idx`` lets a model run on a selected feature subset (used by the
Adaptive Dual-Phase framework).
"""

from __future__ import annotations

import numpy as np

from ..data.dataset import build_sequences, build_test_sequences
from ..forecasting.recursive import recursive_forecast, prediction_intervals
from ..results import build_result, forecast_payload
from .lstm import TorchLSTM, predict_ensemble
from .elm import ELM
from .trees import GBDT, ExtraTrees
from .evaluate import (
    collect_lstm_validation, collect_elm_validation,
    collect_gbdt_validation, collect_extra_trees_validation,
)


def _val_block(dataset, validation):
    """Map a walk-forward validation trace onto calendar dates for plotting."""
    idx = np.asarray(validation.get("tune_idx", []), dtype=int)
    tune_start = int(dataset.extras.get("tune_start", 0))
    dates = []
    for pos in idx:
        g = tune_start + int(pos)
        dates.append(dataset.dates_train[g] if 0 <= g < len(dataset.dates_train) else "")
    return {
        "dates": dates,
        "actual": np.asarray(validation.get("actual", []), float).tolist(),
        "pred": np.asarray(validation.get("pred", []), float).tolist(),
        "rmse": float(validation.get("rmse", 0.0)),
    }


def _extras(dataset, hyper, validation_rmse, feature_idx, val, learning_curve=None, importance=None):
    return {
        "optimizer": {
            "method": str(hyper.get("optimizer_method", "")),
            "history": list(hyper.get("optimizer_history", [])),
            "telemetry": hyper.get("optimizer_telemetry", {}),
        },
        "best_hyper": {k: v for k, v in hyper.items()
                       if k not in ("optimizer_history", "optimizer_telemetry", "feature_idx")},
        "validation_rmse": float(validation_rmse),
        "n_selected": int(len(feature_idx)),
        "val": val,
        "learning_curve": learning_curve or {"train": [], "val": []},
        "feature_importance": importance or [],
    }


# --------------------------------------------------------------------------- #
# LSTM                                                                         #
# --------------------------------------------------------------------------- #
def run_lstm(dataset, hyper, feature_idx=None, n_splits=3):
    fi = dataset.feature_subset(feature_idx if feature_idx is not None else hyper.get("feature_idx"))
    seq_len = int(hyper.get("seq_len", dataset.base_seq_len))
    ensemble = int(hyper.get("ensemble", 1))
    epochs = int(hyper.get("epochs", 60))

    x_train = dataset.x_train[:, fi]
    x_test = dataset.x_test[:, fi]
    x_tr_seq, y_tr_seq = build_sequences(x_train, dataset.y_train_scaled, seq_len)
    x_te_seq, _ = build_test_sequences(x_train, x_test, dataset.y_train_scaled, dataset.y_test_scaled, seq_len)
    if len(x_tr_seq) == 0 or len(x_te_seq) == 0:
        raise ValueError("Sequence length too large for available data.")

    params = {**hyper, "seq_len": seq_len, "epochs": epochs}
    models = [TorchLSTM(params, seed=42 + k).fit(x_tr_seq, y_tr_seq) for k in range(ensemble)]

    pt_scaled = predict_ensemble(models, x_tr_seq)
    pe_scaled = predict_ensemble(models, x_te_seq)
    pred_train = dataset.reconstruct(dataset.prev_train[seq_len:], pt_scaled)
    pred_test = dataset.reconstruct(dataset.prev_test, pe_scaled)
    residual_std = float(np.std(dataset.inverse_target(dataset.y_test_scaled) - dataset.inverse_target(pe_scaled)))

    def predict_next(x_norm_full):
        window = x_norm_full[-seq_len:, fi]
        if len(window) < seq_len:
            return None
        return float(predict_ensemble(models, window.reshape(1, seq_len, len(fi)))[0])

    fut, dates = recursive_forecast(predict_next, dataset, dataset.n_future)
    intervals = prediction_intervals(fut, residual_std)
    forecast = forecast_payload(fut, dates, intervals)

    validation = collect_lstm_validation(dataset, params, fi, n_splits=n_splits)
    val = _val_block(dataset, validation)
    val_rmse = float(hyper.get("validation_rmse", validation["rmse"]))

    return build_result(
        y_test=dataset.y_test, pred_test=pred_test, prev_test=dataset.prev_test, dates_test=dataset.dates_test,
        y_train=dataset.y_train[seq_len:], pred_train=pred_train, dates_train=dataset.dates_train[seq_len:],
        forecast=forecast, val=val, n_features=len(fi),
        extras=_extras(dataset, params, val_rmse, fi, val, learning_curve=models[0].history_),
    )


# --------------------------------------------------------------------------- #
# Flat learners                                                                #
# --------------------------------------------------------------------------- #
def _run_flat(dataset, hyper, feature_idx, build, collector, n_splits, importance_fn=None):
    fi = dataset.feature_subset(feature_idx if feature_idx is not None else hyper.get("feature_idx"))
    x_train = dataset.x_train[:, fi]
    x_test = dataset.x_test[:, fi]
    model = build(hyper).fit(x_train, dataset.y_train_scaled)

    pt_scaled = model.predict(x_train)
    pe_scaled = model.predict(x_test)
    pred_train = dataset.reconstruct(dataset.prev_train, pt_scaled)
    pred_test = dataset.reconstruct(dataset.prev_test, pe_scaled)
    residual_std = float(np.std(dataset.inverse_target(dataset.y_test_scaled) - dataset.inverse_target(pe_scaled)))

    def predict_next(x_norm_full):
        return float(model.predict(x_norm_full[-1:, fi])[0])

    fut, dates = recursive_forecast(predict_next, dataset, dataset.n_future)
    intervals = prediction_intervals(fut, residual_std)
    forecast = forecast_payload(fut, dates, intervals)

    validation = collector(dataset, hyper, fi, n_splits=n_splits)
    val = _val_block(dataset, validation)
    val_rmse = float(hyper.get("validation_rmse", validation["rmse"]))

    importance = importance_fn(model, x_test, dataset.y_test_scaled, fi) if importance_fn else None

    return build_result(
        y_test=dataset.y_test, pred_test=pred_test, prev_test=dataset.prev_test, dates_test=dataset.dates_test,
        y_train=dataset.y_train, pred_train=pred_train, dates_train=dataset.dates_train,
        forecast=forecast, val=val, n_features=len(fi),
        extras=_extras(dataset, hyper, val_rmse, fi, val, importance=importance),
    )


def run_elm(dataset, hyper, feature_idx=None, n_splits=3):
    return _run_flat(dataset, hyper, feature_idx,
                     lambda p: ELM(p, seed=42), collect_elm_validation, n_splits)


def _tree_importance(dataset, fi):
    names = [dataset.feature_cols[i] for i in fi]
    return names


def run_gbdt(dataset, hyper, feature_idx=None, n_splits=3):
    def imp(model, x, y, fi):
        return model.feature_importance(x, y, [dataset.feature_cols[i] for i in fi])
    return _run_flat(dataset, hyper, feature_idx,
                     lambda p: GBDT(p, seed=42), collect_gbdt_validation, n_splits, importance_fn=imp)


def run_extra_trees(dataset, hyper, feature_idx=None, n_splits=3):
    def imp(model, x, y, fi):
        return model.feature_importance(x, y, [dataset.feature_cols[i] for i in fi])
    return _run_flat(dataset, hyper, feature_idx,
                     lambda p: ExtraTrees(p, seed=42), collect_extra_trees_validation, n_splits, importance_fn=imp)
