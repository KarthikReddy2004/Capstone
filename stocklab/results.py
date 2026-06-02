"""Assemble a standardized, JSON-serialisable result record for one model.

Every model runner returns the same schema so the leaderboard, dashboard, and
ensemble can treat models uniformly. Top-level metric keys are duplicated for
convenient sorting while structured blocks (``test``/``train``/``val``/
``forecast``/``optimizer``) drive the charts.
"""

from __future__ import annotations

import numpy as np

from .config import SPOTLIGHT_HORIZONS
from .validation.metrics import compute_metrics, METRIC_KEYS, mse


def _f(arr):
    return np.asarray(arr, float).tolist()


def build_result(
    *,
    y_test, pred_test, prev_test, dates_test,
    y_train, pred_train, dates_train,
    forecast=None, val=None, n_features=0, extras=None,
):
    """Build the canonical result dict for a single model."""

    y_test = np.asarray(y_test, float)
    pred_test = np.asarray(pred_test, float)
    y_train = np.asarray(y_train, float)
    pred_train = np.asarray(pred_train, float)
    metrics = compute_metrics(y_test, pred_test, prev_test)
    residuals = (y_test - pred_test)

    result = {
        # flattened metrics for leaderboard sorting
        **{k: float(metrics[k]) for k in METRIC_KEYS},
        # train vs test MSE expose generalisation gap (overfit detection)
        "train_mse": float(mse(y_train, pred_train)) if y_train.size else 0.0,
        "test_mse": float(metrics["mse"]),
        "validation_rmse": float(metrics["rmse"]),
        "n_features": int(n_features),
        "test": {
            "dates": list(dates_test),
            "actual": _f(y_test),
            "pred": _f(pred_test),
            "prev": _f(prev_test),
        },
        "train": {
            "dates": list(dates_train),
            "actual": _f(y_train),
            "pred": _f(pred_train),
        },
        "val": val or {"dates": [], "actual": [], "pred": []},
        "residuals": _f(residuals),
        "forecast": forecast or _empty_forecast(),
        # optimizer / framework annotations (filled by callers)
        "optimizer": {"method": "", "history": [], "telemetry": {}},
        "best_hyper": {},
        "fusion_weights": [],
        "selected_features": [],
        "n_selected": 0,
        "phase1": {},
        "phase2": {},
        "feature_importance": [],
        "timing_sec": 0.0,
        "memory_mb": 0.0,
        "learning_curve": {"train": [], "val": []},
    }
    if extras:
        result.update(extras)
    return result


def _empty_forecast():
    return {
        "dates": [], "values": [],
        "intervals": {str(p): {"lower": [], "upper": []} for p in (80, 90, 95)},
        "points": [],
    }


def forecast_payload(values, dates, intervals):
    """Package recursive forecast values + prediction intervals + spotlight points."""
    values = [float(v) for v in values]
    dates = list(dates)
    points = []
    for h in SPOTLIGHT_HORIZONS:
        if len(values) >= h:
            pt = {"horizon": int(h), "date": dates[h - 1], "value": values[h - 1]}
            band = intervals.get("90")
            if band and len(band["lower"]) >= h:
                pt["lower"] = float(band["lower"][h - 1])
                pt["upper"] = float(band["upper"][h - 1])
            points.append(pt)
    return {"dates": dates, "values": values, "intervals": intervals, "points": points}
