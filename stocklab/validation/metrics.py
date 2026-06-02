"""Forecast error metrics.

Every metric is a pure NumPy function so it can be unit-tested in isolation and
reused across the optimizer objective, the leaderboard, and the dashboard.

Conventions
-----------
* ``y_true`` / ``y_pred`` are 1-D arrays of the price level being forecast.
* ``prev`` (optional) is the last observed price *before* each target. When it
  is supplied, directional metrics are computed against the realised move from
  the anchor (the realistic "did we call the next move" question) rather than
  against differences of consecutive predictions.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9

METRIC_KEYS = (
    "mse", "rmse", "mae", "mape", "smape", "r2", "theil_u", "arv",
    "dir_acc", "hit_ratio", "bias", "tracking_signal",
)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.mean(err ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(err ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + _EPS))) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred) + _EPS
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / (ss_tot + _EPS))


def arv(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average Relative Variance = SS_res / SS_tot (equivalently 1 − R²).

    Normalises the forecast error by the variance of the series, so it is
    comparable across symbols. ``ARV < 1`` beats a mean (climatology) predictor;
    smaller is better.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(ss_res / (ss_tot + _EPS))


def theil_u(y_true, y_pred, prev=None) -> float:
    """Theil's U2 forecast-accuracy coefficient (model error vs. naive forecast).

    With an anchor ``prev`` (the last observed price before each target) the
    naive forecast is "no change", so this compares the model's relative-change
    error against simply predicting today's price for tomorrow:

        U2 = sqrt(Σ((ŷ−y)/prev)²) / sqrt(Σ((y−prev)/prev)²)

    ``U2 < 1`` beats the random-walk baseline, ``= 1`` matches it, ``> 1`` is
    worse. Without an anchor it falls back to Theil's U1 inequality coefficient
    on price levels.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_true.size == 0:
        return 0.0
    if prev is not None and len(prev) == len(y_true):
        prev = np.asarray(prev, float)
        num = float(np.sqrt(np.mean(((y_pred - y_true) / (prev + _EPS)) ** 2)))
        den = float(np.sqrt(np.mean(((y_true - prev) / (prev + _EPS)) ** 2)))
        return float(num / (den + _EPS))
    num = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    den = float(np.sqrt(np.mean(y_pred ** 2)) + np.sqrt(np.mean(y_true ** 2)))
    return float(num / (den + _EPS))


def _directions(y_true: np.ndarray, y_pred: np.ndarray, prev: np.ndarray | None):
    """Return (actual_dir, pred_dir) sign arrays.

    With ``prev`` we compare the realised move (target vs anchor) against the
    predicted move (prediction vs the same anchor). Without it we fall back to
    differences of consecutive points within the series.
    """

    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if prev is not None and len(prev) == len(y_true):
        prev = np.asarray(prev, float)
        return np.sign(y_true - prev), np.sign(y_pred - prev)
    if len(y_true) < 2:
        return np.array([]), np.array([])
    return np.sign(np.diff(y_true)), np.sign(np.diff(y_pred))


def directional_accuracy(y_true, y_pred, prev=None) -> float:
    """Percentage of correctly predicted up/down moves."""
    act, pred = _directions(y_true, y_pred, prev)
    if act.size == 0:
        return 0.0
    return float(np.mean(act == pred) * 100.0)


def hit_ratio(y_true, y_pred, prev=None) -> float:
    """Fraction of moves where the sign was called correctly, ignoring flats.

    Differs from directional accuracy by excluding bars with no actual move,
    which is the more honest "trading hit rate".
    """
    act, pred = _directions(y_true, y_pred, prev)
    mask = act != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(act[mask] == pred[mask]) * 100.0)


def forecast_bias(y_true, y_pred) -> float:
    """Mean signed error (positive => model over-predicts)."""
    return float(np.mean(np.asarray(y_pred, float) - np.asarray(y_true, float)))


def tracking_signal(y_true, y_pred) -> float:
    """Running-sum-of-errors divided by mean absolute deviation.

    A classic forecast-control statistic: values drifting far from 0 (|TS| > ~4)
    indicate persistent bias rather than random error.
    """
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    rsfe = float(np.sum(err))
    mad = float(np.mean(np.abs(err)))
    return float(rsfe / (mad + _EPS))


def compute_metrics(y_true, y_pred, prev=None) -> dict[str, float]:
    """Compute the full metric suite as a JSON-serialisable dict."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_true.size == 0:
        return {k: 0.0 for k in METRIC_KEYS}
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "theil_u": theil_u(y_true, y_pred, prev),
        "arv": arv(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred, prev),
        "hit_ratio": hit_ratio(y_true, y_pred, prev),
        "bias": forecast_bias(y_true, y_pred),
        "tracking_signal": tracking_signal(y_true, y_pred),
    }
