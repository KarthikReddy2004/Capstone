"""Recursive forecasting loop and residual-based prediction intervals."""

from __future__ import annotations

import numpy as np

from ..data.features import prepare_features
from .synthesis import extend_history, next_business_date


def recursive_forecast(predict_next, dataset, n_future):
    """Roll the model forward ``n_future`` steps.

    ``predict_next(x_norm_full)`` receives the full normalised feature matrix of
    the (growing) history and returns the next-step prediction in *scaled* units.
    The model wrapper decides whether to use the last row (flat learners) or the
    last ``seq_len`` window (sequence learners).
    """

    raw = dataset.history_raw.copy()
    values: list[float] = []
    dates: list[str] = []
    for _ in range(int(n_future)):
        feat = prepare_features(raw)
        x = feat[dataset.feature_cols].to_numpy(float)
        x_norm = np.clip((x - dataset.feature_mean) / dataset.feature_std, -6.0, 6.0)
        scaled = predict_next(x_norm)
        if scaled is None:
            break
        last_close = float(raw["Close"].iloc[-1])
        # Reconstruct the next price from the predicted next-step return.
        value = float(dataset.reconstruct(np.array([last_close]), np.array([scaled]))[0])
        dates.append(next_business_date(raw["Date"].iloc[-1]))
        values.append(value)
        raw = extend_history(raw, value)
    return values, dates


# Two-sided z-multipliers for the requested confidence levels.
_Z = {80: 1.2816, 90: 1.6449, 95: 1.9600}


def prediction_intervals(values, residual_std, levels=(80, 90, 95), multiplicative=True):
    """Grow uncertainty with the forecast horizon (random-walk error accrual).

    The one-step **return** residual standard deviation is inflated by
    ``sqrt(h)`` at horizon ``h`` to reflect compounding error in recursive
    forecasting. With ``multiplicative=True`` (the default for return targets)
    bands are ``price * exp(±z*sigma_h)``, which keeps them positive and
    proportional to the price level; otherwise symmetric additive bands are used.
    """

    values = np.asarray(values, float)
    horizons = np.arange(1, len(values) + 1)
    grow = np.sqrt(horizons) * float(max(residual_std, 1e-9))
    out = {}
    for level in levels:
        z = _Z.get(int(level), 1.96)
        if multiplicative:
            lower = (values * np.exp(-z * grow)).tolist()
            upper = (values * np.exp(z * grow)).tolist()
        else:
            delta = z * grow
            lower = (values - delta).tolist()
            upper = (values + delta).tolist()
        out[str(level)] = {"lower": lower, "upper": upper}
    return out
