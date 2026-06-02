"""Synthesise plausible future OHLCV bars during recursive forecasting.

When the model predicts the next close, the feature engineering for the *next*
step needs a full OHLCV bar. We construct one from the predicted close and a
volatility band estimated from recent history. This keeps the causal feature
pipeline identical between historical fitting and forward forecasting.
"""

from __future__ import annotations

import pandas as pd

from ..data.loaders import RAW_COLS


def next_business_date(date_value) -> str:
    last = pd.Timestamp(date_value)
    return pd.bdate_range(last, periods=2)[-1].strftime("%Y-%m-%d")


def extend_history(raw: pd.DataFrame, next_close: float) -> pd.DataFrame:
    """Append a synthesised bar for ``next_close`` to the raw OHLCV history."""
    raw = raw[[c for c in RAW_COLS if c in raw.columns]].copy()
    prev_close = float(raw["Close"].iloc[-1])
    recent_ret = raw["Close"].pct_change().tail(12).dropna()
    recent_vol = float(recent_ret.std()) if not recent_ret.empty else 0.01
    drift = (float(next_close) - prev_close) / (prev_close + 1e-9)
    band = prev_close * max(0.003, recent_vol, abs(drift) * 0.8)

    next_open = prev_close
    next_high = max(next_open, float(next_close)) + band
    next_low = max(1e-6, min(next_open, float(next_close)) - band)
    next_volume = float(raw["Volume"].tail(5).median()) if len(raw) >= 5 else float(raw["Volume"].iloc[-1])
    row = {
        "Date": next_business_date(raw["Date"].iloc[-1]),
        "Open": next_open,
        "High": next_high,
        "Low": next_low,
        "Close": float(next_close),
        "Volume": max(next_volume, 1.0),
    }
    return pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
