"""Strictly causal technical feature engineering.

Every indicator here uses only information available *up to and including* the
bar it is attached to (trailing rolling windows, backward-shifted lags). The
prediction target is supplied separately by the dataset builder as the *next*
bar's close, so no same-bar or future information leaks into the feature matrix.

The function returns a tidy frame with ``Date``, the raw OHLCV, and the derived
feature columns. Downstream code treats every column that is not ``Date`` or
``Close`` as a model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-9

# Short human-readable descriptions surfaced in the dashboard tooltips.
FEATURE_HELP = {
    "LogReturn": "Daily log return",
    "RSI14": "Relative Strength Index (14)",
    "MACD": "MACD line (EMA12 - EMA26)",
    "ATR14": "Average True Range (14)",
    "CCI20": "Commodity Channel Index (20)",
    "WilliamsR": "Williams %R (14)",
    "OBV": "On-Balance Volume",
    "VWAPDist": "Close distance from rolling VWAP(20)",
    "BBPos": "Position within Bollinger Bands",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the causal feature matrix from a normalized OHLCV frame."""

    frame = df.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)

    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "Close" not in frame.columns:
        raise ValueError("A 'Close' column is required.")

    close_s = frame["Close"].astype(float)
    frame["Open"] = frame["Open"].astype(float) if "Open" in frame.columns else close_s
    frame["High"] = frame["High"].astype(float) if "High" in frame.columns else close_s
    frame["Low"] = frame["Low"].astype(float) if "Low" in frame.columns else close_s
    frame["Volume"] = frame["Volume"].astype(float) if "Volume" in frame.columns else 1.0
    frame = frame.dropna(subset=["Close"]).reset_index(drop=True)

    close = frame["Close"].to_numpy(float)
    open_ = frame["Open"].to_numpy(float)
    high = frame["High"].to_numpy(float)
    low = frame["Low"].to_numpy(float)
    volume = frame["Volume"].to_numpy(float)
    c = pd.Series(close)

    feat = pd.DataFrame({
        "Date": frame["Date"].dt.strftime("%Y-%m-%d"),
        "Close": close,
        "Open": open_, "High": high, "Low": low, "Volume": volume,
    })

    # --- returns ------------------------------------------------------------
    feat["LogReturn"] = np.concatenate([[np.nan], np.log((close[1:] + _EPS) / (close[:-1] + _EPS))])
    feat["Return3"] = c.pct_change(3).to_numpy()
    feat["Return5"] = c.pct_change(5).to_numpy()
    feat["RangePct"] = (high - low) / (close + _EPS)
    feat["BodyPct"] = (close - open_) / (open_ + _EPS)
    feat["GapPct"] = np.concatenate([[np.nan], (open_[1:] - close[:-1]) / (close[:-1] + _EPS)])

    # --- moving averages ----------------------------------------------------
    for w in (5, 10, 20, 50, 100):
        sma = c.rolling(w).mean()
        feat[f"SMA{w}"] = sma.to_numpy()
        feat[f"DistSMA{w}"] = close / (sma.to_numpy() + _EPS)
    for w in (5, 10, 20, 50):
        ema = c.ewm(span=w, adjust=False).mean()
        feat[f"EMA{w}"] = ema.to_numpy()
        feat[f"DistEMA{w}"] = close / (ema.to_numpy() + _EPS)

    # --- momentum / oscillators --------------------------------------------
    feat["RSI14"] = _rsi(close, 14)
    feat["RSI7"] = _rsi(close, 7)
    ema12 = c.ewm(span=12, adjust=False).mean().to_numpy()
    ema26 = c.ewm(span=26, adjust=False).mean().to_numpy()
    feat["MACD"] = ema12 - ema26
    feat["MACDSignal"] = pd.Series(feat["MACD"]).ewm(span=9, adjust=False).mean().to_numpy()
    feat["MACDHist"] = feat["MACD"] - feat["MACDSignal"]
    feat["Momentum10"] = np.concatenate([np.full(10, np.nan), close[10:] - close[:-10]])
    feat["ROC10"] = np.concatenate([np.full(10, np.nan), (close[10:] - close[:-10]) / (close[:-10] + _EPS) * 100])
    feat["ROC5"] = np.concatenate([np.full(5, np.nan), (close[5:] - close[:-5]) / (close[:-5] + _EPS) * 100])

    # --- Bollinger bands ----------------------------------------------------
    sma20 = c.rolling(20).mean().to_numpy()
    std20 = c.rolling(20).std().to_numpy()
    feat["BBUpper"] = sma20 + 2 * std20
    feat["BBLower"] = sma20 - 2 * std20
    feat["BBWidth"] = (feat["BBUpper"] - feat["BBLower"]) / (sma20 + _EPS)
    feat["BBPos"] = (close - feat["BBLower"]) / (feat["BBUpper"] - feat["BBLower"] + _EPS)

    # --- volatility / range indicators -------------------------------------
    feat["StochK"], feat["StochD"] = _stochastic(high, low, close, 14, 3)
    feat["ATR14"] = _atr(high, low, close, 14)
    feat["ATRRatio14"] = feat["ATR14"] / (close + _EPS)
    feat["CCI20"] = _cci(high, low, close, 20)
    feat["WilliamsR"] = _williams_r(high, low, close, 14)

    # --- volume -------------------------------------------------------------
    feat["OBV"] = _obv(close, volume)
    feat["VWAP20"] = _rolling_vwap(high, low, close, volume, 20)
    feat["VWAPDist"] = close / (feat["VWAP20"] + _EPS)
    feat["VolumeChange"] = np.concatenate([[np.nan], np.diff(volume) / (volume[:-1] + _EPS)])
    feat["VolSMA20"] = pd.Series(volume).rolling(20).mean().to_numpy()
    feat["VolumeRatio20"] = volume / (feat["VolSMA20"] + _EPS)
    feat["VolumeZ20"] = _rolling_zscore(volume, 20)

    # --- rolling statistics / z-scores -------------------------------------
    feat["Vol10"] = pd.Series(feat["LogReturn"]).rolling(10).std().to_numpy()
    feat["Vol20"] = pd.Series(feat["LogReturn"]).rolling(20).std().to_numpy()
    feat["VolRatio"] = feat["Vol10"] / (feat["Vol20"] + _EPS)
    feat["CloseZ20"] = _rolling_zscore(close, 20)
    feat["CloseZ50"] = _rolling_zscore(close, 50)
    feat["ReturnZ20"] = _rolling_zscore(feat["LogReturn"].to_numpy(), 20)

    # --- lag features (strictly backward) ----------------------------------
    for lag in (1, 2, 3, 5, 10):
        feat[f"CloseLag{lag}"] = c.shift(lag).to_numpy()
        feat[f"RetLag{lag}"] = pd.Series(feat["LogReturn"]).shift(lag).to_numpy()

    # --- trend / structure --------------------------------------------------
    feat["HighLow20"] = close / (pd.Series(low).rolling(20).min().to_numpy() + _EPS)
    feat["HighHigh20"] = close / (pd.Series(high).rolling(20).max().to_numpy() + _EPS)
    feat["Breakout20"] = close / (pd.Series(high).rolling(20).max().shift(1).to_numpy() + _EPS)
    feat["Drawdown20"] = close / (pd.Series(close).rolling(20).max().to_numpy() + _EPS)
    feat["Trend10"] = c.pct_change(10).to_numpy()
    feat["Trend20"] = c.pct_change(20).to_numpy()
    feat["TrendStrength"] = np.abs(feat["EMA10"] - feat["EMA20"]) / (feat["EMA20"] + _EPS)

    # --- calendar / cyclical ------------------------------------------------
    dow = frame["Date"].dt.dayofweek.to_numpy()
    month = frame["Date"].dt.month.to_numpy()
    feat["DaySin"] = np.sin(2 * np.pi * dow / 7)
    feat["DayCos"] = np.cos(2 * np.pi * dow / 7)
    feat["MonthSin"] = np.sin(2 * np.pi * month / 12)
    feat["MonthCos"] = np.cos(2 * np.pi * month / 12)

    feat = feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return feat


def build_summary(prepared: pd.DataFrame) -> dict:
    """Lightweight dataset summary for the UI data strip."""
    non_feature = {"Date", "Close", "Open", "High", "Low", "Volume"}
    return {
        "rows": int(len(prepared)),
        "features": int(len([c for c in prepared.columns if c not in {"Date", "Close"}])),
        "indicators": int(len([c for c in prepared.columns if c not in non_feature])),
        "start": str(prepared["Date"].iloc[0]),
        "end": str(prepared["Date"].iloc[-1]),
        "lastClose": float(prepared["Close"].iloc[-1]),
    }


# --------------------------------------------------------------------------- #
# Indicator helpers (all causal)                                              #
# --------------------------------------------------------------------------- #
def _rolling_zscore(values, window):
    s = pd.Series(values)
    return ((s - s.rolling(window).mean()) / (s.rolling(window).std() + _EPS)).to_numpy()


def _rsi(close, period=14):
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    rs = avg_gain / (avg_loss + _EPS)
    return np.concatenate([[np.nan], 100 - 100 / (1 + rs)])


def _stochastic(high, low, close, k_period, d_period):
    size = len(close)
    k = np.full(size, np.nan)
    low_s = pd.Series(low).rolling(k_period).min().to_numpy()
    high_s = pd.Series(high).rolling(k_period).max().to_numpy()
    for i in range(k_period - 1, size):
        k[i] = 100 * (close[i] - low_s[i]) / (high_s[i] - low_s[i] + _EPS)
    d = pd.Series(k).rolling(d_period).mean().to_numpy()
    return k, d


def _atr(high, low, close, period):
    size = len(close)
    tr = np.full(size, np.nan)
    for i in range(1, size):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def _cci(high, low, close, period):
    typical = (high + low + close) / 3
    size = len(typical)
    cci = np.full(size, np.nan)
    ts = pd.Series(typical)
    mean_s = ts.rolling(period).mean().to_numpy()
    for i in range(period - 1, size):
        window = typical[i - period + 1:i + 1]
        mad = np.mean(np.abs(window - mean_s[i]))
        cci[i] = (typical[i] - mean_s[i]) / (0.015 * mad + _EPS)
    return cci


def _williams_r(high, low, close, period):
    size = len(close)
    wr = np.full(size, np.nan)
    high_s = pd.Series(high).rolling(period).max().to_numpy()
    low_s = pd.Series(low).rolling(period).min().to_numpy()
    for i in range(period - 1, size):
        wr[i] = -100 * (high_s[i] - close[i]) / (high_s[i] - low_s[i] + _EPS)
    return wr


def _obv(close, volume):
    size = len(close)
    obv = np.zeros(size)
    for i in range(1, size):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def _rolling_vwap(high, low, close, volume, window):
    typical = (high + low + close) / 3
    pv = pd.Series(typical * volume).rolling(window).sum().to_numpy()
    vol = pd.Series(volume).rolling(window).sum().to_numpy()
    return pv / (vol + _EPS)
