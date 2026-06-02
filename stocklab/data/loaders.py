"""Data ingestion: Yahoo Finance download and CSV upload normalization."""

from __future__ import annotations

import pandas as pd

RAW_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame into the canonical schema.

    Accepts common column aliases (adj close, vol, datetime, ...) and validates
    that the minimum required columns are present.
    """

    frame = df.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    alias = {}
    for col in frame.columns:
        key = col.lower().strip()
        if key in ("date", "time", "datetime", "timestamp"):
            alias[col] = "Date"
        elif key == "open":
            alias[col] = "Open"
        elif key == "high":
            alias[col] = "High"
        elif key == "low":
            alias[col] = "Low"
        elif key in ("close", "adj close", "adj_close", "adjclose"):
            alias[col] = "Close"
        elif key in ("volume", "vol"):
            alias[col] = "Volume"
    frame = frame.rename(columns=alias)

    missing = [c for c in ("Date", "Close") if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    keep = [c for c in RAW_COLS if c in frame.columns]
    return frame[keep].reset_index(drop=True)


def load_from_yahoo(symbol: str, start: str = "2018-01-01") -> pd.DataFrame:
    """Download adjusted OHLCV from Yahoo Finance and normalize it."""
    import yfinance as yf

    raw = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No data returned for symbol '{symbol}'.")
    raw = raw.reset_index()
    # yfinance can return a MultiIndex on the columns for single symbols.
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    return normalize_frame(raw)


def load_from_csv(file_like) -> pd.DataFrame:
    """Read an uploaded CSV file object and normalize it."""
    return normalize_frame(pd.read_csv(file_like))
