"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest

from stocklab.data import normalize_frame


@pytest.fixture
def synthetic_ohlcv():
    """A reproducible synthetic OHLCV frame with mild drift."""
    rng = np.random.default_rng(42)
    n = 420
    drift = np.linspace(0, 0.4, n)
    ret = rng.normal(0.0005, 0.012, n) + drift / n
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return normalize_frame(pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=n),
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol,
    }))


@pytest.fixture
def redis_available():
    """True if a live Redis is reachable (used to skip integration tests)."""
    try:
        from stocklab.storage import get_storage
        return get_storage().ping()
    except Exception:
        return False
