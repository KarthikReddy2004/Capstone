"""Dataset assembly: split, scale, and package a leak-free supervised problem.

The supervised target is the **next** bar's close: ``y[t] = Close[t+1]`` while
``X[t]`` holds only causal features known at bar ``t``. Normalisation statistics
are fit on the **training portion only** and then applied to validation/test, so
no scaling information leaks backward in time.

Sequence tensors for the LSTM are built on demand (because the sequence length
is itself a tuned hyper-parameter), while the flat learners (ELM / GBDT / trees)
consume the normalised feature matrix directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import prepare_features
from .loaders import RAW_COLS
from ..validation.splitters import holdout_split


@dataclass
class Dataset:
    """Immutable container of everything a model needs for one run."""

    feature_cols: list[str]
    core_idx: list[int]

    # Normalised feature matrices (train then test, contiguous in time).
    x_train: np.ndarray
    x_test: np.ndarray
    x_tune: np.ndarray            # trailing slice of train used for hyper-search

    # Scaled targets (next-step close, standardised by train statistics).
    y_train_scaled: np.ndarray
    y_test_scaled: np.ndarray
    y_tune_scaled: np.ndarray

    # Raw (price-level) targets and anchors for plotting + directional metrics.
    y_train: np.ndarray
    y_test: np.ndarray
    prev_train: np.ndarray        # close at bar t (anchor of the move)
    prev_test: np.ndarray
    y_target_full: np.ndarray     # full next-step close series
    dates_train: list[str]
    dates_test: list[str]
    dates_full: list[str]

    # Scaling parameters. The supervised target is the next-step LOG-RETURN
    # (stationary), and price is reconstructed as prev_close * exp(return).
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: float
    target_scale: float

    # Forecasting support.
    history_raw: pd.DataFrame
    base_seq_len: int
    n_future: int

    # Convenience metadata.
    target_kind: str = "logreturn"
    n_train: int = 0
    n_test: int = 0

    extras: dict = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------- #
    def inverse_target(self, scaled) -> np.ndarray:
        """Map a scaled model output back to a raw next-step log-return."""
        return np.asarray(scaled, float) * self.target_scale + self.target_mean

    def reconstruct(self, prev_close, scaled) -> np.ndarray:
        """Reconstruct the predicted price level from a scaled return prediction.

        ``price_hat[t+1] = close[t] * exp(return_hat)`` where ``close[t]`` is the
        (actual, for one-step) anchor close. Clipping the return guards against
        runaway recursion during multi-step forecasting.
        """
        ret = np.clip(self.inverse_target(scaled), -0.5, 0.5)
        return np.asarray(prev_close, float) * np.exp(ret)

    def feature_subset(self, feature_idx=None) -> np.ndarray:
        if feature_idx is None:
            return np.arange(len(self.feature_cols))
        return np.asarray(feature_idx, dtype=int)

    def trend_strength(self) -> float:
        """Dimensionless drift-to-volatility ratio of the training window.

        Total log-return divided by the random-walk volatility over the window
        (``return_std * sqrt(n)``) — a Sharpe-like signal-to-noise measure. Used
        by the ensemble's champion-preservation rule to detect strongly trending
        regimes where flat predictors look deceptively accurate. Scale-free, so
        it behaves consistently regardless of price level.
        """
        y = np.asarray(self.y_train, float)
        if len(y) < 2 or y[0] <= 0 or y[-1] <= 0:
            return 0.0
        total_logret = abs(float(np.log(y[-1] / y[0])))
        vol = float(self.target_scale) * np.sqrt(len(y))  # target_scale == return std
        return float(total_logret / (vol + 1e-9))


def build_sequences(x: np.ndarray, y: np.ndarray, seq_len: int):
    """Build overlapping ``(window, features)`` sequences with aligned targets."""
    if len(x) <= seq_len:
        return np.empty((0, seq_len, x.shape[1])), np.empty(0)
    windows = np.stack([x[i - seq_len:i] for i in range(seq_len, len(x))])
    return windows, y[seq_len:]


def build_test_sequences(x_train, x_test, y_train, y_test, seq_len):
    """Join the tail of train onto test so the first test window is complete."""
    joined_x = np.vstack([x_train[-seq_len:], x_test])
    joined_y = np.concatenate([y_train[-seq_len:], y_test])
    return build_sequences(joined_x, joined_y, seq_len)


def _history_frame(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    h["Date"] = pd.to_datetime(h["Date"]).dt.strftime("%Y-%m-%d")
    h = h.sort_values("Date").reset_index(drop=True)
    h["Close"] = pd.to_numeric(h["Close"], errors="coerce")
    for col in ("Open", "High", "Low"):
        h[col] = pd.to_numeric(h[col], errors="coerce") if col in h.columns else h["Close"]
    h["Volume"] = pd.to_numeric(h["Volume"], errors="coerce") if "Volume" in h.columns else 1.0
    h = h.dropna(subset=["Close"]).reset_index(drop=True)
    return h[[c for c in RAW_COLS if c in h.columns]].copy()


def build_dataset(df: pd.DataFrame, cfg, core_features=()) -> Dataset:
    """Assemble a :class:`Dataset` from a normalized OHLCV frame and run config."""

    from ..config import MIN_ROWS

    prepared = prepare_features(df)
    feature_cols = [c for c in prepared.columns if c not in ("Date", "Close")]

    x_all = prepared[feature_cols].to_numpy(float)
    close = prepared["Close"].to_numpy(float)
    dates = prepared["Date"].tolist()

    # Next-step target: the LOG-RETURN to the following bar (stationary).
    # Drop the last bar (no future close available).
    x_all = x_all[:-1]
    y_next = close[1:]                                  # price level at t+1 (for plots/metrics)
    prev = close[:-1]                                   # anchor close at bar t
    ret = np.log((close[1:] + 1e-12) / (close[:-1] + 1e-12))  # supervised target
    target_dates = dates[1:]
    n = len(ret)
    if n < MIN_ROWS:
        raise ValueError(
            f"Only {n} usable rows after feature engineering; need >= {MIN_ROWS}. "
            "Use a longer history or a smaller sequence length."
        )

    n_train, n_test = holdout_split(n, cfg.test_pct)
    if n_train <= cfg.seq_len + 40:
        raise ValueError("Sequence length is too large for the available history.")

    # Fit scalers on the training portion only.
    feat_mean = x_all[:n_train].mean(axis=0)
    feat_std = x_all[:n_train].std(axis=0)
    feat_std = np.where(feat_std < 1e-8, 1.0, feat_std)
    x_norm = np.clip((x_all - feat_mean) / feat_std, -6.0, 6.0)

    target_mean = float(ret[:n_train].mean())
    target_scale = float(max(ret[:n_train].std(), 1e-6))
    y_scaled = (ret - target_mean) / target_scale

    tune_rows = min(n_train, max(180, cfg.seq_len * 5))
    tune_start = n_train - tune_rows

    core_idx = [i for i, name in enumerate(feature_cols) if name in set(core_features)]

    return Dataset(
        feature_cols=feature_cols,
        core_idx=core_idx,
        x_train=x_norm[:n_train],
        x_test=x_norm[n_train:],
        x_tune=x_norm[tune_start:n_train],
        y_train_scaled=y_scaled[:n_train],
        y_test_scaled=y_scaled[n_train:],
        y_tune_scaled=y_scaled[tune_start:n_train],
        y_train=y_next[:n_train],
        y_test=y_next[n_train:],
        prev_train=prev[:n_train],
        prev_test=prev[n_train:],
        y_target_full=y_next,
        dates_train=target_dates[:n_train],
        dates_test=target_dates[n_train:],
        dates_full=target_dates,
        feature_mean=feat_mean,
        feature_std=feat_std,
        target_mean=target_mean,
        target_scale=target_scale,
        history_raw=_history_frame(df),
        base_seq_len=cfg.seq_len,
        n_future=cfg.horizon,
        n_train=n_train,
        n_test=n_test,
        extras={"tune_start": tune_start, "n_total": n},
    )
