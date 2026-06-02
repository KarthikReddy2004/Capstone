"""Leakage / causality guarantees for the feature + dataset pipeline."""

import numpy as np

from stocklab.config import RunConfig, CORE_FEATURES
from stocklab.data import prepare_features, build_dataset


def test_features_are_causal(synthetic_ohlcv):
    """Mutating the LAST bar must not change any earlier feature row.

    A causal (trailing-window) feature only ever depends on the current and past
    bars, so altering the final close can affect only the final feature row.
    """
    feat = prepare_features(synthetic_ohlcv)
    df2 = synthetic_ohlcv.copy()
    df2.loc[df2.index[-1], "Close"] *= 1.5  # large shock to the last bar
    feat2 = prepare_features(df2)

    cols = [c for c in feat.columns if c not in ("Date",)]
    a = feat[cols].to_numpy(float)[:-1]
    b = feat2[cols].to_numpy(float)[:-1]
    assert np.allclose(a, b, atol=1e-9), "Earlier feature rows changed -> look-ahead leakage"


def test_next_step_target_alignment(synthetic_ohlcv):
    """y[t] must equal close[t+1] and the scalers must be train-only."""
    cfg = RunConfig.from_request({"seqLen": 24, "testPct": 20})
    ds = build_dataset(synthetic_ohlcv, cfg, CORE_FEATURES)

    # target is a return whose reconstruction equals the next close
    recon = ds.reconstruct(ds.prev_test, ds.y_test_scaled)
    assert np.allclose(recon, ds.y_test, rtol=1e-6)

    # training feature scaler has ~zero mean on train but not on test (no leak)
    assert abs(float(ds.x_train.mean())) < 0.2
    assert abs(float(ds.x_test.mean())) > 1e-6


def test_train_test_are_contiguous_in_time(synthetic_ohlcv):
    cfg = RunConfig.from_request({"seqLen": 24, "testPct": 20})
    ds = build_dataset(synthetic_ohlcv, cfg, CORE_FEATURES)
    assert ds.dates_train[-1] < ds.dates_test[0]
    assert ds.n_train + ds.n_test == ds.extras["n_total"]
