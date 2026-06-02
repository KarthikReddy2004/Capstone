"""Fast model-runner schema checks and (optional) Redis integration."""

import numpy as np
import pytest

from stocklab.config import RunConfig, CORE_FEATURES
from stocklab.data import build_dataset
from stocklab.models import run_elm, run_gbdt
from stocklab.results import build_result


def _dataset(df):
    return build_dataset(df, RunConfig.from_request({"seqLen": 20, "testPct": 20}), CORE_FEATURES)


def test_elm_runner_schema(synthetic_ohlcv):
    ds = _dataset(synthetic_ohlcv)
    r = run_elm(ds, {"n_hidden": 160, "alpha": 0.1, "scale": 0.25, "activation": "tanh"})
    assert len(r["test"]["pred"]) == len(ds.y_test)
    assert len(r["forecast"]["values"]) == ds.n_future
    for lvl in ("80", "90", "95"):
        assert len(r["forecast"]["intervals"][lvl]["lower"]) == ds.n_future
    assert len(r["forecast"]["points"]) >= 4   # D+1..D+11 spotlights
    assert np.isfinite(r["rmse"]) and r["rmse"] > 0


def test_gbdt_runner_has_importance(synthetic_ohlcv):
    ds = _dataset(synthetic_ohlcv)
    r = run_gbdt(ds, {"n_estimators": 120, "lr": 0.05, "max_depth": 3,
                      "min_samples_leaf": 20, "l2": 0.03})
    assert len(r["feature_importance"]) > 0
    assert "feature" in r["feature_importance"][0]


def test_intervals_are_ordered(synthetic_ohlcv):
    ds = _dataset(synthetic_ohlcv)
    r = run_elm(ds, {"n_hidden": 120, "alpha": 0.1, "scale": 0.25, "activation": "tanh"})
    iv = r["forecast"]["intervals"]
    # wider confidence -> wider band
    w80 = np.array(iv["80"]["upper"]) - np.array(iv["80"]["lower"])
    w95 = np.array(iv["95"]["upper"]) - np.array(iv["95"]["lower"])
    assert np.all(w95 >= w80)


def test_build_result_metrics_flattened():
    y = np.array([10.0, 11, 12, 13])
    p = np.array([10.1, 10.9, 12.2, 12.8])
    res = build_result(y_test=y, pred_test=p, prev_test=np.array([9, 10, 11, 12.0]),
                       dates_test=["a", "b", "c", "d"], y_train=y, pred_train=p,
                       dates_train=["a", "b", "c", "d"])
    for k in ("rmse", "mae", "r2", "dir_acc", "hit_ratio", "tracking_signal"):
        assert k in res


@pytest.mark.parametrize("key", ["job", "data"])
def test_redis_roundtrip(redis_available, key):
    if not redis_available:
        pytest.skip("Redis not reachable")
    from stocklab.storage import get_storage
    s = get_storage()
    s.put_json(f"test:{key}", {"v": 1}, ttl=30)
    assert s.get_json(f"test:{key}") == {"v": 1}
    s.delete(f"test:{key}")
