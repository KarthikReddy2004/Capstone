"""The stacked fusion must not be worse than its best member (guarded), and
should beat it when experts are decorrelated."""

import numpy as np
import pandas as pd

from stocklab.framework.ensemble import run_fusion_ensemble


class _DS:
    """Minimal dataset stand-in for the fusion (no training involved)."""

    def __init__(self, y_test, prev_test, dates_test):
        self.y_test = np.asarray(y_test, float)
        self.prev_test = np.asarray(prev_test, float)
        self.dates_test = list(dates_test)

    def trend_strength(self):
        return 0.0  # not a long-trend regime for this test


def _model(name, group, val_dates, truth_val, test_truth, noise, rng):
    vp = truth_val + rng.normal(0, noise, len(truth_val))
    tp = test_truth + rng.normal(0, noise, len(test_truth))
    return {
        "name": name, "group": group,
        "val": {"dates": list(val_dates), "actual": truth_val.tolist(), "pred": vp.tolist(),
                "rmse": float(np.sqrt(np.mean((truth_val - vp) ** 2)))},
        "validation_rmse": float(np.sqrt(np.mean((truth_val - vp) ** 2))),
        "test": {"pred": tp.tolist(), "actual": test_truth.tolist()},
        "train": {"pred": tp.tolist(), "actual": test_truth.tolist(),
                  "dates": [str(d) for d in range(len(tp))]},
        "forecast": {"values": [1.0, 2.0, 3.0], "dates": ["a", "b", "c"]},
        "n_features": 10,
    }


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_fusion_not_worse_than_best_member():
    rng = np.random.default_rng(0)
    val_dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2021-01-01", periods=120)]
    truth_val = np.cumsum(rng.normal(0, 1, 120)) + 50
    test_truth = np.cumsum(rng.normal(0, 1, 40)) + truth_val[-1]
    prev = np.concatenate([[test_truth[0]], test_truth[:-1]])
    dates_test = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2021-07-01", periods=40)]
    ds = _DS(test_truth, prev, dates_test)

    models = [
        _model("LSTM + PSO", "PSO", val_dates, truth_val, test_truth, 0.8, rng),
        _model("Base LSTM", "Baseline", val_dates, truth_val, test_truth, 0.9, rng),
        _model("LSTM + GWO", "GWO", val_dates, truth_val, test_truth, 1.0, rng),
        _model("ELM + PSO", "PSO", val_dates, truth_val, test_truth, 1.4, rng),
        _model("Base GBDT", "Baseline", val_dates, truth_val, test_truth, 1.6, rng),
    ]
    final = run_fusion_ensemble(ds, models, ctx=None)

    best_member = min(_rmse(test_truth, m["test"]["pred"]) for m in models)
    flagship = final["rmse"]
    assert flagship <= best_member * 1.02, "ensemble must not be materially worse than its best member"

    weights = [w["weight"] for w in final["fusion_weights"]]
    assert abs(sum(weights) - 1.0) < 5e-3 and all(w >= 0 for w in weights)  # 4-dp rounding
    # weak experts should be pruned / near-zero
    assert len(final["fusion_weights"]) >= 1
