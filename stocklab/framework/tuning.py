"""Per-family hyper-parameter tuning with PSO / GWO / hybrid optimizers.

These drive models 4-12 (the PSO/GWO/hybrid-optimized baselines). Each tuner
runs a population search over the family's bounds, decodes the winner, and
attaches optimizer provenance (method, convergence history, telemetry, validation
RMSE) so the dashboard can show how the solution was found.
"""

from __future__ import annotations

import numpy as np

from ..models.decoders import (
    decode_lstm, decode_elm, decode_gbdt, decode_extra_trees,
    LSTM_BOUNDS, ELM_BOUNDS, GBDT_BOUNDS, EXTRA_TREE_BOUNDS,
)
from ..models.evaluate import eval_lstm, eval_elm, eval_gbdt, eval_extra_trees
from ..optimization import run_optimizer, method_label


def _annotate(params, result, method):
    params["optimizer_method"] = method_label(method)
    params["optimizer_history"] = result["history"]
    params["optimizer_telemetry"] = result.get("telemetry", {})
    params["validation_rmse"] = float(result["fitness"])
    return params


def tune_lstm(dataset, cfg, method, seed=11, n_splits=2):
    def objective(values):
        p = decode_lstm(values)
        p.update({"tune_epochs": 22, "ensemble": 1})
        return eval_lstm(dataset, p, n_splits=n_splits)

    result = run_optimizer(LSTM_BOUNDS, objective, cfg, method, seed=seed)
    params = decode_lstm(result["vector"])
    params.update({"epochs": 45, "ensemble": 1})
    return _annotate(params, result, method)


def tune_elm(dataset, cfg, method, seed=23, n_splits=3):
    result = run_optimizer(
        ELM_BOUNDS, lambda v: eval_elm(dataset, decode_elm(v), n_splits=n_splits),
        cfg, method, seed=seed,
    )
    return _annotate(decode_elm(result["vector"]), result, method)


def tune_gbdt(dataset, cfg, method, seed=17, n_splits=3):
    result = run_optimizer(
        GBDT_BOUNDS, lambda v: eval_gbdt(dataset, decode_gbdt(v), n_splits=n_splits),
        cfg, method, seed=seed,
    )
    return _annotate(decode_gbdt(result["vector"]), result, method)


def tune_extra_trees_quick(dataset, seed=101, n_trials=10, n_splits=3):
    """Lightweight preset + random search for the Extra Trees fusion expert."""
    lb = np.array([b[0] for b in EXTRA_TREE_BOUNDS], float)
    ub = np.array([b[1] for b in EXTRA_TREE_BOUNDS], float)
    rng = np.random.default_rng(seed)
    presets = [
        np.array([220, 8, 2, 0.55], float),
        np.array([320, 10, 2, 0.72], float),
        np.array([420, 12, 1, 0.48], float),
        np.array([280, 14, 4, 0.92], float),
    ]
    best_vec, best_fit, history = None, float("inf"), []
    for t in range(max(len(presets), n_trials)):
        vec = presets[t] if t < len(presets) else lb + rng.random(len(lb)) * (ub - lb)
        fit = eval_extra_trees(dataset, decode_extra_trees(vec), n_splits=n_splits)
        if fit < best_fit:
            best_fit, best_vec = float(fit), vec.copy()
        history.append(best_fit)
    params = decode_extra_trees(best_vec)
    params["optimizer_method"] = "Random Search"
    params["optimizer_history"] = history
    params["optimizer_telemetry"] = {"best": history}
    params["validation_rmse"] = float(best_fit)
    return params
