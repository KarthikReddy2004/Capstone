"""The Adaptive Dual-Phase PSO-GWO framework (models 13-15).

Phase 1 - Binary Adaptive PSO-GWO feature selection
    * multi-seed search with stability voting
    * core market-structure features are protected (never dropped)
    * adaptive ``lambda(t) = 1 - (t/T)^2`` mixing
    * objective jointly rewards validation score and penalises feature count

Phase 2 - Continuous Adaptive PSO-GWO hyper-parameter optimisation
    * adaptive hybrid search on the *selected* feature subset
    * multi-seed, then a local refinement stage on the best vectors
    * walk-forward / temporal CV throughout

The same machinery serves all three learner families (LSTM / ELM / GBDT); only
the bounds, decoder, evaluator, runner, and probe parameters differ.
"""

from __future__ import annotations

import numpy as np

from ..models.decoders import (
    decode_lstm, decode_elm, decode_gbdt,
    LSTM_BOUNDS, ELM_BOUNDS, GBDT_BOUNDS,
)
from ..models.evaluate import eval_lstm, eval_elm, eval_gbdt
from ..models.experts import run_lstm, run_elm, run_gbdt
from ..optimization import adaptive_binary_hybrid, hybrid_pso_gwo_optimize, local_refine, scale_optimizer_cfg


def _family_spec(family, dataset):
    if family == "lstm":
        probe = {"n_hidden": 64, "n_layers": 1, "dropout": 0.1, "lr": 1e-3,
                 "batch_size": 32, "seq_len": dataset.base_seq_len, "alpha": 2e-4,
                 "tune_epochs": 16, "ensemble": 1}
        final = {"epochs": 55, "ensemble": 2}
        return dict(bounds=LSTM_BOUNDS, decode=decode_lstm, evaluate=eval_lstm,
                    run=run_lstm, probe=probe, final=final, label="LSTM")
    if family == "elm":
        probe = {"n_hidden": 180, "alpha": 0.1, "scale": 0.3, "activation": "tanh"}
        return dict(bounds=ELM_BOUNDS, decode=decode_elm, evaluate=eval_elm,
                    run=run_elm, probe=probe, final={}, label="ELM")
    probe = {"n_estimators": 160, "lr": 0.06, "max_depth": 3, "min_samples_leaf": 20, "l2": 0.05}
    return dict(bounds=GBDT_BOUNDS, decode=decode_gbdt, evaluate=eval_gbdt,
                run=run_gbdt, probe=probe, final={}, label="GBDT")


def _stability_scores(runs):
    """Fitness-weighted average per-feature selection frequency across seeds."""
    weights = np.array([1.0 / max(float(r["fitness"]), 1e-6) for r in runs], float)
    weights /= weights.sum()
    score = np.zeros(len(runs[0]["select_freq"]), float)
    for w, r in zip(weights, runs):
        score += w * np.asarray(r["select_freq"], float)
    return score


def _select_stable(stability, core_idx, total, min_keep, max_keep):
    stability = np.asarray(stability, float)
    core_idx = np.asarray(core_idx, int)
    selected = np.where(stability >= 0.5)[0]
    selected = np.unique(np.concatenate([selected, core_idx])) if core_idx.size else np.unique(selected)
    if len(selected) < min_keep:
        selected = np.argsort(stability)[-min_keep:]
    if len(selected) > max_keep:
        selected = np.argsort(stability)[-max_keep:]
    if core_idx.size:
        selected = np.unique(np.concatenate([selected, core_idx]))
    return np.sort(selected.astype(int))


def run_dual_phase(dataset, family, base_cfg, ctx=None, seeds=(29, 43), n_splits=2, depth=1.0):
    """Execute both phases for one family and return a packaged result."""

    spec = _family_spec(family, dataset)
    feature_cols = dataset.feature_cols
    core_idx = dataset.core_idx
    total = len(feature_cols)

    def log(msg):
        if ctx:
            ctx.log(msg)

    # ---- Phase 1: binary feature selection -------------------------------- #
    log(f"Adaptive Dual-Phase [{spec['label']}] - Phase 1 feature selection")
    # depth scales the evaluation budget (and thus convergence-curve length). A
    # capped, modest population keeps most of that budget for iterations, giving
    # long, smooth convergence curves for the report at high effort.
    search_cfg = dict(base_cfg)
    search_cfg["pop_size"] = min(int(base_cfg["pop_size"]), 20)
    search_cfg["max_iter"] = 200  # let the eval budget (below) drive iteration count
    p1_cfg = scale_optimizer_cfg(search_cfg, pop_scale=0.5, iter_scale=1.0, min_pop=10, min_iter=6,
                                 max_eval=int(60 * depth))
    cache: dict = {}

    def phase1_objective(mask):
        key = tuple(int(v) for v in (np.asarray(mask) > 0.5))
        if key in cache:
            return cache[key]
        selected = np.where(np.asarray(mask) > 0.5)[0]
        selected = np.unique(np.concatenate([selected, core_idx])) if core_idx else np.unique(selected)
        if len(selected) < max(8, len(core_idx)):
            return 1e9
        penalty = 0.009 * max(0, len(selected) - max(len(core_idx), 16))
        fit = spec["evaluate"](dataset, spec["probe"], selected, n_splits=n_splits) + penalty
        cache[key] = fit
        return fit

    p1_runs = []
    for s in seeds:
        r = adaptive_binary_hybrid(total, phase1_objective, p1_cfg, seed=s, locked_idx=core_idx)
        r["seed"] = s
        p1_runs.append(r)
        log(f"  Phase 1 seed={s} best={r['fitness']:.4f}")
    stability = _stability_scores(p1_runs)
    min_keep = max(16, len(core_idx) + 6)
    max_keep = min(total, max(40, len(core_idx) + 24))
    selected_idx = _select_stable(stability, core_idx, total, min_keep, max_keep)
    best_p1 = min(p1_runs, key=lambda r: float(r["fitness"]))

    # Full-feature fallback: selection must EARN its place. Only keep the reduced
    # subset if it genuinely beats using all features on validation; otherwise
    # use the full set. This guarantees the adaptive model is never handicapped
    # by dropping informative features (the failure mode on single trending
    # stocks where more features help).
    all_idx = np.arange(total)
    sel_score = spec["evaluate"](dataset, spec["probe"], selected_idx, n_splits=n_splits)
    full_score = spec["evaluate"](dataset, spec["probe"], all_idx, n_splits=n_splits)
    if full_score <= sel_score * 1.005:
        selected_idx = all_idx
        log(f"  Phase 1 kept all {total} features (selection gave no validation gain)")
    else:
        gain = (1.0 - sel_score / max(full_score, 1e-9)) * 100.0
        log(f"  Phase 1 selected {len(selected_idx)}/{total} features (+{gain:.1f}% val gain)")
    selected_names = [feature_cols[i] for i in selected_idx]

    # ---- Phase 2: continuous adaptive hyper-parameter search -------------- #
    log(f"Adaptive Dual-Phase [{spec['label']}] - Phase 2 hyper-parameter search")
    p2_cfg = scale_optimizer_cfg(search_cfg, pop_scale=0.5, iter_scale=1.0, min_pop=10, min_iter=7,
                                 max_eval=int(80 * depth))
    p2_cache: dict = {}

    def phase2_objective(values):
        p = spec["decode"](values)
        key = tuple(np.round(np.asarray(values, float), 5).tolist())
        if key in p2_cache:
            return p2_cache[key]
        search_p = dict(p)
        if family == "lstm":
            search_p.update({"tune_epochs": 20, "ensemble": 1})
        fit = spec["evaluate"](dataset, search_p, selected_idx, n_splits=n_splits)
        p2_cache[key] = fit
        return fit

    p2_runs = []
    for s in tuple(int(x) + 2 for x in seeds):
        r = hybrid_pso_gwo_optimize(spec["bounds"], phase2_objective, p2_cfg, seed=s, adaptive=True)
        r["seed"] = s
        p2_runs.append(r)
        log(f"  Phase 2 seed={s} best={r['fitness']:.4f}")
    p2_runs.sort(key=lambda r: float(r["fitness"]))
    for rank, r in enumerate(p2_runs[:2]):
        refined = local_refine(spec["bounds"], r["vector"], phase2_objective,
                               rounds=int(14 * depth), seed=73 + rank * 19)
        if refined["fitness"] < r["fitness"]:
            r["vector"], r["fitness"] = refined["vector"], refined["fitness"]
            log(f"  Phase 2 local refine improved rank {rank + 1} -> {refined['fitness']:.4f}")
    p2_runs.sort(key=lambda r: float(r["fitness"]))
    best_p2 = p2_runs[0]

    params = spec["decode"](best_p2["vector"])
    params.update(spec["final"])
    params["feature_idx"] = list(map(int, selected_idx))
    params["optimizer_method"] = "Adaptive Dual-Phase PSO-GWO"
    params["optimizer_history"] = best_p2["history"]
    params["optimizer_telemetry"] = best_p2.get("telemetry", {})
    params["validation_rmse"] = float(best_p2["fitness"])

    # ---- Build the family's adaptive model on selected features ----------- #
    result = spec["run"](dataset, params, feature_idx=selected_idx, n_splits=3)
    result["selected_features"] = selected_names
    result["n_selected"] = len(selected_idx)
    result["phase1"] = {
        "history": best_p1["history"],
        "telemetry": best_p1.get("telemetry", {}),
        "lam_curve": best_p1.get("lam_curve", []),
        "stability": stability.round(4).tolist(),
        "feature_names": feature_cols,
        "selected": list(map(int, selected_idx)),
        "seeds": list(seeds),
        "select_freq_by_seed": [np.asarray(r["select_freq"], float).round(4).tolist() for r in p1_runs],
    }
    result["phase2"] = {
        "history": best_p2["history"],
        "telemetry": best_p2.get("telemetry", {}),
        "lam_curve": best_p2.get("lam_curve", []),
        "seeds": [int(r["seed"]) for r in p2_runs],
    }
    result["optimizer"] = {
        "method": "Adaptive Dual-Phase PSO-GWO",
        "history": best_p2["history"],
        "telemetry": best_p2.get("telemetry", {}),
    }
    # Expose the tuned spec so the fusion ensemble can reuse it.
    result["_tuned_params"] = params
    result["_family"] = family
    result["_selected_idx"] = list(map(int, selected_idx))
    return result
