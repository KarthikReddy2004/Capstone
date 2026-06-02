"""Benchmark runner: trains all 16 models on one shared dataset.

The runner is the single worker function handed to the job manager. It builds a
leak-free dataset once, scales optimizer effort to the configured budget, trains
every model in order while streaming progress/results to Redis, and finally
assembles the leaderboard, runtime/memory benchmark tables, and diagnostics.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from ..config import RunConfig, CORE_FEATURES
from ..reproducibility import seed_everything
from ..data.dataset import build_dataset
from ..optimization import scale_optimizer_cfg
from ..models import run_lstm, run_elm, run_gbdt
from ..framework import tune_lstm, tune_elm, tune_gbdt, tune_extra_trees_quick, run_dual_phase, run_fusion_ensemble
from .diagnostics import compute_diagnostics

# Ordered model registry (display name, group).
MODEL_GROUPS = ["Baseline", "PSO", "GWO", "Hybrid PSO-GWO", "Adaptive Dual-Phase", "Ensemble"]
MODEL_NAMES = [
    "Base LSTM", "Base ELM", "Base GBDT",
    "LSTM + PSO", "ELM + PSO", "GBDT + PSO",
    "LSTM + GWO", "ELM + GWO", "GBDT + GWO",
    "LSTM + PSO-GWO", "ELM + PSO-GWO", "GBDT + PSO-GWO",
    "Adaptive Dual-Phase PSO-GWO + LSTM",
    "Adaptive Dual-Phase PSO-GWO + ELM",
    "Adaptive Dual-Phase PSO-GWO + GBDT",
    "Adaptive Dual-Phase PSO-GWO Expert Fusion Ensemble",
]
_GROUP_OF = (["Baseline"] * 3 + ["PSO"] * 3 + ["GWO"] * 3 + ["Hybrid PSO-GWO"] * 3
             + ["Adaptive Dual-Phase"] * 3 + ["Ensemble"])

# Default (untuned) hyper-parameters for the three baseline models.
_BASE_LSTM = {"n_hidden": 64, "n_layers": 1, "dropout": 0.1, "lr": 1e-3,
              "batch_size": 32, "alpha": 2e-4, "epochs": 45, "ensemble": 1}
_BASE_ELM = {"n_hidden": 160, "alpha": 0.1, "scale": 0.25, "activation": "tanh"}
_BASE_GBDT = {"n_estimators": 160, "lr": 0.05, "max_depth": 3, "min_samples_leaf": 20, "l2": 0.03}

_EFFORT = {"fast": 0.6, "balanced": 1.0, "thorough": 1.6, "max": 3.0}
# (lstm tune evals, flat tune evals, min iterations) per effort tier.
_TUNE_BUDGET = {"fast": (10, 16, 2), "balanced": (18, 28, 4),
                "thorough": (60, 70, 8), "max": (130, 150, 14)}
# Phase-1 stability-voting seeds per effort tier (more seeds -> richer heatmap).
_ADP_SEEDS = {"fast": (29, 43), "balanced": (29, 43),
              "thorough": (29, 43, 59), "max": (29, 43, 59, 67, 83)}


def run_benchmark(ctx, df, cfg_dict):
    """Worker entry point invoked by the job manager."""

    cfg = RunConfig.from_request(cfg_dict)
    seed_everything(cfg.seed)
    ctx.set_stage("Engineering features")
    ctx.log("Building leak-free dataset (next-step target, causal features)")
    dataset = build_dataset(df, cfg, CORE_FEATURES)
    ctx.log(f"Dataset ready: {len(dataset.feature_cols)} features, "
            f"{dataset.n_train} train / {dataset.n_test} test rows")

    base_cfg = cfg.optimizer.as_dict()
    eff = _EFFORT.get(cfg.effort, 1.0)
    # Per-effort tuning budget (lstm evals, flat evals, min iterations). A higher
    # min-iteration count on Thorough yields longer, smoother optimizer
    # convergence curves for the report; Fast stays cheap.
    le, fe, mi = _TUNE_BUDGET.get(cfg.effort, _TUNE_BUDGET["balanced"])
    lstm_tune_cfg = scale_optimizer_cfg(base_cfg, 0.5, 1.0, 5, mi, le)
    flat_tune_cfg = scale_optimizer_cfg(base_cfg, 0.5, 1.0, 5, mi, fe)
    base_cfg = scale_optimizer_cfg(base_cfg, eff, eff, 6, 4, int(round(base_cfg["pop_size"] * base_cfg["max_iter"])))

    tuned: dict = {}

    def get_tuned(family, method):
        key = (family, method)
        if key not in tuned:
            ctx.log(f"Tuning {family.upper()} + {method.upper()}")
            if family == "lstm":
                tuned[key] = tune_lstm(dataset, lstm_tune_cfg, method)
            elif family == "elm":
                tuned[key] = tune_elm(dataset, flat_tune_cfg, method)
            else:
                tuned[key] = tune_gbdt(dataset, flat_tune_cfg, method)
        return tuned[key]

    adaptive: dict = {}

    def make_runner(idx):
        name = MODEL_NAMES[idx]
        if idx == 0:
            return lambda: run_lstm(dataset, {**_BASE_LSTM, "seq_len": dataset.base_seq_len})
        if idx == 1:
            return lambda: run_elm(dataset, dict(_BASE_ELM))
        if idx == 2:
            return lambda: run_gbdt(dataset, dict(_BASE_GBDT))
        # optimized baselines 3..11 (PSO / GWO / hybrid x LSTM / ELM / GBDT)
        if 3 <= idx <= 11:
            fam = ["lstm", "elm", "gbdt"][idx % 3]
            method = ["pso", "gwo", "hybrid"][(idx - 3) // 3]
            runfn = {"lstm": run_lstm, "elm": run_elm, "gbdt": run_gbdt}[fam]
            return lambda: runfn(dataset, dict(get_tuned(fam, method)))
        if idx in (12, 13, 14):
            fam2 = ["lstm", "elm", "gbdt"][idx - 12]
            # Higher effort -> more stability-voting seeds (richer Phase-1
            # heatmaps) and a deeper search budget (longer convergence curves).
            adp_seeds = _ADP_SEEDS.get(cfg.effort, (29, 43))
            def _adaptive():
                res = run_dual_phase(dataset, fam2, base_cfg, ctx=ctx, seeds=adp_seeds, depth=eff)
                adaptive[fam2] = res
                return res
            return _adaptive
        # 15 - flagship: stacked CV fusion over all already-trained models
        # (baselines + PSO/GWO/hybrid + the 3 Adaptive Dual-Phase models).
        def _fusion():
            return run_fusion_ensemble(dataset, list(results), ctx=ctx)
        return _fusion

    results = []
    n = len(MODEL_NAMES)
    for idx, name in enumerate(MODEL_NAMES):
        ctx.set_stage(f"[{idx + 1}/{n}] {name}")
        ctx.set_model_status(name, "running")
        ctx.set_progress(idx / n)
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            result = make_runner(idx)()
            elapsed = time.perf_counter() - t0
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result["name"] = name
            result["index"] = idx
            result["group"] = _GROUP_OF[idx]
            result["timing_sec"] = round(float(elapsed), 3)
            result["memory_mb"] = round(peak / (1024 * 1024), 2)
            results.append(result)
            ctx.push_result(result)
            if hasattr(ctx, "push_summary"):
                ctx.push_summary(_strip_for_leaderboard(result))
            ctx.set_model_status(name, "done")
            ctx.set_progress((idx + 1) / n)
            ctx.log(f"Done {name}: RMSE={result['rmse']:.4f} R2={result['r2']:.4f} "
                    f"DirAcc={result['dir_acc']:.1f}% ({elapsed:.1f}s)")
        except Exception as err:  # noqa: BLE001
            tracemalloc.stop()
            ctx.set_model_status(name, "error")
            ctx.log(f"FAILED {name}: {err}")
            import traceback
            traceback.print_exc()

    ctx.set_stage("Assembling leaderboard")
    final = _build_final(dataset, results, cfg)
    ctx.set_final(final)
    ctx.log(f"Benchmark complete: {len(results)}/{n} models")


def _strip_for_leaderboard(r):
    return {
        "name": r["name"], "index": r["index"], "group": r.get("group", ""),
        "train_mse": r.get("train_mse", 0.0), "test_mse": r.get("test_mse", r.get("mse", 0.0)),
        "mse": r.get("mse", 0.0), "rmse": r["rmse"], "mae": r["mae"],
        "mape": r["mape"], "smape": r["smape"], "r2": r["r2"],
        "theil_u": r.get("theil_u", 0.0), "arv": r.get("arv", 0.0),
        "dir_acc": r["dir_acc"], "hit_ratio": r["hit_ratio"],
        "bias": r["bias"], "tracking_signal": r["tracking_signal"],
        "timing_sec": r.get("timing_sec", 0.0), "memory_mb": r.get("memory_mb", 0.0),
        "n_features": r.get("n_features", 0), "n_selected": r.get("n_selected", 0),
    }


def _build_final(dataset, results, cfg):
    leaderboard = sorted([_strip_for_leaderboard(r) for r in results], key=lambda d: d["rmse"])
    for rank, row in enumerate(leaderboard, 1):
        row["rank"] = rank
    best = leaderboard[0] if leaderboard else None
    ensemble_row = next((r for r in leaderboard if r["index"] == 15), None)
    return {
        "dates_full": list(dataset.dates_full),
        "y_full": np.asarray(dataset.y_target_full, float).tolist(),
        "split_index": int(dataset.n_train),
        "models": results,
        "leaderboard": leaderboard,
        "best_name": best["name"] if best else None,
        "best_rmse": best["rmse"] if best else None,
        "flagship_name": ensemble_row["name"] if ensemble_row else None,
        "flagship_rank": ensemble_row["rank"] if ensemble_row else None,
        "groups": MODEL_GROUPS,
        "diagnostics": compute_diagnostics(dataset),
        "config": cfg.as_dict(),
        "runtime_table": [{"name": r["name"], "timing_sec": r.get("timing_sec", 0.0),
                           "memory_mb": r.get("memory_mb", 0.0),
                           "n_features": r.get("n_features", 0)} for r in results],
    }
