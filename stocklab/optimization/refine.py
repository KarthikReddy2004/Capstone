"""Optimizer dispatch, effort scaling, and a local refinement stage."""

from __future__ import annotations

import numpy as np

from .swarm import pso_optimize, gwo_optimize, hybrid_pso_gwo_optimize


def method_label(method: str) -> str:
    return {"pso": "PSO", "gwo": "GWO", "hybrid": "PSO-GWO"}.get(method, str(method).upper())


def run_optimizer(bounds, objective, cfg, method, seed):
    """Dispatch to the requested optimizer by short name."""
    if method == "pso":
        return pso_optimize(bounds, objective, cfg, seed=seed)
    if method == "gwo":
        return gwo_optimize(bounds, objective, cfg, seed=seed)
    # The per-family hybrid baselines use a fixed lambda=0.5; the Adaptive
    # Dual-Phase framework uses adaptive=True explicitly.
    return hybrid_pso_gwo_optimize(bounds, objective, cfg, seed=seed, adaptive=False)


def scale_optimizer_cfg(cfg, pop_scale=1.0, iter_scale=1.0, min_pop=6, min_iter=4, max_eval=64):
    """Down-scale optimizer effort to bound the number of objective evaluations.

    Objective calls (model fits) dominate runtime, so we cap
    ``pop_size * max_iter`` at ``max_eval`` to keep every search tractable on the
    target CPU while preserving the optimizer's qualitative behaviour.
    """

    pop = max(min_pop, int(round(cfg["pop_size"] * pop_scale)))
    it = max(min_iter, int(round(cfg["max_iter"] * iter_scale)))
    if pop * it > max_eval:
        it = max(min_iter, max_eval // pop)
    return {
        "pop_size": max(4, pop),
        "max_iter": max(2, it),
        "c1": float(cfg["c1"]),
        "c2": float(cfg["c2"]),
        "w": float(cfg["w"]),
    }


def local_refine(bounds, start_vector, objective, rounds=20, seed=73):
    """Gaussian random-walk refinement around a promising solution.

    Shrinks the perturbation scale over ``rounds`` to polish the best vector
    found by the population search (Phase-2 exploitation step).
    """

    lb = np.array([b[0] for b in bounds], float)
    ub = np.array([b[1] for b in bounds], float)
    span = ub - lb
    rng = np.random.default_rng(seed)
    best = np.clip(np.asarray(start_vector, float), lb, ub)
    best_fit = float(objective(best))
    history = [best_fit]
    for step in range(max(1, int(rounds))):
        scale = 0.14 * (1 - step / max(1, rounds - 1)) + 0.02
        trial = np.clip(best + rng.normal(0, scale, len(best)) * span, lb, ub)
        f = float(objective(trial))
        if f < best_fit:
            best_fit, best = f, trial
        history.append(best_fit)
    return {"vector": best, "fitness": best_fit, "history": history}
