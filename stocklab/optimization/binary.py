"""Binary Adaptive PSO-GWO for feature selection (Phase 1 of the framework).

Particles live in a continuous space and are mapped to ``{0,1}`` feature masks
through a hyperbolic-tangent transfer function. Core market-structure features
are *locked on* so the selector can never discard them. The same adaptive
``lambda(t)`` schedule as the continuous hybrid governs the PSO/GWO blend.

The objective is supplied by the caller and typically combines a validation
error with a compactness penalty, so the search jointly maximises predictive
quality and minimises feature count.
"""

from __future__ import annotations

import numpy as np

from .telemetry import SearchTelemetry, encircle


def adaptive_binary_hybrid(dim, objective, cfg, seed=42, locked_idx=None):
    rng = np.random.default_rng(seed)
    lb = np.zeros(dim)
    ub = np.ones(dim)
    tele = SearchTelemetry(lb, ub)
    locked = np.array(sorted(locked_idx or []), dtype=int)

    def to_mask(values):
        transfer = np.abs(np.tanh(values))
        mask = (rng.random(dim) < transfer).astype(float)
        if locked.size:
            mask[locked] = 1.0
        return mask

    pop = rng.random((cfg["pop_size"], dim))
    vel = np.zeros_like(pop)
    masks = np.array([to_mask(x) for x in pop])
    p_best = pop.copy()
    p_mask = masks.copy()
    p_fit = np.array([objective(m) for m in masks], float)
    g_idx = int(np.argmin(p_fit))
    g_best, g_mask, g_fit = p_best[g_idx].copy(), p_mask[g_idx].copy(), float(p_fit[g_idx])

    # Per-feature selection frequency across all evaluated masks (stability data).
    select_freq = masks.mean(axis=0)
    n_eval_rounds = 1
    tele.record(pop, p_fit, g_fit, lam=1.0)

    for step in range(cfg["max_iter"]):
        ratio = step / max(1, cfg["max_iter"] - 1)
        lam = float(1 - ratio ** 2)
        a = 2 - 2 * ratio
        order = np.argsort(p_fit)
        alpha, beta, delta = p_best[order[0]], p_best[order[1]], p_best[order[2]]

        fit = np.empty(len(pop), float)
        round_masks = np.empty_like(masks)
        for i in range(len(pop)):
            r1, r2 = rng.random(dim), rng.random(dim)
            vel[i] = cfg["w"] * vel[i] + cfg["c1"] * r1 * (p_best[i] - pop[i]) + cfg["c2"] * r2 * (g_best - pop[i])
            pso_pos = pop[i] + vel[i]
            gwo_pos = (encircle(rng, pop[i], alpha, a)
                       + encircle(rng, pop[i], beta, a)
                       + encircle(rng, pop[i], delta, a)) / 3
            pop[i] = np.clip(lam * pso_pos + (1 - lam) * gwo_pos, 0.0, 1.0)
            mask = to_mask(pop[i])
            round_masks[i] = mask
            f = float(objective(mask))
            fit[i] = f
            if f < p_fit[i]:
                p_best[i], p_mask[i], p_fit[i] = pop[i].copy(), mask, f
                if f < g_fit:
                    g_fit, g_best, g_mask = f, pop[i].copy(), mask.copy()
        select_freq += round_masks.mean(axis=0)
        n_eval_rounds += 1
        tele.record(pop, fit, g_fit, lam=lam)

    return {
        "vector": g_best,
        "binary": g_mask,
        "fitness": g_fit,
        "history": tele.best,
        "telemetry": tele.to_dict(),
        "lam_curve": tele.lam,
        "select_freq": (select_freq / n_eval_rounds).tolist(),
    }
