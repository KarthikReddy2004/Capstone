"""Continuous swarm optimizers: PSO, GWO, and the adaptive PSO-GWO hybrid.

All three minimise ``objective(vector) -> float`` over box ``bounds`` and return
a uniform result dict::

    {"vector", "fitness", "history", "telemetry"}

``history`` keeps the classic best-fitness-per-iteration curve (for quick
plotting / backward compatibility) while ``telemetry`` holds the richer
diagnostics produced by :class:`SearchTelemetry`.
"""

from __future__ import annotations

import numpy as np

from .telemetry import SearchTelemetry, bounds_arrays, spawn, encircle


def pso_optimize(bounds, objective, cfg, seed=42):
    lb, ub = bounds_arrays(bounds)
    rng = np.random.default_rng(seed)
    tele = SearchTelemetry(lb, ub)

    pop = spawn(rng, lb, ub, cfg["pop_size"])
    vel = np.zeros_like(pop)
    p_best = pop.copy()
    p_fit = np.array([objective(x) for x in pop], float)
    g_idx = int(np.argmin(p_fit))
    g_best, g_fit = p_best[g_idx].copy(), float(p_fit[g_idx])
    tele.record(pop, p_fit, g_fit)

    for step in range(cfg["max_iter"]):
        ratio = step / max(1, cfg["max_iter"] - 1)
        inertia = float(cfg["w"] * (1 - ratio) + 0.38 * ratio)  # linearly decaying inertia
        r1, r2 = rng.random(pop.shape), rng.random(pop.shape)
        vel = inertia * vel + cfg["c1"] * r1 * (p_best - pop) + cfg["c2"] * r2 * (g_best - pop)
        pop = np.clip(pop + vel, lb, ub)
        fit = np.array([objective(x) for x in pop], float)
        improved = fit < p_fit
        if np.any(improved):
            p_best[improved] = pop[improved]
            p_fit[improved] = fit[improved]
            g_idx = int(np.argmin(p_fit))
            if p_fit[g_idx] < g_fit:
                g_fit, g_best = float(p_fit[g_idx]), p_best[g_idx].copy()
        tele.record(pop, fit, g_fit)

    return {"vector": g_best, "fitness": g_fit, "history": tele.best, "telemetry": tele.to_dict()}


def gwo_optimize(bounds, objective, cfg, seed=42):
    lb, ub = bounds_arrays(bounds)
    rng = np.random.default_rng(seed)
    tele = SearchTelemetry(lb, ub)

    wolves = spawn(rng, lb, ub, cfg["pop_size"])
    fit = np.array([objective(x) for x in wolves], float)
    order = np.argsort(fit)
    alpha, beta, delta = wolves[order[0]].copy(), wolves[order[1]].copy(), wolves[order[2]].copy()
    alpha_fit = float(fit[order[0]])
    tele.record(wolves, fit, alpha_fit, a=2.0)

    for step in range(cfg["max_iter"]):
        a = 2 - 2 * step / max(1, cfg["max_iter"] - 1)  # encircling pressure 2 -> 0
        for i in range(len(wolves)):
            x1 = encircle(rng, wolves[i], alpha, a)
            x2 = encircle(rng, wolves[i], beta, a)
            x3 = encircle(rng, wolves[i], delta, a)
            wolves[i] = np.clip((x1 + x2 + x3) / 3, lb, ub)
        fit = np.array([objective(x) for x in wolves], float)
        order = np.argsort(fit)
        alpha, beta, delta = wolves[order[0]].copy(), wolves[order[1]].copy(), wolves[order[2]].copy()
        alpha_fit = float(fit[order[0]])
        tele.record(wolves, fit, alpha_fit, a=a)

    return {"vector": alpha, "fitness": alpha_fit, "history": tele.best, "telemetry": tele.to_dict()}


def hybrid_pso_gwo_optimize(bounds, objective, cfg, seed=42, adaptive=True):
    """Hybrid PSO-GWO with adaptive mixing ``lambda(t) = 1 - (t/T)^2``.

    Each particle's next position blends a PSO velocity step with a GWO
    leader-encircling step: ``x = lam * pso + (1 - lam) * gwo``. With
    ``adaptive=True`` the schedule starts PSO-heavy (broad exploration) and ends
    GWO-heavy (sharp exploitation); with ``adaptive=False`` lambda is fixed at
    0.5 (a plain 50/50 hybrid baseline).
    """

    lb, ub = bounds_arrays(bounds)
    rng = np.random.default_rng(seed)
    tele = SearchTelemetry(lb, ub)

    pop = spawn(rng, lb, ub, cfg["pop_size"])
    vel = np.zeros_like(pop)
    p_best = pop.copy()
    p_fit = np.array([objective(x) for x in pop], float)
    g_idx = int(np.argmin(p_fit))
    g_best, g_fit = p_best[g_idx].copy(), float(p_fit[g_idx])
    tele.record(pop, p_fit, g_fit, lam=1.0, a=2.0)

    for step in range(cfg["max_iter"]):
        ratio = step / max(1, cfg["max_iter"] - 1)
        lam = float(1 - ratio ** 2) if adaptive else 0.5
        a = 2 - 2 * ratio
        order = np.argsort(p_fit)
        alpha, beta, delta = p_best[order[0]], p_best[order[1]], p_best[order[2]]

        fit = np.empty(len(pop), float)
        for i in range(len(pop)):
            r1, r2 = rng.random(pop.shape[1]), rng.random(pop.shape[1])
            vel[i] = cfg["w"] * vel[i] + cfg["c1"] * r1 * (p_best[i] - pop[i]) + cfg["c2"] * r2 * (g_best - pop[i])
            pso_pos = pop[i] + vel[i]
            gwo_pos = (encircle(rng, pop[i], alpha, a)
                       + encircle(rng, pop[i], beta, a)
                       + encircle(rng, pop[i], delta, a)) / 3
            pop[i] = np.clip(lam * pso_pos + (1 - lam) * gwo_pos, lb, ub)
            f = float(objective(pop[i]))
            fit[i] = f
            if f < p_fit[i]:
                p_best[i], p_fit[i] = pop[i].copy(), f
                if f < g_fit:
                    g_fit, g_best = f, pop[i].copy()
        tele.record(pop, fit, g_fit, lam=lam, a=a)

    out = {"vector": g_best, "fitness": g_fit, "history": tele.best, "telemetry": tele.to_dict()}
    out["lam_curve"] = tele.lam
    return out
