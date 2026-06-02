"""Search telemetry shared by every optimizer.

Captures, per iteration, everything the dashboard's convergence and
search-behaviour charts need:

* best / mean / worst fitness  (convergence envelope)
* population diversity          (exploration signal)
* exploration / exploitation %  (diversity vs. its running max)
* lambda(t)                     (PSO<->GWO mixing, hybrid only)
* coefficient a(t)              (GWO encircling pressure)
* particle/wolf trajectories    (positions of a capped sample, normalised 0-1)
"""

from __future__ import annotations

import numpy as np


def bounds_arrays(bounds):
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    return lb, ub


def spawn(rng, lb, ub, size):
    return lb + rng.random((size, len(lb))) * (ub - lb)


def encircle(rng, current, leader, a):
    """GWO position update toward a leader wolf."""
    a_vec = 2 * a * rng.random(len(current)) - a
    c_vec = 2 * rng.random(len(current))
    return leader - a_vec * np.abs(c_vec * leader - current)


def population_diversity(pop, lb, ub) -> float:
    """Mean distance to the centroid, normalised by the bounds diagonal -> [0,1]."""
    span = np.maximum(ub - lb, 1e-9)
    norm = (pop - lb) / span
    centroid = norm.mean(axis=0)
    dist = np.sqrt(((norm - centroid) ** 2).sum(axis=1))
    return float(np.mean(dist) / (np.sqrt(norm.shape[1]) + 1e-9))


class SearchTelemetry:
    """Accumulates per-iteration optimizer diagnostics."""

    def __init__(self, lb, ub, track_dims=(0, 1), max_particles=12):
        self.lb = np.asarray(lb, float)
        self.ub = np.asarray(ub, float)
        self.dims = [d for d in track_dims if d < len(self.lb)][:2]
        self.max_particles = max_particles
        self.best: list[float] = []
        self.mean: list[float] = []
        self.worst: list[float] = []
        self.diversity: list[float] = []
        self.exploration: list[float] = []
        self.exploitation: list[float] = []
        self.lam: list[float] = []
        self.a: list[float] = []
        self.trajectories: list[list[list[float]]] = []
        self._div_max = 1e-9

    def record(self, pop, fitness, best_fit, lam=None, a=None):
        fitness = np.asarray(fitness, float)
        finite = fitness[np.isfinite(fitness)]
        self.best.append(float(best_fit))
        self.mean.append(float(np.mean(finite)) if finite.size else float(best_fit))
        self.worst.append(float(np.max(finite)) if finite.size else float(best_fit))

        div = population_diversity(pop, self.lb, self.ub)
        self.diversity.append(div)
        self._div_max = max(self._div_max, div)
        expl = float(div / self._div_max)
        self.exploration.append(expl)
        self.exploitation.append(float(1.0 - expl))

        if lam is not None:
            self.lam.append(float(lam))
        if a is not None:
            self.a.append(float(a))

        if self.dims:
            span = np.maximum(self.ub[self.dims] - self.lb[self.dims], 1e-9)
            sample = pop[: self.max_particles][:, self.dims]
            norm = (sample - self.lb[self.dims]) / span
            self.trajectories.append(norm.round(4).tolist())

    def to_dict(self) -> dict:
        return {
            "best": self.best,
            "mean": self.mean,
            "worst": self.worst,
            "diversity": self.diversity,
            "exploration": self.exploration,
            "exploitation": self.exploitation,
            "lam": self.lam,
            "a": self.a,
            "trajectories": self.trajectories,
            "track_dims": self.dims,
        }
