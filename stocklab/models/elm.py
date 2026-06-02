"""Extreme Learning Machine (single hidden layer, closed-form ridge solution).

A random projection into a non-linear hidden space followed by a regularised
linear read-out solved in closed form. Training is a single pseudo-inverse, so
ELMs are extremely fast and make excellent diverse members of the fusion pool.
The activation, hidden width, and ridge coefficient are all tunable.
"""

from __future__ import annotations

import numpy as np

_ACTIVATIONS = ("tanh", "sigmoid", "relu")


def _activate(z, name):
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    if name == "relu":
        return np.maximum(0.0, z)
    return np.tanh(z)


class ELM:
    def __init__(self, params: dict, seed: int = 42):
        self.p = params
        self.seed = seed
        self.weights = None
        self.bias = None
        self.beta = None
        self.activation = params.get("activation", "tanh")

    def fit(self, x: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        n_hidden = int(self.p.get("n_hidden", 160))
        scale = float(self.p.get("scale", 0.22))
        alpha = float(self.p.get("alpha", 0.05))
        n_features = x.shape[1]
        self.weights = rng.normal(0.0, scale, size=(n_features, n_hidden))
        self.bias = rng.normal(0.0, scale, size=(n_hidden,))
        h = _activate(x @ self.weights + self.bias, self.activation)
        ridge = alpha * np.eye(n_hidden)
        self.beta = np.linalg.pinv(h.T @ h + ridge) @ h.T @ np.asarray(y, float)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = _activate(x @ self.weights + self.bias, self.activation)
        return h @ self.beta
