"""Gradient-boosted and randomised-tree learners (scikit-learn).

* ``GBDT``        -> :class:`HistGradientBoostingRegressor` (fast, binned boosting)
* ``ExtraTrees``  -> :class:`ExtraTreesRegressor` (low-variance randomised forest)

Both expose a uniform ``fit``/``predict`` plus a ``feature_importance`` helper.
HistGB has no native importances, so we use a cheap permutation importance on a
held-out slice; ExtraTrees uses its impurity-based importances directly.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance


class GBDT:
    def __init__(self, params: dict, seed: int = 42):
        self.p = params
        self.seed = seed
        self.model = None

    def fit(self, x, y):
        self.model = HistGradientBoostingRegressor(
            learning_rate=float(self.p.get("lr", 0.05)),
            max_depth=int(self.p.get("max_depth", 3)),
            max_iter=int(self.p.get("n_estimators", 160)),
            min_samples_leaf=int(self.p.get("min_samples_leaf", 20)),
            l2_regularization=float(self.p.get("l2", 0.03)),
            early_stopping=False,
            random_state=self.seed,
        )
        self.model.fit(x, np.asarray(y, float))
        return self

    def predict(self, x):
        return self.model.predict(x)

    def feature_importance(self, x, y, feature_names, max_samples=200):
        x = np.asarray(x, float)
        if len(x) > max_samples:
            x, y = x[-max_samples:], np.asarray(y, float)[-max_samples:]
        try:
            r = permutation_importance(self.model, x, y, n_repeats=3,
                                       random_state=self.seed, n_jobs=1)
            imp = r.importances_mean
        except Exception:
            imp = np.zeros(x.shape[1])
        return _rank_importance(imp, feature_names)


class ExtraTrees:
    def __init__(self, params: dict, seed: int = 42):
        self.p = params
        self.seed = seed
        self.model = None

    def fit(self, x, y):
        self.model = ExtraTreesRegressor(
            n_estimators=int(self.p.get("n_estimators", 260)),
            max_depth=int(self.p.get("max_depth", 12)),
            min_samples_leaf=int(self.p.get("min_samples_leaf", 2)),
            max_features=float(self.p.get("max_features", 0.72)),
            bootstrap=False,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model.fit(x, np.asarray(y, float))
        return self

    def predict(self, x):
        return self.model.predict(x)

    def feature_importance(self, x, y, feature_names, max_samples=None):
        return _rank_importance(self.model.feature_importances_, feature_names)


def _rank_importance(importances, feature_names, top=20):
    importances = np.asarray(importances, float)
    importances = np.where(np.isfinite(importances), importances, 0.0)
    order = np.argsort(importances)[::-1][:top]
    return [
        {"feature": str(feature_names[i]), "importance": float(importances[i])}
        for i in order
    ]
