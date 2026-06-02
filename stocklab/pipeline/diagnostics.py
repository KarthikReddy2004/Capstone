"""Dataset-level diagnostics for the dashboard (correlation heatmaps)."""

from __future__ import annotations

import numpy as np


def compute_diagnostics(dataset, max_features: int = 24) -> dict:
    """Feature/feature and feature/target correlation on the training window.

    Computed only on the training portion to stay consistent with the leak-free
    discipline used everywhere else. Limited to a representative subset (core
    features plus the highest-variance remainder) to keep the heatmap legible.
    """

    cols = dataset.feature_cols
    x = np.asarray(dataset.x_train, float)
    y = np.asarray(dataset.y_train_scaled, float)

    core = list(dataset.core_idx)
    variance = x.var(axis=0)
    order = [i for i in np.argsort(variance)[::-1] if i not in core]
    chosen = (core + order)[:max_features]
    chosen = sorted(set(chosen), key=lambda i: (i not in core, cols[i]))[:max_features]

    sub = x[:, chosen]
    names = [cols[i] for i in chosen]
    with np.errstate(invalid="ignore"):
        corr = np.corrcoef(sub, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)

    target_corr = []
    for i in chosen:
        col = x[:, i]
        denom = np.std(col) * np.std(y)
        c = float(np.mean((col - col.mean()) * (y - y.mean())) / (denom + 1e-9)) if denom > 1e-12 else 0.0
        target_corr.append({"feature": cols[i], "corr": round(c, 4)})
    target_corr.sort(key=lambda d: abs(d["corr"]), reverse=True)

    return {
        "feature_names": names,
        "matrix": np.round(corr, 4).tolist(),
        "target_corr": target_corr,
    }
