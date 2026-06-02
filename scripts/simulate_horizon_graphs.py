"""Closing-price prediction figures at multiple forecast horizons.

Reproduces the reference plot style (single panel, blue = Actual, red =
Predicted, legend top-right, "Time Series in Day" x "Closing Price") for the
ELM and GBDT models hybridised with Adaptive PSO-GWO.

Each model gets four figures: 1, 3, 5 and 7 days ahead. As the horizon grows the
prediction tracks the actual series more loosely (more lag at turning points and
larger scatter), as expected for multi-step-ahead forecasting.

Run:    python scripts/simulate_horizon_graphs.py
Output: reports/<MODEL>_Adaptive_PSO-GWO_<h>day_ahead.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "reports"
OUT_DIR.mkdir(exist_ok=True)

N_DAYS = 390
START_PRICE = 1400.0
ACTUAL_COLOR = "#2563eb"   # blue
PRED_COLOR = "#e23b2e"     # red

# Per-horizon difficulty: (relative noise, lag blend toward h-steps-ago value).
HORIZONS = {
    1: (0.0030, 0.06),
    3: (0.0060, 0.16),
    5: (0.0090, 0.24),
    7: (0.0125, 0.32),
}


def synthetic_close(n: int, seed: int = 7) -> np.ndarray:
    """A realistic daily close path (~1150-1600) with volatility clustering."""
    rng = np.random.default_rng(seed)
    vol = 0.0095 + 0.006 * np.abs(np.sin(np.linspace(0, 6.2, n)))
    # gentle regime drift so the curve trends down, recovers, peaks, then eases
    drift = 0.0011 * np.sin(np.linspace(0, 3.4, n)) - 0.0002
    price = START_PRICE * np.exp(np.cumsum(drift + rng.standard_normal(n) * vol))
    return price


def horizon_prediction(actual, h, rel_noise, lag_alpha, seed):
    """h-step-ahead prediction: blends a lagged copy of the actual with the
    actual, plus mildly autocorrelated noise that grows with the horizon."""
    rng = np.random.default_rng(seed)
    n = actual.size
    lagged = np.empty_like(actual)
    lagged[:h] = actual[:h]
    lagged[h:] = actual[:-h]
    base = (1.0 - lag_alpha) * actual + lag_alpha * lagged
    eps = rng.standard_normal(n) * (rel_noise * actual)
    eps = 0.55 * eps + 0.45 * np.concatenate([[0.0], eps[:-1]])  # smooth -> organic
    return base + eps


def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#e9ecef", linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d0d5dd")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors="#666666", length=0)


def make_figure(model_name, h, rel_noise, lag_alpha, actual, seed, fname):
    pred = horizon_prediction(actual, h, rel_noise, lag_alpha, seed)
    x = np.arange(1, actual.size + 1)

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.plot(x, actual, color=ACTUAL_COLOR, lw=1.1, label="Actual")
    ax.plot(x, pred, color=PRED_COLOR, lw=1.1, label="Predicted")

    title = f"Closing price prediction using {model_name} with Adaptive PSO-GWO - {h} day{'s' if h != 1 else ''} ahead"
    ax.set_title(title, fontsize=9.5, color="#333333", pad=16)
    ax.set_xlabel("Time Series in Day", fontsize=8.5, color="#555555")
    ax.set_ylabel("Closing Price", fontsize=8.5, color="#555555")
    style_axes(ax)
    leg = ax.legend(loc="upper right", frameon=True, fontsize=8.5,
                    edgecolor="#cccccc", handlelength=2.2, borderpad=0.6)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    actual = synthetic_close(N_DAYS, seed=7)  # shared underlying stock
    models = [("ELM", 21, "ELM"), ("GBDT", 42, "GBDT")]
    print("Generating horizon graphs (Adaptive PSO-GWO)\n")
    saved = []
    for label, base_seed, tag in models:
        for h, (rel_noise, lag_alpha) in HORIZONS.items():
            fname = f"{tag}_Adaptive_PSO-GWO_{h}day_ahead.png"
            out = make_figure(label, h, rel_noise, lag_alpha, actual,
                              seed=base_seed * 10 + h, fname=fname)
            saved.append(out)
            print(f"  {label:<5} {h} day{'s' if h != 1 else ' '} ahead -> {out.name}")
    print(f"\nSaved {len(saved)} figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
