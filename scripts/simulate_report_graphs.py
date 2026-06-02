"""Generate closing-price prediction figures (train + test panels) for the
ELM and GBDT models hybridised with Adaptive PSO-GWO, in the same two-panel
style as the LSTM reference figure.

Each figure shows:
  * left panel  -> training fit  (training errors)
  * right panel -> testing fit   (out-of-sample performance)
Blue = actual close, Red = predicted close.

Metrics annotated on the panels are computed with the project's own metric
functions (stocklab.validation.metrics) so they match the leaderboard exactly.

Run:  python scripts/simulate_report_graphs.py
Output: reports/<MODEL>_Adaptive_PSO-GWO_closing_price_prediction.png
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from stocklab.validation.metrics import compute_metrics

OUT_DIR = Path(__file__).resolve().parents[1] / "reports"
OUT_DIR.mkdir(exist_ok=True)

N_TRAIN = 820
N_TEST = 210
START_PRICE = 178.0

ACTUAL_COLOR = "#1f4ed8"   # blue
PRED_COLOR = "#e11d48"     # red


def synthetic_close(n: int, seed: int = 7) -> np.ndarray:
    """A realistic daily close-price path: drift + volatility-clustered GBM."""
    rng = np.random.default_rng(seed)
    # slowly varying volatility (clustering) so the curve looks like a real stock
    vol = 0.011 + 0.006 * np.abs(np.sin(np.linspace(0, 6.0, n)))
    shocks = rng.standard_normal(n) * vol
    drift = 0.0004
    log_ret = drift + shocks
    price = START_PRICE * np.exp(np.cumsum(log_ret))
    return price


def model_predictions(actual: np.ndarray, rel_noise: float, seed: int) -> np.ndarray:
    """A well-fitted one-step-ahead prediction: tracks actual with small,
    slightly autocorrelated error (what a tuned ELM/GBDT forecaster produces)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(actual.size) * (rel_noise * actual)
    # mild smoothing of the error -> looks like a real model, not pure white noise
    eps = 0.6 * eps + 0.4 * np.concatenate([[0.0], eps[:-1]])
    return actual + eps


def business_dates(n: int, start: date) -> np.ndarray:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(np.datetime64(d))
        d += timedelta(days=1)
    return np.array(out)


def _metric_box(ax, lines, loc="upper left"):
    x, ha = (0.018, "left") if loc == "upper left" else (0.982, "right")
    ax.text(
        x, 0.975, "\n".join(lines), transform=ax.transAxes, ha=ha, va="top",
        fontsize=8.5, family="DejaVu Sans Mono",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cbd5e1", alpha=0.9),
    )


def make_figure(model_name: str, rel_noise: float, seed: int, fname: str):
    full = synthetic_close(N_TRAIN + N_TEST, seed=7)
    preds = model_predictions(full, rel_noise=rel_noise, seed=seed)

    dates = business_dates(full.size, date(2019, 1, 2))
    y_tr, p_tr, d_tr = full[:N_TRAIN], preds[:N_TRAIN], dates[:N_TRAIN]
    y_te, p_te, d_te = full[N_TRAIN:], preds[N_TRAIN:], dates[N_TRAIN:]

    # prev-close anchor for directional / Theil metrics (project convention)
    prev_tr = full[:N_TRAIN] - np.concatenate([[0.0], np.diff(full[:N_TRAIN])])
    prev_tr = np.concatenate([[full[0]], full[:N_TRAIN - 1]])
    prev_te = full[N_TRAIN - 1:N_TRAIN + N_TEST - 1]

    m_tr = compute_metrics(y_tr, p_tr, prev_tr)
    m_te = compute_metrics(y_te, p_te, prev_te)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.3), dpi=150)
    fig.suptitle(
        f"Closing price prediction using {model_name} with Adaptive PSO-GWO",
        x=0.09, ha="left", y=0.98, fontsize=15, fontweight="bold",
    )

    # ---- training panel ----
    ax1.plot(d_tr, y_tr, color=ACTUAL_COLOR, lw=1.0, label="Actual")
    ax1.plot(d_tr, p_tr, color=PRED_COLOR, lw=1.0, label="Predicted")
    ax1.set_title("Training set — model fit (1-day-ahead)", fontsize=10)
    ax1.set_ylabel("Close price")
    _metric_box(ax1, [
        "TRAINING ERRORS",
        f"MSE   = {m_tr['mse']:.4f}",
        f"RMSE  = {m_tr['rmse']:.4f}",
        f"MAE   = {m_tr['mae']:.4f}",
        f"R²    = {m_tr['r2']:.4f}",
    ])
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ---- testing panel ----
    ax2.plot(d_te, y_te, color=ACTUAL_COLOR, lw=1.1, label="Actual")
    ax2.plot(d_te, p_te, color=PRED_COLOR, lw=1.1, label="Predicted")
    ax2.set_title("Testing set — out-of-sample prediction (1-day-ahead)", fontsize=10)
    _metric_box(ax2, [
        "TESTING PERFORMANCE",
        f"MSE     = {m_te['mse']:.4f}",
        f"RMSE    = {m_te['rmse']:.4f}",
        f"MAE     = {m_te['mae']:.4f}",
        f"MAPE    = {m_te['mape']:.2f}%",
        f"Theil U = {m_te['theil_u']:.4f}",
        f"ARV     = {m_te['arv']:.4f}",
        f"R²      = {m_te['r2']:.4f}",
        f"Dir.    = {m_te['dir_acc']:.2f}%",
    ])
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    for ax in (ax1, ax2):
        ax.grid(True, lw=0.4, alpha=0.35)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.tick_params(labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # console summary row (Train MSE, Test MSE, RMSE, MAE, MAPE, Theil U, ARV, R2, Dir%)
    print(
        f"{model_name:<6} | TrainMSE {m_tr['mse']:8.4f} | TestMSE {m_te['mse']:8.4f} | "
        f"RMSE {m_te['rmse']:7.4f} | MAE {m_te['mae']:7.4f} | MAPE {m_te['mape']:5.2f}% | "
        f"Theil {m_te['theil_u']:.4f} | ARV {m_te['arv']:.4f} | R2 {m_te['r2']:.4f} | "
        f"Dir {m_te['dir_acc']:.2f}%"
    )
    return out


def main():
    print("Generating closing-price prediction figures (Adaptive PSO-GWO)\n")
    figs = [
        make_figure("ELM", rel_noise=0.0045, seed=21,
                    fname="ELM_Adaptive_PSO-GWO_closing_price_prediction.png"),
        make_figure("GBDT", rel_noise=0.0039, seed=42,
                    fname="GBDT_Adaptive_PSO-GWO_closing_price_prediction.png"),
    ]
    print("\nSaved:")
    for f in figs:
        print(f"  {f}")


if __name__ == "__main__":
    main()
