"""Offline end-to-end smoke benchmark on synthetic data (no network/Redis).

Runs the full 16-model pipeline with a small budget and prints the leaderboard.
Useful for verifying correctness and the flagship ensemble's standing without a
browser or a live data feed.

    uv run python scripts/smoke_benchmark.py
"""

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from stocklab.data import normalize_frame
from stocklab.pipeline import run_benchmark, MODEL_NAMES


class FakeCtx:
    def __init__(self):
        self.results = []
        self.final = None

    def log(self, m): print("  ", m)
    def set_stage(self, s): print("STAGE:", s)
    def set_progress(self, p): pass
    def set_model_status(self, n, s): pass
    def push_result(self, r): self.results.append(r)
    def push_summary(self, r): pass
    def set_final(self, f): self.final = f


def synthetic(n=460, seed=11):
    rng = np.random.default_rng(seed)
    drift = np.linspace(0, 0.5, n)
    ret = rng.normal(0.0006, 0.012, n) + drift / n
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    vol = rng.integers(1e6, 5e6, n).astype(float)
    return normalize_frame(pd.DataFrame({
        "Date": pd.bdate_range("2020-01-01", periods=n),
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol,
    }))


def main():
    cfg = {"seqLen": 20, "testPct": 20, "popSize": 8, "maxIter": 8, "effort": "fast", "seed": 1337}
    ctx = FakeCtx()
    t0 = time.time()
    run_benchmark(ctx, synthetic(), cfg)
    print(f"\n==== {time.time()-t0:.1f}s · {len(ctx.results)}/{len(MODEL_NAMES)} models ====")
    print("Rank | RMSE | R2 | DirAcc | name")
    for r in ctx.final["leaderboard"]:
        print("  %2d | %7.4f | %6.3f | %5.1f | %s" % (r["rank"], r["rmse"], r["r2"], r["dir_acc"], r["name"]))
    fr = ctx.final["flagship_rank"]
    print(f"\nFlagship Expert Fusion Ensemble rank: {fr} / {len(ctx.final['leaderboard'])}")
    fus = next((r for r in ctx.results if r["index"] == 15), None)
    if fus:
        print("Fusion weights:", [(w["name"], w["weight"]) for w in fus["fusion_weights"]])


if __name__ == "__main__":
    main()
