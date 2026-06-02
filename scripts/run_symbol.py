"""Run the full 16-model benchmark on a real Yahoo Finance symbol (offline ctx).

    uv run python scripts/run_symbol.py RELIANCE.NS 2018-01-01

Prints the leaderboard and the flagship ensemble's standing. Requires network
access for the Yahoo Finance download; no Redis needed (uses an in-memory ctx).
"""

import sys
import time

sys.path.insert(0, ".")
from stocklab.data import load_from_yahoo
from stocklab.pipeline import run_benchmark, MODEL_NAMES


class FakeCtx:
    def __init__(self):
        self.results, self.final = [], None

    def log(self, m): print("  ", m, flush=True)
    def set_stage(self, s): print("STAGE:", s, flush=True)
    def set_progress(self, p): pass
    def set_model_status(self, n, s): pass
    def push_result(self, r): self.results.append(r)
    def push_summary(self, r): pass
    def set_final(self, f): self.final = f


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    start = sys.argv[2] if len(sys.argv) > 2 else "2018-01-01"
    cfg = {"seqLen": 24, "testPct": 18, "popSize": 10, "maxIter": 10, "effort": "fast", "seed": 1337}

    print(f"Fetching {symbol} from {start} …", flush=True)
    df = load_from_yahoo(symbol, start)
    print(f"  {len(df)} rows downloaded", flush=True)

    ctx = FakeCtx()
    t0 = time.time()
    run_benchmark(ctx, df, cfg)
    dt = time.time() - t0

    print(f"\n==== {symbol} · {dt:.1f}s · {len(ctx.results)}/{len(MODEL_NAMES)} models ====")
    print("Rank | RMSE | MAE | R2 | DirAcc | Hit | name")
    for r in ctx.final["leaderboard"]:
        print("  %2d | %8.3f | %7.3f | %6.3f | %5.1f | %5.1f | %s"
              % (r["rank"], r["rmse"], r["mae"], r["r2"], r["dir_acc"], r["hit_ratio"], r["name"]))
    print(f"\nFlagship Expert Fusion Ensemble rank: {ctx.final['flagship_rank']} / {len(ctx.final['leaderboard'])}")
    fus = next((r for r in ctx.results if r["index"] == 15), None)
    if fus:
        print("Fusion weights:", [(w["name"], w["weight"]) for w in fus["fusion_weights"]])
        print("Selected features:", fus["n_selected"], "of", fus["best_hyper"].get("selected_features", "?"))
        pts = fus["forecast"]["points"]
        print("Forecast spotlights:", [(f"D+{p['horizon']}", round(p["value"], 2)) for p in pts])


if __name__ == "__main__":
    main()
