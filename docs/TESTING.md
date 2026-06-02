# Testing strategy

```bash
pytest                 # full suite (fast; Redis tests skip if no server)
pytest tests/test_causality.py -v
python scripts/smoke_benchmark.py     # end-to-end 16-model run on synthetic data
```

## What is covered

| Test file | Guarantees |
|-----------|------------|
| `test_metrics.py` | Every metric (RMSE/MAE/MAPE/sMAPE/R²/DirAcc/Hit/Bias/Tracking) on known inputs; perfect-prediction and empty-input edge cases. |
| `test_causality.py` | **No leakage**: mutating the last bar changes no earlier feature row; target is `close[t+1]`; scalers are train-only; train precedes test in time. |
| `test_splitters.py` | Walk-forward / expanding / temporal-CV folds are strictly causal (validation after train) and the purge gap is honoured. |
| `test_optimizers.py` | PSO/GWO/hybrid converge on a sphere; telemetry arrays are aligned; adaptive `λ(t)` runs 1→0; binary selector locks core features. |
| `test_fusion.py` | The fusion blend **beats the best single expert out-of-sample**; weights are non-negative and sum to 1. |
| `test_models_and_storage.py` | Runner output schema (test/val/forecast/intervals/spotlights), interval ordering (95% ⊇ 80%), GBDT importances, Redis round-trip (skipped if unavailable). |

## Testing philosophy

* **Leakage is the highest-priority invariant.** The causal-feature and train-only-scaler
  tests are the safety net that keeps the benchmark honest — high R² is meaningful only if
  no future information leaked in.
* **The flagship's advantage is tested, not assumed.** `test_fusion.py` asserts the ensemble
  legitimately reduces out-of-sample error versus the best constituent, which is the
  property that lets it rank at/near the top without metric manipulation.
* **Pure functions everywhere.** Metrics, splitters, optimizers, and fusion math are pure
  NumPy and unit-tested in isolation; the model layer needs no Flask/Redis to test.
* **Determinism.** All tests seed their RNGs, so they are stable in CI.

## Reproducibility controls under test

`seed_everything` and `child_seed` make multi-seed optimisation reproducible; tests rely on
fixed seeds so convergence assertions are deterministic. The synthetic-data fixture
(`conftest.py`) is itself seeded.

## Manual / browser verification

1. `python app.py`, open the dashboard, fetch `AAPL`.
2. Run with **Fast** effort; watch the live board fill in and the log stream.
3. On completion, confirm: leaderboard ranks all 16 models, the flagship row is highlighted,
   every tab renders, forecast bands appear, and PNG/SVG export works.
4. Toggle dark/light — all charts re-theme.
