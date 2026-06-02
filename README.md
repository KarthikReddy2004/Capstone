# Adaptive Dual-Phase PSO-GWO Forecasting Lab

A production-grade, reproducible research platform that benchmarks **16 stock-forecasting
models** on a single shared, leak-free dataset and presents the results in a professional,
interactive financial-analytics dashboard.

The flagship model is the **Adaptive Dual-Phase PSO-GWO Expert Fusion Ensemble**: a
two-phase optimisation framework (binary feature selection + continuous hyper-parameter
search) whose tuned family models are blended by a temporal-cross-validated, non-negative
ridge ensemble. It is designed to rank at or near the top of the leaderboard through
legitimate optimisation, feature selection, and variance-reducing fusion — not metric
manipulation.

---

## Quick start

```bash
# 1. Install (CPU-only PyTorch — no GPU required)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Point at a Redis instance (managed endpoint or local)
export STOCKLAB_REDIS_URL="redis://:password@host:port/0"   # PowerShell: $env:STOCKLAB_REDIS_URL=...

# 3. Run
python app.py
# open http://127.0.0.1:5000
```

### Docker (bundles Redis)

```bash
docker compose up --build
# open http://127.0.0.1:5000
```

`docker compose` starts a Redis container and the app together; override
`STOCKLAB_REDIS_URL` to use a managed Redis instead.

### Offline smoke test (no browser, no network)

```bash
python scripts/smoke_benchmark.py     # runs the full 16-model pipeline on synthetic data
pytest                                # fast unit + integration tests
```

---

## The 16 models

| # | Model | Group |
|---|-------|-------|
| 1 | LSTM | Baseline |
| 2 | ELM | Baseline |
| 3 | GBDT | Baseline |
| 4 | LSTM + PSO | PSO |
| 5 | ELM + PSO | PSO |
| 6 | GBDT + PSO | PSO |
| 7 | LSTM + GWO | GWO |
| 8 | ELM + GWO | GWO |
| 9 | GBDT + GWO | GWO |
| 10 | LSTM + PSO-GWO | Hybrid |
| 11 | ELM + PSO-GWO | Hybrid |
| 12 | GBDT + PSO-GWO | Hybrid |
| 13 | Adaptive Dual-Phase PSO-GWO + LSTM | Adaptive Dual-Phase |
| 14 | Adaptive Dual-Phase PSO-GWO + ELM | Adaptive Dual-Phase |
| 15 | Adaptive Dual-Phase PSO-GWO + GBDT | Adaptive Dual-Phase |
| 16 | **Adaptive Dual-Phase PSO-GWO Expert Fusion Ensemble** | Ensemble (flagship) |

* **LSTM** is a real recurrent network (PyTorch, CPU) — stacked LSTM cells + MLP head,
  Adam, dropout, early stopping.
* **ELM** is an Extreme Learning Machine (random hidden projection + closed-form ridge).
* **GBDT** is scikit-learn `HistGradientBoostingRegressor`; the ensemble pool also uses
  `ExtraTreesRegressor`.

---

## The Adaptive Dual-Phase framework

### Phase 1 — Binary Adaptive PSO-GWO feature selection
* Multi-seed binary swarm search with **stability voting** across seeds.
* **Core market features protected** (never dropped): OHLCV, returns, key MAs, RSI, MACD,
  ATR, VWAP distance.
* **Adaptive mixing** `λ(t) = 1 − (t/T)²` (PSO-led exploration → GWO-led exploitation).
* Objective jointly **maximises validation score** and **penalises feature count**.

### Phase 2 — Continuous Adaptive PSO-GWO hyper-parameter optimisation
* Adaptive hybrid search on the selected feature subset, **multi-seed + local refinement**.
* Optimises per family: LSTM (sequence length, hidden units, layers, dropout, learning
  rate, batch size, L2); ELM (hidden neurons, activation, ridge); GBDT (estimators,
  learning rate, max depth, leaf size, L2).
* **Walk-forward / temporal cross-validation** throughout.

### Ensemble — Expert Fusion
Builds a diverse expert pool — Adaptive LSTM/ELM/GBDT, full-feature LSTM, selected-feature
LSTM, short-/long-window LSTM, Extra Trees, HistGradientBoosting — and selects the
weighting scheme with the best **temporal-CV out-of-sample** error among ridge-regularised
and equal-weight blends. Weights are **non-negative and sum to one**. A
**champion-preservation** rule prevents trend-aware sequence models from being dominated by
flat predictors in strongly trending regimes.

---

## Data pipeline (leak-free)

* **Sources:** Yahoo Finance download or CSV upload (OHLCV, alias-tolerant).
* **~80 strictly causal features:** RSI, MACD, Bollinger, ATR, CCI, Williams %R, OBV, ROC,
  momentum, EMA/SMA, rolling VWAP, z-scores, rolling statistics, lag features, calendar &
  cyclical features.
* **Target leakage protection:** the supervised target is the **next-step log-return**
  (`y[t] = log(close[t+1]/close[t])`); features at bar `t` use only data through `t`.
  Price is reconstructed as `close[t] · exp(return)`. Feature/target scalers are fit on the
  **training window only**.

### Validation & metrics
Walk-forward, expanding-window, and blocked temporal CV (with optional purge gap).
Metrics: **RMSE, MAE, MAPE, sMAPE, R², Directional Accuracy, Hit Ratio, Forecast Bias,
Tracking Signal**.

### Forecasting
Recursive **11-day** forecast with spotlight horizons **D+1, D+2, D+3, D+5, D+7, D+11** and
**80 / 90 / 95% prediction intervals** (return-space uncertainty grown by √h).

---

## Dashboard (all 20 visualisation categories)

Actual-vs-predicted (train/val/test) · forecast with confidence bands · all-model comparison
· sortable leaderboard · error-distribution (RMSE/MAE/residual histogram/KDE) · residual
analysis (vs time, vs prediction, Q–Q) · feature importance (GBDT / Extra Trees / selection
frequency) · optimisation convergence (best/mean/worst) for PSO/GWO/PSO-GWO/Adaptive ·
search-behaviour (particle & wolf trajectories, λ evolution, exploration vs exploitation) ·
hyper-parameter/search evolution · feature-selection stability heatmaps · correlation
heatmaps · learning curves · model-complexity (accuracy vs runtime, accuracy vs features) ·
ensemble weight distribution (pie + bar) · forecast accuracy radar · cumulative error ·
directional accuracy · runtime benchmark table · memory consumption table.

Interactive Plotly throughout: zoom, hover tooltips, PNG **and** SVG export, plus a
**dark / light** theme toggle.

---

## HTTP API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET  | `/api/health` | Redis + model registry health |
| GET  | `/api/models` | Model names & groups |
| POST | `/api/fetch_data` | Download from Yahoo Finance, cache, return summary |
| POST | `/api/upload_data` | Upload OHLCV CSV, cache, return summary |
| POST | `/api/run` | Submit a benchmark job → `{job_id}` |
| GET  | `/api/status/<job_id>` | Light poll: progress, stage, per-model summaries, log |
| GET  | `/api/results/<job_id>` | Full results + final payload (fetched once on completion) |

---

## Project structure

```
Capstone/
├── app.py                     # entry point -> stocklab.web:create_app
├── pyproject.toml · requirements.txt · Dockerfile · docker-compose.yml · .env.example
├── scripts/smoke_benchmark.py # offline end-to-end run
├── static/                    # single-page dashboard
│   ├── index.html
│   └── app.js
├── tests/                     # pytest suite (causality, metrics, optimizers, fusion, …)
├── docs/                      # ARCHITECTURE · RUNTIME · TESTING
└── stocklab/                  # application package
    ├── config.py · reproducibility.py · results.py
    ├── data/        # loaders, causal features, dataset builder
    ├── validation/  # metrics, temporal splitters
    ├── optimization/# pso, gwo, hybrid, binary, telemetry, refine
    ├── models/      # lstm (torch), elm, trees, decoders, evaluate, experts
    ├── forecasting/ # recursive forecast, intervals, bar synthesis
    ├── framework/   # tuning, dual_phase, ensemble (fusion)
    ├── pipeline/    # runner (16-model orchestration), diagnostics
    ├── storage/     # Redis backend + interface
    ├── jobs/        # thread-safe job manager / queue / cache
    └── web/         # Flask factory + routes
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/RUNTIME.md`](docs/RUNTIME.md),
and [`docs/TESTING.md`](docs/TESTING.md) for the deep dive.

---

## Reproducibility

`STOCKLAB_SEED` seeds Python, NumPy, and PyTorch; optimizers take explicit derived seeds, so
multi-seed searches reproduce run to run. Feature/target scalers are fit on training data
only. Torch thread count is pinned for stable timings and deterministic CPU kernels.

## Tips

* Indian equities use `.NS` (e.g. `RELIANCE.NS`); crypto pairs like `BTC-USD` work too.
* Use **Fast** effort for quick demos and **Thorough** for deeper optimisation.
* Longer history (≥ 2 years) improves feature stability and convergence quality.
