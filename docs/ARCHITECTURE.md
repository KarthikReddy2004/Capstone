# Architecture

## Overview

The platform is a modular Python package (`stocklab`) behind a thin Flask API and a
single-page Plotly dashboard. All shared state lives in **Redis** — there is no global
mutable Python state — so the system is safe across worker threads and horizontally
restartable.

```
Browser (SPA)  ──HTTP──▶  Flask routes  ──▶  JobManager (ThreadPool)  ──▶  run_benchmark
     ▲                         │                      │                          │
     │  poll /status           │  cache dataset       │  job state / progress    │ models,
     └─────────────────────────┴──────────────────────┴──────────────────────────┘ telemetry
                                          Redis (single source of truth)
```

## Layered design

| Layer | Package | Responsibility |
|-------|---------|----------------|
| Web | `stocklab.web` | Flask factory, JSON API, static serving |
| Jobs | `stocklab.jobs` | Thread-safe queue, progress, dataset cache |
| Storage | `stocklab.storage` | `StorageBackend` interface + Redis implementation |
| Pipeline | `stocklab.pipeline` | Orchestrates 16 models, leaderboard, diagnostics |
| Framework | `stocklab.framework` | Dual-phase tuning, feature selection, fusion ensemble |
| Models | `stocklab.models` | LSTM/ELM/GBDT/ExtraTrees, decoders, CV objectives, runners |
| Optimization | `stocklab.optimization` | PSO/GWO/hybrid/binary + telemetry |
| Forecasting | `stocklab.forecasting` | Recursive forecast, intervals, bar synthesis |
| Validation | `stocklab.validation` | Metrics + temporal splitters |
| Data | `stocklab.data` | Loaders, causal features, dataset builder |

Dependencies flow downward only (web → jobs → pipeline → framework → models → …), which
keeps the core ML code importable and testable without Flask or Redis.

## Data flow of a run

1. `POST /api/fetch_data|upload_data` → load OHLCV, cache the raw frame in Redis under a
   `cache_key`, return a summary.
2. `POST /api/run` → `JobManager.submit(run_benchmark, df, cfg)` schedules the worker on a
   bounded `ThreadPoolExecutor` and returns a `job_id`.
3. The worker owns a `JobContext` (the **only** writer of that job's record) and streams
   stage/progress/log/per-model summary/full-result/final to Redis.
4. The SPA polls `GET /api/status/<job_id>` for the light live board, then fetches
   `GET /api/results/<job_id>` once on completion for the full payload.

## Leakage discipline

* Target = next-step **log-return**; features are strictly causal (trailing windows,
  backward-shifted lags).
* All scalers (feature and target) are fit on the **training window only**.
* Every CV splitter is causal (validation strictly after train); temporal CV supports a
  purge gap for the fusion-weight calibration.

## The flagship, end to end

`run_dual_phase(family)` runs Phase 1 (binary feature selection, multi-seed, stability
voting, core protection) and Phase 2 (continuous adaptive hybrid hyper-search + local
refine) for LSTM, ELM, and GBDT → models 13/14/15. `run_fusion_ensemble` then builds a
diverse expert pool, fits **non-negative ridge / equal-weight** blends, selects the scheme
with the best **temporal-CV** error, and applies champion-preservation → model 16. Because
the blend is chosen out-of-sample and reduces variance across decorrelated experts, it
generalises to the test set and ranks at/near the top legitimately.

## Telemetry

Every optimizer records, per iteration: best/mean/worst fitness, population diversity,
exploration vs exploitation ratio, `λ(t)`, GWO pressure `a(t)`, and a capped sample of
particle/wolf trajectories. The binary selector additionally records per-feature selection
frequency for the stability heatmaps. This is what powers the optimisation and
search-behaviour charts.

## Swapping the storage backend

`StorageBackend` (in `stocklab/storage/backend.py`) is a small interface (`put_json`,
`get_json`, `list_push`, `list_all`, `put_raw`, `get_raw`, `exists`, `delete`, `ping`). The
Redis implementation is the default; any conforming backend (e.g. an in-memory test double)
can be injected into `JobManager` without touching the rest of the system.
