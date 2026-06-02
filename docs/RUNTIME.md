# Runtime & performance notes

Target hardware: **Intel i5-9400F (6 cores), 16 GB RAM, GTX 1050 Ti, Windows 11** — and the
system is explicitly **CPU-only** (no GPU dependency).

## Where the time goes

A full 16-model benchmark is dominated by the LSTM work: every population-based search
evaluation is one or more LSTM fits across walk-forward folds, and the Adaptive Dual-Phase
LSTM plus the fusion pool train several LSTMs each. The tree/ELM learners are comparatively
free.

## How runtime is kept bounded

* **Evaluation budgeting** — `scale_optimizer_cfg` caps `pop_size × max_iter` (objective
  evaluations) per search. LSTM tuning gets the tightest cap; ELM/GBDT more.
* **Tiered epochs** — LSTM uses few epochs with early stopping during tuning, more for the
  final fit; small ensembles (1 during search, 2 for final).
* **Objective caching** — Phase-1 masks and Phase-2 hyper-vectors are memoised so repeated
  candidates are free.
* **Thread pinning** — `STOCKLAB_TORCH_THREADS` / `OMP_NUM_THREADS` pin intra-op threads so
  the 6 cores are not oversubscribed when the worker pool and library threads overlap.
* **Effort switch** — `fast` / `balanced` / `thorough` scales all search budgets from one
  control.
* **Bounded worker pool** — `STOCKLAB_MAX_WORKERS` (default 2) keeps concurrent jobs from
  thrashing the CPU; one full benchmark already saturates several cores.

### Indicative timings (i5-class CPU, `balanced`, ~500 rows)

| Stage | Approx. time |
|-------|--------------|
| 3 baselines | seconds |
| 9 PSO/GWO/hybrid tunes | the LSTM tunes dominate this block |
| 3 Adaptive Dual-Phase models | the heaviest single block (LSTM dual-phase) |
| Fusion ensemble | trains ~6 LSTM experts + walk-forward collectors |
| **Total (Fast, ~500 rows)** | ~8-15 min |
| **Total (Balanced / Thorough)** | ~20-45 min |

The real PyTorch LSTM is the cost driver: every search evaluation is an LSTM fit across
walk-forward folds. ELM/GBDT tunes are near-free. Use **Fast** for classroom/viva demos;
**Thorough** only when you want the optimisers to search deeply and can wait. Reducing the
sequence length, population, or iterations cuts time roughly linearly.

## Memory

Per-model peak Python allocation is measured with `tracemalloc` and surfaced in the
dashboard's memory table. Sequence tensors are float32 and modest (`rows × seq_len ×
features`); nothing approaches the 16 GB budget. Datasets and job state live in Redis with
TTLs (`STOCKLAB_CACHE_TTL`, `STOCKLAB_JOB_TTL`), so memory does not accumulate across runs.

## Scaling knobs (env vars)

| Variable | Default | Effect |
|----------|---------|--------|
| `STOCKLAB_MAX_WORKERS` | 2 | Concurrent benchmark jobs |
| `STOCKLAB_TORCH_THREADS` | 4 | Torch/OMP intra-op threads |
| `STOCKLAB_SEED` | 1337 | Global reproducibility seed |
| `STOCKLAB_CACHE_TTL` | 21600 | Cached dataset lifetime (s) |
| `STOCKLAB_JOB_TTL` | 86400 | Finished job lifetime (s) |

UI controls (population, iterations, effort, sequence length, test %) feed the same budget
machinery, so the operator can trade accuracy for speed without code changes.
