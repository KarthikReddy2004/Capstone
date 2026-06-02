# Model Suite: Architecture, Methodology, and Comparative Analysis

## 1. Overview

The forecasting platform evaluates sixteen models on a single, shared, leak-free
dataset. The suite is constructed systematically as a factorial combination of
**three base learners** ("engines") and **five optimization strategies**, with a
final **ensemble** that fuses the resulting models:

| Base learner / Strategy | Baseline | PSO | GWO | Hybrid PSO–GWO | Adaptive Dual-Phase |
| :-- | :--: | :--: | :--: | :--: | :--: |
| LSTM (sequential) | M1 | M4 | M7 | M10 | M13 |
| ELM (flat) | M2 | M5 | M8 | M11 | M14 |
| GBDT (tree-based) | M3 | M6 | M9 | M12 | M15 |

Model M16 — the *Adaptive Dual-Phase PSO–GWO Expert Fusion Ensemble* — is a
stacked combination of all fifteen preceding models.

This design isolates two sources of variation. Holding the strategy fixed and
changing the engine measures the effect of the **learning algorithm**; holding
the engine fixed and changing the strategy measures the effect of the
**optimization method**. All comparisons are therefore controlled.

## 2. Common Experimental Framework

To ensure that the leaderboard reflects model quality rather than differences in
data handling, every model shares the following protocol:

1. **Prediction target.** Each model forecasts the next-day closing price,
   expressed internally as a one-step (log) return and reconstructed to a price
   level using the previous close. Modelling the return rather than the raw price
   preserves stationarity and prevents look-ahead leakage.
2. **Feature set.** A set of thirteen protected market-structure features
   (price, volume, lagged returns, and standard technical indicators such as
   SMA, EMA, RSI, MACD, ATR, and VWAP distance) is supplemented by additional
   engineered indicators. The protected features are never removed by any
   selection procedure.
3. **Validation.** Model selection and optimizer objectives use walk-forward
   (temporal) cross-validation exclusively. No random shuffling is applied, so
   temporal ordering — and therefore causality — is preserved throughout.
4. **Optimization objective.** Every optimizer minimizes the walk-forward
   validation root-mean-square error (RMSE).
5. **Outputs and metrics.** Each model produces aligned training, validation, and
   test predictions; an eleven-day recursive forecast with 80%, 90%, and 95%
   prediction intervals; and a uniform metric suite: Train MSE, Test MSE, RMSE,
   MAE, MAPE, Theil's U, Average Relative Variance (ARV), R², and directional
   accuracy.

## 3. Base Learners

### 3.1 Long Short-Term Memory Network (LSTM)

The LSTM is a recurrent neural network implemented in PyTorch. It receives a
rolling window of recent observations (default length 32) across all features and
processes them through a recurrent layer (64 hidden units) followed by a compact
multilayer-perceptron regression head. Training uses the Adam optimizer with
mean-squared-error loss, gradient-norm clipping, and early stopping on a temporal
validation tail; predictions may be averaged across multiple random seeds for
stability.

- **Distinguishing property.** The LSTM is the only engine that explicitly
  models temporal dependence; it learns sequential patterns such as momentum and
  mean reversion across days.
- **Advantages.** Captures non-linear temporal dynamics; typically strongest on
  trending or autocorrelated series; incorporates internal regularization
  (dropout, early stopping).
- **Limitations.** Highest computational cost; sensitive to hyperparameter
  choices; requires sufficient history to form sequences; susceptible to
  overfitting on noisy data.

### 3.2 Extreme Learning Machine (ELM)

The ELM is a single-hidden-layer feed-forward network whose input weights are
assigned randomly and held fixed; only the output weights are trained, by a
closed-form ridge-regularized least-squares solution (a single matrix
pseudo-inverse). It therefore requires no iterative optimization.

- **Distinguishing property.** Training reduces to one linear solve, making the
  ELM orders of magnitude faster than the other engines. It operates on the
  current observation only, without a temporal window.
- **Advantages.** Extremely fast and deterministic; provides predictive
  diversity that is valuable to the ensemble because its errors are weakly
  correlated with those of the other engines.
- **Limitations.** The random projection yields lower precision than a tuned
  tree or recurrent model; it has no temporal memory and is sensitive to the
  hidden-layer width and input scaling.

### 3.3 Gradient-Boosted Decision Trees (GBDT)

The GBDT engine uses a histogram-based gradient-boosting regressor. It builds an
additive sequence of shallow regression trees, each fitted to the residuals of
the current ensemble, with a learning rate, limited tree depth, leaf-size
constraints, and L2 regularization.

- **Distinguishing property.** Excels at modelling non-linear interactions among
  tabular features, evaluated on a per-observation basis without any sequential
  structure.
- **Advantages.** Strong performance on tabular data; robust with minimal
  preprocessing; provides interpretable feature-importance estimates.
- **Limitations.** Ignores temporal structure; extrapolates poorly beyond the
  training range, which can cause lag on strongly trending series; requires
  regularization to control overfitting.

## 4. Metaheuristic Optimizers

The optimizers search the bounded hyperparameter space of each engine and
minimize validation RMSE. They differ in how candidate solutions are updated.

### 4.1 Particle Swarm Optimization (PSO)

A population of particles carries position and velocity. Each particle is
attracted toward its own best-known position (cognitive term) and the swarm's
global best (social term). The inertia weight decays linearly over the search,
shifting behaviour from exploration to exploitation. PSO converges smoothly and
retains memory of promising regions, but may stagnate in local optima if the
swarm collapses prematurely.

### 4.2 Grey Wolf Optimizer (GWO)

A population of search agents is ranked into three leaders (alpha, beta, delta).
Each agent repositions itself by encircling the three leaders, with the
encircling coefficient decreasing over the search to intensify exploitation.
GWO uses no velocity or individual memory; it offers strong late-stage
convergence with few control parameters, but can lose diversity and converge
early.

### 4.3 Hybrid PSO–GWO

Each candidate is updated by computing both a PSO velocity step and a GWO
encircling step and combining them with a mixing coefficient lambda:

> position = lambda · (PSO step) + (1 − lambda) · (GWO step)

The treatment of lambda is the principal distinction between two model families:

- In the **hybrid baseline family (M10–M12)**, lambda is fixed at 0.5, giving a
  static equal blend of the two search behaviours.
- In the **Adaptive Dual-Phase framework (M13–M16)**, lambda follows an adaptive
  schedule, lambda(t) = 1 − (t/T)², beginning PSO-dominant (broad exploration)
  and ending GWO-dominant (focused exploitation). This adaptive schedule is the
  principal mechanism behind the framework's improved performance over the static
  hybrid.

## 5. Model Families

### 5.1 Baseline Models (M1–M3)

Each engine is trained with fixed default hyperparameters on the full feature
set, without any search procedure.

- **Role.** A control group establishing the performance obtainable without
  optimization.
- **Advantages.** Immediate to train; simple and fully reproducible.
- **Limitations.** Generally the weakest configuration of each engine; the
  improvement of the optimized variants over these baselines quantifies the value
  of optimization.

### 5.2 PSO-Optimized Models (M4–M6)

The three engines are retained, but their hyperparameters are selected by a PSO
search that minimizes walk-forward validation RMSE.

- **Distinction from baseline.** Hyperparameters are tuned rather than fixed.
- **Distinction within the family.** Only the engine differs; the optimizer is
  identical across the three members.
- **Advantages and limitations.** Inherit the characteristics of the underlying
  engine; PSO typically yields reliable, smooth improvements but may not locate
  the global optimum.

### 5.3 GWO-Optimized Models (M7–M9)

Identical in construction to the PSO family, but tuned with the Grey Wolf
Optimizer. Comparing this family with the PSO family isolates the effect of the
search algorithm on each engine. GWO may locate sharper optima through stronger
exploitation, at the risk of premature convergence.

### 5.4 Hybrid PSO–GWO Optimized Models (M10–M12)

The three engines are tuned by the static hybrid optimizer (lambda = 0.5),
combining PSO's memory with GWO's leadership-based exploitation.

- **Significance.** This family is the direct experimental control for the
  Adaptive Dual-Phase framework: it shares the hybrid search mechanics but
  excludes the adaptive schedule and the additional phases. The performance gap
  between this family and the next therefore measures the specific contribution
  of the adaptive dual-phase design.

### 5.5 Adaptive Dual-Phase PSO–GWO Models (M13–M15)

The principal framework of the study. Each engine is optimized through a
two-phase adaptive procedure.

**Phase 1 — Adaptive binary feature selection.** Candidate solutions in a
continuous space are mapped to binary feature masks through a transfer function.
Protected core features are locked on and cannot be removed. The search is run
across multiple random seeds and consolidated by fitness-weighted stability
voting, so that only consistently selected features are retained. The objective
combines validation error with a compactness penalty, favouring small yet
predictive subsets. A full-feature fallback is enforced: the reduced subset is
adopted only if it demonstrably outperforms the complete feature set on
validation, ensuring the model is never disadvantaged by feature removal.

**Phase 2 — Adaptive continuous hyperparameter optimization.** Using the
selected features, the hybrid optimizer with the adaptive lambda schedule tunes
the engine's hyperparameters across multiple seeds. The best candidates are then
refined by a local search (a shrinking Gaussian random walk) for final
exploitation. The evaluation budget is larger than that of the baseline
families.

- **Distinction from the hybrid family.** Four enhancements over M10–M12:
  (i) the adaptive lambda schedule in place of a fixed value; (ii) an explicit
  feature-selection phase; (iii) multi-seed stability voting with local
  refinement; and (iv) a deeper search budget.
- **Advantages.** Highest single-model accuracy; produces compact, informative
  feature subsets; substantially more robust against local optima.
- **Limitations.** The most computationally expensive family, owing to two
  searches, multiple seeds, and the refinement stage.

### 5.6 Adaptive Dual-Phase PSO–GWO Expert Fusion Ensemble (M16)

The final model is a stacked generalization over the fifteen preceding models. It
does not retrain any learner. Instead, it collects each model's out-of-fold
validation predictions, aligns them on common dates, and fits non-negative
weights that sum to one by temporal cross-validation. The resulting weights are
applied to the shared test predictions and to the recursive forecast.

Three safeguards ensure the combination is sound:

1. **Pruning.** Models whose validation RMSE is far from the best are excluded,
   preventing weak learners from degrading the blend.
2. **Guarding.** The selected weighting scheme is compared against the single
   best model under the same cross-validation, and is adopted only if it does not
   underperform it. The ensemble can therefore never be worse than its best
   member.
3. **Regime awareness.** In strongly trending conditions, non-sequential models
   that fail trend criteria are excluded so that the sequential model is not
   diluted.

- **Advantages.** Combines complementary strengths across engines; reduces
  variance; is robust and, by construction, at least as accurate as its best
  constituent.
- **Limitations.** Its quality depends on the strength and diversity of the
  constituent models; it introduces additional complexity and performs no new
  learning of its own.

## 6. Comparative Summary

### 6.1 Strategy comparison

| Models | Strategy | Defining characteristic | Primary strength |
| :-- | :-- | :-- | :-- |
| M1–M3 | Baseline | Default hyperparameters, all features | Reference performance |
| M4–M6 | PSO | Hyperparameters via particle swarm | Stable, reliable tuning |
| M7–M9 | GWO | Hyperparameters via grey-wolf hierarchy | Strong exploitation |
| M10–M12 | Hybrid (lambda = 0.5) | Static blend of PSO and GWO | Robust single search |
| M13–M15 | Adaptive Dual-Phase | Adaptive blend + feature selection + refinement | Best single-model accuracy |
| M16 | Fusion Ensemble | Guarded weighted stack of all models | Best overall robustness |

### 6.2 Engine comparison

| Engine | Temporal modelling | Training cost | Principal strength | Principal limitation |
| :-- | :--: | :--: | :-- | :-- |
| LSTM | Yes (sequences) | High | Non-linear temporal dynamics | Cost; overfitting risk |
| ELM | No (single step) | Very low | Speed; ensemble diversity | Lower precision (random projection) |
| GBDT | No (single step) | Low–moderate | Feature interactions; robustness | No sequence; weak extrapolation |

### 6.3 Concluding remarks

The suite is a controlled factorial experiment: three learning algorithms are
each subjected to four optimization strategies of increasing sophistication, and
the resulting models are finally combined by a guarded ensemble. The progression
from baseline to single-optimizer tuning, to static hybrid search, to the
adaptive dual-phase framework, and finally to fusion provides a transparent
account of where predictive gains originate and isolates the contribution of each
methodological component.
