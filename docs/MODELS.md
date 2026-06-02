# Chapter 3 — Methodology: Model Architectures and Optimization Framework

This chapter presents the complete methodology of the forecasting platform. It
formalises the prediction problem, the common experimental protocol, the three
base learners, the three metaheuristic optimizers, the sixteen models that
constitute the benchmark, and the evaluation criteria. Mathematical formulations
are provided for every component, and design choices are related to the relevant
literature.

---

## 3.1 Experimental Design

The benchmark is organised as a controlled factorial experiment. Three base
learners (referred to as *engines*) are each paired with five optimization
*strategies*, and the resulting models are finally combined by an ensemble:

| Engine \ Strategy | Baseline | PSO | GWO | Hybrid PSO–GWO | Adaptive Dual-Phase |
| :-- | :--: | :--: | :--: | :--: | :--: |
| LSTM (sequential) | M1 | M4 | M7 | M10 | M13 |
| ELM (flat) | M2 | M5 | M8 | M11 | M14 |
| GBDT (tree-based) | M3 | M6 | M9 | M12 | M15 |

Model **M16** is the *Adaptive Dual-Phase PSO–GWO Expert Fusion Ensemble*, a
stacked generalisation over M1–M15.

Fixing the strategy and varying the engine isolates the effect of the **learning
algorithm**; fixing the engine and varying the strategy isolates the effect of
the **optimization method**. Because all sixteen models share an identical data
pipeline, target, validation protocol, and metric suite, the leaderboard
constitutes a fair comparison.

---

## 3.2 Problem Formulation and Common Framework

### 3.2.1 Forecasting target

Let $C_t$ denote the closing price on day $t$. Rather than regressing the raw
price, each model predicts the one-step logarithmic return

$$
r_t = \ln\!\left(\frac{C_t}{C_{t-1}}\right),
$$

which is approximately stationary. The predicted return $\hat r_t$ is mapped back
to a price through the previous close,

$$
\hat C_t = C_{t-1}\,\exp(\hat r_t).
$$

Returns are standardised before training and inverted at reconstruction.
Predicting returns and reconstructing from the *observed* previous close removes
the spurious autocorrelation that arises when forecasting price levels directly,
and prevents look-ahead leakage.

### 3.2.2 Feature space

Each observation is described by a vector $\mathbf{x}_t \in \mathbb{R}^{d}$
comprising thirteen protected market-structure features — open, high, low,
volume, logarithmic return, lagged close and return, and the technical
indicators SMA(10), EMA(20), RSI(14), MACD, ATR(14), and VWAP distance —
together with additional engineered indicators. The protected subset
$\mathcal{C}$ (the *core* features) is never removed by feature selection.

### 3.2.3 Validation protocol

All model selection and optimizer objectives use **walk-forward (temporal)
cross-validation**: each validation fold lies strictly in the future relative to
its training fold, and no random shuffling is applied. This is the recommended
practice for evaluating predictors on serially dependent data
(Bergmeir & Benítez, 2012). Every optimizer minimises the walk-forward
validation root-mean-square error.

### 3.2.4 Evaluation metrics

Given test targets $y_i$, predictions $\hat y_i$, the previous-close anchor
$p_i$, and $n$ test points, the platform reports:

$$
\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2, \qquad
\mathrm{RMSE} = \sqrt{\mathrm{MSE}}, \qquad
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\lvert y_i-\hat y_i\rvert,
$$

$$
\mathrm{MAPE} = \frac{100}{n}\sum_{i=1}^{n}\frac{\lvert y_i-\hat y_i\rvert}{\lvert y_i\rvert+\varepsilon},
\qquad
R^2 = 1-\frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}.
$$

The **Average Relative Variance** normalises error by the variance of the series
and is the complement of $R^2$ (Lapedes & Farber, 1987):

$$
\mathrm{ARV} = \frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2} = 1-R^2 .
$$

**Theil's U** (the $U_2$ inequality coefficient; Theil, 1966) compares the model
against a naïve random-walk forecast that predicts the anchor $p_i$:

$$
U = \frac{\sqrt{\dfrac{1}{n}\sum_i \left(\dfrac{\hat y_i-y_i}{p_i}\right)^2}}
         {\sqrt{\dfrac{1}{n}\sum_i \left(\dfrac{y_i-p_i}{p_i}\right)^2}} .
$$

$U<1$ indicates that the model outperforms the random walk, $U=1$ matches it, and
$U>1$ is worse. Finally, **directional accuracy** measures the fraction of
correctly predicted movement signs relative to the anchor:

$$
\mathrm{DIR} = \frac{100}{n}\sum_{i=1}^{n}
\mathbf{1}\!\left[\operatorname{sign}(y_i-p_i)=\operatorname{sign}(\hat y_i-p_i)\right].
$$

---

## 3.3 Base Learners

### 3.3.1 Long Short-Term Memory network (LSTM)

The recurrent engine (Hochreiter & Schmidhuber, 1997) consumes a window of the
$L$ most recent feature vectors $\{\mathbf{x}_{t-L+1},\dots,\mathbf{x}_t\}$. At
each step it updates its gates and cell state:

$$
\begin{aligned}
\mathbf{i}_t &= \sigma(\mathbf{W}_i\mathbf{x}_t+\mathbf{U}_i\mathbf{h}_{t-1}+\mathbf{b}_i), &
\mathbf{f}_t &= \sigma(\mathbf{W}_f\mathbf{x}_t+\mathbf{U}_f\mathbf{h}_{t-1}+\mathbf{b}_f),\\
\mathbf{o}_t &= \sigma(\mathbf{W}_o\mathbf{x}_t+\mathbf{U}_o\mathbf{h}_{t-1}+\mathbf{b}_o), &
\tilde{\mathbf{c}}_t &= \tanh(\mathbf{W}_c\mathbf{x}_t+\mathbf{U}_c\mathbf{h}_{t-1}+\mathbf{b}_c),\\
\mathbf{c}_t &= \mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t, &
\mathbf{h}_t &= \mathbf{o}_t\odot\tanh(\mathbf{c}_t),
\end{aligned}
$$

where $\sigma$ is the logistic sigmoid and $\odot$ is the elementwise product.
The final hidden state $\mathbf{h}_L$ is passed to a two-layer regression head,

$$
\hat r = \mathbf{w}_2^{\top}\,\mathrm{ReLU}(\mathbf{W}_1\mathbf{h}_L+\mathbf{b}_1)+b_2 ,
$$

with dropout applied before the output layer. Parameters are trained by the Adam
optimizer on the mean-squared-error loss, with gradient-norm clipping
$\lVert\nabla\rVert_2\le 3$ and early stopping on a temporal validation tail;
predictions may be averaged over several random seeds.

- **Distinguishing property.** The only engine that explicitly models temporal
  dependence, learning sequential structure such as momentum and mean reversion.
- **Advantages.** Captures non-linear temporal dynamics; typically strongest on
  trending or autocorrelated series; internally regularised by dropout and early
  stopping.
- **Limitations.** Highest computational cost; sensitive to hyperparameters;
  requires sufficient history; prone to overfitting noisy data.

### 3.3.2 Extreme Learning Machine (ELM)

The ELM (Huang, Zhu, & Siew, 2006) is a single-hidden-layer feed-forward network
whose input weights $\mathbf{W}\sim\mathcal{N}(0,s^2)$ and biases
$\mathbf{b}\sim\mathcal{N}(0,s^2)$ are drawn randomly and held **fixed**. Only the
output weights are trained, in closed form. For inputs $\mathbf{X}$ the hidden
representation is

$$
\mathbf{H} = g(\mathbf{X}\mathbf{W}+\mathbf{b}), \qquad g\in\{\tanh,\ \sigma,\ \mathrm{ReLU}\},
$$

and the output weights follow from the ridge-regularised least-squares solution

$$
\boldsymbol{\beta} = \left(\mathbf{H}^{\top}\mathbf{H}+\alpha\mathbf{I}\right)^{-1}\mathbf{H}^{\top}\mathbf{y},
$$

evaluated through the Moore–Penrose pseudo-inverse. Prediction is
$\hat y(\mathbf{x}) = g(\mathbf{x}^{\top}\mathbf{W}+\mathbf{b})\,\boldsymbol{\beta}$.

- **Distinguishing property.** Training is a single matrix solve, with no
  iterative optimization; it uses the current observation only, without a window.
- **Advantages.** Extremely fast and deterministic; supplies predictive
  diversity to the ensemble because its errors are weakly correlated with the
  other engines.
- **Limitations.** Lower precision than tuned recurrent or tree models owing to
  the random projection; no temporal memory; sensitive to hidden width and scale.

### 3.3.3 Gradient-Boosted Decision Trees (GBDT)

The GBDT engine implements histogram-based gradient boosting
(Friedman, 2001; Ke et al., 2017). Starting from a constant
$F_0(\mathbf{x})=\arg\min_\gamma\sum_i L(y_i,\gamma)$, it adds shallow regression
trees that fit the negative gradient of the loss. For the squared-error loss the
pseudo-residuals reduce to ordinary residuals,

$$
r_{im} = -\left[\frac{\partial L(y_i,F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)}\right]_{F=F_{m-1}} = y_i-F_{m-1}(\mathbf{x}_i),
$$

and each tree $h_m$ is fitted to $\{(\mathbf{x}_i,r_{im})\}$, giving the additive
update with learning rate $\nu$:

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu\,h_m(\mathbf{x}), \qquad
F_M(\mathbf{x}) = F_0(\mathbf{x}) + \nu\sum_{m=1}^{M} h_m(\mathbf{x}).
$$

Tree depth, minimum leaf size, and $L_2$ shrinkage regularise the model;
continuous features are binned into histograms for efficient split finding.

- **Distinguishing property.** Models non-linear feature interactions on a
  per-observation basis, with no sequential structure.
- **Advantages.** Strong tabular performance; robust with minimal preprocessing;
  provides interpretable feature-importance estimates.
- **Limitations.** Ignores temporal order; extrapolates poorly beyond the
  training range, producing lag on strongly trending series; needs regularisation.

---

## 3.4 Metaheuristic Optimizers

Each optimizer searches the bounded hyperparameter space
$[\mathbf{lb},\mathbf{ub}]\subset\mathbb{R}^{d}$ and minimises validation RMSE.
Let $t=0,\dots,T-1$ index iterations and $\tau=t/(T-1)\in[0,1]$ denote normalised
progress. The projection $\Pi_{[\mathbf{lb},\mathbf{ub}]}$ clips a vector to the
feasible box, and $\mathbf{r}_{(\cdot)}\sim\mathcal{U}(0,1)^{d}$ are independent
random vectors.

### 3.4.1 Particle Swarm Optimization (PSO)

PSO (Kennedy & Eberhart, 1995) maintains particles with positions
$\mathbf{x}_i$ and velocities $\mathbf{v}_i$, each attracted to its personal best
$\mathbf{p}_i$ and the global best $\mathbf{g}$. A linearly decaying inertia
weight (Shi & Eberhart, 1998) balances exploration and exploitation:

$$
w(\tau) = w_0(1-\tau) + 0.38\,\tau,
$$

$$
\mathbf{v}_i \leftarrow w(\tau)\,\mathbf{v}_i + c_1\mathbf{r}_1\odot(\mathbf{p}_i-\mathbf{x}_i) + c_2\mathbf{r}_2\odot(\mathbf{g}-\mathbf{x}_i),
\qquad
\mathbf{x}_i \leftarrow \Pi(\mathbf{x}_i+\mathbf{v}_i),
$$

with cognitive and social coefficients $c_1=1.5$, $c_2=2.0$ and initial inertia
$w_0=0.72$. PSO converges smoothly and retains memory of promising regions, but
can stagnate if diversity collapses early.

### 3.4.2 Grey Wolf Optimizer (GWO)

GWO (Mirjalili, Mirjalili, & Lewis, 2014) ranks the population into three leaders
$\boldsymbol{\alpha},\boldsymbol{\beta},\boldsymbol{\delta}$ (the three best
solutions). The encircling coefficient decreases linearly,

$$
a(\tau) = 2 - 2\tau,
$$

and each agent moves toward every leader $\boldsymbol{\ell}$ by

$$
\mathbf{A}_\ell = 2a(\tau)\mathbf{r}_1 - a(\tau), \quad
\mathbf{C}_\ell = 2\mathbf{r}_2, \quad
\mathbf{X}_\ell = \boldsymbol{\ell} - \mathbf{A}_\ell\odot\bigl\lvert \mathbf{C}_\ell\odot\boldsymbol{\ell} - \mathbf{x}_i\bigr\rvert,
$$

with the next position the leaders' average:

$$
\mathbf{x}_i \leftarrow \Pi\!\left(\tfrac{1}{3}\bigl(\mathbf{X}_\alpha+\mathbf{X}_\beta+\mathbf{X}_\delta\bigr)\right).
$$

GWO has few control parameters and exploits strongly in late iterations, but
carries no velocity or individual memory and may converge prematurely.

### 3.4.3 Hybrid PSO–GWO and adaptive mixing

The hybrid (after Şenel et al., 2019) computes both a PSO step and a GWO step for
each agent and blends them with a mixing coefficient $\lambda$. The PSO component
uses the global best as above; the GWO component encircles the three best
*personal* bests. Writing
$\mathbf{p}^{\text{pso}}_i=\mathbf{x}_i+\mathbf{v}_i$ and
$\mathbf{p}^{\text{gwo}}_i=\tfrac{1}{3}(\mathbf{X}_\alpha+\mathbf{X}_\beta+\mathbf{X}_\delta)$,

$$
\mathbf{x}_i \leftarrow \Pi\!\left(\lambda\,\mathbf{p}^{\text{pso}}_i + (1-\lambda)\,\mathbf{p}^{\text{gwo}}_i\right).
$$

The treatment of $\lambda$ defines two of the model families:

$$
\lambda =
\begin{cases}
\tfrac{1}{2}, & \text{static hybrid baseline (M10–M12),}\\[4pt]
1-\tau^{2}, & \text{Adaptive Dual-Phase framework (M13–M16).}
\end{cases}
$$

The adaptive schedule begins PSO-dominant ($\lambda\!\approx\!1$, broad
exploration) and ends GWO-dominant ($\lambda\!\approx\!0$, focused exploitation).
This schedule is the principal mechanism by which the framework improves on the
static hybrid.

---

## 3.5 Model Families

### 3.5.1 Baseline models (M1–M3)

Each engine is trained with fixed default hyperparameters on the full feature set,
with no search. They form the control group: the gains of the optimized variants
over these baselines quantify the value of optimization. They are immediate to
train and fully reproducible, but generally the weakest configuration of each
engine.

### 3.5.2 PSO-optimized models (M4–M6)

The three engines, with hyperparameters chosen by PSO (§3.4.1) minimising
walk-forward validation RMSE. They differ from the baselines only in being tuned,
and differ from one another only in the engine. PSO yields reliable, smooth
improvements but does not guarantee the global optimum.

### 3.5.3 GWO-optimized models (M7–M9)

Constructed identically to §3.5.2 but tuned with GWO (§3.4.2). Comparison with
the PSO family isolates the effect of the search algorithm on each engine. GWO
can locate sharper optima through stronger exploitation, at the risk of premature
convergence.

### 3.5.4 Hybrid PSO–GWO models (M10–M12)

The three engines tuned by the static hybrid ($\lambda=\tfrac12$, §3.4.3),
combining PSO's memory with GWO's leader-based exploitation. This family is the
direct control for the Adaptive Dual-Phase framework: it shares the hybrid
mechanics but omits the adaptive schedule and the additional phases, so the gap
between this family and the next measures the specific contribution of the
adaptive dual-phase design.

### 3.5.5 Adaptive Dual-Phase PSO–GWO models (M13–M15)

The principal framework, applied to each engine in two phases.

**Phase 1 — Adaptive binary feature selection.** Continuous positions are mapped
to binary masks through a $V$-shaped transfer function
(Mirjalili & Lewis, 2013):

$$
T(v_j) = \lvert\tanh(v_j)\rvert, \qquad
m_j =
\begin{cases}
1, & \text{if } u_j < T(v_j) \text{ or } j\in\mathcal{C},\\
0, & \text{otherwise,}
\end{cases}\quad u_j\sim\mathcal{U}(0,1),
$$

so that core features $\mathcal{C}$ are locked on. With selected set
$\mathcal{S}(m)$ the objective rewards validation accuracy and penalises size:

$$
\Phi(m) = E_{\mathrm{cv}}\!\bigl(\mathcal{S}(m)\bigr) + 0.009\,\max\!\bigl(0,\ \lvert\mathcal{S}\rvert-\max(\lvert\mathcal{C}\rvert,16)\bigr).
$$

The search is repeated over several seeds $k$ and consolidated by
fitness-weighted stability voting, where $\phi_{k,j}$ is the selection frequency
of feature $j$ under seed $k$ and $f_k$ its best fitness:

$$
\omega_k = \frac{1/f_k}{\sum_{k'} 1/f_{k'}}, \qquad
s_j = \sum_k \omega_k\,\phi_{k,j}, \qquad
\hat{\mathcal{S}} = \{\,j : s_j\ge 0.5\,\}\cup\mathcal{C}.
$$

A **full-feature fallback** guarantees the subset must earn its place: the reduced
set is retained only if it beats the complete set on validation,
$E_{\mathrm{cv}}(\hat{\mathcal{S}}) < E_{\mathrm{cv}}(\text{all})/1.005$;
otherwise all features are used.

**Phase 2 — Adaptive continuous hyperparameter optimization.** On the selected
features, the adaptive hybrid ($\lambda=1-\tau^2$, §3.4.3) tunes the engine over
several seeds. The best candidates are then polished by a local Gaussian
random-walk refinement with annealed step size,

$$
\mathbf{x}' = \Pi\!\bigl(\mathbf{x}^{*} + \boldsymbol{\eta}\odot(\mathbf{ub}-\mathbf{lb})\bigr),
\qquad \boldsymbol{\eta}\sim\mathcal{N}(0,\sigma_t^2\mathbf{I}),\quad
\sigma_t = 0.14\left(1-\tfrac{t}{R-1}\right)+0.02,
$$

accepting $\mathbf{x}'$ when it lowers the objective.

Relative to the static-hybrid family, this framework adds four enhancements:
(i) the adaptive $\lambda$ schedule; (ii) an explicit feature-selection phase;
(iii) multi-seed stability voting with local refinement; and (iv) a larger search
budget. It attains the highest single-model accuracy and yields compact,
informative feature subsets, at the highest computational cost.

### 3.5.6 Adaptive Dual-Phase PSO–GWO Expert Fusion Ensemble (M16)

The flagship is a **stacked generalisation** (Wolpert, 1992; Breiman, 1996) over
M1–M15 that retrains nothing. Let $\mathbf{P}\in\mathbb{R}^{n\times K}$ collect the
$K$ experts' out-of-fold validation predictions aligned on common dates, with
target $\mathbf{a}$. Non-negative weights summing to one are obtained from a
ridge-regularised solve followed by clipping and renormalisation:

$$
\mathbf{w}_{\text{raw}} = \bigl(\mathbf{P}^{\top}\mathbf{P}+\rho\mathbf{I}\bigr)^{-1}\mathbf{P}^{\top}\mathbf{a},
\qquad
\mathbf{w} = \frac{\max(\mathbf{w}_{\text{raw}},0)}{\mathbf{1}^{\top}\max(\mathbf{w}_{\text{raw}},0)},
\qquad \mathbf{w}\ge 0,\ \mathbf{1}^{\top}\mathbf{w}=1.
$$

Several weighting schemes (ridge blends, equal-weight top-$k$, and the single best
expert) are scored by temporal cross-validation, and the test forecast is the
weighted blend $\hat{\mathbf{y}}^{\text{test}} = \mathbf{P}^{\text{test}}\mathbf{w}$.
Three safeguards ensure soundness:

1. **Pruning** — experts whose validation RMSE is far from the best are excluded,
   so weak learners cannot degrade the blend.
2. **Guarding** — the chosen scheme is compared against the single best expert
   under the same cross-validation and is adopted only if it does not lose, so the
   ensemble can never be worse than its best member.
3. **Regime awareness** — in strongly trending conditions, non-sequential experts
   failing trend criteria are excluded so the sequential model is not diluted.

The ensemble combines complementary strengths and reduces variance, and is by
construction at least as accurate as its best constituent; its quality depends on
the strength and diversity of the pool, and it performs no new learning itself.

---

## 3.6 Comparative Summary

**Table 3.1 — Optimization strategies.**

| Models | Strategy | Defining characteristic | Primary strength |
| :-- | :-- | :-- | :-- |
| M1–M3 | Baseline | Default hyperparameters, all features | Reference performance |
| M4–M6 | PSO | Particle-swarm tuning | Stable, reliable tuning |
| M7–M9 | GWO | Grey-wolf tuning | Strong exploitation |
| M10–M12 | Hybrid ($\lambda=\tfrac12$) | Static PSO–GWO blend | Robust single search |
| M13–M15 | Adaptive Dual-Phase | Adaptive blend + selection + refinement | Best single-model accuracy |
| M16 | Fusion Ensemble | Guarded weighted stack | Best overall robustness |

**Table 3.2 — Base learners.**

| Engine | Temporal modelling | Training cost | Principal strength | Principal limitation |
| :-- | :--: | :--: | :-- | :-- |
| LSTM | Yes (sequences) | High | Non-linear temporal dynamics | Cost; overfitting risk |
| ELM | No (single step) | Very low | Speed; ensemble diversity | Lower precision (random projection) |
| GBDT | No (single step) | Low–moderate | Feature interactions; robustness | No sequence; weak extrapolation |

The suite therefore subjects three learning algorithms to four optimization
strategies of increasing sophistication and finally combines them by a guarded
ensemble. The progression — baseline, single-optimizer tuning, static hybrid,
adaptive dual-phase, and fusion — provides a transparent account of where
predictive gains originate and isolates the contribution of each methodological
component.

---

## References

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time
series predictor evaluation. *Information Sciences, 191*, 192–213.

Breiman, L. (1996). Stacked regressions. *Machine Learning, 24*(1), 49–64.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting
machine. *Annals of Statistics, 29*(5), 1189–1232.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
*Neural Computation, 9*(8), 1735–1780.

Huang, G.-B., Zhu, Q.-Y., & Siew, C.-K. (2006). Extreme learning machine: Theory
and applications. *Neurocomputing, 70*(1–3), 489–501.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y.
(2017). LightGBM: A highly efficient gradient boosting decision tree.
*Advances in Neural Information Processing Systems, 30*.

Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
*Proceedings of the IEEE International Conference on Neural Networks, 4*,
1942–1948.

Lapedes, A., & Farber, R. (1987). *Nonlinear signal processing using neural
networks: Prediction and system modelling* (Technical Report LA-UR-87-2662).
Los Alamos National Laboratory.

Mirjalili, S., & Lewis, A. (2013). S-shaped versus V-shaped transfer functions
for binary particle swarm optimization. *Swarm and Evolutionary Computation, 9*,
1–14.

Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer.
*Advances in Engineering Software, 69*, 46–61.

Şenel, F. A., Gökçe, F., Yüksel, A. S., & Yiğit, T. (2019). A novel hybrid
PSO–GWO algorithm for optimization problems. *Engineering with Computers, 35*(4),
1359–1373.

Shi, Y., & Eberhart, R. (1998). A modified particle swarm optimizer.
*Proceedings of the IEEE International Conference on Evolutionary Computation*,
69–73.

Theil, H. (1966). *Applied Economic Forecasting*. North-Holland.

Wolpert, D. H. (1992). Stacked generalization. *Neural Networks, 5*(2), 241–259.
