# SPEC.md — FinSentNet-C Single Source of Truth

> **This file is normative.** Every constant quoted in the paper, the README, or any
> figure MUST equal the value in `configs/base.yaml`. `tests/test_spec_matches_config.py`
> enforces this mechanically. If a number appears in a manuscript and not here, it is
> not a result.

**Version:** 0.3.0 · **Status:** research rebuild (FinSentNet V2 → FinSentNet-C)

---

## 0. The single claim

> Calibrated probability, not raw accuracy, determines whether a directional equity
> signal survives position sizing. We show (a) cross-modal gating improves calibration
> more than it improves accuracy, and (b) calibration-aware growth-optimal sizing
> converts a small, statistically fragile predictive edge into a materially better
> risk-adjusted outcome than the same predictions sized by uncalibrated softmax
> confidence.

Every design decision below serves that claim. Components that do not serve it live in
an appendix or are deleted.

---

## 1. Resolved contradictions (previously inconsistent across paper / math doc / code)

These eight items were mutually contradictory in the V2 artifacts. Each now has exactly
one value, and that value is enforced by a test.

| # | Item | **Frozen decision** | Enforced by |
|---|------|---------------------|-------------|
| 1 | Label convention | `0 = DOWN`, `1 = NEUTRAL`, `2 = UP` | `tests/test_label_convention.py` |
| 2 | Class weights | **Inverse frequency.** Majority class (NEUTRAL) receives the *smallest* weight. The V2 vector `(0.3, 0.4, 0.3)` up-weighted the majority class and is deleted. | `tests/test_losses.py` |
| 3 | Target / sizing | No barrier-odds Kelly. Continuous growth-optimal sizing `f = kappa * mu_hat / sigma2_hat`, clipped. `m_T`, `P0*(1+|r|)` and the ATR-target formula are deleted from the research path. | `tests/test_growth_theory.py` |
| 4 | Loss weights | One objective: `L = L_dir + 1.0*L_NLL + 0.5*L_MMCE + 0.2*L_rank`. All other weightings are ablations. | `configs/base.yaml::loss` |
| 5 | Early-stopping patience | **8**, everywhere (text, figures, code). V2 had 10 in prose and 15 in a figure. | `configs/base.yaml::train.patience` |
| 6 | Feature count `F` | **12**, cross-sectionally ranked. V2 claimed 15 / 20 / ~35 in three places. | `tests/test_spec_matches_config.py` |
| 7 | Confidence multiplier | Deleted. Sizing does **not** multiply by a separate confidence scalar: a calibrated probability already carries that information and multiplying double-counts it. | `finsent/decision/sizing.py` |
| 8 | Text path at inference | The **same** encoder path runs at train and inference. The lexicon-score → fake-token-id bridge is deleted with prejudice. | `finsent/models/text_encoder.py` |

---

## 2. Frozen constants

### 2.1 Label convention (never re-derive this)

```
DOWN = 0
NEUTRAL = 1
UP = 2
```

### 2.2 Shapes and horizons

| Symbol | Value | Meaning |
|---|---|---|
| `L` | 60 | price lookback in trading days |
| `F` | 12 | features per day |
| `K` | 8 | max headlines retained per (ticker, day) |
| `h` | 5 | label horizon in trading days (**primary**; 1 and 21 are robustness rows) |
| `d_model` | 128 | model width throughout |

`L = 60` is not arbitrary. The TCN receptive field is

```
R = 1 + 2*(k-1)*sum(dilations) = 1 + 2*2*(1+2+4+8) = 61 > 60
```

so every output position sees the entire window. At the V2 value `L = 30`, the deepest
dilations read only zero-padding — a silent architectural defect.

### 2.3 Timestamp discipline

A headline published at `pub_ts` may inform the decision executed at the open of
session `t` if and only if

```
open_ts(t) >= pub_ts + min_lag_hours,      min_lag_hours = 24
```

Returns are measured **open-to-open**, from `open(t)` to `open(t+h)`. Close-to-close
labelling with a decision nominally taken "at t" is the most common hidden leak in this
literature and is prohibited.

**Feature lag (found by this specification's own audit threshold).** Daily features dated
`t` are computed from the **close** of day `t`, but an open-to-open label starting at `t`
begins *before* that close. Decomposing the label into close-to-close terms,

```
log(O[t+5] / O[t]) = r[t] + r[t+1] + ... + r[t+4] + overnight terms
ret_5[t]           = r[t-4] + r[t-3] + ... + r[t]
```

the two share `r[t]`. On pure random-walk data this alone produced a cross-sectional
Rank-IC of **0.108** for `ret_5` — five times a realistic value and entirely spurious.
Every feature is therefore lagged one session
(`finsent.data.features_causal.lag_for_open_execution`, applied by default), after which
the same measurement gives **0.004**. A decision executed at the open of `t` may use
information through the close of `t-1` and no later. Regression test:
`tests/test_causality.py::test_features_must_be_lagged_for_an_open_executed_decision`.

This is worth recording rather than quietly fixing: no splitter catches it, the model
trains happily, and the only symptom is a headline number that looks good.

### 2.4 Labels

Volatility-scaled dead band (causal):

```
theta[i,t] = k_band * sigma_hat[i,t] * sqrt(h)
sigma_hat[i,t] = EWMA_60(|r[i,.]|)          # causal, strictly past
k_band = 0.6

y = UP      if  r_fwd >  +theta
y = DOWN    if  r_fwd <  -theta
y = NEUTRAL otherwise
```

A fixed 0.5% band (V2) makes "UP" mean different things for a 15%-vol utility and an
80%-vol biotech, and different things in 2018 and 2020; class balance then becomes a
volatility proxy rather than a signal property.

The full triple barrier (upper / lower / vertical) is implemented as an option and is
required whenever barrier-defined odds are used anywhere.

### 2.5 Splits

```
|<-- TRAIN (expanding, >= 3y) -->|purge h|<- INNER VAL 6m ->|purge h + embargo|<- TEST 6m ->|
```

- `purge_days = 5` (equal to `h`)
- `embargo_pct = 0.01`
- `refit_every_months = 6` → **18 folds** over 2016-01-01 … 2025-06-30
- Hyperparameters, early stopping, vector scaling, and conformal quantiles are **all**
  fitted on the inner-validation split. The test split is touched exactly once, at
  report time. This is what removes the V2 triple-dipping defect.

### 2.6 Feature set (`F = 12`)

```
ret_1, ret_5, ret_21, mom_12_1, rev_5, ewma_vol_20,
atr14_norm, gk_vol_20, macd_hist_norm, rsi_14, bb_pctb, dollar_vol_log
```

All causal. All cross-sectionally ranked per day to `[-0.5, 0.5]`. Winsorised at 1%.
Candlestick pattern flags are deleted from the default path (near-zero documented
predictive power on daily bars).

---

## 3. Architecture — FinSentNet-C

### 3.1 Price encoder

```
X [60,12] -> LayerNorm
          -> multi-scale causal Conv1d, kernels {3,7,15}, 32ch each -> concat 96
          -> pointwise Conv1d 96->64, GELU
          -> dilated causal TCN: 4 residual blocks, dilations {1,2,4,8}, k=3, 64ch
          -> GRU(64 -> 128), unidirectional, 1 layer
          -> H [60,128];  p = LayerNorm(h_60) [128]
```

### 3.2 Text encoder

FinBERT is **frozen and cached offline**; it never enters the training graph.

```
offline:  e_j = FinBERT_CLS(headline_j) [768]   -> parquet cache
online:   e~_j = LayerNorm(W_e e_j) [128]
          a_j = softmax_j( w^T tanh(W_a [e~_j ; phi(dt_j) ; src_j]) )   (masked)
          s   = LayerNorm(MLP(sum_j a_j e~_j)) [128]
          s   = s_null  if the day has no headlines   (learned parameter)
```

Deleted from V2: TextCNN, the 3-layer BiLSTM stack, the 8-head self-attention block on
top of FinBERT, the sinusoidal positional encoding after a recurrent layer, and the
three-way gated sentence combination. FinBERT already contextualises; that stack was
overfitting surface on ~12-word headlines.

### 3.3 Gated cross-modal fusion (one block)

```
c = MHA(Q = s, K = H, V = H)      4 heads, d_k = 32
g = sigmoid(W_g [s ; p ; c])      elementwise gate in (0,1)^128
f = LayerNorm(g * c + (1 - g) * p)
```

**Modality dropout** (training only): independently with p = 0.15 replace `s <- s_null`,
and with p = 0.15 replace `p <- 0`. Never both. This buys (i) no modal collapse,
(ii) graceful degradation when the news feed fails, (iii) a test-time single-modality
ablation that needs no retraining.

### 3.4 Heads

```
z = Dropout(GELU(W_t f))
direction logits u = W_d z            [3]
conditional mean  mu_hat = w_mu^T z   [1]
log variance      logvar = clamp(w_v^T z, -10, 2)   [1]
```

The heteroscedastic third head is the single most important change: it supplies the
`sigma_hat` that continuous Kelly requires and it eliminates the incoherent
"confidence sigmoid with a learnable temperature" of V2.

### 3.5 Parameter budget (trainable)

Realised counts, emitted by `FinSentNetC.parameter_counts()` and asserted against this
table by `tests/test_model.py::test_parameter_count_matches_the_specification`.

| Module | Params | % |
|---|---|---|
| price: multi-scale conv | 9,696 | 2.2 |
| price: pointwise mixer | 6,208 | 1.4 |
| price: TCN (4 blocks) | 99,328 | 22.1 |
| price: GRU(64→128) | 74,496 | 16.6 |
| price: norms | 280 | 0.1 |
| text: projection 768→128 | 98,688 | 22.0 |
| text: attention pooling + MLP | 26,624 | 5.9 |
| text: source embeddings | 264 | 0.1 |
| text: null embedding | 128 | 0.0 |
| fusion: MHA (Q,K,V,O) | 66,048 | 14.7 |
| fusion: gate + norm | 49,536 | 11.0 |
| heads: trunk + 3 outputs | 17,157 | 3.8 |
| **TOTAL** | **448,453** | 100 |

Receptive field: **61** timesteps ≥ `L = 60`, verified at runtime.

Frozen FinBERT (110M) is an offline cache and is not part of the training graph.

**Capacity argument for the paper:** ~452k parameters against ~880k training samples at
average uniqueness ≈ 0.25 → ≈ 220k effective independent samples, a ~2:1
sample-to-parameter ratio. V2 was roughly 20× worse.

---

## 4. Objective

```
L = L_dir + lambda_reg * L_NLL + lambda_cal * L_MMCE + lambda_rank * L_rank
    lambda_reg  = 1.0
    lambda_cal  = 0.5
    lambda_rank = 0.2
```

- `L_dir` — cross-entropy, **inverse-frequency** class weights, per-sample average
  uniqueness weights.
- `L_NLL` — Gaussian negative log-likelihood `0.5*(logvar + (r-mu)^2 * exp(-logvar))`,
  with a **3-epoch sigma warmup** (logvar detached and zeroed). Without the warmup the
  model minimises NLL by inflating variance on hard samples and never learns the mean.
- `L_MMCE` — Kumar, Sarawagi & Jain (ICML 2018), Laplacian kernel. Differentiable,
  unlike hard-binned ECE, which V2 incorrectly used as a training penalty.
- `L_rank` — soft-Spearman within each date; optimises the Rank-IC that we report.

Focal loss is an ablation flag, **off by default**, with the Mukhoti et al. (2020)
calibration interaction documented.

---

## 5. Decision layer

### 5.1 Score

```
score[i,t] = mu_hat[i,t]                     (primary)
score[i,t] = P_up[i,t] - P_down[i,t]         (robustness)
```

Never `P_up` alone: V2's rule assigned identical actions to `(0.35, 0.60, 0.05)` and
`(0.35, 0.05, 0.60)`, which are opposite risk profiles.

### 5.2 Conformal gate

Split conformal (APS, **randomised**) fitted on the inner-validation fold; **trade only
when the prediction set is a singleton**.

The randomisation is not optional. With three classes the non-randomised APS score is a
coarse discrete variable, and measured coverage came out at 1.000 against a nominal 0.90
— the predictor simply admitted every class, and the singleton gate never fired. The
randomised score restores coverage to within 0.005 of nominal across α ∈ {0.05, 0.10,
0.20}. The draw is seeded, so trading decisions stay reproducible.

Note also that coverage is floored by top-1 accuracy, because a set is never allowed to
be empty. At α = 0.5 the nominal level is below that floor and the predictor
over-covers; report that rather than presenting the floor as a result.

Adaptive update under shift (Gibbs & Candès, 2021):

```
alpha[t+1] = alpha[t] + gamma * (alpha_target - 1{y_t not in C_t}),   gamma = 0.005
```

### 5.3 Growth-optimal sizing

```
f* = mu / sigma^2                 G(f*) = mu^2 / (2 sigma^2) = SR^2 / 2
f_used = clip(kappa * mu_hat / sigma2_hat, -f_max, f_max)
kappa = 0.25,  f_max = 0.10
```

### 5.4 The three propositions (paper §6, implemented in `finsent/decision/growth_theory.py`)

**Proposition 1 (mean miscalibration).**
`dG = (mu - mu_hat)^2 / (2 sigma^2)`
The log-growth penalty is exactly one half the squared standardised forecast error.
Via Murphy's (1973) decomposition, the calibration component of the loss is literally a
quantity of money per unit time.

**Proposition 2 (variance miscalibration).** With `u = sigma^2 / sigma2_hat`:
`G(f_hat) = G(f*) * (2u - u^2)`  hence  `dG / G(f*) = (1 - u)^2`
Corollaries: underestimating variance by 30% destroys 18% of the growth rate;
underestimating it by **half** (`u = 2`) destroys **100%** of it — `G = 0` exactly —
while the signal is still genuinely positive.

**Proposition 3 (discrete case).** For binary Kelly with odds `b`,
`f*(p) = ((b+1)p - 1)/b` is linear in `p`, so
`dG ≈ 0.5 * |G''(f*)| * ((b+1)/b)^2 * (p - p_hat)^2`  and  `E[dG] >= (C/2) * ECE^2`.

### 5.5 Portfolio and costs

- Covariance: **Ledoit–Wolf shrinkage**. `Sigma + 1e-8 I` is cosmetic and is deleted.
- Constrained mean-variance QP with box limits, gross-leverage cap, dollar neutrality
  and an L1 turnover penalty. The V2 analytic clip-and-renormalise is not the
  constrained max-Sharpe portfolio and is demoted to a baseline.
- Costs: `c = 0.5*spread + eta * sigma * sqrt(Q / ADV)`, `eta = 0.5`. Reported as a
  net-Sharpe-vs-cost **curve** over `{0,2,5,10,15,20,25}` bps, not a single number.

---

## 6. Reporting rules

1. **No number is ever typed by hand.** Every table and figure is emitted by
   `finsent/eval/report.py` and carries `git_sha`, `config_hash`, `data_hash` and the
   seed list.
2. Every reported quantity carries a confidence interval and ≥ 3 seeds.
3. The number of evaluated configurations (`eval.n_configs_evaluated`) is **declared**
   and fed to the Deflated Sharpe Ratio. Declaring it is what separates a scientist
   from a data miner.
4. Mandatory outputs: Rank-IC / ICIR, decile spreads, Diebold–Mariano vs each baseline,
   Hansen SPA across the model zoo, Deflated & Probabilistic Sharpe, FF5+MOM+STR alpha
   with HAC t-statistics, turnover, net-of-cost curve, capacity, ECE/MCE/Brier,
   reliability diagrams, conformal coverage.

## 7. Expected magnitudes (audit thresholds)

If a run lands outside these bands, **audit for leakage before celebrating**.

| Metric | Realistic | Audit if above |
|---|---|---|
| 3-class accuracy | 41–44% | > 48% |
| Binary accuracy (excl. neutral) | 51–53% | > 56% |
| Daily Rank-IC | 0.015–0.035 | **> 0.06** |
| Net Sharpe @10bps, h=5 | 0.3–0.8 | > 1.2 |
| ECE after calibration | 0.010–0.030 | — |

For reference, the V2 manuscript reported IC = 0.168 and accuracy = 57.4%, which are
3–8× above the audit threshold. Those figures were never produced by an experiment and
have been removed from every artifact in this repository.

---

## 8. Prohibited (do not reintroduce)

- Any hand-tuned points-based scoring heuristic ("TechScore": +20 for RSI<30, etc.).
- Five-level BUY/SELL bucket mapping anywhere in the research path (UI concern only).
- TextCNN / BiLSTM stack / self-attention on top of FinBERT / positional encoding after
  a recurrent layer.
- Any sentiment-score → token-id bridge.
- `if/else` regime classifiers — use the Gaussian HMM.
- Random (non-temporal) shuffling of any split.
- A current constituent list applied retroactively to a historical period.
- Early stopping on a financial metric computed on data also used for calibration.
