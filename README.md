# FinSentNet-C

**Calibration-aware cross-modal learning for equity direction: gated news–price fusion,
conformal abstention, and growth-optimal position sizing.**

---

## Results status

> **There are no results in this repository yet, and none may be quoted from it.**
>
> Every table and figure currently produced comes from synthetic data or from a
> zero-signal panel, and is stamped as such. The real study — a point-in-time universe
> over 2016–2025 under purged, embargoed walk-forward retraining — has not been run.
>
> The previous version of this project (V2) circulated a manuscript containing a Sharpe
> ratio, an accuracy and an information coefficient that **were not produced by any
> experiment**. Those figures have been removed from every artifact here. `SPEC.md`
> section 7 records what they were and why they were implausible, because deleting the
> record would be its own kind of dishonesty.

## The claim this repository exists to test

Calibrated probability, not raw accuracy, determines whether a directional equity signal
survives position sizing. Specifically:

1. cross-modal gating improves **calibration** more than it improves accuracy; and
2. calibration-aware growth-optimal sizing converts a small, statistically fragile
   predictive edge into a materially better risk-adjusted outcome than the *same*
   predictions sized by uncalibrated softmax confidence.

The second claim is the headline experiment, and it is deliberately chosen to be
testable at a **small** edge. A daily-to-weekly equity model produces a rank information
coefficient around 0.02–0.04; a paper claiming to beat the market on that dies on
contact with a referee. A paper measuring *what miscalibration costs* thrives on it,
because the subject is the sizing, not the alpha.

## Why calibration is a quantity of money

Three propositions, derived in `finsent/decision/growth_theory.py` and verified against
simulation in `tests/test_growth_theory.py`.

For a position `f` in an asset with conditional mean `mu` and variance `sigma^2`:

```
G(f) = f*mu - 0.5*f^2*sigma^2       f* = mu/sigma^2       G(f*) = SR^2 / 2
```

**Proposition 1.** A wrong mean costs `dG = (mu - mu_hat)^2 / (2 sigma^2)` — exactly one
half the squared standardised forecast error. Since mean squared error decomposes into
calibration plus refinement, the *calibration* component of a forecaster's loss is
literally money per unit time.

**Proposition 2.** With `u = sigma^2 / sigma2_hat`, realised growth is `G(f*)(2u - u^2)`,
so the fraction destroyed is `(1-u)^2`. Underestimating variance by 30% destroys 18% of
the growth rate; underestimating it by **half destroys all of it** — `G = 0` exactly, on
a genuinely profitable signal. This is why Kelly sizing on overconfident probabilities
can lose to flat equal weighting.

**Proposition 3.** In the discrete case `E[dG] >= (C/2) * ECE^2`, the theory curve
plotted against measured fold-level ECE in Figure F6.

## Architecture

448,453 trainable parameters. Frozen FinBERT (110M) is cached offline and never enters
the training graph.

```
price (B,60,12) -> multi-scale causal conv {3,7,15} -> dilated TCN {1,2,4,8} -> GRU(128)
                                                        |  (B,60,128) and (B,128)
cached FinBERT (B,8,768) -> project -> attention pool -> (B,128)
                                                        |
                        gated cross-modal fusion: text queries price
                        c = MHA(Q=s, K=H, V=H);   g = sigmoid(W[s;p;c])
                        f = LayerNorm(g*c + (1-g)*p)      + modality dropout
                                                        |
                    direction logits (3) | mu | log-variance
```

The third head is the consequential one: `f = mu/sigma^2` needs a conditional variance,
and Proposition 2 prices getting it wrong.

| Component | V2 | Now | Why |
|---|---|---|---|
| Trainable parameters | ~8–12M | **448k** | ~220k effectively independent samples after overlap correction |
| Text branch | FinBERT + TextCNN + 3xBiLSTM + 8-head MHA + positional encoding | frozen cached FinBERT -> attention pooling | FinBERT already contextualises 12-word headlines; positional encoding after a recurrence is self-contradictory |
| Text at inference | lexicon score -> fabricated token id | the same encoder as training | the deployed system must be the one described |
| Lookback | 30 | **60** | TCN receptive field is 61; at 30 the deepest dilations read padding |
| Labels | fixed 0.5% band | volatility-scaled band + triple barrier | a fixed band makes class balance a volatility proxy |
| Splits | one 60/20/20 | purged + embargoed walk-forward, refit twice a year | a single split is not walk-forward |
| Sizing | binary Kelly on barrier odds x confidence | `f = kappa*mu/sigma^2`, clipped | a horizon probability with first-passage odds is a category error; the confidence multiplier double-counts |
| Early stopping | validation Sharpe | inner-validation NLL | stopping and calibrating on the same data is triple-dipping |
| Covariance | `Sigma + 1e-8*I` | Ledoit–Wolf shrinkage | the ridge is cosmetic |
| Regimes | if/else on moving averages | Gaussian HMM (Hamilton, 1989) | hand-tuned thresholds are uncounted fitted parameters |

## Two bugs this harness caught, recorded rather than quietly fixed

**A look-ahead leak worth five times the real signal.** Features dated `t` use the
**close** of day `t`, but an open-to-open label starting at `t` begins before that close,
so the two share day `t`'s return. On pure random-walk data this alone gave `ret_5` a
cross-sectional Rank-IC of **0.108**. After lagging one session it is **0.004**. No
splitter catches this; the model trains happily; the only symptom is a good-looking
number. Pinned by
`tests/test_causality.py::test_features_must_be_lagged_for_an_open_executed_decision`.

**Post-hoc calibration making calibration worse.** An unregularised six-parameter vector
scaling fitted on a small validation block cut inner-validation ECE from 0.020 to 0.004
while *raising* test ECE from 0.003 to 0.056 — it had learned the validation fold's class
balance, which drifted. Fixed with positive-constrained scales, L2 toward identity, and
guarded selection against an identity baseline on a later slice of the validation block.

## Layout

```
SPEC.md                  the single source of truth; every frozen constant
configs/base.yaml        the operational half of SPEC.md
finsent/
  config.py              typed config + validation + config_hash
  data/                  PIT universe, timestamp alignment, labels, purged splits,
                         causal indicators, uniqueness weights, panel
  models/                price/text encoders, gated fusion, three heads, baselines
  training/              objectives, calibration, conformal, walk-forward
  decision/              growth theory, sizing, costs, portfolio, regimes
  eval/                  metrics, significance tests, deflated Sharpe, backtest, report
experiments/             runnable studies
tests/                   190 tests, including the leakage and causality contracts
```

`finsent.config`, `finsent.data`, `finsent.decision` and `finsent.eval` depend only on
NumPy/pandas/SciPy. The entire evaluation and decision stack — purged splits, metrics,
significance tests, calibration, conformal prediction, backtest — reproduces **without
PyTorch**. Only the network itself needs it.

## Reproduce

```bash
pip install -e ".[dev]"          # add [torch] to train the network
pytest                           # 190 tests
python -m finsent.eval.report --signal random --out paper --label skeleton
python experiments/00_smoke_end_to_end.py --out runs/smoke
```

The second command is the Phase-1 acceptance test: the complete set of nine tables and
the figures renders on a **zero-signal** panel. The manuscript skeleton therefore exists
before any result does, which is what makes it structurally awkward to write tables first
and produce the numbers later.

The third runs the whole protocol on synthetic data with a known planted signal and
recovers a Rank-IC consistent with what was planted.

## Reporting rules (`SPEC.md` section 6)

1. No number is ever typed by hand. Every table and figure carries `git_sha`,
   `config_hash`, `data_hash` and the seed list.
2. Every reported quantity carries a confidence interval and at least three seeds.
3. The number of evaluated configurations is **declared** and fed to the Deflated Sharpe
   Ratio.
4. Mandatory: Rank-IC/ICIR, decile spreads, Diebold–Mariano against each baseline,
   Hansen SPA across the model zoo, Deflated and Probabilistic Sharpe, factor-regression
   alpha with HAC t-statistics, turnover, the net-of-cost curve, ECE/MCE/Brier,
   reliability diagrams, conformal coverage.

## Expected magnitudes, and when to suspect yourself

If a run lands outside these bands, audit for leakage **before** celebrating.

| Metric | Realistic | Audit above |
|---|---|---|
| 3-class accuracy | 41–44% | 48% |
| Binary accuracy (excluding neutral) | 51–53% | 56% |
| Daily Rank-IC | 0.015–0.035 | **0.06** |
| Net Sharpe @10bps, h=5 | 0.3–0.8 | 1.2 |
| ECE after calibration | 0.010–0.030 | — |

Daily rebalancing is uninvestable here and the arithmetic says so plainly: a 0.02 IC
against 2% cross-sectional dispersion yields roughly 9 bps a day gross, which 100% daily
turnover at 10 bps consumes entirely. The primary horizon is therefore **five days**.

## Not investment advice

Research code. No capital has been deployed against it. Nothing in this repository is a
recommendation to buy or sell any security.

## License

MIT.
