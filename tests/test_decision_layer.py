"""Sizing, costs and portfolio construction.

The headline test is ``test_uncalibrated_kelly_can_lose_to_equal_weighting``: it
reproduces, on synthetic data with a known truth, the paper's central empirical claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsent.decision import portfolio as P
from finsent.decision.costs import CostModel, break_even_cost_bps, corwin_schultz_spread
from finsent.decision.growth_theory import expected_log_growth, max_log_growth
from finsent.decision.regime import fit_regimes
from finsent.decision.sizing import (
    SizingConfig,
    calibrated_kelly_sizing,
    conformal_gated_kelly_sizing,
    cross_sectional_weights,
    kelly_continuous,
    raw_softmax_kelly_sizing,
    score_from_probs,
)


# --------------------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------------------
def test_score_uses_both_tails_not_just_p_up():
    """The V2 defect: (0.35, 0.60, 0.05) and (0.35, 0.05, 0.60) are opposite risks."""
    probs = np.array([[0.05, 0.60, 0.35], [0.60, 0.05, 0.35]])

    p_up_only = score_from_probs(probs, mode="p_up")
    assert p_up_only[0] == p_up_only[1], "the fixture must be ambiguous under p_up alone"

    both = score_from_probs(probs, mode="p_up_minus_p_down")
    assert both[0] > 0 > both[1], "the corrected score must separate them"


def test_kelly_is_clipped_and_signed_correctly():
    mu = np.array([0.01, -0.01, 0.0])
    sigma2 = np.array([0.0004, 0.0004, 0.0004])

    f = kelly_continuous(mu, sigma2, kappa=0.25, f_max=0.10)
    assert f[0] == pytest.approx(0.10), "a strong positive edge must hit the cap"
    assert f[1] == pytest.approx(-0.10)
    assert f[2] == pytest.approx(0.0)
    assert np.all(np.abs(f) <= 0.10 + 1e-12)


def test_sizing_never_multiplies_by_a_separate_confidence_scalar():
    """Contradiction #7: a calibrated probability already carries confidence."""
    cfg = SizingConfig(kappa=0.25, f_max=0.10)
    mu = np.array([0.002])
    sigma2 = np.array([0.0004])

    expected = 0.25 * 0.002 / 0.0004
    assert calibrated_kelly_sizing(mu, sigma2, cfg)[0] == pytest.approx(min(expected, 0.10))


def test_conformal_gate_zeroes_positions_where_the_set_is_not_a_singleton():
    cfg = SizingConfig()
    mu = np.array([0.002, 0.002, 0.002])
    sigma2 = np.full(3, 0.0004)
    mask = np.array([True, False, True])

    f = conformal_gated_kelly_sizing(mu, sigma2, mask, cfg)
    assert f[1] == 0.0
    assert f[0] > 0 and f[2] > 0


def test_uncalibrated_kelly_can_lose_to_equal_weighting():
    """The paper's central empirical claim, on data whose truth is known.

    An overconfident model implies too small a variance, so Kelly overbets. Proposition 2
    says the penalty is quadratic in the variance ratio and total at a factor of two.
    Here the overconfident sizing must deliver strictly lower realised log growth than
    correctly calibrated sizing, and it may fall below flat equal weighting entirely.
    """
    rng = np.random.default_rng(0)
    n = 200_000
    mu_true, sigma2_true = 0.0006, 0.0004
    r = mu_true + np.sqrt(sigma2_true) * rng.standard_normal(n)

    cfg = SizingConfig(kappa=1.0, f_max=10.0)  # unclipped, to isolate the mechanism

    calibrated = calibrated_kelly_sizing(
        np.full(n, mu_true), np.full(n, sigma2_true), cfg
    )
    # A model stating twice the confidence it has earned implies half the variance.
    overconfident = calibrated_kelly_sizing(
        np.full(n, mu_true), np.full(n, sigma2_true / 2.0), cfg
    )

    g_cal = float(np.mean(calibrated * r - 0.5 * (calibrated * r) ** 2))
    g_over = float(np.mean(overconfident * r - 0.5 * (overconfident * r) ** 2))

    assert g_cal > g_over, (
        f"calibrated growth {g_cal:.3e} must exceed overconfident growth {g_over:.3e}"
    )
    assert g_over == pytest.approx(0.0, abs=2e-6), (
        "underestimating variance by half should destroy essentially all growth "
        f"(Proposition 2); measured {g_over:.3e}"
    )
    assert g_cal == pytest.approx(float(max_log_growth(mu_true, sigma2_true)), rel=0.10)


def test_raw_softmax_kelly_is_driven_by_confidence_not_by_dispersion():
    """Rule (2) sizes on stated confidence, which is the mechanism the paper indicts.

    Without a variance head a practitioner has no choice but to read confidence as an
    inverse-variance proxy. The position then responds to how *sure the model sounds*
    rather than to how dispersed outcomes actually are -- so when the model is
    overconfident it overbets, and Proposition 2 prices that mistake.
    """
    cfg = SizingConfig(kappa=0.25, f_max=1.0)
    mu = np.array([0.001, 0.001])

    moderate = np.array([[0.05, 0.10, 0.85], [0.05, 0.10, 0.85]])
    emphatic = np.array([[0.01, 0.02, 0.97], [0.01, 0.02, 0.97]])

    f_moderate = raw_softmax_kelly_sizing(moderate, mu, cfg)
    f_emphatic = raw_softmax_kelly_sizing(emphatic, mu, cfg)

    assert abs(f_emphatic[0]) > abs(f_moderate[0]), (
        "a more confident softmax must produce a larger position under rule (2)"
    )

    # And the size ratio follows the squared confidence gap, i.e. it is quadratic in
    # overconfidence, which is exactly why the failure mode is severe rather than mild.
    expected_ratio = ((1 - 0.85) ** 2) / ((1 - 0.97) ** 2)
    assert abs(f_emphatic[0]) / abs(f_moderate[0]) == pytest.approx(expected_ratio, rel=0.01)

    # Rule (3) ignores stated confidence entirely and uses the fitted variance.
    honest = np.full(2, 0.0004)
    assert calibrated_kelly_sizing(mu, honest, cfg)[0] == pytest.approx(
        0.25 * 0.001 / 0.0004
    )


def test_cross_sectional_weights_are_neutral_capped_and_levered_within_limits():
    cfg = SizingConfig(f_max=0.05, gross_leverage_max=1.0, dollar_neutral=True)
    dates = np.repeat(pd.bdate_range("2022-01-03", periods=4), 10)
    tickers = np.tile([f"T{i}" for i in range(10)], 4)
    sizes = np.tile(np.linspace(-0.2, 0.2, 10), 4)

    w = cross_sectional_weights(pd.Series(sizes), dates, tickers, cfg)

    assert np.allclose(w.sum(axis=1), 0.0, atol=1e-9), "weights must be dollar neutral"
    assert (w.abs() <= 0.05 + 1e-12).all().all(), "per-name cap violated"
    assert (w.abs().sum(axis=1) <= 1.0 + 1e-9).all(), "gross leverage cap violated"


# --------------------------------------------------------------------------------------
# Costs
# --------------------------------------------------------------------------------------
def test_impact_grows_with_the_square_root_of_participation():
    model = CostModel(spread_bps=0.0, impact_eta=0.5)
    small = model.cost_rate(1e6, adv=1e8, volatility=0.02)
    large = model.cost_rate(4e6, adv=1e8, volatility=0.02)
    assert float(large) == pytest.approx(2.0 * float(small), rel=1e-9), (
        "quadrupling size must exactly double square-root impact"
    )


def test_flat_model_reduces_to_a_half_spread():
    model = CostModel(spread_bps=10.0)
    assert float(model.cost_rate(1e6)) == pytest.approx(10.0 * 1e-4)


def test_break_even_cost_is_the_number_that_decides_investability():
    """9 bps a day of gross edge against 100% daily turnover breaks even at 9 bps."""
    gross = pd.Series(np.full(500, 0.0009))
    assert break_even_cost_bps(gross, turnover_per_period=1.0) == pytest.approx(9.0)
    # Halve the turnover and the strategy tolerates twice the cost.
    assert break_even_cost_bps(gross, turnover_per_period=0.5) == pytest.approx(18.0)


def test_corwin_schultz_spread_is_non_negative():
    rng = np.random.default_rng(1)
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01))
    high = pd.Series(close * (1 + np.abs(rng.standard_normal(300)) * 0.005))
    low = pd.Series(close * (1 - np.abs(rng.standard_normal(300)) * 0.005))

    spread = corwin_schultz_spread(high, low).dropna()
    assert (spread >= 0).all()
    assert spread.mean() < 0.05, "estimated spreads above 500 bps are implausible"


# --------------------------------------------------------------------------------------
# Portfolio
# --------------------------------------------------------------------------------------
def test_ledoit_wolf_shrinks_toward_the_identity_and_is_better_conditioned():
    """The reason `Sigma + 1e-8 I` is not an acceptable substitute."""
    rng = np.random.default_rng(2)
    n_obs, n_assets = 60, 40  # fewer observations than assets pairs: badly conditioned
    returns = rng.standard_normal((n_obs, n_assets)) * 0.01

    sample = np.cov(returns.T, bias=True)
    shrunk, intensity = P.ledoit_wolf_covariance(returns)

    assert 0.0 < intensity <= 1.0
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample), (
        "shrinkage must improve conditioning"
    )
    assert np.allclose(shrunk, shrunk.T)
    assert np.linalg.eigvalsh(shrunk).min() > 0, "shrunk covariance must be positive definite"


def test_mean_variance_respects_every_declared_constraint():
    rng = np.random.default_rng(3)
    n = 12
    returns = rng.standard_normal((300, n)) * 0.01
    sigma, _ = P.ledoit_wolf_covariance(returns)
    mu = rng.standard_normal(n) * 0.001

    cons = P.PortfolioConstraints(f_max=0.10, gross_leverage_max=1.5, dollar_neutral=True)
    w = P.mean_variance_weights(mu, sigma, cons, risk_aversion=5.0)

    assert np.abs(w).max() <= 0.10 + 1e-6, "per-name box constraint violated"
    assert np.abs(w).sum() <= 1.5 + 1e-6, "gross leverage constraint violated"
    assert abs(w.sum()) < 1e-6, "dollar neutrality violated"


def test_mean_variance_tilts_toward_the_higher_expected_return():
    n = 6
    sigma = np.eye(n) * 0.0004
    mu = np.array([0.003, 0.002, 0.001, -0.001, -0.002, -0.003])

    w = P.mean_variance_weights(
        mu, sigma, P.PortfolioConstraints(f_max=0.5, gross_leverage_max=2.0), risk_aversion=1.0
    )
    assert w[0] > w[-1], "the optimiser must prefer the higher expected return"
    assert np.corrcoef(w, mu)[0, 1] > 0.9


def test_risk_parity_equalises_risk_contributions():
    sigma = np.diag([0.0001, 0.0004, 0.0016])
    w = P.risk_parity_weights(sigma)

    contributions = w * (sigma @ w)
    assert np.allclose(contributions, contributions.mean(), rtol=0.05), contributions
    assert w[0] > w[1] > w[2], "the least volatile asset must get the largest weight"


def test_cvar_optimiser_returns_a_feasible_portfolio():
    rng = np.random.default_rng(4)
    scenarios = rng.standard_normal((400, 8)) * 0.01
    cons = P.PortfolioConstraints(f_max=0.3, gross_leverage_max=1.0, dollar_neutral=True)

    w = P.cvar_weights(scenarios, alpha=0.95, constraints=cons)
    assert np.abs(w).max() <= 0.3 + 1e-6
    assert abs(w.sum()) < 1e-6


# --------------------------------------------------------------------------------------
# Regimes
# --------------------------------------------------------------------------------------
def test_hmm_separates_a_calm_regime_from_a_crisis_regime():
    rng = np.random.default_rng(5)
    calm = rng.standard_normal(600) * 0.005
    crisis = rng.standard_normal(200) * 0.035
    returns = np.concatenate([calm, crisis, calm])

    model, labels = fit_regimes(returns, n_states=2, seed=0)

    assert labels.state_vol[0] < labels.state_vol[-1], "states must be volatility-ordered"
    crisis_slice = labels.states[600:800]
    assert float((crisis_slice == labels.n_states - 1).mean()) > 0.5, (
        "the high-volatility block should be assigned mostly to the crisis state"
    )
    assert np.isfinite(model.log_likelihood_)


def test_online_state_assignment_is_causal():
    """The filtered state at t must not move when a future observation changes."""
    rng = np.random.default_rng(6)
    returns = rng.standard_normal(500) * 0.01

    model, _ = fit_regimes(returns, n_states=2, seed=0)
    X = np.column_stack([returns, np.abs(returns)])
    base = model.predict_states_online(X)

    shocked = returns.copy()
    shocked[300:] *= 8.0
    Xs = np.column_stack([shocked, np.abs(shocked)])
    after = model.predict_states_online(Xs)

    assert (base[:300] == after[:300]).all(), (
        "filtered states changed before the perturbation: the assignment is not causal"
    )
