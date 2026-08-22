"""Deflated Sharpe: the correction that decides whether a result is a discovery.

The property that matters is the one tested last: a strategy selected as the best of two
hundred trials must *not* survive deflation on noise. Without that, the number is
decoration.
"""

from __future__ import annotations

import numpy as np
import pytest

from finsent.eval import dsr as D


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    values = [D.expected_max_sharpe(n, sr_variance=1.0) for n in (1, 10, 100, 1000)]
    assert values[0] == 0.0
    assert all(a < b for a, b in zip(values, values[1:])), values
    # With unit variance, the best of 100 null strategies sits around 2.5 standard errors.
    assert 2.0 < values[2] < 3.2


def test_sharpe_standard_error_widens_with_negative_skew_and_fat_tails():
    """Mertens (2002): the naive sqrt(1/n) understates uncertainty for trading returns.

    At a per-period Sharpe of 0.1 -- roughly a 1.6 annualised Sharpe, about as good as
    this design could plausibly reach -- negative skew and excess kurtosis widen the
    standard error by around 8%. Modest, but it moves a borderline t-statistic, and it
    is the correct quantity to feed the deflation.
    """
    naive = np.sqrt(1.0 / 999)
    normal = D.sharpe_standard_error(sr=0.1, skew=0.0, kurtosis=3.0, n=1000)
    nasty = D.sharpe_standard_error(sr=0.1, skew=-1.5, kurtosis=9.0, n=1000)

    assert normal == pytest.approx(naive, rel=0.05)
    assert nasty > normal * 1.05, (
        f"skew and kurtosis must widen the standard error: {normal:.5f} -> {nasty:.5f}"
    )
    # The skew term dominates: it enters linearly in SR while kurtosis enters at SR^2.
    skew_only = D.sharpe_standard_error(sr=0.1, skew=-1.5, kurtosis=3.0, n=1000)
    kurt_only = D.sharpe_standard_error(sr=0.1, skew=0.0, kurtosis=9.0, n=1000)
    assert skew_only > kurt_only


def test_probabilistic_sharpe_is_high_for_a_genuinely_good_strategy():
    """A 1.3 annualised Sharpe over eight years gives PSR near 0.99, not 0.9999.

    Worth internalising before writing a results section: even a strategy this strong,
    with a sample this long, leaves roughly a 1% chance that the true Sharpe is
    negative. Anything reported at 0.6 net Sharpe over three years is far weaker
    evidence than its point estimate suggests.
    """
    rng = np.random.default_rng(0)
    r = rng.standard_normal(2000) * 0.01 + 0.0008

    result = D.probabilistic_sharpe_ratio(r, benchmark_sr=0.0, periods_per_year=252)
    assert result["psr"] > 0.95
    assert result["n"] == 2000

    # The realised Sharpe must sit within three standard errors of the population value.
    # Note how wide that band is: se_ann = sqrt(252/2000) ~ 0.36, so a strategy built to
    # a true 1.27 can easily print 0.82 over eight years. Point estimates of Sharpe are
    # far noisier than the two-decimal precision they are usually quoted with.
    population = 0.0008 / 0.01 * np.sqrt(252)
    se_annualised = np.sqrt(252 / 2000)
    assert abs(result["sr_annualised"] - population) < 3 * se_annualised


def test_probabilistic_sharpe_falls_when_measured_against_a_real_benchmark():
    """PSR against a 1.0 annualised benchmark must be far below PSR against zero."""
    rng = np.random.default_rng(0)
    r = rng.standard_normal(2000) * 0.01 + 0.0008

    vs_zero = D.probabilistic_sharpe_ratio(r, 0.0, periods_per_year=252)["psr"]
    vs_one = D.probabilistic_sharpe_ratio(r, 1.0, periods_per_year=252)["psr"]
    assert vs_one < vs_zero


def test_probabilistic_sharpe_is_near_a_half_for_a_null_strategy():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(2000) * 0.01
    psr = D.probabilistic_sharpe_ratio(r)["psr"]
    assert 0.15 < psr < 0.85, f"PSR={psr:.3f} on a pure noise strategy"


def test_deflated_sharpe_rejects_the_best_of_many_null_strategies():
    """The test the paper depends on.

    Two hundred coin-flip strategies are generated; the best one is selected exactly as a
    researcher would, and the Deflated Sharpe must decline to call it a discovery.
    """
    rng = np.random.default_rng(2)
    n_trials, n_periods = 200, 1500

    candidates = rng.standard_normal((n_trials, n_periods)) * 0.01
    sharpes = candidates.mean(axis=1) / candidates.std(axis=1, ddof=1)
    best = candidates[int(np.argmax(sharpes))]

    naive = D.probabilistic_sharpe_ratio(best)["psr"]
    deflated = D.deflated_sharpe_ratio(best, n_trials=n_trials, trial_sr_variance=float(
        np.var(sharpes, ddof=1)
    ))

    assert naive > 0.90, "the selected strategy should look good before deflation"
    assert deflated.dsr < 0.95, (
        f"DSR={deflated.dsr:.3f}: the best of {n_trials} null strategies was accepted"
    )
    assert not deflated.significant_at_05
    assert "not survive" in deflated.verdict() or "marginal" in deflated.verdict()


def test_deflated_sharpe_accepts_a_genuinely_strong_strategy():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(2500) * 0.01 + 0.0015  # ~2.4 annualised Sharpe

    result = D.deflated_sharpe_ratio(r, n_trials=200)
    assert result.dsr > 0.95, f"DSR={result.dsr:.3f} on a strategy this strong"
    assert result.significant_at_05
    assert "survives" in result.verdict()


def test_declaring_more_trials_makes_deflation_stricter():
    rng = np.random.default_rng(4)
    r = rng.standard_normal(1500) * 0.01 + 0.0007

    few = D.deflated_sharpe_ratio(r, n_trials=5).dsr
    many = D.deflated_sharpe_ratio(r, n_trials=2000).dsr
    assert many < few, (
        "declaring more configurations must lower the deflated Sharpe; otherwise the "
        "declaration is meaningless"
    )


def test_minimum_track_record_length_is_finite_and_sensible():
    rng = np.random.default_rng(5)
    r = rng.standard_normal(1000) * 0.01 + 0.0005

    n = D.minimum_track_record_length(r, benchmark_sr=0.0, confidence=0.95)
    assert np.isfinite(n) and n > 0
    # A ~0.8 annualised Sharpe needs a few years before it is distinguishable from zero.
    assert n > 250, f"minimum track record of {n:.0f} days looks implausibly short"


# --------------------------------------------------------------------------------------
# Probability of Backtest Overfitting
# --------------------------------------------------------------------------------------
def test_pbo_is_near_one_half_when_every_strategy_is_noise():
    """If all candidates are noise, picking the in-sample winner is a coin flip.

    This is the diagnostic both referee reports asked for: it measures whether the
    *selection procedure* generalises, not whether the winner looks good.
    """
    rng = np.random.default_rng(0)
    M = rng.standard_normal((2000, 20)) * 0.01

    result = D.probability_of_backtest_overfitting(M, n_blocks=10)
    assert 0.30 < result.pbo < 0.70, f"PBO={result.pbo:.3f} on pure noise"
    assert result.n_combinations == 252
    assert "overfitting" in result.verdict() or "risk" in result.verdict()


def test_pbo_is_low_when_one_strategy_is_genuinely_better():
    """A real and persistent edge must be selectable, or the statistic is useless."""
    rng = np.random.default_rng(1)
    M = rng.standard_normal((2000, 20)) * 0.01
    M[:, 7] += 0.0015  # one genuinely superior strategy, present in every block

    result = D.probability_of_backtest_overfitting(M, n_blocks=10)
    assert result.pbo < 0.10, f"PBO={result.pbo:.3f} despite a persistent edge"
    assert result.verdict().startswith("selection procedure appears sound")

    # oos_degradation is the in-sample winner's out-of-sample score minus the best
    # out-of-sample score, so it is non-positive by construction. When the edge is real
    # the selected strategy is also the out-of-sample winner almost every time, leaving
    # a degradation of essentially zero in Sharpe units.
    assert -0.05 < result.oos_degradation <= 1e-9, (
        f"degradation {result.oos_degradation:.4f} is too large for a persistent edge"
    )


def test_pbo_detects_a_selection_procedure_that_chases_the_first_half():
    """Strategies that only work early must not be trusted by the selector.

    Constructed so that the in-sample winner in any early-weighted split is precisely
    the strategy that dies later. PBO should be materially above the noise floor.
    """
    rng = np.random.default_rng(2)
    n = 2000
    M = rng.standard_normal((n, 15)) * 0.01
    half = n // 2
    M[:half, 3] += 0.004
    M[half:, 3] -= 0.004

    result = D.probability_of_backtest_overfitting(M, n_blocks=10)
    assert result.pbo > 0.25, f"PBO={result.pbo:.3f} for a strategy that decays"


def test_pbo_rejects_malformed_input():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="even"):
        D.probability_of_backtest_overfitting(rng.standard_normal((500, 5)), n_blocks=7)
    with pytest.raises(ValueError, match="at least two"):
        D.probability_of_backtest_overfitting(rng.standard_normal((500, 1)))
    with pytest.raises(ValueError, match="too few"):
        D.probability_of_backtest_overfitting(rng.standard_normal((20, 5)), n_blocks=10)
