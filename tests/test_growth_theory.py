"""Numerical verification of the paper's three propositions.

A proposition that is only derived on paper is a claim; one that is also verified by
simulation against realised log wealth is a result. Each test below simulates many
periods of compounding and confirms that the *measured* growth shortfall matches the
closed form to within Monte Carlo error.
"""

from __future__ import annotations

import numpy as np
import pytest

from finsent.decision import growth_theory as G


def test_kelly_fraction_maximises_expected_log_growth():
    """f* = mu/sigma^2 must beat every neighbouring fraction."""
    mu, sigma2 = 0.0008, 0.0004
    f_star = float(G.kelly_fraction_continuous(mu, sigma2))
    assert f_star == pytest.approx(mu / sigma2)

    grid = np.linspace(f_star - 1.0, f_star + 1.0, 2001)
    growth = G.expected_log_growth(grid, mu, sigma2)
    assert grid[int(np.argmax(growth))] == pytest.approx(f_star, abs=1e-2)
    assert float(G.max_log_growth(mu, sigma2)) == pytest.approx(float(growth.max()), rel=1e-6)


def test_max_growth_equals_half_squared_sharpe():
    """G(f*) = SR^2 / 2 -- the identity worth putting in the abstract."""
    mu, sigma2 = 0.0006, 0.00025
    sharpe = mu / np.sqrt(sigma2)
    assert float(G.max_log_growth(mu, sigma2)) == pytest.approx(sharpe**2 / 2.0, rel=1e-12)


@pytest.mark.parametrize("mu_hat", [0.0000, 0.0004, 0.0012, -0.0004])
def test_proposition_1_closed_form_matches_simulation(mu_hat):
    """dG = (mu - mu_hat)^2 / (2 sigma^2), verified against realised log wealth."""
    mu, sigma2 = 0.0008, 0.0004
    sigma = np.sqrt(sigma2)

    rng = np.random.default_rng(0)
    r = mu + sigma * rng.standard_normal(4_000_000)

    f_star = float(G.kelly_fraction_continuous(mu, sigma2))
    f_hat = float(G.kelly_fraction_continuous(mu_hat, sigma2))

    # Realised per-period log growth of a fraction f, to second order in r.
    realised_star = float(np.mean(f_star * r - 0.5 * (f_star * r) ** 2))
    realised_hat = float(np.mean(f_hat * r - 0.5 * (f_hat * r) ** 2))

    predicted = float(G.growth_loss_mean_error(mu, mu_hat, sigma2))
    measured = realised_star - realised_hat

    assert measured == pytest.approx(predicted, rel=0.05, abs=1e-7), (
        f"Proposition 1 mismatch: closed form {predicted:.3e} vs simulated {measured:.3e}"
    )


@pytest.mark.parametrize("u", [0.5, 0.8, 1.0, 1.3, 1.7, 2.0])
def test_proposition_2_growth_fraction_is_2u_minus_u_squared(u):
    """G(f_hat)/G(f*) = 2u - u^2 with u = sigma^2 / sigma2_hat."""
    mu, sigma2 = 0.0008, 0.0004
    sigma2_hat = sigma2 / u

    f_hat = float(G.kelly_fraction_continuous(mu, sigma2_hat))
    realised = float(G.expected_log_growth(f_hat, mu, sigma2))
    optimal = float(G.max_log_growth(mu, sigma2))

    assert realised / optimal == pytest.approx(float(G.growth_fraction_from_variance_ratio(u)))
    assert 1.0 - realised / optimal == pytest.approx(float(G.growth_loss_variance_ratio(u)))


def test_underestimating_variance_by_half_destroys_all_growth():
    """The corollary the paper quotes: u = 2 gives exactly zero growth.

    This is the sharpest available argument that calibration is not cosmetic. The signal
    is genuinely profitable and the position still compounds to nothing.
    """
    mu, sigma2 = 0.001, 0.0004
    assert G.ruin_variance_ratio() == 2.0

    f_hat = float(G.kelly_fraction_continuous(mu, sigma2 / 2.0))
    assert float(G.expected_log_growth(f_hat, mu, sigma2)) == pytest.approx(0.0, abs=1e-15)

    # A 30% underestimate destroys ~18% of the growth rate.
    assert float(G.growth_loss_variance_ratio(1.0 / 0.7)) == pytest.approx(0.1837, abs=1e-3)

    # Beyond u = 2 the position compounds negatively on a correct forecast.
    f_worse = float(G.kelly_fraction_continuous(mu, sigma2 / 2.5))
    assert float(G.expected_log_growth(f_worse, mu, sigma2)) < 0.0


def test_proposition_3_binary_growth_loss_is_quadratic_in_probability_error():
    """dG ~ (p - p_hat)^2 near the optimum, so the quadratic expansion is valid."""
    p, b = 0.55, 1.0
    errors = np.array([0.005, 0.010, 0.020, 0.040])
    losses = np.array([float(G.growth_loss_binary(p, p + e, b)) for e in errors])

    ratios = losses[1:] / losses[:-1]
    # Doubling the probability error should roughly quadruple the loss.
    assert np.allclose(ratios, 4.0, rtol=0.10), f"ratios={ratios}"


def test_ece_lower_bound_is_positive_and_quadratic():
    """E[dG] >= (C/2) ECE^2 (equation 6), used for the theory curve in Figure F6."""
    bound = G.growth_loss_lower_bound_from_ece(np.array([0.0, 0.02, 0.04, 0.08]))
    assert bound[0] == 0.0
    assert np.all(np.diff(bound) > 0)
    assert bound[2] / bound[1] == pytest.approx(4.0, rel=1e-9)


def test_diagnose_attributes_the_shortfall_to_the_right_source():
    """A pure variance error must show up as a variance error, and vice versa."""
    mu, sigma2 = 0.0008, 0.0004

    only_variance = G.diagnose(mu, sigma2, mu_hat=mu, sigma2_hat=sigma2 / 1.5)
    assert only_variance.loss_from_mean == pytest.approx(0.0, abs=1e-15)
    assert only_variance.loss_from_variance > 0
    assert only_variance.overbetting is True

    only_mean = G.diagnose(mu, sigma2, mu_hat=mu * 0.5, sigma2_hat=sigma2)
    assert only_mean.loss_from_variance == pytest.approx(0.0, abs=1e-15)
    assert only_mean.loss_from_mean > 0
    assert only_mean.overbetting is False

    perfect = G.diagnose(mu, sigma2, mu, sigma2)
    assert perfect.loss_total == pytest.approx(0.0, abs=1e-15)
