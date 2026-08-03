"""What miscalibration costs: the paper's theoretical core.

Three propositions, each implemented and numerically verified in
``tests/test_growth_theory.py``. They matter because they make the central claim
falsifiable and because they hold at *any* signal quality -- so the paper survives even
if the measured edge turns out to be small, which for daily equity data it will be.

Setup
-----
For a position of fraction ``f`` in an asset with conditional mean ``mu`` and variance
``sigma^2`` per period, expected log growth is

    G(f) = f*mu - 0.5 * f^2 * sigma^2                                          (1)

maximised at the growth-optimal (Kelly) fraction

    f* = mu / sigma^2,        G(f*) = mu^2 / (2 sigma^2) = SR^2 / 2            (2)

Equation (2) already says something worth putting in an abstract: **the maximum
achievable log-growth rate is half the squared Sharpe ratio.**

Proposition 1 (mean miscalibration)
-----------------------------------
Betting ``f_hat = mu_hat / sigma^2`` with a wrong mean costs

    dG = G(f*) - G(f_hat) = (mu - mu_hat)^2 / (2 sigma^2)                      (3)

*Proof.* ``G`` is exactly quadratic, so ``G(f*) - G(f) = 0.5 * sigma^2 * (f* - f)^2``.
Substituting ``f* - f_hat = (mu - mu_hat)/sigma^2`` gives (3). QED

**Interpretation.** The log-growth penalty is exactly one half the squared standardised
forecast error. Because mean squared error decomposes into calibration plus refinement
(Murphy, 1973), the *calibration* component of a forecaster's loss is literally a
quantity of money per unit time. That sentence is the thesis of the paper.

Proposition 2 (variance miscalibration)
---------------------------------------
With the correct mean but a wrong variance, writing ``u = sigma^2 / sigma2_hat``,

    G(f_hat) = G(f*) * (2u - u^2),      dG / G(f*) = (1 - u)^2                 (4)

*Proof.* ``f_hat = mu / sigma2_hat = u * f*``. Substituting into (1) and dividing by
``G(f*) = mu^2/(2 sigma^2)`` yields ``2u - u^2``. QED

**Corollaries a practitioner feels immediately.** Underestimating variance by 30%
(``u = 1.43``) destroys 18% of the growth rate. Underestimating it by *half*
(``u = 2``) destroys **100%** of it: ``G = 0`` exactly, while the signal is still
genuinely positive. Past that, growth is negative and capital compounds toward ruin on
a correct forecast. This is why raw-softmax Kelly can lose to equal weighting:
overconfidence *is* variance underestimation, and Kelly punishes it quadratically and
then catastrophically.

Proposition 3 (discrete case, and the link to ECE)
--------------------------------------------------
For binary Kelly with net odds ``b``, ``f*(p) = ((b+1)p - 1)/b`` is linear in ``p``, so
a second-order expansion about the optimum gives

    dG ~= 0.5 * |G''(f*)| * ((b+1)/b)^2 * (p - p_hat)^2                        (5)
    E[dG] >= (C/2) * ECE^2                                                     (6)

where (6) follows by Jensen, since ECE is an L1 calibration error and
``E[(p-p_hat)^2] >= (E|p-p_hat|)^2``. Equation (6) is the bound plotted against
measured fold-level ECE in the paper's Figure F6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "kelly_fraction_continuous",
    "expected_log_growth",
    "max_log_growth",
    "growth_loss_mean_error",
    "growth_fraction_from_variance_ratio",
    "growth_loss_variance_ratio",
    "ruin_variance_ratio",
    "kelly_fraction_binary",
    "expected_log_growth_binary",
    "growth_loss_binary",
    "growth_loss_lower_bound_from_ece",
    "GrowthDiagnostics",
    "diagnose",
]

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Continuous outcomes
# --------------------------------------------------------------------------------------
def kelly_fraction_continuous(mu: np.ndarray | float, sigma2: np.ndarray | float) -> np.ndarray:
    """Growth-optimal fraction ``f* = mu / sigma^2`` (equation 2)."""
    return np.asarray(mu, dtype=float) / np.maximum(np.asarray(sigma2, dtype=float), _EPS)


def expected_log_growth(
    f: np.ndarray | float,
    mu: np.ndarray | float,
    sigma2: np.ndarray | float,
) -> np.ndarray:
    """``G(f) = f*mu - 0.5*f^2*sigma^2`` (equation 1)."""
    f = np.asarray(f, dtype=float)
    return f * np.asarray(mu, dtype=float) - 0.5 * f**2 * np.asarray(sigma2, dtype=float)


def max_log_growth(mu: np.ndarray | float, sigma2: np.ndarray | float) -> np.ndarray:
    """``G(f*) = mu^2 / (2 sigma^2) = SR^2 / 2`` (equation 2)."""
    return np.asarray(mu, dtype=float) ** 2 / (2.0 * np.maximum(np.asarray(sigma2, float), _EPS))


def growth_loss_mean_error(
    mu_true: np.ndarray | float,
    mu_hat: np.ndarray | float,
    sigma2: np.ndarray | float,
) -> np.ndarray:
    """**Proposition 1**: ``dG = (mu - mu_hat)^2 / (2 sigma^2)`` (equation 3)."""
    err = np.asarray(mu_true, dtype=float) - np.asarray(mu_hat, dtype=float)
    return err**2 / (2.0 * np.maximum(np.asarray(sigma2, dtype=float), _EPS))


def growth_fraction_from_variance_ratio(u: np.ndarray | float) -> np.ndarray:
    """**Proposition 2**: realised growth as a fraction of the optimum, ``2u - u^2``."""
    u = np.asarray(u, dtype=float)
    return 2.0 * u - u**2


def growth_loss_variance_ratio(u: np.ndarray | float) -> np.ndarray:
    """**Proposition 2**: fractional growth destroyed, ``(1 - u)^2`` (equation 4).

    ``u = sigma^2 / sigma2_hat``; ``u > 1`` means the model *underestimated* variance
    and is therefore overbetting.
    """
    return (1.0 - np.asarray(u, dtype=float)) ** 2


def ruin_variance_ratio() -> float:
    """The variance ratio at which all growth is destroyed: ``u = 2``.

    Underestimating conditional variance by a factor of two takes ``G`` to exactly zero
    on a genuinely profitable signal. Quoted in the paper as the sharpest available
    argument for why calibration is not a cosmetic concern.
    """
    return 2.0


# --------------------------------------------------------------------------------------
# Discrete outcomes (binary Kelly), and the ECE bound
# --------------------------------------------------------------------------------------
def kelly_fraction_binary(p: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """``f*(p) = ((b+1)p - 1) / b`` -- win probability ``p``, net odds ``b``."""
    p = np.asarray(p, dtype=float)
    b = np.maximum(np.asarray(b, dtype=float), _EPS)
    return ((b + 1.0) * p - 1.0) / b


def expected_log_growth_binary(
    f: np.ndarray | float,
    p: np.ndarray | float,
    b: np.ndarray | float,
) -> np.ndarray:
    """``G(f) = p log(1 + b f) + (1-p) log(1 - f)``."""
    f = np.asarray(f, dtype=float)
    p = np.asarray(p, dtype=float)
    b = np.asarray(b, dtype=float)
    up = np.maximum(1.0 + b * f, _EPS)
    dn = np.maximum(1.0 - f, _EPS)
    return p * np.log(up) + (1.0 - p) * np.log(dn)


def growth_loss_binary(
    p_true: np.ndarray | float,
    p_hat: np.ndarray | float,
    b: np.ndarray | float,
) -> np.ndarray:
    """Exact growth loss from betting the fraction implied by a miscalibrated ``p_hat``."""
    p_true = np.asarray(p_true, dtype=float)
    f_star = kelly_fraction_binary(p_true, b)
    f_hat = kelly_fraction_binary(np.asarray(p_hat, dtype=float), b)
    g_star = expected_log_growth_binary(f_star, p_true, b)
    g_hat = expected_log_growth_binary(f_hat, p_true, b)
    return g_star - g_hat


def growth_loss_lower_bound_from_ece(
    ece: np.ndarray | float,
    p_ref: float = 0.55,
    b: float = 1.0,
) -> np.ndarray:
    """**Proposition 3**, equation (6): ``E[dG] >= (C/2) * ECE^2``.

    ``C = |G''(f*)| * ((b+1)/b)^2`` is evaluated at a reference operating point
    ``p_ref`` -- the curvature varies slowly over the range of probabilities a
    directional model actually emits, so a single reference point is adequate for the
    bound plotted in Figure F6.
    """
    f_star = float(kelly_fraction_binary(p_ref, b))
    up = max(1.0 + b * f_star, _EPS)
    dn = max(1.0 - f_star, _EPS)
    g2 = -(p_ref * b**2) / up**2 - (1.0 - p_ref) / dn**2  # G''(f*) < 0
    curvature = abs(g2) * ((b + 1.0) / b) ** 2
    return 0.5 * curvature * np.asarray(ece, dtype=float) ** 2


# --------------------------------------------------------------------------------------
# Diagnostics used by the report
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GrowthDiagnostics:
    """Decomposition of realised growth shortfall into mean and variance error."""

    g_optimal: float
    g_realised: float
    loss_total: float
    loss_from_mean: float
    loss_from_variance: float
    variance_ratio: float
    overbetting: bool

    def summary(self) -> str:
        pct = 100.0 * self.loss_total / self.g_optimal if self.g_optimal > _EPS else np.nan
        state = "OVERBETTING" if self.overbetting else "underbetting"
        return (
            f"G* = {self.g_optimal:.6f}  realised = {self.g_realised:.6f}  "
            f"shortfall = {self.loss_total:.6f} ({pct:.1f}% of optimal)\n"
            f"  from mean error     : {self.loss_from_mean:.6f}\n"
            f"  from variance error : {self.loss_from_variance:.6f}\n"
            f"  variance ratio u    : {self.variance_ratio:.3f}  [{state}]"
        )


def diagnose(
    mu_true: float,
    sigma2_true: float,
    mu_hat: float,
    sigma2_hat: float,
) -> GrowthDiagnostics:
    """Attribute the growth shortfall of a forecast to mean vs variance miscalibration.

    The two components are computed by holding the other input at its true value, so
    they are marginal contributions and need not sum exactly to the total when both
    errors are large -- which is itself worth reporting.
    """
    f_star = float(kelly_fraction_continuous(mu_true, sigma2_true))
    f_hat = float(kelly_fraction_continuous(mu_hat, sigma2_hat))

    g_star = float(max_log_growth(mu_true, sigma2_true))
    g_hat = float(expected_log_growth(f_hat, mu_true, sigma2_true))

    loss_mean = float(growth_loss_mean_error(mu_true, mu_hat, sigma2_true))
    u = float(sigma2_true / max(sigma2_hat, _EPS))
    loss_var = float(g_star * growth_loss_variance_ratio(u))

    return GrowthDiagnostics(
        g_optimal=g_star,
        g_realised=g_hat,
        loss_total=g_star - g_hat,
        loss_from_mean=loss_mean,
        loss_from_variance=loss_var,
        variance_ratio=u,
        overbetting=bool(abs(f_hat) > abs(f_star)),
    )
