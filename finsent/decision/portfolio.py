"""Portfolio construction: shrunk covariance and a genuinely constrained optimiser.

Two V2 defects are corrected here.

**Covariance.** ``Sigma + 1e-8 * I`` is cosmetic: on a handful of names with a few
hundred observations the sample covariance is badly conditioned and its extreme
eigenvectors are noise, which a mean-variance optimiser then levers into. Ledoit & Wolf
(2004) shrinkage toward a scaled identity target is the standard fix and is implemented
from the paper's formulae below.

**Optimiser.** V2 computed the unconstrained analytic solution and then clipped and
renormalised it. That is not the constrained maximum-Sharpe portfolio, and a referee
will say so. This module solves

    max_w   w'mu - (lambda/2) w'Sigma w - tau ||w - w_prev||_1
    s.t.    |w_i| <= f_max,  ||w||_1 <= L,  1'w = 0 (optional)

with cvxpy when it is installed and with a projected proximal-gradient method otherwise,
so the repository has no hard dependency on a solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ledoit_wolf_covariance",
    "PortfolioConstraints",
    "mean_variance_weights",
    "min_variance_weights",
    "risk_parity_weights",
    "cvar_weights",
    "project_onto_constraints",
]

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Covariance
# --------------------------------------------------------------------------------------
def ledoit_wolf_covariance(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity target.

    Returns ``(sigma_shrunk, shrinkage_intensity)``. The intensity is reported in the
    paper because it tells the reader how little the sample covariance could be trusted:
    values near 1 mean the estimator is essentially discarding the sample structure.

    Implementation follows "A well-conditioned estimator for large-dimensional
    covariance matrices", JMVA 88(2):365-411, equations for ``mu``, ``d2``, ``b2``,
    ``a2``, with the shrinkage constant ``b2/d2`` truncated to [0, 1].
    """
    X = np.asarray(returns, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"returns must be (n_obs, n_assets); got {X.shape}")
    n, p = X.shape
    if n < 2:
        raise ValueError("need at least two observations")

    Xc = X - X.mean(axis=0, keepdims=True)
    sample = (Xc.T @ Xc) / n

    mu = float(np.trace(sample) / p)                      # average eigenvalue
    target = mu * np.eye(p)

    d2 = float(np.sum((sample - target) ** 2) / p)        # dispersion from the target
    if d2 < _EPS:
        return sample, 0.0

    # b2: average squared deviation of the per-observation covariances from the sample.
    b2_sum = 0.0
    for t in range(n):
        xt = Xc[t][:, None]
        b2_sum += float(np.sum((xt @ xt.T - sample) ** 2))
    b2 = min(b2_sum / (n**2 * p), d2)

    shrinkage = float(np.clip(b2 / d2, 0.0, 1.0))
    return shrinkage * target + (1.0 - shrinkage) * sample, shrinkage


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PortfolioConstraints:
    f_max: float = 0.10
    gross_leverage_max: float = 2.0
    dollar_neutral: bool = True
    long_only: bool = False

    def as_bounds(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        lo = np.zeros(n) if self.long_only else np.full(n, -self.f_max)
        return lo, np.full(n, self.f_max)


def project_onto_constraints(
    w: np.ndarray,
    cons: PortfolioConstraints,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> np.ndarray:
    """Project onto the box, dollar-neutrality and gross-leverage constraint set.

    Demeaning and clipping are **alternated to convergence** rather than applied once.
    Doing them in sequence is what an earlier revision did, and it silently breaks the
    property it was supposed to establish: clipping after demeaning reintroduces a net
    exposure, so a nominally dollar-neutral book carried a residual long tilt whose
    return is market beta, not alpha.

    Both sets are convex, so alternating projection converges (von Neumann). The final
    leverage rescaling is a positive scalar multiplication, which preserves both
    ``sum(w) = 0`` and the box, so it is safe to apply last.
    """
    v = np.asarray(w, dtype=float).copy()
    if v.size == 0:
        return v

    lo, hi = cons.as_bounds(v.size)
    for _ in range(max_iter):
        prev = v
        if cons.dollar_neutral:
            v = v - v.mean()
        v = np.clip(v, lo, hi)
        if np.max(np.abs(v - prev)) < tol:
            break

    if cons.dollar_neutral and abs(v.sum()) > 1e-9:
        # Alternating projection can stall when the box is saturated on one side; shed
        # the residual across the names that still have headroom.
        residual = v.sum()
        headroom = np.where(residual > 0, v - lo, hi - v)
        total = headroom.sum()
        if total > _EPS:
            v = v - residual * headroom / total

    gross = np.abs(v).sum()
    if gross > cons.gross_leverage_max and gross > _EPS:
        v = v * (cons.gross_leverage_max / gross)
    return v


# --------------------------------------------------------------------------------------
# Optimisers
# --------------------------------------------------------------------------------------
def _mean_variance_cvxpy(
    mu: np.ndarray,
    sigma: np.ndarray,
    cons: PortfolioConstraints,
    risk_aversion: float,
    turnover_penalty: float,
    w_prev: np.ndarray | None,
):
    try:
        import cvxpy as cp
    except ImportError:
        return None

    n = mu.size
    w = cp.Variable(n)
    sigma_psd = cp.psd_wrap(sigma) if hasattr(cp, "psd_wrap") else sigma
    objective = mu @ w - 0.5 * risk_aversion * cp.quad_form(w, sigma_psd)
    if turnover_penalty > 0 and w_prev is not None:
        objective = objective - turnover_penalty * cp.norm1(w - w_prev)

    lo, hi = cons.as_bounds(n)
    constraints = [w >= lo, w <= hi, cp.norm1(w) <= cons.gross_leverage_max]
    if cons.dollar_neutral:
        constraints.append(cp.sum(w) == 0)

    problem = cp.Problem(cp.Maximize(objective), constraints)
    try:
        problem.solve(solver=cp.OSQP, verbose=False)
    except Exception:
        try:
            problem.solve(verbose=False)
        except Exception:
            return None
    if w.value is None:
        return None
    return np.asarray(w.value, dtype=float).ravel()


def _mean_variance_proximal(
    mu: np.ndarray,
    sigma: np.ndarray,
    cons: PortfolioConstraints,
    risk_aversion: float,
    turnover_penalty: float,
    w_prev: np.ndarray | None,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> np.ndarray:
    """Projected proximal gradient ascent, used when cvxpy is unavailable.

    Step size is ``1/L`` with ``L`` the Lipschitz constant of the quadratic term, so no
    line search is needed. The L1 turnover term is handled by soft-thresholding around
    ``w_prev``.
    """
    n = mu.size
    w = np.zeros(n) if w_prev is None else np.asarray(w_prev, dtype=float).copy()
    prev = np.zeros(n) if w_prev is None else np.asarray(w_prev, dtype=float)

    lipschitz = max(float(np.linalg.eigvalsh(sigma).max()) * risk_aversion, _EPS)
    step = 1.0 / lipschitz

    for _ in range(max_iter):
        grad = mu - risk_aversion * (sigma @ w)
        cand = w + step * grad
        if turnover_penalty > 0:
            thresh = step * turnover_penalty
            delta = cand - prev
            cand = prev + np.sign(delta) * np.maximum(np.abs(delta) - thresh, 0.0)
        new = project_onto_constraints(cand, cons)
        if np.max(np.abs(new - w)) < tol:
            w = new
            break
        w = new
    return w


def mean_variance_weights(
    mu: np.ndarray,
    sigma: np.ndarray,
    constraints: PortfolioConstraints | None = None,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.0,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """Constrained mean-variance portfolio, exact when cvxpy is present."""
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (mu.size, mu.size):
        raise ValueError(f"sigma must be ({mu.size},{mu.size}); got {sigma.shape}")
    cons = constraints or PortfolioConstraints()

    sigma = 0.5 * (sigma + sigma.T)
    eig_min = float(np.linalg.eigvalsh(sigma).min())
    if eig_min < 0:  # numerical asymmetry only; shrinkage should prevent real ones
        sigma = sigma + (abs(eig_min) + 1e-10) * np.eye(mu.size)

    w = _mean_variance_cvxpy(mu, sigma, cons, risk_aversion, turnover_penalty, w_prev)
    if w is None:
        w = _mean_variance_proximal(mu, sigma, cons, risk_aversion, turnover_penalty, w_prev)
    return project_onto_constraints(w, cons)


def min_variance_weights(
    sigma: np.ndarray,
    constraints: PortfolioConstraints | None = None,
) -> np.ndarray:
    """Minimum-variance portfolio under the same constraint set."""
    n = sigma.shape[0]
    return mean_variance_weights(
        np.zeros(n), sigma, constraints, risk_aversion=1.0, turnover_penalty=0.0
    )


def risk_parity_weights(sigma: np.ndarray, max_iter: int = 1000, tol: float = 1e-12) -> np.ndarray:
    """Equal risk contribution weights. Long-only and fully invested by construction.

    Uses the damped multiplicative update ``w <- w * sqrt(target / rc)``. The naive
    fixed point ``w <- normalise(1 / Sigma w)`` is what an earlier revision used, and on
    a diagonal covariance it **oscillates with period two**: it overshoots to the exact
    reciprocal weighting, then bounces straight back to equal weights, so the loop
    terminates at whatever iteration parity it happens to stop on. The square root halves
    the step in log space and converges monotonically instead.
    """
    s = np.asarray(sigma, dtype=float)
    n = s.shape[0]
    w = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        mrc = s @ w
        rc = w * mrc
        target = rc.mean()
        if target <= _EPS:
            break

        ratio = np.where(np.abs(rc) > _EPS, target / np.where(np.abs(rc) > _EPS, rc, 1.0), 1.0)
        new = w * np.sqrt(np.clip(ratio, 1e-8, 1e8))
        total = new.sum()
        if total < _EPS:
            break
        new = new / total

        if np.max(np.abs(new - w)) < tol:
            w = new
            break
        w = new
    return w


def cvar_weights(
    scenario_returns: np.ndarray,
    alpha: float = 0.95,
    constraints: PortfolioConstraints | None = None,
    target_return: float | None = None,
) -> np.ndarray:
    """Minimum-CVaR portfolio via the Rockafellar & Uryasev (2000) linear program.

    ``scenario_returns`` is ``(n_scenarios, n_assets)``; historical returns are the
    usual choice of scenarios. Reported as a robustness row against the mean-variance
    result, since CVaR is the risk measure a risk desk actually monitors.
    """
    from scipy.optimize import linprog

    R = np.asarray(scenario_returns, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"scenario_returns must be 2-D; got {R.shape}")
    n_scen, n = R.shape
    cons = constraints or PortfolioConstraints()

    # Variables: [w (n), zeta (1), u (n_scen)] ; minimise zeta + 1/((1-a) S) sum u
    c = np.concatenate([np.zeros(n), [1.0], np.full(n_scen, 1.0 / ((1.0 - alpha) * n_scen))])

    # u >= -R w - zeta   ->   -R w - zeta - u <= 0
    A_ub = np.hstack([-R, -np.ones((n_scen, 1)), -np.eye(n_scen)])
    b_ub = np.zeros(n_scen)

    if target_return is not None:
        mu = R.mean(axis=0)
        A_ub = np.vstack([A_ub, np.concatenate([-mu, [0.0], np.zeros(n_scen)])])
        b_ub = np.concatenate([b_ub, [-target_return]])

    A_eq, b_eq = None, None
    if cons.dollar_neutral:
        A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(n_scen)])[None, :]
        b_eq = np.array([0.0])

    lo, hi = cons.as_bounds(n)
    bounds = (
        [(lo[i], hi[i]) for i in range(n)] + [(None, None)] + [(0.0, None)] * n_scen
    )

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return project_onto_constraints(np.zeros(n), cons)
    return project_onto_constraints(np.asarray(res.x[:n], dtype=float), cons)
