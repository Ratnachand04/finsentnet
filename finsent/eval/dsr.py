"""Probabilistic and Deflated Sharpe ratios (Bailey & Lopez de Prado, 2014).

Why the paper cannot omit this: the reported Sharpe of a strategy selected as the best
of many trials is upward biased, and the bias grows with the number of trials and with
the non-normality of returns. The Deflated Sharpe Ratio corrects for both. The trial
count comes from ``eval.n_configs_evaluated`` in the configuration -- declaring it
honestly is the difference between a scientist and a data miner.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats as sps

__all__ = [
    "sharpe_standard_error",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "minimum_track_record_length",
    "DSRResult",
    "PBOResult",
    "probability_of_backtest_overfitting",
]

_EPS = 1e-12
_EULER_MASCHERONI = 0.5772156649015329


def _moments(returns: np.ndarray) -> tuple[float, float, float, int]:
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    n = r.size
    if n < 4:
        return np.nan, np.nan, np.nan, n
    sr = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > _EPS else np.nan
    skew = float(sps.skew(r, bias=False))
    kurt = float(sps.kurtosis(r, fisher=False, bias=False))  # non-excess
    return sr, skew, kurt, n


def sharpe_standard_error(sr: float, skew: float, kurtosis: float, n: int) -> float:
    """Standard error of a Sharpe estimate under non-normal returns (Mertens, 2002).

    ``se = sqrt( (1 - skew*SR + (kurt-1)/4 * SR^2) / (n-1) )`` with non-excess kurtosis.
    Negative skew and fat tails -- both endemic to trading strategies -- widen this
    materially versus the naive ``sqrt(1/n)``.
    """
    if not np.isfinite(sr) or n < 3:
        return np.nan
    var = 1.0 - skew * sr + 0.25 * (kurtosis - 1.0) * sr**2
    return float(np.sqrt(max(var, _EPS) / (n - 1)))


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    periods_per_year: int | None = None,
) -> dict[str, float]:
    """P(true Sharpe > benchmark) given the observed sample.

    ``benchmark_sr`` is expressed in the same (per-period) units as the returns unless
    ``periods_per_year`` is supplied, in which case an annualised benchmark is
    de-annualised for you.
    """
    sr, skew, kurt, n = _moments(returns)
    if not np.isfinite(sr):
        return {"psr": np.nan, "sr": np.nan, "n": n}

    bench = benchmark_sr
    if periods_per_year:
        bench = benchmark_sr / np.sqrt(periods_per_year)

    se = sharpe_standard_error(sr, skew, kurt, n)
    if not np.isfinite(se) or se < _EPS:
        return {"psr": np.nan, "sr": sr, "n": n}

    z = (sr - bench) / se
    return {
        "psr": float(sps.norm.cdf(z)),
        "sr": float(sr),
        "sr_annualised": float(sr * np.sqrt(periods_per_year)) if periods_per_year else np.nan,
        "se": float(se),
        "skew": float(skew),
        "kurtosis": float(kurt),
        "n": int(n),
    }


def expected_max_sharpe(n_trials: int, sr_variance: float = 1.0) -> float:
    """Expected maximum Sharpe across ``n_trials`` independent null strategies.

    Uses the standard extreme-value approximation

    ``E[max] = sqrt(V) * [ (1-g) Z(1 - 1/N) + g Z(1 - 1/(N e)) ]``

    with ``g`` the Euler-Mascheroni constant. This is the benchmark the observed Sharpe
    must beat before it counts as a discovery rather than the luckiest of many draws.
    """
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.0
    sd = np.sqrt(max(sr_variance, _EPS))
    z1 = sps.norm.ppf(1.0 - 1.0 / n)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sd * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2))


@dataclass(frozen=True)
class DSRResult:
    dsr: float
    sr: float
    sr_annualised: float
    expected_max_sr: float
    n_trials: int
    n_periods: int
    skew: float
    kurtosis: float
    significant_at_05: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)

    def verdict(self) -> str:
        if not np.isfinite(self.dsr):
            return "undetermined"
        if self.dsr >= 0.95:
            return "survives deflation at 5%"
        if self.dsr >= 0.90:
            return "marginal after deflation"
        return "does not survive deflation - report as such"


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    trial_sr_variance: float | None = None,
    periods_per_year: int = 252,
) -> DSRResult:
    """Deflated Sharpe Ratio: PSR against the expected maximum of ``n_trials`` nulls.

    ``trial_sr_variance`` is the cross-trial variance of the Sharpe estimates. When it
    is unknown, ``1/(n-1)`` is used, i.e. the sampling variance of a Sharpe under the
    null -- a conservative and commonly used default.
    """
    sr, skew, kurt, n = _moments(returns)
    if not np.isfinite(sr):
        return DSRResult(np.nan, np.nan, np.nan, np.nan, n_trials, n, np.nan, np.nan, False)

    variance = trial_sr_variance if trial_sr_variance is not None else 1.0 / max(n - 1, 1)
    sr0 = expected_max_sharpe(n_trials, variance)

    se = sharpe_standard_error(sr, skew, kurt, n)
    dsr = float(sps.norm.cdf((sr - sr0) / se)) if np.isfinite(se) and se > _EPS else np.nan

    return DSRResult(
        dsr=dsr,
        sr=float(sr),
        sr_annualised=float(sr * np.sqrt(periods_per_year)),
        expected_max_sr=float(sr0),
        n_trials=int(n_trials),
        n_periods=int(n),
        skew=float(skew),
        kurtosis=float(kurt),
        significant_at_05=bool(np.isfinite(dsr) and dsr >= 0.95),
    )


def minimum_track_record_length(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Periods needed before the observed Sharpe is significant at ``confidence``.

    A blunt and useful reviewer answer: if this exceeds the length of your sample, the
    result is not yet evidence.
    """
    sr, skew, kurt, n = _moments(returns)
    if not np.isfinite(sr) or abs(sr - benchmark_sr) < _EPS:
        return np.inf
    z = sps.norm.ppf(confidence)
    numer = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2
    return float(1.0 + max(numer, _EPS) * (z / (sr - benchmark_sr)) ** 2)

# --------------------------------------------------------------------------------------
# Probability of Backtest Overfitting
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PBOResult:
    """Combinatorially symmetric cross-validation result (Bailey et al., 2017)."""

    pbo: float
    n_combinations: int
    n_strategies: int
    n_periods: int
    median_logit: float
    oos_degradation: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def verdict(self) -> str:
        if not np.isfinite(self.pbo):
            return "undetermined"
        if self.pbo <= 0.10:
            return "selection procedure appears sound"
        if self.pbo <= 0.50:
            return "material overfitting risk"
        return "the selection procedure is overfitting; the winner is noise"


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_blocks: int = 10,
    metric: str = "sharpe",
) -> PBOResult:
    """PBO by combinatorially symmetric cross-validation.

    The question this answers is not "is my best strategy good?" but "does my *selection
    procedure* pick winners that stay winners?". The sample is cut into ``n_blocks``
    contiguous blocks; every balanced split into in-sample and out-of-sample halves is
    enumerated; in each split the strategy with the best in-sample metric is chosen, and
    its **rank among all strategies out-of-sample** is recorded. If choosing the
    in-sample winner is no better than choosing at random, that rank sits at the median
    half the time and PBO approaches 0.5.

    Both referee reports on the predecessor of this work asked for this statistic
    specifically, because a paper that evaluates a dozen configurations on one slice and
    reports the best has performed a selection whose reliability is unmeasured.

    Parameters
    ----------
    returns_matrix
        ``(n_periods, n_strategies)`` per-period returns, one column per configuration.
    n_blocks
        Number of contiguous blocks; must be even. ``C(n_blocks, n_blocks/2)``
        combinations are enumerated, so 10 gives 252 and 16 gives 12,870.
    """
    from itertools import combinations

    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"returns_matrix must be (n_periods, n_strategies); got {M.shape}")
    n_periods, n_strategies = M.shape
    if n_strategies < 2:
        raise ValueError("PBO requires at least two competing strategies")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even so the split is symmetric")
    if n_periods < n_blocks * 4:
        raise ValueError(
            f"{n_periods} periods is too few for {n_blocks} blocks; use fewer blocks"
        )

    bounds = np.linspace(0, n_periods, n_blocks + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_blocks)]

    def score(sub: np.ndarray) -> np.ndarray:
        if metric == "sharpe":
            sd = sub.std(axis=0, ddof=1)
            return np.where(sd > _EPS, sub.mean(axis=0) / np.maximum(sd, _EPS), -np.inf)
        return sub.mean(axis=0)

    logits: list[float] = []
    degradations: list[float] = []

    for combo in combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in combo])

        is_score = score(M[is_idx])
        oos_score = score(M[oos_idx])
        if not np.isfinite(is_score).any() or not np.isfinite(oos_score).any():
            continue

        best = int(np.nanargmax(is_score))
        # Rank of the in-sample winner among out-of-sample results, 1 = worst.
        order = np.argsort(np.argsort(oos_score))
        rank = float(order[best]) + 1.0

        omega = rank / (n_strategies + 1.0)
        omega = float(np.clip(omega, 1e-6, 1 - 1e-6))
        logits.append(float(np.log(omega / (1.0 - omega))))
        degradations.append(float(oos_score[best] - np.nanmax(oos_score)))

    if not logits:
        return PBOResult(np.nan, 0, n_strategies, n_periods, np.nan, np.nan)

    arr = np.asarray(logits, dtype=float)
    return PBOResult(
        pbo=float((arr <= 0.0).mean()),
        n_combinations=int(arr.size),
        n_strategies=int(n_strategies),
        n_periods=int(n_periods),
        median_logit=float(np.median(arr)),
        oos_degradation=float(np.mean(degradations)),
    )
