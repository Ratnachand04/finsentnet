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
