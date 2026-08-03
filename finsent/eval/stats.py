"""Statistical tests that turn "57.4 > 54.8" into a claim.

Implemented from the source papers rather than imported, so each formula is checkable:

- ``newey_west_lrv`` / ``newey_west_tstat`` — Newey & West (1987) HAC long-run variance.
- ``diebold_mariano`` — Diebold & Mariano (1995), with the Harvey, Leybourne & Newbold
  (1997) small-sample correction.
- ``stationary_bootstrap_indices`` — Politis & Romano (1994).
- ``whites_reality_check`` — White (2000).
- ``hansen_spa`` — Hansen (2005) Superior Predictive Ability, studentised and recentred.
- ``paired_bootstrap_diff`` — the paired test the sizing experiment needs: predictions
  are held fixed and only the sizing rule varies, so the comparison's variance is far
  smaller than an unpaired strategy-vs-strategy test.

Why these are mandatory here: with ~500 test days the standard error of an accuracy
estimate is about 2.2 percentage points, so most raw comparisons in this literature sit
inside their own noise band. Reporting a difference without one of these tests is not a
finding.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Sequence

import numpy as np
from scipy import stats as sps

__all__ = [
    "newey_west_lrv",
    "newey_west_tstat",
    "DMResult",
    "diebold_mariano",
    "stationary_bootstrap_indices",
    "block_bootstrap_ci",
    "paired_bootstrap_diff",
    "whites_reality_check",
    "hansen_spa",
]

_EPS = 1e-12


def _default_lags(n: int) -> int:
    """Newey-West automatic lag rule, ``floor(4 (n/100)^(2/9))``."""
    return max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def newey_west_lrv(x: np.ndarray, lags: int | None = None) -> float:
    """HAC long-run variance of a series (Bartlett kernel).

    ``lrv = gamma_0 + 2 * sum_l (1 - l/(L+1)) * gamma_l``
    """
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return np.nan
    if lags is None:
        lags = _default_lags(n)
    lags = int(min(max(lags, 0), n - 1))

    dev = v - v.mean()
    lrv = float(dev @ dev) / n
    for lag in range(1, lags + 1):
        gamma = float(dev[lag:] @ dev[:-lag]) / n
        lrv += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma
    return float(max(lrv, _EPS))


def newey_west_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """HAC t-statistic and two-sided p-value for ``H0: mean(x) == 0``."""
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n < 3:
        return np.nan, np.nan
    lrv = newey_west_lrv(v, lags)
    se = np.sqrt(lrv / n)
    if se < _EPS:
        return np.nan, np.nan
    t = float(v.mean() / se)
    p = float(2.0 * (1.0 - sps.t.cdf(abs(t), df=n - 1)))
    return t, p


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    mean_diff: float
    n: int
    lags: int
    better: str

    def to_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    lags: int | None = None,
    small_sample_correction: bool = True,
    labels: tuple[str, str] = ("A", "B"),
) -> DMResult:
    """Diebold-Mariano test of equal predictive accuracy.

    Inputs are **per-period losses** (lower is better), for example squared error or
    negative log-likelihood, aligned on the same dates. A negative statistic favours
    model A.

    ``small_sample_correction`` applies the Harvey, Leybourne & Newbold (1997) factor
    and switches the reference distribution to Student-t, which matters at the few
    hundred observations typical of a walk-forward test fold.
    """
    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"loss arrays must align: {a.shape} vs {b.shape}")

    d = a - b
    d = d[np.isfinite(d)]
    n = d.size
    if n < 5:
        return DMResult(np.nan, np.nan, np.nan, n, 0, "undetermined")

    if lags is None:
        lags = _default_lags(n)
    lrv = newey_west_lrv(d, lags)
    se = np.sqrt(lrv / n)
    if se < _EPS:
        return DMResult(np.nan, np.nan, float(d.mean()), n, lags, "undetermined")

    stat = float(d.mean() / se)
    if small_sample_correction:
        h = lags + 1
        factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
        stat *= float(np.sqrt(max(factor, _EPS)))
        p = float(2.0 * (1.0 - sps.t.cdf(abs(stat), df=n - 1)))
    else:
        p = float(2.0 * (1.0 - sps.norm.cdf(abs(stat))))

    better = labels[0] if d.mean() < 0 else labels[1]
    return DMResult(stat, p, float(d.mean()), n, int(lags), better)


def stationary_bootstrap_indices(
    n: int,
    block_mean: int = 10,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap index matrix ``(n_resamples, n)``.

    Block lengths are geometric with mean ``block_mean``, which preserves the serial
    dependence of daily strategy returns. An i.i.d. bootstrap would understate
    uncertainty here by a wide margin.
    """
    rng = rng or np.random.default_rng(0)
    if n <= 0:
        return np.zeros((n_resamples, 0), dtype=int)
    p = 1.0 / max(block_mean, 1)

    idx = np.empty((n_resamples, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_resamples)
    if n > 1:
        restart = rng.random((n_resamples, n - 1)) < p
        fresh = rng.integers(0, n, size=(n_resamples, n - 1))
        for t in range(1, n):
            cont = (idx[:, t - 1] + 1) % n
            idx[:, t] = np.where(restart[:, t - 1], fresh[:, t - 1], cont)
    return idx


def block_bootstrap_ci(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    block_mean: int = 10,
    n_resamples: int = 5000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile confidence interval for any statistic of a serially dependent series."""
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n < 10:
        return {"point": float(statistic(v)) if n else np.nan, "lo": np.nan, "hi": np.nan, "n": n}

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, block_mean, n_resamples, rng)
    draws = np.array([statistic(v[row]) for row in idx], dtype=float)
    draws = draws[np.isfinite(draws)]
    alpha = (1.0 - ci_level) / 2.0
    return {
        "point": float(statistic(v)),
        "lo": float(np.quantile(draws, alpha)) if draws.size else np.nan,
        "hi": float(np.quantile(draws, 1.0 - alpha)) if draws.size else np.nan,
        "se": float(draws.std(ddof=1)) if draws.size > 1 else np.nan,
        "n": int(n),
    }


def paired_bootstrap_diff(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    block_mean: int = 10,
    n_resamples: int = 5000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Paired block bootstrap for ``statistic(x) - statistic(y)`` on common dates.

    This is the test behind the paper's headline sizing experiment. Because both series
    come from the *same* predictions and differ only in the sizing rule, the resampling
    is paired and most of the common variance cancels -- which is precisely why this
    comparison has power where an unpaired alpha test does not.
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired series must align: {a.shape} vs {b.shape}")
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = a.size
    if n < 10:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p_value": np.nan, "n": n}

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, block_mean, n_resamples, rng)
    point = float(statistic(a) - statistic(b))
    draws = np.array([statistic(a[row]) - statistic(b[row]) for row in idx], dtype=float)
    draws = draws[np.isfinite(draws)]

    if draws.size == 0:
        # Every resample produced an undefined statistic -- typically a strategy that
        # never trades, so its return series is identically zero and Sharpe is undefined.
        # Report that honestly rather than emitting a NaN-laden p-value from an empty
        # slice.
        return {"diff": point, "lo": np.nan, "hi": np.nan, "se": np.nan,
                "p_value": np.nan, "n": int(n)}

    alpha = (1.0 - ci_level) / 2.0
    # Two-sided bootstrap p-value by inverting the centred distribution.
    centred = draws - draws.mean()
    p = float((np.abs(centred) >= abs(point)).mean())
    return {
        "diff": point,
        "lo": float(np.quantile(draws, alpha)) if draws.size else np.nan,
        "hi": float(np.quantile(draws, 1.0 - alpha)) if draws.size else np.nan,
        "se": float(draws.std(ddof=1)) if draws.size > 1 else np.nan,
        "p_value": p,
        "n": int(n),
    }


def whites_reality_check(
    benchmark_loss: np.ndarray,
    model_losses: np.ndarray,
    block_mean: int = 10,
    n_resamples: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """White (2000) Reality Check for data snooping.

    ``model_losses`` has shape ``(n_models, n_periods)``. Tests the null that no model
    in the collection beats the benchmark, correcting for the fact that the best of
    many models looks good by construction.
    """
    bench = np.asarray(benchmark_loss, dtype=float).ravel()
    losses = np.atleast_2d(np.asarray(model_losses, dtype=float))
    if losses.shape[1] != bench.size:
        raise ValueError("model_losses must be (n_models, n_periods) aligned to benchmark")

    # f_k = benchmark loss - model loss; positive means the model is better.
    f = bench[None, :] - losses
    n = f.shape[1]
    fbar = f.mean(axis=1)
    stat = float(np.sqrt(n) * fbar.max())

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, block_mean, n_resamples, rng)
    draws = np.empty(n_resamples, dtype=float)
    for i, row in enumerate(idx):
        boot = f[:, row].mean(axis=1)
        draws[i] = np.sqrt(n) * (boot - fbar).max()

    return {
        "statistic": stat,
        "p_value": float((draws >= stat).mean()),
        "best_model": int(np.argmax(fbar)),
        "best_mean_outperformance": float(fbar.max()),
        "n_models": int(f.shape[0]),
        "n_periods": int(n),
    }


def hansen_spa(
    benchmark_loss: np.ndarray,
    model_losses: np.ndarray,
    block_mean: int = 10,
    n_resamples: int = 5000,
    seed: int = 0,
) -> dict[str, float]:
    """Hansen (2005) Superior Predictive Ability test.

    Improves on White's Reality Check by studentising each model's outperformance and
    recentring so that hopelessly poor models stop diluting the null distribution. This
    is the right multiple-comparison control when a paper reports a zoo of variants,
    which this one does.
    """
    bench = np.asarray(benchmark_loss, dtype=float).ravel()
    losses = np.atleast_2d(np.asarray(model_losses, dtype=float))
    if losses.shape[1] != bench.size:
        raise ValueError("model_losses must be (n_models, n_periods) aligned to benchmark")

    f = bench[None, :] - losses
    k, n = f.shape
    fbar = f.mean(axis=1)
    omega = np.array([np.sqrt(max(newey_west_lrv(f[j], None), _EPS)) for j in range(k)])
    omega = np.maximum(omega, _EPS)

    stat = float(max(0.0, np.max(np.sqrt(n) * fbar / omega)))

    # Hansen's consistent recentring threshold.
    threshold = -np.sqrt((omega**2 / n) * 2.0 * np.log(max(np.log(n), 1.0001)))
    mu_c = np.where(fbar >= threshold, fbar, 0.0)

    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, block_mean, n_resamples, rng)
    draws = np.empty(n_resamples, dtype=float)
    for i, row in enumerate(idx):
        boot = f[:, row].mean(axis=1)
        z = np.sqrt(n) * (boot - mu_c) / omega
        draws[i] = max(0.0, float(z.max()))

    return {
        "statistic": stat,
        "p_value": float((draws >= stat).mean()),
        "best_model": int(np.argmax(fbar / omega)),
        "n_models": int(k),
        "n_periods": int(n),
    }
