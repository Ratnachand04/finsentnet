"""Predictive, calibration and performance metrics.

Everything here is deliberately dependency-light (numpy / pandas / scipy) and
hand-rolled from definitions, so that a referee can check each formula against the
source without chasing a library version.

Metric groups
-------------
cross-sectional : ``information_coefficient``, ``ic_summary``, ``quantile_spread``
classification  : ``confusion_matrix``, ``classification_metrics``
calibration     : ``brier_score``, ``expected_calibration_error``, ``reliability_curve``
performance     : ``sharpe_ratio``, ``sortino_ratio``, ``max_drawdown``, ``turnover``

Convention reminder (SPEC.md 2.1): ``0 = DOWN``, ``1 = NEUTRAL``, ``2 = UP``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

__all__ = [
    "information_coefficient",
    "ic_summary",
    "ICSummary",
    "quantile_spread",
    "confusion_matrix",
    "classification_metrics",
    "brier_score",
    "multiclass_nll",
    "expected_calibration_error",
    "reliability_curve",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "hit_rate",
    "turnover",
    "annualised_return",
    "annualised_volatility",
    "performance_summary",
]

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Cross-sectional (the industry currency)
# --------------------------------------------------------------------------------------
def information_coefficient(
    scores: pd.Series,
    forward_returns: pd.Series,
    dates: pd.Series | np.ndarray,
    method: str = "spearman",
    min_names: int = 5,
) -> pd.Series:
    """Per-date cross-sectional correlation between score and realised forward return.

    ``method="spearman"`` gives the Rank-IC, which is what practitioners quote because
    it is robust to the heavy tails of daily equity returns.

    Returns a Series indexed by date. Dates with fewer than ``min_names`` observations
    are dropped rather than reported as noisy singletons.
    """
    if method not in {"spearman", "pearson"}:
        raise ValueError(f"unknown method {method!r}")

    frame = pd.DataFrame(
        {
            "date": pd.Series(np.asarray(dates)).to_numpy(),
            "score": np.asarray(scores, dtype=float),
            "fwd": np.asarray(forward_returns, dtype=float),
        }
    ).dropna()

    out: dict[Any, float] = {}
    for date, grp in frame.groupby("date", sort=True):
        if len(grp) < min_names:
            continue
        s, r = grp["score"].to_numpy(), grp["fwd"].to_numpy()
        if np.std(s) < _EPS or np.std(r) < _EPS:
            continue
        if method == "spearman":
            value = sps.spearmanr(s, r).statistic
        else:
            value = float(np.corrcoef(s, r)[0, 1])
        if np.isfinite(value):
            out[date] = float(value)

    return pd.Series(out, name=f"{method}_ic").sort_index()


@dataclass(frozen=True)
class ICSummary:
    """Summary of an information-coefficient series."""

    mean: float
    std: float
    icir: float
    icir_annualised: float
    t_stat: float
    p_value: float
    hit_rate: float
    n_periods: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def ic_summary(
    ic: pd.Series,
    periods_per_year: int = 252,
    hac_lags: int | None = 10,
) -> ICSummary:
    """Summarise an IC series with a HAC (Newey-West) t-statistic.

    The plain t-statistic overstates significance because IC is autocorrelated; the
    HAC correction is what a referee expects to see.
    """
    from finsent.eval.stats import newey_west_tstat  # local import avoids a cycle

    values = np.asarray(ic.dropna(), dtype=float)
    n = values.size
    if n == 0:
        return ICSummary(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else np.nan
    icir = mean / std if std and std > _EPS else np.nan
    t_stat, p_value = newey_west_tstat(values, lags=hac_lags)

    return ICSummary(
        mean=mean,
        std=std,
        icir=float(icir),
        icir_annualised=float(icir * np.sqrt(periods_per_year)) if np.isfinite(icir) else np.nan,
        t_stat=float(t_stat),
        p_value=float(p_value),
        hit_rate=float((values > 0).mean()),
        n_periods=int(n),
    )


def quantile_spread(
    scores: pd.Series,
    forward_returns: pd.Series,
    dates: pd.Series | np.ndarray,
    n_quantiles: int = 10,
    min_names: int = 20,
) -> tuple[pd.Series, pd.DataFrame]:
    """Long-short top-minus-bottom quantile return, per date.

    Returns ``(spread, per_quantile)`` where ``spread`` is the daily top-decile minus
    bottom-decile mean return and ``per_quantile`` holds each quantile's mean return.
    Monotonicity across quantiles is the sanity check that a signal is real rather than
    an artefact of a handful of extreme names.
    """
    frame = pd.DataFrame(
        {
            "date": pd.Series(np.asarray(dates)).to_numpy(),
            "score": np.asarray(scores, dtype=float),
            "fwd": np.asarray(forward_returns, dtype=float),
        }
    ).dropna()

    spreads: dict[Any, float] = {}
    rows: list[dict[str, Any]] = []

    for date, grp in frame.groupby("date", sort=True):
        if len(grp) < max(min_names, n_quantiles):
            continue
        ranks = grp["score"].rank(method="first", pct=True).to_numpy()
        bucket = np.clip((ranks * n_quantiles).astype(int), 0, n_quantiles - 1)
        fwd = grp["fwd"].to_numpy()

        means = np.full(n_quantiles, np.nan)
        for q in range(n_quantiles):
            sel = bucket == q
            if sel.any():
                means[q] = fwd[sel].mean()
        if np.isfinite(means[0]) and np.isfinite(means[-1]):
            spreads[date] = float(means[-1] - means[0])
        rows.append({"date": date, **{f"q{q + 1}": means[q] for q in range(n_quantiles)}})

    per_quantile = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    return pd.Series(spreads, name="ls_spread").sort_index(), per_quantile


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Rows are true classes, columns are predictions."""
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_pred, dtype=int).ravel()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (yt, yp), 1)
    return cm


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 3,
    neutral_class: int = 1,
) -> dict[str, Any]:
    """Accuracy, balanced accuracy, macro-F1, MCC and the majority-class baseline.

    ``accuracy`` alone is misleading here because the NEUTRAL class dominates; the
    paper reports balanced accuracy and MCC alongside it for exactly that reason.
    ``binary_accuracy`` excludes NEUTRAL truths and is the number an interviewer asks
    for, since it isolates directional skill.
    """
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_pred, dtype=int).ravel()
    if yt.size == 0:
        return {"n": 0}

    cm = confusion_matrix(yt, yp, n_classes)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    correct = np.diag(cm)
    total = cm.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, correct / np.maximum(support, 1), np.nan)
        precision = np.where(predicted > 0, correct / np.maximum(predicted, 1), np.nan)
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / np.maximum(precision + recall, _EPS),
            0.0,
        )

    # Matthews correlation coefficient, multiclass form (Gorodkin, 2004).
    c, s = float(correct.sum()), float(total)
    pk, tk = predicted.astype(float), support.astype(float)
    num = c * s - float(pk @ tk)
    den = np.sqrt(max(s**2 - float(pk @ pk), 0.0)) * np.sqrt(max(s**2 - float(tk @ tk), 0.0))
    mcc = num / den if den > _EPS else 0.0

    directional = yt != neutral_class
    if directional.any():
        binary_acc = float((yt[directional] == yp[directional]).mean())
    else:
        binary_acc = np.nan

    return {
        "n": int(total),
        "accuracy": float(c / s),
        "balanced_accuracy": float(np.nanmean(recall)),
        "majority_baseline": float(support.max() / s),
        "macro_f1": float(np.nanmean(f1)),
        "mcc": float(mcc),
        "binary_accuracy": binary_acc,
        "per_class_f1": [float(v) for v in f1],
        "per_class_recall": [float(v) for v in recall],
        "per_class_precision": [float(v) for v in precision],
        "support": [int(v) for v in support],
        "confusion_matrix": cm.tolist(),
    }


# --------------------------------------------------------------------------------------
# Calibration — the metric family this project's central claim rests on
# --------------------------------------------------------------------------------------
def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot outcome."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, dtype=int).ravel()
    onehot = np.zeros_like(p)
    onehot[np.arange(y.size), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def multiclass_nll(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean negative log-likelihood of the observed class."""
    p = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0)
    y = np.asarray(y_true, dtype=int).ravel()
    return float(-np.mean(np.log(p[np.arange(y.size), y])))


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Equal-width binned ECE and MCE over top-class confidence.

    Note for the manuscript: this estimator uses hard bins and is therefore **not
    differentiable**. It is a reporting metric only. The training-time calibration
    penalty is MMCE (``finsent.training.losses.mmce_loss``); using binned ECE as a
    training objective, as the V2 draft described, is not well defined.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, dtype=int).ravel()
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    acc = (pred == y).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    rows: list[dict[str, float]] = []
    n = conf.size
    for b in range(n_bins):
        sel = idx == b
        count = int(sel.sum())
        if count == 0:
            rows.append(
                {
                    "bin_lower": float(edges[b]),
                    "bin_upper": float(edges[b + 1]),
                    "count": 0,
                    "confidence": np.nan,
                    "accuracy": np.nan,
                    "gap": np.nan,
                }
            )
            continue
        bin_conf = float(conf[sel].mean())
        bin_acc = float(acc[sel].mean())
        gap = abs(bin_acc - bin_conf)
        ece += (count / n) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "count": count,
                "confidence": bin_conf,
                "accuracy": bin_acc,
                "gap": gap,
            }
        )

    return {
        "ece": float(ece),
        "mce": float(mce),
        "n_bins": n_bins,
        "bins": pd.DataFrame(rows),
        "mean_confidence": float(conf.mean()),
        "mean_accuracy": float(acc.mean()),
        "overconfidence": float(conf.mean() - acc.mean()),
    }


def reliability_curve(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Bin-level (confidence, accuracy) pairs for a reliability diagram."""
    return expected_calibration_error(probs, y_true, n_bins)["bins"]


# --------------------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------------------
def annualised_return(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan
    return float(np.expm1(np.log1p(r).sum() * periods_per_year / r.size))


def annualised_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio. ``risk_free_rate`` is annualised and de-annualised here."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    excess = r - risk_free_rate / periods_per_year
    sd = excess.std(ddof=1)
    if sd < _EPS:
        return np.nan
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    excess = r - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    if downside.size == 0:
        return np.inf
    dd = np.sqrt(np.mean(downside**2))
    if dd < _EPS:
        return np.nan
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough decline of the compounded equity curve (positive number)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd < _EPS:
        return np.nan
    return float(annualised_return(returns, periods_per_year) / mdd)


def hit_rate(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r) & (r != 0.0)]
    return float((r > 0).mean()) if r.size else np.nan


def turnover(weights: np.ndarray) -> float:
    """Mean one-period two-sided turnover, ``mean(sum |w_t - w_{t-1}|)``.

    Reported in the paper because a signal with 100% daily turnover is uninvestable at
    10 bps regardless of its gross Sharpe.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(w, axis=0)).sum(axis=1)))


def performance_summary(
    returns: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """One dict with every performance number the paper's Table 6 needs."""
    r = np.asarray(returns, dtype=float)
    out = {
        "n_periods": int(np.isfinite(r).sum()),
        "ann_return": annualised_return(r, periods_per_year),
        "ann_volatility": annualised_volatility(r, periods_per_year),
        "sharpe": sharpe_ratio(r, periods_per_year, risk_free_rate),
        "sortino": sortino_ratio(r, periods_per_year, risk_free_rate),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r, periods_per_year),
        "hit_rate": hit_rate(r),
        "skew": float(sps.skew(r[np.isfinite(r)])) if np.isfinite(r).sum() > 2 else np.nan,
        "kurtosis": float(sps.kurtosis(r[np.isfinite(r)])) if np.isfinite(r).sum() > 3 else np.nan,
    }
    if weights is not None:
        out["turnover"] = turnover(weights)
        out["turnover_annualised"] = out["turnover"] * periods_per_year
    return out
