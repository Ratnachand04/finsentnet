"""Labels: volatility-scaled dead bands and the triple barrier.

Why the V2 fixed 0.5% band was replaced
---------------------------------------
A constant threshold makes "UP" mean different things for a 15%-volatility utility and
an 80%-volatility biotech, and different things in 2018 than in March 2020. Class
balance then becomes a *volatility proxy* rather than a property of the signal, with two
consequences: accuracy stops being comparable across periods, and the class imbalance
that motivated the augmentation GAN was largely manufactured by the labelling scheme
itself.

The replacement scales the band by a causal volatility estimate::

    theta[i,t] = k_band * EWMA_span(|r[i,.]|) * sqrt(h)

With ``k_band`` around 0.6 the three classes stay roughly stable through time.

Triple barrier
--------------
``triple_barrier_labels`` implements the upper / lower / vertical construction of
Lopez de Prado (2018, s. 3.4). It matters here for a specific reason: any use of
barrier-defined odds in position sizing requires a label that refers to *the same
event*. Feeding a terminal-horizon class probability into barrier odds, as V2 did, is a
category error, because hitting a target before a stop is a first-passage event.

Label convention is frozen: ``0 = DOWN``, ``1 = NEUTRAL``, ``2 = UP`` (SPEC.md 2.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finsent.config import LABEL_DOWN, LABEL_NEUTRAL, LABEL_UP

__all__ = [
    "causal_volatility",
    "forward_log_return",
    "vol_scaled_labels",
    "triple_barrier_labels",
    "LabelResult",
    "class_balance",
]

_EPS = 1e-12


def causal_volatility(
    returns: pd.Series, span: int = 60, min_periods: int | None = None
) -> pd.Series:
    """EWMA of absolute returns, **shifted by one** so that ``sigma[t]`` uses only ``<t``.

    The shift is the whole point. An unshifted EWMA at ``t`` includes ``r[t]``, and since
    the label at ``t`` is a function of returns starting at ``t``, that would let the
    threshold peek at the very move it is meant to classify.

    ``min_periods`` defaults to ``max(2, span // 3)`` rather than a fixed constant, so
    that shrinking ``span`` actually shortens the warm-up instead of silently producing
    an all-NaN threshold and therefore an all-NaN label column.
    """
    r = pd.Series(returns, dtype=float)
    mp = min_periods if min_periods is not None else max(2, span // 3)
    ewma = r.abs().ewm(span=span, min_periods=mp, adjust=False).mean()
    return ewma.shift(1)


def forward_log_return(prices: pd.Series, horizon: int) -> pd.Series:
    """``log(P[t+h] / P[t])`` — the return *earned* from a decision at ``t``.

    ``prices`` should be the execution price series (opens, if decisions execute at the
    open), not closes. SPEC.md 2.3 requires open-to-open measurement.
    """
    p = pd.Series(prices, dtype=float)
    return np.log(p.shift(-horizon) / p)


@dataclass(frozen=True)
class LabelResult:
    """Labels plus everything needed to audit or re-derive them."""

    y: pd.Series
    forward_return: pd.Series
    threshold: pd.Series
    horizon: int
    method: str
    t1: pd.Series  # index position at which each label is resolved

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "y": self.y,
                "fwd_ret": self.forward_return,
                "theta": self.threshold,
                "t1": self.t1,
            }
        )

    def balance(self) -> dict[str, float]:
        return class_balance(self.y)


def vol_scaled_labels(
    prices: pd.Series,
    horizon: int = 5,
    k_band: float = 0.6,
    vol_span: int = 60,
    returns: pd.Series | None = None,
) -> LabelResult:
    """Three-class labels with a causal, volatility-scaled dead band.

    ``prices`` is the execution price series. ``returns`` may be supplied when the
    volatility estimate should come from a different series (for example close-to-close
    returns while execution is at the open).
    """
    p = pd.Series(prices, dtype=float)
    r = pd.Series(returns, dtype=float) if returns is not None else np.log(p / p.shift(1))

    sigma = causal_volatility(r, span=vol_span)
    theta = k_band * sigma * np.sqrt(horizon)
    fwd = forward_log_return(p, horizon)

    y = pd.Series(np.full(len(p), np.nan), index=p.index, dtype=float)
    valid = fwd.notna() & theta.notna()
    y[valid & (fwd > theta)] = LABEL_UP
    y[valid & (fwd < -theta)] = LABEL_DOWN
    y[valid & (fwd.abs() <= theta)] = LABEL_NEUTRAL

    positions = pd.Series(np.arange(len(p)), index=p.index)
    t1 = (positions + horizon).where(valid)

    return LabelResult(
        y=y, forward_return=fwd, threshold=theta, horizon=horizon,
        method="vol_scaled_band", t1=t1,
    )


def triple_barrier_labels(
    close: pd.Series,
    horizon: int = 5,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    vol_span: int = 60,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
) -> LabelResult:
    """Upper / lower / vertical barrier labels (Lopez de Prado, 2018, s. 3.4).

    Barrier widths are ``pt_mult * sigma`` and ``sl_mult * sigma`` with the same causal
    volatility estimate used elsewhere, so a label reflects what would actually have
    happened to a stop-and-target trade. When ``high``/``low`` are supplied the touch
    test uses the intraday range, which is stricter and more realistic than testing
    closes only.

    ``t1`` records the position at which each label resolved -- the first barrier touch,
    not the vertical barrier -- and feeds ``uniqueness.average_uniqueness``, because
    overlap between labels is what inflates the effective sample size.
    """
    c = pd.Series(close, dtype=float)
    r = np.log(c / c.shift(1))
    sigma = causal_volatility(r, span=vol_span)

    hi = pd.Series(high, dtype=float) if high is not None else c
    lo = pd.Series(low, dtype=float) if low is not None else c

    n = len(c)
    y = np.full(n, np.nan)
    fwd = np.full(n, np.nan)
    t1 = np.full(n, np.nan)
    upper = np.full(n, np.nan)

    c_arr, hi_arr, lo_arr, sig_arr = (
        c.to_numpy(), hi.to_numpy(), lo.to_numpy(), sigma.to_numpy()
    )

    for i in range(n):
        s = sig_arr[i]
        if not np.isfinite(s) or s <= _EPS or i + 1 >= n:
            continue
        entry = c_arr[i]
        pt = entry * np.exp(pt_mult * s)
        sl = entry * np.exp(-sl_mult * s)
        upper[i] = pt

        end = min(i + horizon, n - 1)
        label, resolved = LABEL_NEUTRAL, end
        for j in range(i + 1, end + 1):
            touched_up = hi_arr[j] >= pt
            touched_dn = lo_arr[j] <= sl
            if touched_up and touched_dn:
                # Both barriers inside one bar: unresolvable from daily data. Treat as
                # NEUTRAL rather than guessing an ordering that the data cannot support.
                label, resolved = LABEL_NEUTRAL, j
                break
            if touched_up:
                label, resolved = LABEL_UP, j
                break
            if touched_dn:
                label, resolved = LABEL_DOWN, j
                break

        y[i] = label
        t1[i] = resolved
        fwd[i] = np.log(c_arr[resolved] / entry)

    idx = c.index
    return LabelResult(
        y=pd.Series(y, index=idx),
        forward_return=pd.Series(fwd, index=idx),
        threshold=pd.Series(upper, index=idx),
        horizon=horizon,
        method="triple_barrier",
        t1=pd.Series(t1, index=idx),
    )


def class_balance(y: pd.Series | np.ndarray) -> dict[str, float]:
    """Class shares, used in Table 1 and to derive inverse-frequency loss weights."""
    v = pd.Series(np.asarray(y, dtype=float)).dropna().astype(int)
    if v.empty:
        return {"DOWN": np.nan, "NEUTRAL": np.nan, "UP": np.nan, "n": 0}
    counts = v.value_counts(normalize=True)
    return {
        "DOWN": float(counts.get(LABEL_DOWN, 0.0)),
        "NEUTRAL": float(counts.get(LABEL_NEUTRAL, 0.0)),
        "UP": float(counts.get(LABEL_UP, 0.0)),
        "n": int(v.size),
    }
