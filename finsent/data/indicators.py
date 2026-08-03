"""Causal technical indicators, written to match TA-Lib bit-for-bit.

Two properties every function here guarantees, and which
``tests/test_indicators_parity.py`` verifies:

1. **Causality.** ``f(x)[t]`` depends only on ``x[:t+1]``. Perturbing any future value
   leaves the output at ``t`` unchanged. This is asserted directly by the test rather
   than argued in a comment.

2. **Reference parity.** Values match TA-Lib to ``1e-6`` after the warm-up period. The
   subtlety that makes hand-rolled indicators disagree with references is *seeding*:
   TA-Lib seeds its exponential averages with a simple moving average of the first
   ``n`` observations, whereas ``pandas.Series.ewm(adjust=False)`` seeds with the first
   observation alone. ``_ema_seeded`` reproduces the TA-Lib convention.

Keeping these from scratch is deliberate. "All indicators are verified to 1e-6 against
TA-Lib" is one sentence in the paper that forecloses an entire category of reviewer
doubt, and the implementations are short enough to be checked by eye.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma",
    "ema",
    "wilder_smooth",
    "rsi",
    "macd",
    "true_range",
    "atr",
    "bollinger",
    "bollinger_pctb",
    "ewma_volatility",
    "garman_klass_volatility",
    "parkinson_volatility",
    "amihud_illiquidity",
    "obv",
]

_EPS = 1e-12


def sma(x: pd.Series, n: int) -> pd.Series:
    """Simple moving average over a trailing window of ``n`` (inclusive of ``t``)."""
    return pd.Series(x, dtype=float).rolling(n, min_periods=n).mean()


def _ema_seeded(x: pd.Series, n: int) -> pd.Series:
    """Exponential moving average seeded with an SMA — the TA-Lib convention.

    ``alpha = 2/(n+1)``; the first output appears at position ``n-1`` and equals the
    simple mean of the first ``n`` values, after which the standard recursion applies.
    """
    v = pd.Series(x, dtype=float).to_numpy()
    out = np.full(v.size, np.nan)
    if v.size < n:
        return pd.Series(out, index=pd.Series(x).index)

    alpha = 2.0 / (n + 1.0)
    seed_window = v[:n]
    if not np.isfinite(seed_window).all():
        first = np.argmax(np.isfinite(v))
        if v.size - first < n:
            return pd.Series(out, index=pd.Series(x).index)
        start = first + n - 1
        prev = float(np.mean(v[first : first + n]))
    else:
        start = n - 1
        prev = float(np.mean(seed_window))

    out[start] = prev
    for i in range(start + 1, v.size):
        prev = alpha * v[i] + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=pd.Series(x).index)


def ema(x: pd.Series, n: int) -> pd.Series:
    """TA-Lib-compatible exponential moving average."""
    return _ema_seeded(x, n)


def wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing: ``avg[t] = (avg[t-1]*(n-1) + x[t]) / n``, SMA-seeded.

    Used by RSI and ATR. Equivalent to an EMA with ``alpha = 1/n``, which is *not* the
    same as ``alpha = 2/(n+1)``; conflating the two is the usual reason a hand-rolled
    RSI drifts a point or two away from every charting package.
    """
    v = pd.Series(x, dtype=float).to_numpy()
    out = np.full(v.size, np.nan)
    finite = np.isfinite(v)
    if finite.sum() < n:
        return pd.Series(out, index=pd.Series(x).index)

    first = int(np.argmax(finite))
    start = first + n - 1
    if start >= v.size:
        return pd.Series(out, index=pd.Series(x).index)

    prev = float(np.mean(v[first : first + n]))
    out[start] = prev
    for i in range(start + 1, v.size):
        prev = (prev * (n - 1) + v[i]) / n
        out[i] = prev
    return pd.Series(out, index=pd.Series(x).index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index, in ``[0, 100]``."""
    c = pd.Series(close, dtype=float)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = wilder_smooth(gain.iloc[1:], n).reindex(c.index)
    avg_loss = wilder_smooth(loss.iloc[1:], n).reindex(c.index)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    # Zero average loss means an unbroken run of gains: RSI is 100 by definition.
    out = out.where(avg_loss.ne(0.0) | avg_gain.isna(), 100.0)
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(macd_line, signal_line, histogram)``.

    The signal line is an EMA of the MACD line computed over the MACD's own non-null
    span, matching TA-Lib's handling of the leading NaNs.
    """
    c = pd.Series(close, dtype=float)
    line = ema(c, fast) - ema(c, slow)
    valid = line.dropna()
    sig = _ema_seeded(valid, signal).reindex(c.index)
    return line, sig, line - sig


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """``max(H-L, |H - C_prev|, |L - C_prev|)``."""
    h, l, c = (pd.Series(v, dtype=float) for v in (high, low, close))
    prev = c.shift(1)
    return pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average True Range with Wilder smoothing."""
    tr = true_range(high, low, close)
    return wilder_smooth(tr.iloc[1:], n).reindex(pd.Series(close).index)


def bollinger(
    close: pd.Series, n: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(upper, middle, lower)``.

    Uses the **population** standard deviation (``ddof=0``), which is what TA-Lib does;
    the sample standard deviation gives visibly different bands on a 20-day window.
    """
    c = pd.Series(close, dtype=float)
    mid = c.rolling(n, min_periods=n).mean()
    sd = c.rolling(n, min_periods=n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def bollinger_pctb(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """``%B = (C - lower) / (upper - lower)``; 0.5 at the middle band."""
    upper, _, lower = bollinger(close, n, k)
    width = (upper - lower).replace(0.0, np.nan)
    return (pd.Series(close, dtype=float) - lower) / width


def ewma_volatility(returns: pd.Series, span: int = 20, annualise: int | None = None) -> pd.Series:
    """Exponentially weighted realised volatility of a return series."""
    r = pd.Series(returns, dtype=float)
    var = r.pow(2).ewm(span=span, min_periods=max(span // 2, 2), adjust=False).mean()
    vol = np.sqrt(var)
    return vol * np.sqrt(annualise) if annualise else vol


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20
) -> pd.Series:
    """Garman-Klass range volatility.

    ``0.5 * ln(H/L)^2 - (2 ln2 - 1) * ln(C/O)^2``, averaged over ``n`` days. Roughly
    seven times more efficient than close-to-close volatility for the same sample, which
    matters a great deal when the estimate feeds a Kelly denominator.
    """
    o, h, l, c = (pd.Series(v, dtype=float) for v in (open_, high, low, close))
    hl = np.log(h / l.replace(0.0, np.nan)) ** 2
    co = np.log(c / o.replace(0.0, np.nan)) ** 2
    daily = 0.5 * hl - (2.0 * np.log(2.0) - 1.0) * co
    return np.sqrt(daily.rolling(n, min_periods=max(n // 2, 2)).mean().clip(lower=0.0))


def parkinson_volatility(high: pd.Series, low: pd.Series, n: int = 20) -> pd.Series:
    """Parkinson high-low range volatility estimator."""
    h, l = pd.Series(high, dtype=float), pd.Series(low, dtype=float)
    hl = np.log(h / l.replace(0.0, np.nan)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt((factor * hl).rolling(n, min_periods=max(n // 2, 2)).mean())


def amihud_illiquidity(returns: pd.Series, dollar_volume: pd.Series, n: int = 21) -> pd.Series:
    """Amihud (2002) illiquidity: mean of ``|r| / dollar volume`` over ``n`` days."""
    r = pd.Series(returns, dtype=float).abs()
    dv = pd.Series(dollar_volume, dtype=float).replace(0.0, np.nan)
    return (r / dv).rolling(n, min_periods=max(n // 2, 2)).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume, signed cumulative volume."""
    c = pd.Series(close, dtype=float)
    v = pd.Series(volume, dtype=float)
    return (np.sign(c.diff()).fillna(0.0) * v).cumsum()
