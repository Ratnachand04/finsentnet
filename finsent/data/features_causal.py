"""The twelve-feature causal set, and cross-sectional normalisation.

V2 carried roughly 35 computed features, of which 15 or 20 reached the model depending
on which document you read. Most were the same factor expressed several ways --
``SMA5/10/20/50/200`` plus ``EMA5/.../200`` plus price-versus-each is one trend signal
written nine times -- which inflates the parameter count without adding information and
makes any attention or importance analysis uninterpretable.

The replacement is twelve features spanning six economically distinct groups:

=================  =====================================================================
group              features
=================  =====================================================================
returns            ``ret_1``, ``ret_5``, ``ret_21``
momentum           ``mom_12_1`` (12-month, skipping the last month), ``rev_5``
volatility         ``ewma_vol_20``, ``atr14_norm``, ``gk_vol_20``
trend              ``macd_hist_norm``
mean reversion     ``rsi_14``, ``bb_pctb``
liquidity          ``dollar_vol_log``
=================  =====================================================================

Note on ``rev_5``: defined as the trailing five-day sum of **overnight gap returns**
``log(O_t / C_{t-1})``, not as ``-ret_5``. The negation would be perfectly collinear
with ``ret_5`` and a referee would say so; the overnight-gap variant is a genuinely
distinct and documented effect.

Cross-sectional ranking
-----------------------
Every feature is ranked within each day across the universe and mapped to
``[-0.5, 0.5]``. This does three things at once: it removes the market factor, it makes
each feature stationary regardless of regime, and it makes the model learn *relative*
attractiveness -- which is what a cross-sectional portfolio actually trades. It is also
what the asset-pricing ML literature does (Gu, Kelly & Xiu, 2020), so a reviewer
recognises it immediately.

Causality is the invariant that matters most here and is asserted by
``tests/test_causality.py``: no feature at ``t`` may read a price after ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finsent.data import indicators as ind

__all__ = [
    "FEATURE_NAMES",
    "compute_features",
    "cross_sectional_rank",
    "winsorize",
    "build_feature_panel",
    "lag_for_open_execution",
]

FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_21",
    "mom_12_1",
    "rev_5",
    "ewma_vol_20",
    "atr14_norm",
    "gk_vol_20",
    "macd_hist_norm",
    "rsi_14",
    "bb_pctb",
    "dollar_vol_log",
)

_EPS = 1e-12


def compute_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Twelve causal features for one ticker from an OHLCV frame.

    ``ohlcv`` needs columns ``open``, ``high``, ``low``, ``close``, ``volume`` indexed
    by session date. Every column of the result at row ``t`` is a function of rows
    ``<= t`` only.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv is missing columns: {sorted(missing)}")

    o, h, l = ohlcv["open"], ohlcv["high"], ohlcv["low"]
    c, v = ohlcv["close"], ohlcv["volume"]

    log_close = np.log(c.replace(0.0, np.nan))
    ret_1 = log_close.diff()

    atr14 = ind.atr(h, l, c, 14)
    _, _, macd_hist = ind.macd(c, 12, 26, 9)

    out = pd.DataFrame(index=ohlcv.index)
    out["ret_1"] = ret_1
    out["ret_5"] = log_close.diff(5)
    out["ret_21"] = log_close.diff(21)
    # 12-1 momentum: skip the most recent month to avoid the short-term reversal effect.
    out["mom_12_1"] = log_close.shift(21) - log_close.shift(252)
    # Overnight-gap reversal; distinct from ret_5 rather than its negation.
    out["rev_5"] = np.log(o / c.shift(1).replace(0.0, np.nan)).rolling(5, min_periods=3).sum()
    out["ewma_vol_20"] = ind.ewma_volatility(ret_1, span=20)
    out["atr14_norm"] = atr14 / c.replace(0.0, np.nan)
    out["gk_vol_20"] = ind.garman_klass_volatility(o, h, l, c, 20)
    # Normalising the MACD histogram by ATR makes it comparable across price levels and
    # volatility regimes; the raw histogram is in dollars and is not.
    out["macd_hist_norm"] = macd_hist / atr14.replace(0.0, np.nan)
    out["rsi_14"] = ind.rsi(c, 14) / 100.0
    out["bb_pctb"] = ind.bollinger_pctb(c, 20, 2.0)
    out["dollar_vol_log"] = np.log1p((c * v).clip(lower=0.0))

    return out[list(FEATURE_NAMES)].replace([np.inf, -np.inf], np.nan)


def winsorize(frame: pd.DataFrame, limit: float = 0.01) -> pd.DataFrame:
    """Clip each column to its ``[limit, 1-limit]`` quantiles, computed per row group.

    Applied cross-sectionally (one row = one day) so the clipping uses no time-series
    information and therefore cannot look ahead.
    """
    if limit <= 0:
        return frame
    lo = frame.quantile(limit, axis=1)
    hi = frame.quantile(1.0 - limit, axis=1)
    return frame.clip(lower=lo, upper=hi, axis=0)


def cross_sectional_rank(values: pd.Series, min_names: int = 5) -> pd.Series:
    """Rank to ``[-0.5, 0.5]`` within the group, exactly centred; NaNs stay NaN.

    Uses ``(rank - 0.5) / n - 0.5`` rather than ``rank_pct - 0.5``. The latter maps
    ranks to ``{1/n, ..., 1}`` and therefore leaves a mean of ``+1/(2n)`` on every day --
    a constant that a dollar-neutral portfolio would have to cancel and that shows up as
    a small spurious long bias on narrow universes. Negligible at 400 names, material at
    8, and free to avoid.

    Days with fewer than ``min_names`` observations return NaN rather than a rank: a
    "decile" of three names is not a decile.
    """
    v = pd.Series(values, dtype=float)
    n = int(v.notna().sum())
    if n < min_names:
        return pd.Series(np.nan, index=v.index)
    return (v.rank(method="average") - 0.5) / n - 0.5


def lag_for_open_execution(panel: pd.DataFrame, sessions: int = 1) -> pd.DataFrame:
    """Shift every feature forward by one session, per ticker. **Required**, not optional.

    The bug this prevents, stated precisely
    ---------------------------------------
    Features are computed from daily bars, so a feature dated ``t`` uses the **close** of
    day ``t``. The label is measured open-to-open, so the return attributed to a decision
    at ``t`` begins at the **open** of day ``t`` -- which comes *before* that close.
    Decomposing an open-to-open five-day return into close-to-close terms,

        log(O[t+5] / O[t]) = r[t] + r[t+1] + ... + r[t+4] + overnight terms
        ret_5[t]           = r[t-4] + r[t-3] + ... + r[t]

    the two share ``r[t]``. On pure random-walk data that alone produces a cross-sectional
    rank information coefficient near 0.11 -- roughly five times a realistic value, and
    entirely spurious. It was caught here by the audit threshold in SPEC.md section 7,
    which is what that threshold is for.

    Lagging by one session restores the contract: a decision executed at the open of
    day ``t`` may use information through the close of day ``t-1`` and no later.
    """
    if sessions < 1:
        raise ValueError("open-executed decisions require a lag of at least one session")

    out = panel.sort_values(["ticker", "date"]).copy()
    cols = [c for c in FEATURE_NAMES if c in out.columns]
    out[cols] = out.groupby("ticker", sort=False)[cols].shift(sessions)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_feature_panel(
    ohlcv_by_ticker: dict[str, pd.DataFrame],
    cross_sectional: bool = True,
    winsorize_limit: float = 0.01,
    min_names: int = 5,
    lag_sessions: int = 1,
) -> pd.DataFrame:
    """Assemble a tidy ``[date, ticker, <12 features>]`` panel for the whole universe.

    When ``cross_sectional`` is set, each feature is winsorised and ranked within each
    date across the names present on that date -- the point-in-time universe, so names
    that had not yet listed or have delisted simply do not participate.

    ``lag_sessions`` defaults to 1 and applies :func:`lag_for_open_execution`, without
    which a feature dated ``t`` would carry day ``t``'s close into a decision executed at
    day ``t``'s open. Set it to 0 only when labels are measured close-to-close from
    ``t+1``, and say so in the paper.
    """
    frames = []
    for ticker, ohlcv in ohlcv_by_ticker.items():
        feats = compute_features(ohlcv)
        feats = feats.assign(ticker=str(ticker))
        feats.index.name = "date"
        frames.append(feats.reset_index())

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", *FEATURE_NAMES])

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    if lag_sessions:
        panel = lag_for_open_execution(panel, lag_sessions)

    if not cross_sectional:
        return panel

    for name in FEATURE_NAMES:
        col = panel[name]
        if winsorize_limit > 0:
            bounds = panel.groupby("date")[name].transform(
                lambda s: s.clip(s.quantile(winsorize_limit), s.quantile(1 - winsorize_limit))
            )
            col = bounds
        panel[name] = col.groupby(panel["date"]).transform(
            lambda s: cross_sectional_rank(s, min_names=min_names)
        )

    return panel
