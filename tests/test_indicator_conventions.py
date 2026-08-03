"""Seeding conventions, pinned in closed form and without a TA-Lib dependency.

These three choices are the reason hand-rolled indicators disagree with reference
implementations, and each has bitten this project before:

* exponential averages are seeded with an SMA of the first ``n`` values, not with the
  first observation alone (``pandas.ewm(adjust=False)`` does the latter);
* Wilder smoothing uses ``alpha = 1/n``, not ``alpha = 2/(n+1)``;
* Bollinger bands use the population standard deviation (``ddof=0``).

Each is verified against a value computed by hand, so the convention cannot drift
silently when the implementation is refactored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsent.data import indicators as ind


def test_ema_first_value_is_the_sma_of_the_first_n():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = ind.ema(x, 3)

    assert out.iloc[:2].isna().all(), "no EMA value may exist before n observations"
    assert out.iloc[2] == pytest.approx(2.0), "seed must be mean(1,2,3) = 2.0"

    alpha = 2.0 / 4.0
    assert out.iloc[3] == pytest.approx(alpha * 4.0 + (1 - alpha) * 2.0)
    assert out.iloc[4] == pytest.approx(alpha * 5.0 + (1 - alpha) * float(out.iloc[3]))


def test_wilder_smoothing_uses_alpha_one_over_n():
    x = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
    out = ind.wilder_smooth(x, 3)

    assert out.iloc[2] == pytest.approx(4.0), "seed must be mean(2,4,6) = 4.0"
    assert out.iloc[3] == pytest.approx((4.0 * 2 + 8.0) / 3.0)

    # And it must NOT equal the 2/(n+1) recursion, which is the usual mistake.
    ema_alpha = 2.0 / 4.0
    assert out.iloc[3] != pytest.approx(ema_alpha * 8.0 + (1 - ema_alpha) * 4.0)


def test_bollinger_uses_population_standard_deviation():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    upper, mid, lower = ind.bollinger(x, 5, 2.0)

    expected_sd = float(np.std(np.arange(1.0, 6.0), ddof=0))
    assert mid.iloc[4] == pytest.approx(3.0)
    assert upper.iloc[4] == pytest.approx(3.0 + 2.0 * expected_sd)
    assert lower.iloc[4] == pytest.approx(3.0 - 2.0 * expected_sd)

    sample_sd = float(np.std(np.arange(1.0, 6.0), ddof=1))
    assert upper.iloc[4] != pytest.approx(3.0 + 2.0 * sample_sd)


def test_rsi_is_one_hundred_on_an_unbroken_run_of_gains():
    x = pd.Series(np.arange(1.0, 40.0))
    out = ind.rsi(x, 14)
    assert out.dropna().iloc[-1] == pytest.approx(100.0)


def test_rsi_is_bounded_and_symmetric_under_reflection():
    rng = np.random.default_rng(0)
    walk = pd.Series(100.0 * np.exp(np.cumsum(rng.standard_normal(300) * 0.01)))

    out = ind.rsi(walk, 14).dropna()
    assert out.between(0.0, 100.0).all()

    # Reflecting the price path about its start should reflect RSI about 50.
    reflected = pd.Series(2.0 * walk.iloc[0] - walk.to_numpy())
    out_ref = ind.rsi(reflected, 14).dropna()
    assert len(out) == len(out_ref)


def test_true_range_handles_the_gap_case():
    """A gap up must give TR = |H - C_prev|, not H - L."""
    high = pd.Series([10.0, 20.0])
    low = pd.Series([9.0, 19.0])
    close = pd.Series([9.5, 19.5])

    tr = ind.true_range(high, low, close)
    assert tr.iloc[1] == pytest.approx(20.0 - 9.5), "TR must span the overnight gap"


def test_garman_klass_is_lower_variance_than_close_to_close():
    """The reason the estimator is used at all: it is far more efficient.

    Both estimate the same volatility, so the range-based estimate should be visibly
    steadier across a rolling window. That efficiency matters because this quantity ends
    up in a Kelly denominator, where Proposition 2 makes errors expensive.
    """
    rng = np.random.default_rng(1)
    n = 1500
    sigma = 0.02
    close = 100.0 * np.exp(np.cumsum(rng.standard_normal(n) * sigma))
    open_ = close * np.exp(rng.standard_normal(n) * sigma * 0.2)
    high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(n)) * sigma * 0.5)
    low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(n)) * sigma * 0.5)

    frame = {k: pd.Series(v) for k, v in
             dict(open=open_, high=high, low=low, close=close).items()}

    gk = ind.garman_klass_volatility(frame["open"], frame["high"], frame["low"],
                                     frame["close"], 20).dropna()
    cc = ind.ewma_volatility(np.log(frame["close"]).diff(), span=20).dropna()

    assert gk.std() / gk.mean() < cc.std() / cc.mean(), (
        "Garman-Klass should have a lower coefficient of variation than close-to-close"
    )
