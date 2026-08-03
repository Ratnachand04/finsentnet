"""Reference parity for the hand-rolled indicators.

The paper claims "all indicators are verified to 1e-6 against TA-Lib". This is the test
that makes that sentence true, and the whole module skips when TA-Lib is not installed.

The seeding conventions that make hand-rolled indicators drift from every charting
package are pinned separately, without a TA-Lib dependency, in
``test_indicator_conventions.py``. Both files must pass before the parity claim may
appear in the manuscript.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finsent.data import indicators as ind
from finsent.data.synthetic import make_ohlcv_panel

talib = pytest.importorskip("talib", reason="TA-Lib not installed; parity checks skipped")

WARMUP = 260
ATOL = 1e-6


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    return make_ohlcv_panel(n_names=1, n_days=600, seed=13)["SYN000"]


def _compare(mine: pd.Series, theirs: np.ndarray, name: str) -> None:
    a = mine.to_numpy(dtype=float)[WARMUP:]
    b = np.asarray(theirs, dtype=float)[WARMUP:]
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 100, f"{name}: too few comparable observations"
    diff = np.abs(a[both] - b[both])
    assert diff.max() < ATOL, f"{name}: max deviation {diff.max():.3e} exceeds {ATOL}"


def test_rsi_matches_talib(ohlcv):
    _compare(ind.rsi(ohlcv["close"], 14), talib.RSI(ohlcv["close"].to_numpy(), 14), "RSI")


def test_ema_matches_talib(ohlcv):
    _compare(ind.ema(ohlcv["close"], 12), talib.EMA(ohlcv["close"].to_numpy(), 12), "EMA")


def test_sma_matches_talib(ohlcv):
    _compare(ind.sma(ohlcv["close"], 20), talib.SMA(ohlcv["close"].to_numpy(), 20), "SMA")


def test_macd_matches_talib(ohlcv):
    line, signal, hist = ind.macd(ohlcv["close"], 12, 26, 9)
    t_line, t_signal, t_hist = talib.MACD(ohlcv["close"].to_numpy(), 12, 26, 9)
    _compare(line, t_line, "MACD line")
    _compare(signal, t_signal, "MACD signal")
    _compare(hist, t_hist, "MACD histogram")


def test_atr_matches_talib(ohlcv):
    mine = ind.atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], 14)
    theirs = talib.ATR(
        ohlcv["high"].to_numpy(), ohlcv["low"].to_numpy(), ohlcv["close"].to_numpy(), 14
    )
    _compare(mine, theirs, "ATR")


def test_bollinger_matches_talib(ohlcv):
    upper, mid, lower = ind.bollinger(ohlcv["close"], 20, 2.0)
    t_up, t_mid, t_lo = talib.BBANDS(ohlcv["close"].to_numpy(), 20, 2.0, 2.0, 0)
    _compare(upper, t_up, "BB upper")
    _compare(mid, t_mid, "BB middle")
    _compare(lower, t_lo, "BB lower")


def test_true_range_matches_talib(ohlcv):
    mine = ind.true_range(ohlcv["high"], ohlcv["low"], ohlcv["close"])
    theirs = talib.TRANGE(
        ohlcv["high"].to_numpy(), ohlcv["low"].to_numpy(), ohlcv["close"].to_numpy()
    )
    _compare(mine, theirs, "TRANGE")
