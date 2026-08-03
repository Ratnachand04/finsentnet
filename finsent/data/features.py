"""
Technical indicator computation — vectorized, zero look-ahead.
=============================================================

All indicators are computed using ONLY past data at each point t.
No future information leaks into any feature.

Implemented from scratch using NumPy for transparency and control.
No TA-Lib or similar black boxes.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_returns(close: np.ndarray, period: int = 1) -> np.ndarray:
    """Simple log returns. ret[t] = ln(close[t] / close[t-period]).
    
    First `period` values are NaN to prevent look-ahead.
    """
    ret = np.full_like(close, np.nan, dtype=np.float64)
    ret[period:] = np.log(close[period:] / close[:-period])
    return ret


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index using exponential moving average of gains/losses.
    
    RSI = 100 - 100 / (1 + RS)
    RS = EMA(gains, period) / EMA(losses, period)
    
    Uses Wilder's smoothing (alpha = 1/period).
    """
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    rsi = np.full_like(close, np.nan, dtype=np.float64)
    
    if len(close) <= period:
        return rsi
    
    # Seed with simple average
    avg_gain = np.mean(gains[1:period + 1])
    avg_loss = np.mean(losses[1:period + 1])
    
    # Wilder's smoothing
    for i in range(period, len(close)):
        if i == period:
            ag, al = avg_gain, avg_loss
        else:
            ag = (ag * (period - 1) + gains[i]) / period
            al = (al * (period - 1) + losses[i]) / period
        
        if al == 0:
            rsi[i] = 100.0
        else:
            rs = ag / al
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    
    return rsi


def compute_macd(
    close: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD = EMA(fast) - EMA(slow), Signal = EMA(MACD, signal).
    
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands = SMA ± num_std * rolling_std.
    
    Returns: (upper, middle, lower)
    """
    n = len(close)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        window = close[i - period + 1: i + 1]
        mu = np.mean(window)
        sigma = np.std(window, ddof=1)
        middle[i] = mu
        upper[i] = mu + num_std * sigma
        lower[i] = mu - num_std * sigma
    
    return upper, middle, lower


def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range — measures volatility.
    
    TR = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = Wilder's smoothed average of TR.
    """
    n = len(close)
    tr = np.full(n, np.nan, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    
    atr = np.full(n, np.nan, dtype=np.float64)
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def compute_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume — cumulative volume with sign determined by price change.
    
    OBV[t] = OBV[t-1] + sign(close[t] - close[t-1]) * volume[t]
    """
    obv = np.zeros_like(close, dtype=np.float64)
    
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    
    return obv


def compute_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Volume-Weighted Average Price (rolling cumulative intraday).
    
    VWAP = cumsum(typical_price * volume) / cumsum(volume)
    typical_price = (high + low + close) / 3
    """
    typical_price = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)
    
    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, np.nan)
    return vwap


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full feature set from OHLCV DataFrame.
    
    Input columns: Open, High, Low, Close, Volume
    Output: Original columns + 10 technical indicators = 15 features.
    
    All features are strictly causal (no look-ahead).
    """
    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    volume = df["Volume"].values.astype(np.float64)
    
    # Log returns
    df["Returns"] = compute_returns(close, period=1)
    
    # RSI
    df["RSI"] = compute_rsi(close, period=14)
    
    # MACD
    macd_line, signal_line, _ = compute_macd(close)
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal_line
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = compute_bollinger_bands(close, period=20)
    df["BB_Upper"] = bb_upper
    df["BB_Mid"] = bb_mid
    df["BB_Lower"] = bb_lower
    
    # ATR
    df["ATR"] = compute_atr(high, low, close, period=14)
    
    # OBV
    df["OBV"] = compute_obv(close, volume)
    
    # VWAP
    df["VWAP"] = compute_vwap(high, low, close, volume)
    
    return df


def normalize_features(
    df: pd.DataFrame,
    feature_cols: list,
    method: str = "zscore",
    window: int = 252,
) -> pd.DataFrame:
    """Rolling normalization to prevent look-ahead bias.
    
    At each point t, normalize using statistics from [t-window, t-1].
    This is critical: using full-sample statistics leaks future info.
    
    Methods:
        zscore: (x - rolling_mean) / rolling_std
        minmax: (x - rolling_min) / (rolling_max - rolling_min)
    """
    df_norm = df.copy()
    
    for col in feature_cols:
        series = df[col].values.astype(np.float64)
        normalized = np.full_like(series, np.nan)
        
        for t in range(window, len(series)):
            past = series[t - window: t]  # strictly past data
            
            if method == "zscore":
                mu = np.nanmean(past)
                sigma = np.nanstd(past)
                if sigma > 1e-8:
                    normalized[t] = (series[t] - mu) / sigma
                else:
                    normalized[t] = 0.0
            elif method == "minmax":
                mn = np.nanmin(past)
                mx = np.nanmax(past)
                rng = mx - mn
                if rng > 1e-8:
                    normalized[t] = (series[t] - mn) / rng
                else:
                    normalized[t] = 0.0
        
        df_norm[col] = normalized
    
    return df_norm


# ─── Internal helpers ──────────────────────────────────────────────

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average.
    
    EMA[t] = alpha * data[t] + (1 - alpha) * EMA[t-1]
    alpha = 2 / (period + 1)
    """
    alpha = 2.0 / (period + 1)
    ema = np.full_like(data, np.nan, dtype=np.float64)
    ema[period - 1] = np.mean(data[:period])
    
    for i in range(period, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    
    return ema
