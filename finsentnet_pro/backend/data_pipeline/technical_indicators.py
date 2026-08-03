"""
FINSENT NET PRO — Technical Indicators Engine
All indicators computed from raw OHLCV. No look-ahead contamination.
Vectorized NumPy/Pandas operations throughout.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class TechnicalIndicators:
    """
    Complete technical analysis toolkit.
    All methods are strictly causal — use only past data.
    """

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the full indicator suite used by FINSENT price branch.
        Input: DataFrame with [Open, High, Low, Close, Volume]
        Output: DataFrame with ~35+ feature columns appended
        """
        result = df.copy()
        TI = TechnicalIndicators

        # --- Momentum ---
        result["RSI_14"] = TI.rsi(df["Close"], 14)
        result["RSI_21"] = TI.rsi(df["Close"], 21)

        # --- Trend ---
        macd_line, signal_line, histogram = TI.macd(df["Close"])
        result["MACD"] = macd_line
        result["MACD_Signal"] = signal_line
        result["MACD_Hist"] = histogram

        # --- Volatility ---
        bb_upper, bb_mid, bb_lower = TI.bollinger_bands(df["Close"])
        result["BB_Upper"] = bb_upper
        result["BB_Mid"] = bb_mid
        result["BB_Lower"] = bb_lower
        result["BB_Width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
        result["BB_Position"] = (df["Close"] - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
        result["ATR_14"] = TI.atr(df["High"], df["Low"], df["Close"], 14)

        # --- Volume ---
        result["OBV"] = TI.obv(df["Close"], df["Volume"])
        result["VWAP"] = TI.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
        result["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
        result["Volume_Ratio"] = df["Volume"] / result["Volume_SMA_20"].replace(0, np.nan)

        # --- Moving Averages ---
        for period in [5, 10, 20, 50, 200]:
            result[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()
            result[f"SMA_{period}"] = df["Close"].rolling(period).mean()

        # --- Price Position Relative to MAs ---
        result["Price_vs_SMA50"] = (df["Close"] - result["SMA_50"]) / result["SMA_50"].replace(0, np.nan)
        result["Price_vs_SMA200"] = (df["Close"] - result["SMA_200"]) / result["SMA_200"].replace(0, np.nan)

        # --- Cross Signals ---
        result["GoldenCross"] = (result["SMA_50"] > result["SMA_200"]).astype(int)

        # --- Candlestick Patterns ---
        result["Doji"] = TI.doji_pattern(df)
        result["Hammer"] = TI.hammer_pattern(df)
        result["Engulfing"] = TI.engulfing_pattern(df)

        # --- Returns & Volatility ---
        result["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        result["Volatility_20"] = result["Log_Return"].rolling(20).std() * np.sqrt(252)

        # --- Stochastic Oscillator ---
        result["Stoch_K"], result["Stoch_D"] = TI.stochastic(
            df["High"], df["Low"], df["Close"]
        )

        return result.dropna()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index — Wilder's smoothing."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD with signal line and histogram."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        close: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return upper, mid, lower

    @staticmethod
    def atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Average True Range — Wilder's smoothing."""
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.where(close.diff() >= 0, 1, -1)
        return (direction * volume).cumsum()

    @staticmethod
    def vwap(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum().replace(0, np.nan)

    @staticmethod
    def stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series,
        k_period: int = 14, d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator (%K, %D)."""
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def doji_pattern(df: pd.DataFrame) -> pd.Series:
        """Doji: Open ≈ Close (body < 10% of range)."""
        body = (df["Close"] - df["Open"]).abs()
        range_ = df["High"] - df["Low"] + 1e-10
        return (body / range_ < 0.1).astype(int)

    @staticmethod
    def hammer_pattern(df: pd.DataFrame) -> pd.Series:
        """Hammer: Small body at top, long lower shadow."""
        body = (df["Close"] - df["Open"]).abs()
        lower_shadow = df[["Open", "Close"]].min(axis=1) - df["Low"]
        upper_shadow = df["High"] - df[["Open", "Close"]].max(axis=1)
        return ((lower_shadow >= 2 * body) & (upper_shadow < body)).astype(int)

    @staticmethod
    def engulfing_pattern(df: pd.DataFrame) -> pd.Series:
        """Bullish Engulfing: current candle body engulfs previous."""
        prev_open = df["Open"].shift(1)
        prev_close = df["Close"].shift(1)
        bullish = (
            (df["Close"] > df["Open"])
            & (prev_close < prev_open)
            & (df["Close"] > prev_open)
            & (df["Open"] < prev_close)
        )
        return bullish.astype(int)
