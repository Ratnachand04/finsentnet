"""
FINSENT NET PRO — Market Regime Detector
Identifies current market regime: BULL, BEAR, VOLATILE, TRANSITIONAL.
Uses rolling statistical features and optional HMM.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from enum import Enum


class MarketRegime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    VOLATILE = "VOLATILE"
    TRANSITIONAL = "TRANSITIONAL"


class RegimeDetector:
    """
    Rule-based market regime detector.
    Uses trend, volatility, and momentum features.
    """

    def __init__(
        self,
        vol_lookback: int = 20,
        trend_lookback: int = 50,
        vol_high_threshold: float = 0.25,
    ):
        self.vol_lookback = vol_lookback
        self.trend_lookback = trend_lookback
        self.vol_high_threshold = vol_high_threshold

    def detect(self, df: pd.DataFrame) -> Dict:
        """
        Detect current market regime from OHLCV + indicators.

        Returns dict with regime, confidence, and supporting metrics.
        """
        close = df["Close"]
        n = len(close)

        if n < self.trend_lookback + 10:
            return self._default_result()

        # --- Trend Analysis ---
        sma50 = close.rolling(50).mean().iloc[-1]
        sma200 = close.rolling(200).mean().iloc[-1] if n >= 210 else close.rolling(min(n - 1, 100)).mean().iloc[-1]
        current_price = close.iloc[-1]

        trend_above_50 = current_price > sma50
        trend_above_200 = current_price > sma200
        sma50_above_200 = sma50 > sma200

        # --- Volatility ---
        log_returns = np.log(close / close.shift(1)).dropna()
        ann_vol = log_returns.iloc[-self.vol_lookback:].std() * np.sqrt(252)
        vol_zscore = (ann_vol - log_returns.rolling(60).std().iloc[-1] * np.sqrt(252)) / (log_returns.rolling(60).std().iloc[-1] * np.sqrt(252) + 1e-8)

        # --- Momentum ---
        returns_20d = (close.iloc[-1] / close.iloc[-21] - 1) if n > 21 else 0
        returns_60d = (close.iloc[-1] / close.iloc[-61] - 1) if n > 61 else 0

        # --- Classify Regime ---
        if ann_vol > self.vol_high_threshold:
            regime = MarketRegime.VOLATILE
            confidence = min(0.5 + abs(vol_zscore) * 0.15, 0.95)
            detail = f"High Volatility ({ann_vol:.1%} annualized)"
        elif trend_above_50 and sma50_above_200 and returns_20d > 0:
            regime = MarketRegime.BULL
            if ann_vol < 0.15:
                detail = "Low Volatility"
            else:
                detail = "Normal"
            confidence = 0.6 + min(returns_20d * 5, 0.3)
        elif not trend_above_50 and not sma50_above_200 and returns_20d < 0:
            regime = MarketRegime.BEAR
            detail = "Confirmed Downtrend"
            confidence = 0.6 + min(abs(returns_20d) * 5, 0.3)
        else:
            regime = MarketRegime.TRANSITIONAL
            detail = "Mixed Signals"
            confidence = 0.4

        return {
            "regime": regime.value,
            "detail": f"{regime.value} — {detail}",
            "confidence": round(float(confidence), 3),
            "metrics": {
                "annualized_volatility": round(float(ann_vol), 4),
                "volatility_zscore": round(float(vol_zscore), 2),
                "return_20d": round(float(returns_20d) * 100, 2),
                "return_60d": round(float(returns_60d) * 100, 2),
                "price_vs_sma50": round(float((current_price / sma50 - 1) * 100), 2),
                "golden_cross": bool(sma50_above_200),
            },
        }

    def _default_result(self) -> Dict:
        return {
            "regime": "UNKNOWN",
            "detail": "UNKNOWN — Insufficient data",
            "confidence": 0.0,
            "metrics": {},
        }
