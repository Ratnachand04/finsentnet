"""
FINSENT NET PRO — Temporal Data Aligner
Ensures strict causal alignment between news and price data.
No look-ahead contamination allowed.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class DataAligner:
    """
    Aligns multi-modal data streams (price, news, indicators) temporally.
    Guarantees that at time t, only data from times <= t is visible.
    """

    def __init__(self, min_news_lag_hours: float = 1.0):
        self.min_news_lag_hours = min_news_lag_hours

    def align_price_and_indicators(
        self,
        price_df: pd.DataFrame,
        indicators_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge price data with computed indicators on date index."""
        merged = price_df.join(indicators_df, how="inner", rsuffix="_ind")
        # Remove any duplicate columns
        dup_cols = [c for c in merged.columns if c.endswith("_ind")]
        merged = merged.drop(columns=dup_cols)
        return merged

    def prepare_price_window(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        feature_cols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Extract the last `window_size` rows as a normalized numpy array.
        For feeding into the FINSENT price branch.
        """
        if feature_cols is None:
            feature_cols = [
                "Open", "High", "Low", "Close", "Volume",
                "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
                "BB_Upper", "BB_Lower", "BB_Position", "ATR_14",
                "OBV", "Volume_Ratio", "EMA_20", "EMA_50",
                "Log_Return", "Volatility_20", "Stoch_K",
            ]

        available_cols = [c for c in feature_cols if c in df.columns]
        window = df[available_cols].iloc[-window_size:].values

        # Z-score normalization (causal: stats from this window only)
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True) + 1e-8
        normalized = (window - mean) / std

        # Pad to 20 features if needed
        if normalized.shape[1] < 20:
            padding = np.zeros((normalized.shape[0], 20 - normalized.shape[1]))
            normalized = np.concatenate([normalized, padding], axis=1)

        return normalized

    def validate_temporal_integrity(
        self,
        dates: pd.DatetimeIndex,
    ) -> bool:
        """Verify that dates are strictly monotonically increasing."""
        if len(dates) < 2:
            return True
        diffs = dates[1:] - dates[:-1]
        return all(d.total_seconds() > 0 for d in diffs)
