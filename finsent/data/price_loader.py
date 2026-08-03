"""
Price data loader — fetches and caches OHLCV data.
==================================================

Supports:
  - Yahoo Finance (via yfinance) for daily/intraday
  - CSV file loading for custom datasets
  - Proper temporal alignment and split logic
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

from finsent.data.features import compute_all_features, normalize_features


class PriceDataLoader:
    """Load, cache, and preprocess OHLCV price data.
    
    Design principles:
    1. Data is cached locally to avoid repeated API calls.
    2. Features are computed AFTER loading raw data (separation of concerns).
    3. Normalization uses rolling windows (no look-ahead).
    4. Walk-forward splits ensure temporal ordering.
    """
    
    REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
    FEATURE_COLS = [
        "Open", "High", "Low", "Close", "Volume",
        "Returns", "RSI", "MACD", "MACD_Signal",
        "BB_Upper", "BB_Mid", "BB_Lower", "ATR", "OBV", "VWAP",
    ]
    
    def __init__(
        self,
        cache_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ):
        self.cache_dir = Path(cache_dir)
        self.processed_dir = Path(processed_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_yahoo(
        self,
        ticker: str,
        start: str = "2010-01-01",
        end: str = "2024-12-31",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Yahoo Finance with local caching.
        
        Args:
            ticker: Stock symbol (e.g., "AAPL").
            start/end: Date range strings.
            interval: "1d", "1h", "5m", etc.
            force_refresh: Re-download even if cached.
        """
        if not HAS_YFINANCE:
            raise ImportError("yfinance required. Install: pip install yfinance")
        
        cache_file = self.cache_dir / f"{ticker}_{interval}_{start}_{end}.parquet"
        
        if cache_file.exists() and not force_refresh:
            df = pd.read_parquet(cache_file)
            print(f"[PriceLoader] Loaded cached {ticker}: {len(df)} bars")
            return df
        
        print(f"[PriceLoader] Downloading {ticker} ({start} → {end}, {interval})...")
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        # Standardize columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        
        # Remove zero-volume days (holidays/errors)
        df = df[df["Volume"] > 0]
        
        # Cache
        df.to_parquet(cache_file)
        print(f"[PriceLoader] Cached {ticker}: {len(df)} bars → {cache_file}")
        
        return df
    
    def load_csv(self, filepath: str, date_col: str = "Date") -> pd.DataFrame:
        """Load OHLCV from CSV file.
        
        Expects columns: Date, Open, High, Low, Close, Volume.
        """
        df = pd.read_csv(filepath, parse_dates=[date_col])
        df = df.set_index(date_col).sort_index()
        
        missing = set(self.REQUIRED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        df = df[self.REQUIRED_COLS]
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        ticker: str = "unknown",
        norm_window: int = 252,
    ) -> pd.DataFrame:
        """Compute technical indicators and normalize.
        
        Pipeline:
        1. Compute raw technical indicators (all causal)
        2. Rolling z-score normalization (no look-ahead)
        3. Drop NaN warmup period
        
        Args:
            df: Raw OHLCV DataFrame.
            ticker: For logging/caching.
            norm_window: Rolling normalization lookback (252 = 1 year).
        """
        # Step 1: Technical indicators
        df = compute_all_features(df)
        
        # Step 2: Rolling normalization
        df = normalize_features(
            df,
            feature_cols=self.FEATURE_COLS,
            method="zscore",
            window=norm_window,
        )
        
        # Step 3: Drop NaN warmup rows
        warmup = norm_window + 30  # extra buffer for indicator warmup
        df = df.iloc[warmup:].copy()
        
        # Verify no NaNs remain
        nan_count = df[self.FEATURE_COLS].isna().sum().sum()
        if nan_count > 0:
            print(f"[WARNING] {ticker}: {nan_count} NaN values remaining, forward-filling")
            df[self.FEATURE_COLS] = df[self.FEATURE_COLS].ffill().bfill()
        
        # Save processed
        out_path = self.processed_dir / f"{ticker}_features.parquet"
        df.to_parquet(out_path)
        print(f"[PriceLoader] Processed {ticker}: {len(df)} samples, {len(self.FEATURE_COLS)} features")
        
        return df
    
    def fetch_multiple(
        self,
        tickers: List[str],
        start: str = "2010-01-01",
        end: str = "2024-12-31",
        **kwargs,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch and process multiple tickers."""
        data = {}
        for ticker in tickers:
            try:
                raw = self.fetch_yahoo(ticker, start, end, **kwargs)
                processed = self.prepare_features(raw, ticker=ticker)
                data[ticker] = processed
            except Exception as e:
                print(f"[ERROR] Failed to process {ticker}: {e}")
        return data
    
    def create_walk_forward_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Temporal walk-forward split (no shuffle — prevents look-ahead).
        
        CRITICAL: Financial data MUST be split chronologically.
        Random shuffling would allow the model to "see" future data.
        
        Returns: (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()
        
        print(f"[Split] Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} samples)")
        print(f"[Split] Val:   {val.index[0].date()} → {val.index[-1].date()} ({len(val)} samples)")
        print(f"[Split] Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} samples)")
        
        # Sanity check: no temporal overlap
        assert train.index[-1] < val.index[0], "Train/val temporal overlap detected!"
        assert val.index[-1] < test.index[0], "Val/test temporal overlap detected!"
        
        return train, val, test
