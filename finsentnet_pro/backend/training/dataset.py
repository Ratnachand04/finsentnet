"""
FINSENT NET PRO — Training Dataset
Sliding-window PyTorch dataset for supervised training on stock data.

Label creation:
  - Direction: UP (0) if forward_return > +threshold
               NEUTRAL (1) if within threshold
               DOWN (2) if forward_return < -threshold
  - Return:    Actual forward return as regression target

Temporal integrity: strict causal windows — no look-ahead.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════
#  Feature columns used by price branch
# ═══════════════════════════════════════════════════════════
PRICE_FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Upper", "BB_Lower", "BB_Position", "ATR_14",
    "OBV", "Volume_Ratio", "EMA_20", "EMA_50",
    "Log_Return", "Volatility_20", "Stoch_K",
]

NUM_PRICE_FEATURES = 20  # Pad / trim to this


class StockTradingDataset(Dataset):
    """
    Sliding-window dataset for FINSENT model training.

    Each sample:
      - price_sequence: (window_size, NUM_PRICE_FEATURES) — z-score normalized
      - text_tokens:    (text_seq_len,) — synthetic token IDs for training
      - direction_label: int in {0=UP, 1=NEUTRAL, 2=DOWN}
      - return_label:    float — forward return magnitude

    Args:
        df:           DataFrame with OHLCV + technical indicators (index = dates)
        window_size:  Number of trading days per input window (default 30)
        horizon:      Forward-looking period in days (default 5)
        up_thresh:    Return threshold for UP classification (default 0.01 = 1%)
        down_thresh:  Return threshold for DOWN classification (default -0.01)
        text_seq_len: Length of synthetic text token sequence (default 50)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        horizon: int = 5,
        up_thresh: float = 0.01,
        down_thresh: float = -0.01,
        text_seq_len: int = 50,
    ):
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon
        self.text_seq_len = text_seq_len

        # ── Build feature matrix ──
        available_cols = [c for c in PRICE_FEATURE_COLS if c in df.columns]
        price_data = df[available_cols].values.astype(np.float32)
        closes = df["Close"].values.astype(np.float64)

        # ── Build samples ──
        self.price_windows: List[np.ndarray] = []
        self.text_tokens_list: List[np.ndarray] = []
        self.direction_labels: List[int] = []
        self.return_labels: List[float] = []

        for i in range(window_size, len(df) - horizon):
            # --- Price window ---
            window = price_data[i - window_size: i].copy()

            # Causal z-score normalization (stats from this window only)
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True) + 1e-8
            window_norm = (window - mean) / std

            # Pad or trim to NUM_PRICE_FEATURES
            n_feats = window_norm.shape[1]
            if n_feats < NUM_PRICE_FEATURES:
                pad = np.zeros((window_size, NUM_PRICE_FEATURES - n_feats), dtype=np.float32)
                window_norm = np.concatenate([window_norm, pad], axis=1)
            elif n_feats > NUM_PRICE_FEATURES:
                window_norm = window_norm[:, :NUM_PRICE_FEATURES]

            # --- Forward return label ---
            future_close = closes[i + horizon]
            current_close = closes[i]
            forward_return = (future_close / current_close) - 1.0

            if forward_return > up_thresh:
                direction = 0  # UP
            elif forward_return < down_thresh:
                direction = 2  # DOWN
            else:
                direction = 1  # NEUTRAL

            # --- Synthetic text tokens ---
            # Use a seeded RNG based on date index for reproducibility
            rng = np.random.RandomState(seed=(abs(hash(str(i))) % 2**31))
            text_tokens = rng.randint(0, 30000, size=(text_seq_len,))

            self.price_windows.append(window_norm)
            self.text_tokens_list.append(text_tokens)
            self.direction_labels.append(direction)
            self.return_labels.append(float(forward_return))

        # Convert to numpy arrays for faster indexing
        if len(self.price_windows) > 0:
            self._price_np = np.stack(self.price_windows)
            self._text_np = np.stack(self.text_tokens_list)
            self._dir_np = np.array(self.direction_labels, dtype=np.int64)
            self._ret_np = np.array(self.return_labels, dtype=np.float32)
        else:
            self._price_np = np.empty((0, window_size, NUM_PRICE_FEATURES))
            self._text_np = np.empty((0, text_seq_len), dtype=np.int64)
            self._dir_np = np.empty(0, dtype=np.int64)
            self._ret_np = np.empty(0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._dir_np)

    def __getitem__(self, idx: int) -> dict:
        return {
            "price_sequence": torch.from_numpy(self._price_np[idx]),
            "text_tokens": torch.from_numpy(self._text_np[idx]),
            "direction_label": torch.tensor(self._dir_np[idx], dtype=torch.long),
            "return_label": torch.tensor(self._ret_np[idx], dtype=torch.float32),
        }

    @property
    def class_distribution(self) -> dict:
        """Return label distribution for monitoring class imbalance."""
        unique, counts = np.unique(self._dir_np, return_counts=True)
        labels = {0: "UP", 1: "NEUTRAL", 2: "DOWN"}
        return {labels.get(int(u), str(u)): int(c) for u, c in zip(unique, counts)}

    @staticmethod
    def temporal_train_val_split(
        dataset: "StockTradingDataset",
        val_ratio: float = 0.2,
    ) -> Tuple["StockTradingDataset", "StockTradingDataset"]:
        """
        Time-series split: first (1-val_ratio) for train, last val_ratio for val.
        NO shuffling — preserves temporal order.
        """
        n = len(dataset)
        split_idx = int(n * (1 - val_ratio))

        train_ds = StockTradingDataset.__new__(StockTradingDataset)
        train_ds.window_size = dataset.window_size
        train_ds.horizon = dataset.horizon
        train_ds.text_seq_len = dataset.text_seq_len
        train_ds._price_np = dataset._price_np[:split_idx]
        train_ds._text_np = dataset._text_np[:split_idx]
        train_ds._dir_np = dataset._dir_np[:split_idx]
        train_ds._ret_np = dataset._ret_np[:split_idx]
        train_ds.price_windows = []
        train_ds.text_tokens_list = []
        train_ds.direction_labels = []
        train_ds.return_labels = []

        val_ds = StockTradingDataset.__new__(StockTradingDataset)
        val_ds.window_size = dataset.window_size
        val_ds.horizon = dataset.horizon
        val_ds.text_seq_len = dataset.text_seq_len
        val_ds._price_np = dataset._price_np[split_idx:]
        val_ds._text_np = dataset._text_np[split_idx:]
        val_ds._dir_np = dataset._dir_np[split_idx:]
        val_ds._ret_np = dataset._ret_np[split_idx:]
        val_ds.price_windows = []
        val_ds.text_tokens_list = []
        val_ds.direction_labels = []
        val_ds.return_labels = []

        return train_ds, val_ds
