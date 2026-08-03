"""
PyTorch Dataset and DataLoader for FinSentNet.
===============================================

Implements:
  - FinSentDataset: Multi-modal dataset (text + price → labels)
  - Temporal stratified sampling for class balance without shuffling
  - Collate functions for variable-length news
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Optional, Dict, List, Tuple


class FinSentDataset(Dataset):
    """Multi-modal financial dataset combining text and price features.
    
    Each sample contains:
        - price_window: (window_size, n_features) tensor of normalized price features
        - text_ids: (max_seq_length,) tensor of tokenized news text
        - text_mask: (max_seq_length,) attention mask (1 = real token, 0 = padding)
        - label: direction class (0=Down, 1=Neutral, 2=Up)
        - forward_return: actual forward return (for backtesting)
        - date: timestamp (for temporal tracking)
    
    Design:
        - Price window is a sliding lookback window at each timestep
        - News is aggregated over configurable lookback period
        - Labels are STRICTLY forward-looking (prediction target)
    """
    
    def __init__(
        self,
        price_df: pd.DataFrame,
        labels: pd.Series,
        forward_returns: pd.Series,
        news_encoded: Optional[Dict[pd.Timestamp, np.ndarray]] = None,
        feature_cols: Optional[List[str]] = None,
        window_size: int = 30,
        max_seq_length: int = 512,
    ):
        """
        Args:
            price_df: Processed price DataFrame with features.
            labels: Direction labels indexed by date.
            forward_returns: Actual forward returns for evaluation.
            news_encoded: dict[date] → encoded text array.
            feature_cols: Column names to use as price features.
            window_size: Number of past days in price window.
        """
        # Filter to valid labels only
        valid_mask = labels >= 0
        self.dates = price_df.index[valid_mask].tolist()
        self.labels = labels[valid_mask].values
        self.forward_returns = forward_returns[valid_mask].values
        
        # Price features matrix
        if feature_cols is None:
            feature_cols = [
                "Open", "High", "Low", "Close", "Volume",
                "Returns", "RSI", "MACD", "MACD_Signal",
                "BB_Upper", "BB_Mid", "BB_Lower", "ATR", "OBV", "VWAP",
            ]
        
        self.price_matrix = price_df[feature_cols].values.astype(np.float32)
        self.all_dates = price_df.index.tolist()
        self.window_size = window_size
        self.max_seq_length = max_seq_length
        self.news_encoded = news_encoded or {}
        
        # Build index mapping: valid sample index → position in price_matrix
        self._indices = []
        for date in self.dates:
            pos = self.all_dates.index(date)
            if pos >= window_size:
                self._indices.append(pos)
        
        # Trim labels/returns to match valid indices
        n_valid = len(self._indices)
        self.labels = self.labels[-n_valid:]
        self.forward_returns = self.forward_returns[-n_valid:]
        self.dates = self.dates[-n_valid:]
        
        print(f"[Dataset] {len(self)} samples, window={window_size}, "
              f"features={len(feature_cols)}")
    
    def __len__(self) -> int:
        return len(self._indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pos = self._indices[idx]
        date = self.dates[idx]
        
        # Price window: (window_size, n_features) — strictly past data
        price_window = self.price_matrix[pos - self.window_size: pos]
        price_tensor = torch.from_numpy(price_window).float()
        
        # Text encoding
        if date in self.news_encoded:
            text_ids = self.news_encoded[date]
        else:
            text_ids = np.zeros(self.max_seq_length, dtype=np.int64)
        
        text_tensor = torch.from_numpy(text_ids).long()
        text_mask = (text_tensor != 0).float()  # 1 where not padding
        
        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        fwd_ret = torch.tensor(self.forward_returns[idx], dtype=torch.float32)
        
        return {
            "price": price_tensor,        # (window_size, n_features)
            "text_ids": text_tensor,       # (max_seq_length,)
            "text_mask": text_mask,        # (max_seq_length,)
            "label": label,               # scalar
            "forward_return": fwd_ret,     # scalar
            "date_idx": torch.tensor(idx, dtype=torch.long),
        }


class TemporalStratifiedSampler(Sampler):
    """Sampler that respects temporal ordering while balancing classes.
    
    Problem: Random shuffling breaks temporal dependencies.
    But class imbalance hurts training.
    
    Solution: Divide timeline into temporal blocks. Within each block,
    oversample minority classes. Blocks are served in chronological order.
    
    This is a compromise:
    - Temporal structure is preserved at the block level
    - Class balance is improved within blocks
    - No look-ahead: each block only contains past data relative to future blocks
    """
    
    def __init__(
        self,
        labels: np.ndarray,
        block_size: int = 252,  # ~1 year of trading days
        shuffle_within_blocks: bool = True,
    ):
        self.labels = labels
        self.block_size = block_size
        self.shuffle_within_blocks = shuffle_within_blocks
        self.n_samples = len(labels)
    
    def __iter__(self):
        indices = []
        
        for start in range(0, self.n_samples, self.block_size):
            end = min(start + self.block_size, self.n_samples)
            block_indices = list(range(start, end))
            block_labels = self.labels[start:end]
            
            if self.shuffle_within_blocks:
                # Oversample minority within block
                unique, counts = np.unique(block_labels, return_counts=True)
                max_count = counts.max()
                
                balanced_indices = []
                for cls in unique:
                    cls_indices = [i for i in block_indices if self.labels[i] == cls]
                    # Oversample to match majority class
                    if len(cls_indices) < max_count:
                        extra = np.random.choice(cls_indices,
                                                  size=max_count - len(cls_indices),
                                                  replace=True).tolist()
                        cls_indices = cls_indices + extra
                    balanced_indices.extend(cls_indices)
                
                np.random.shuffle(balanced_indices)
                indices.extend(balanced_indices)
            else:
                indices.extend(block_indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        # Approximate: oversampling may increase count
        return self.n_samples


def create_dataloaders(
    train_dataset: FinSentDataset,
    val_dataset: FinSentDataset,
    test_dataset: FinSentDataset,
    batch_size: int = 64,
    num_workers: int = 0,
    use_temporal_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders with proper sampling strategies.
    
    Training: Temporal stratified sampling
    Validation/Test: Sequential (preserves temporal order for evaluation)
    """
    if use_temporal_sampling:
        train_sampler = TemporalStratifiedSampler(
            labels=train_dataset.labels,
            block_size=252,
            shuffle_within_blocks=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # NEVER shuffle financial time series
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"[DataLoaders] Train: {len(train_loader)} batches, "
          f"Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader
