"""
Data pipeline orchestrator.
============================

Coordinates the full data pipeline:
  Raw data → Feature engineering → Temporal alignment → Labels → Dataset

Single entry point for reproducible data preparation.
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

from finsent.data.price_loader import PriceDataLoader
from finsent.data.news_loader import NewsDataLoader, Vocabulary
from finsent.data.features import compute_all_features, normalize_features
from finsent.data.temporal_align import (
    validate_temporal_ordering,
    create_labels,
    compute_class_weights,
)
from finsent.data.dataset import FinSentDataset, create_dataloaders


class DataPipeline:
    """End-to-end data pipeline for FinSentNet.
    
    Usage:
        pipeline = DataPipeline("config/config.yaml")
        train_loader, val_loader, test_loader = pipeline.run()
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.price_loader = PriceDataLoader(
            cache_dir="data/raw",
            processed_dir="data/processed",
        )
        self.news_loader = NewsDataLoader(
            max_seq_length=self.config["data"]["max_seq_length"],
            news_lookback_hours=self.config["data"]["news_lookback_hours"],
            max_news_lag_hours=self.config["data"]["max_news_lag_hours"],
        )
    
    def run(
        self,
        ticker: str = "AAPL",
        use_synthetic_news: bool = True,
        news_csv_path: Optional[str] = None,
    ) -> Tuple:
        """Execute full pipeline.
        
        Steps:
        1. Fetch/load price data
        2. Compute technical indicators
        3. Create direction labels
        4. Load/generate news data
        5. Build vocabulary (TRAIN ONLY)
        6. Temporally align and encode news
        7. Walk-forward split
        8. Create PyTorch datasets and loaders
        
        Returns:
            (train_loader, val_loader, test_loader, metadata)
        """
        cfg = self.config["data"]
        
        # ─── Step 1: Price Data ───────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 1: Loading Price Data")
        print("=" * 60)
        
        raw_df = self.price_loader.fetch_yahoo(
            ticker=ticker,
            start=cfg["start_date"],
            end=cfg["end_date"],
        )
        
        # ─── Step 2: Feature Engineering ──────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 2: Computing Technical Indicators")
        print("=" * 60)
        
        featured_df = self.price_loader.prepare_features(raw_df, ticker=ticker)
        
        # ─── Step 3: Walk-Forward Split ───────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 3: Walk-Forward Temporal Split")
        print("=" * 60)
        
        train_df, val_df, test_df = self.price_loader.create_walk_forward_splits(
            featured_df,
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            test_ratio=cfg["test_ratio"],
        )
        
        # Validate temporal ordering
        validate_temporal_ordering(
            train_df.index, val_df.index, test_df.index
        )
        
        # ─── Step 4: Create Labels ────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 4: Creating Direction Labels")
        print("=" * 60)
        
        threshold = self.config["dual_head"]["neutral_threshold"]
        
        # Target prediction horizon
        target_horizon = self.config["dual_head"].get("target_horizon", "24h")
        
        # Create multi-horizon labels for each split independently
        horizons = ["1h", "4h", "24h", "5d"]
        train_labels_df, train_fwd_df = create_labels(train_df["Close"], horizons=horizons, neutral_threshold=threshold)
        val_labels_df, val_fwd_df = create_labels(val_df["Close"], horizons=horizons, neutral_threshold=threshold)
        test_labels_df, test_fwd_df = create_labels(test_df["Close"], horizons=horizons, neutral_threshold=threshold)
        
        # Extract the target series
        train_labels = train_labels_df[f"label_{target_horizon}"]
        train_fwd = train_fwd_df[f"ret_{target_horizon}"]
        val_labels = val_labels_df[f"label_{target_horizon}"]
        val_fwd = val_fwd_df[f"ret_{target_horizon}"]
        test_labels = test_labels_df[f"label_{target_horizon}"]
        test_fwd = test_fwd_df[f"ret_{target_horizon}"]
        
        # Class weights from training set only
        class_weights = compute_class_weights(train_labels)
        
        # ─── Step 5: News Data ────────────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 5: Processing News Data")
        print("=" * 60)
        
        if use_synthetic_news:
            news_df = self.news_loader.generate_synthetic_news(raw_df)
        elif news_csv_path:
            news_df = self.news_loader.load_csv(news_csv_path)
        else:
            news_df = None
        
        news_encoded_train = {}
        news_encoded_val = {}
        news_encoded_test = {}
        
        if news_df is not None:
            # Build vocabulary from TRAINING news only (prevent leakage)
            train_news_mask = news_df["datetime"] < val_df.index[0]
            train_texts = news_df.loc[train_news_mask, "text"].tolist()
            
            vocab = self.news_loader.build_vocabulary(
                train_texts,
                max_size=cfg["vocab_size"],
                min_freq=cfg["min_word_freq"],
            )
            
            # Align and encode for each split
            for split_df, split_name, encoded_dict in [
                (train_df, "train", news_encoded_train),
                (val_df, "val", news_encoded_val),
                (test_df, "test", news_encoded_test),
            ]:
                aligned = self.news_loader.align_news_to_prices(
                    news_df, split_df.index, ticker=None
                )
                for date, texts in aligned.items():
                    encoded_dict[date] = self.news_loader.encode_texts(texts)
                print(f"  [{split_name}] Encoded {len(encoded_dict)} date-text pairs")
        
        # ─── Step 6: Create Datasets ─────────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 6: Creating PyTorch Datasets")
        print("=" * 60)
        
        window = cfg["price_window"]
        
        train_dataset = FinSentDataset(
            train_df, train_labels, train_fwd,
            news_encoded=news_encoded_train, window_size=window,
        )
        val_dataset = FinSentDataset(
            val_df, val_labels, val_fwd,
            news_encoded=news_encoded_val, window_size=window,
        )
        test_dataset = FinSentDataset(
            test_df, test_labels, test_fwd,
            news_encoded=news_encoded_test, window_size=window,
        )
        
        # ─── Step 7: Create DataLoaders ───────────────────────────────
        print("\n" + "=" * 60)
        print("STEP 7: Creating DataLoaders")
        print("=" * 60)
        
        batch_size = self.config["training"]["batch_size"]
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset,
            batch_size=batch_size,
            use_temporal_sampling=self.config["training"]["temporal_stratified_sampling"],
        )
        
        # Metadata
        metadata = {
            "ticker": ticker,
            "class_weights": class_weights,
            "vocab_size": vocab.size if news_df is not None else 0,
            "n_features": cfg["price_features"],
            "window_size": window,
            "train_dates": (train_df.index[0], train_df.index[-1]),
            "val_dates": (val_df.index[0], val_df.index[-1]),
            "test_dates": (test_df.index[0], test_df.index[-1]),
        }
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        
        return train_loader, val_loader, test_loader, metadata
