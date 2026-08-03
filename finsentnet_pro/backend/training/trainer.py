"""
FINSENT NET PRO — Model Trainer
Per-company training pipeline with checkpoint management.

Features:
  - Fetches historical data → computes indicators → creates labeled dataset
  - Full PyTorch training loop with AdamW + cosine annealing
  - Direction (CrossEntropy) + Return (Huber) joint loss
  - Gradient clipping, early stopping, best-model checkpointing
  - Per-company model saving / loading for incremental learning
  - Real-time progress tracking via callback
"""

import os
import time
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from datetime import datetime

from .dataset import StockTradingDataset

logger = logging.getLogger("finsent.training")

# ── Default directories ──────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


class TrainingProgress:
    """Tracks training progress for API reporting."""

    def __init__(self, ticker: str, total_epochs: int):
        self.ticker = ticker
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.val_accuracy = 0.0
        self.best_val_loss = float("inf")
        self.status = "initializing"  # initializing | fetching_data | training | completed | failed
        self.message = ""
        self.started_at = datetime.utcnow().isoformat()
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "status": self.status,
            "message": self.message,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "train_loss": round(self.train_loss, 6),
            "val_loss": round(self.val_loss, 6),
            "val_accuracy": round(self.val_accuracy, 2),
            "best_val_loss": round(self.best_val_loss, 6) if self.best_val_loss < float("inf") else None,
            "progress_pct": round(self.current_epoch / max(self.total_epochs, 1) * 100, 1),
            "started_at": self.started_at,
            "history": self.history,
        }


class ModelTrainer:
    """
    Per-company FINSENT model trainer.

    Usage:
        trainer = ModelTrainer(model, fetcher, indicators)
        progress = trainer.train("AAPL", "SP500", epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        fetcher,
        indicators,
        device: str = "cpu",
        checkpoint_dir: str = CHECKPOINT_DIR,
    ):
        self.model = model
        self.fetcher = fetcher
        self.indicators = indicators
        self.device = torch.device(device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Move model to device
        self.model.to(self.device)

        # Active training progress trackers {ticker: TrainingProgress}
        self.active_training: Dict[str, TrainingProgress] = {}

        # Cache of currently loaded checkpoint to avoid repeated disk reloads.
        self._loaded_ticker: Optional[str] = None
        self._loaded_ckpt_path: Optional[str] = None
        self._loaded_ckpt_mtime: Optional[float] = None

    # ═══════════════════════════════════════════════════════
    #  PUBLIC API
    # ═══════════════════════════════════════════════════════

    def train(
        self,
        ticker: str,
        market: str = "SP500",
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        window_size: int = 30,
        horizon: int = 5,
        val_ratio: float = 0.2,
        patience: int = 10,
        period: str = "5y",
        progress_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Full training pipeline for a single company.

        1. Fetch 5y OHLCV → compute indicators
        2. Create labeled sliding-window dataset
        3. Train model with early stopping
        4. Save best checkpoint
        5. Return training report

        Args:
            ticker:   Stock ticker symbol
            market:   Market identifier (SP500, NSE, etc.)
            epochs:   Maximum training epochs
            period:   Data fetch period ('2y', '5y', etc.)

        Returns:
            dict with training metrics and status
        """
        progress = TrainingProgress(ticker, epochs)
        self.active_training[ticker] = progress

        try:
            # ── Step 1: Fetch & prepare data ──
            progress.status = "fetching_data"
            progress.message = f"Fetching {period} historical data for {ticker}..."
            logger.info(progress.message)

            df = self.fetcher.fetch_ohlcv(ticker, market=market, period=period)
            if df.empty or len(df) < window_size + horizon + 50:
                raise ValueError(
                    f"Insufficient data for {ticker}: got {len(df)} rows, "
                    f"need at least {window_size + horizon + 50}"
                )

            df = self.indicators.compute_all(df)
            progress.message = f"Data ready: {len(df)} trading days with 35+ indicators"
            logger.info(progress.message)

            # ── Step 2: Create dataset ──
            progress.status = "training"
            progress.message = "Creating training dataset..."

            full_dataset = StockTradingDataset(
                df,
                window_size=window_size,
                horizon=horizon,
                up_thresh=0.01,
                down_thresh=-0.01,
            )

            if len(full_dataset) < 50:
                raise ValueError(
                    f"Dataset too small for {ticker}: {len(full_dataset)} samples"
                )

            train_ds, val_ds = StockTradingDataset.temporal_train_val_split(
                full_dataset, val_ratio=val_ratio
            )

            logger.info(
                f"Dataset: {len(train_ds)} train / {len(val_ds)} val | "
                f"Distribution: {full_dataset.class_distribution}"
            )

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=False,
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=False,
            )

            # ── Step 3: Load existing checkpoint if available ──
            existing_ckpt = self._load_checkpoint(ticker)
            if existing_ckpt:
                progress.message = f"Loaded existing checkpoint — continuing training"
                logger.info(progress.message)

            # ── Step 4: Training loop ──
            report = self._training_loop(
                train_loader=train_loader,
                val_loader=val_loader,
                ticker=ticker,
                epochs=epochs,
                lr=learning_rate,
                weight_decay=weight_decay,
                patience=patience,
                progress=progress,
                progress_callback=progress_callback,
            )

            # ── Step 5: Save final checkpoint ──
            self._save_checkpoint(ticker, market, report)

            progress.status = "completed"
            progress.message = (
                f"Training complete! "
                f"Val Accuracy: {report['best_val_accuracy']:.1f}% | "
                f"Val Loss: {report['best_val_loss']:.4f}"
            )
            logger.info(progress.message)

            return {
                "status": "success",
                "ticker": ticker,
                "market": market,
                **report,
                "checkpoint_path": self._checkpoint_path(ticker),
                "dataset_info": {
                    "total_samples": len(full_dataset),
                    "train_samples": len(train_ds),
                    "val_samples": len(val_ds),
                    "class_distribution": full_dataset.class_distribution,
                    "data_period": period,
                    "window_size": window_size,
                    "horizon": horizon,
                },
            }

        except Exception as e:
            progress.status = "failed"
            progress.message = f"Training failed: {str(e)}"
            logger.error(progress.message, exc_info=True)
            return {"status": "failed", "ticker": ticker, "error": str(e)}

    def get_progress(self, ticker: str) -> Optional[Dict]:
        """Get current training progress for a ticker."""
        if ticker in self.active_training:
            return self.active_training[ticker].to_dict()
        return None

    def is_model_trained(self, ticker: str) -> bool:
        """Check if a trained checkpoint exists for this ticker."""
        return os.path.exists(self._checkpoint_path(ticker))

    def get_checkpoint_mtime(self, ticker: str) -> Optional[float]:
        """Return checkpoint mtime for cache invalidation, if checkpoint exists."""
        ckpt_path = self._checkpoint_path(ticker)
        if not os.path.exists(ckpt_path):
            return None
        try:
            return os.path.getmtime(ckpt_path)
        except Exception:
            return None

    def load_trained_model(self, ticker: str, force: bool = False) -> bool:
        """Load a trained model for inference, skipping redundant reloads when possible."""
        ckpt_path = self._checkpoint_path(ticker)
        if not os.path.exists(ckpt_path):
            return False

        current_mtime = self.get_checkpoint_mtime(ticker)
        if (
            not force
            and self._loaded_ticker == ticker
            and self._loaded_ckpt_path == ckpt_path
            and current_mtime is not None
            and self._loaded_ckpt_mtime is not None
            and abs(self._loaded_ckpt_mtime - current_mtime) < 1e-9
        ):
            return True

        return self._load_checkpoint(ticker) is not None

    # ═══════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ═══════════════════════════════════════════════════════

    def _training_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        ticker: str,
        epochs: int,
        lr: float,
        weight_decay: float,
        patience: int,
        progress: TrainingProgress,
        progress_callback: Optional[Callable],
    ) -> Dict:
        """Core training loop with joint direction + return loss."""

        self.model.train()
        self.model.to(self.device)

        # ── Optimizer & Scheduler ──
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(epochs // 3, 5), T_mult=2
        )

        # ── Loss functions ──
        # Class weights to handle imbalance (UP / NEUTRAL / DOWN)
        direction_loss_fn = nn.CrossEntropyLoss()
        return_loss_fn = nn.HuberLoss(delta=0.05)  # Huber is more robust than MSE

        # ── Training state ──
        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        train_start = time.time()

        for epoch in range(1, epochs + 1):
            # ──── TRAIN ────
            self.model.train()
            epoch_train_loss = 0.0
            n_train_batches = 0

            for batch in train_loader:
                price_seq = batch["price_sequence"].to(self.device)
                text_tok = batch["text_tokens"].to(self.device)
                dir_label = batch["direction_label"].to(self.device)
                ret_label = batch["return_label"].to(self.device)

                optimizer.zero_grad()

                output = self.model(text_tok, price_seq)

                # Joint loss: direction classification + return regression
                dir_loss = direction_loss_fn(output["direction_logits"], dir_label)
                ret_loss = return_loss_fn(
                    output["return_pred"].squeeze(-1), ret_label
                )
                loss = dir_loss + 0.5 * ret_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_train_loss += loss.item()
                n_train_batches += 1

            avg_train_loss = epoch_train_loss / max(n_train_batches, 1)

            # ──── VALIDATE ────
            val_loss, val_acc, val_dir_acc = self._validate(
                val_loader, direction_loss_fn, return_loss_fn
            )

            scheduler.step()

            # ──── Progress update ────
            progress.current_epoch = epoch
            progress.train_loss = avg_train_loss
            progress.val_loss = val_loss
            progress.val_accuracy = val_dir_acc
            progress.history["train_loss"].append(round(avg_train_loss, 6))
            progress.history["val_loss"].append(round(val_loss, 6))
            progress.history["val_acc"].append(round(val_dir_acc, 2))
            max_history_points = 160
            if len(progress.history["train_loss"]) > max_history_points:
                progress.history["train_loss"] = progress.history["train_loss"][-max_history_points:]
                progress.history["val_loss"] = progress.history["val_loss"][-max_history_points:]
                progress.history["val_acc"] = progress.history["val_acc"][-max_history_points:]
            progress.message = (
                f"Epoch {epoch}/{epochs} — "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_dir_acc:.1f}%"
            )

            if progress_callback:
                progress_callback(progress.to_dict())

            logger.info(
                f"[{ticker}] Epoch {epoch:3d}/{epochs} | "
                f"TrLoss={avg_train_loss:.4f} | ValLoss={val_loss:.4f} | "
                f"ValAcc={val_dir_acc:.1f}% | LR={scheduler.get_last_lr()[0]:.2e}"
            )

            # ──── Early stopping ────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_dir_acc
                patience_counter = 0
                # Save best model state
                self._save_best_state(ticker)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        f"[{ticker}] Early stopping at epoch {epoch} "
                        f"(patience={patience})"
                    )
                    break

        # Restore best model
        self._restore_best_state(ticker)

        training_time = time.time() - train_start
        return {
            "epochs_trained": epoch,
            "best_val_loss": round(best_val_loss, 6),
            "best_val_accuracy": round(best_val_acc, 2),
            "final_train_loss": round(avg_train_loss, 6),
            "training_time_seconds": round(training_time, 1),
            "early_stopped": patience_counter >= patience,
        }

    @torch.no_grad()
    def _validate(self, val_loader, direction_loss_fn, return_loss_fn):
        """Run validation pass. Returns (total_loss, return_mae, direction_accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            price_seq = batch["price_sequence"].to(self.device)
            text_tok = batch["text_tokens"].to(self.device)
            dir_label = batch["direction_label"].to(self.device)
            ret_label = batch["return_label"].to(self.device)

            output = self.model(text_tok, price_seq)

            dir_loss = direction_loss_fn(output["direction_logits"], dir_label)
            ret_loss = return_loss_fn(
                output["return_pred"].squeeze(-1), ret_label
            )
            loss = dir_loss + 0.5 * ret_loss
            total_loss += loss.item()

            # Accuracy
            preds = output["direction_logits"].argmax(dim=1)
            correct += (preds == dir_label).sum().item()
            total += dir_label.size(0)

        n_batches = max(len(val_loader), 1)
        avg_loss = total_loss / n_batches
        accuracy = (correct / max(total, 1)) * 100

        self.model.train()
        return avg_loss, 0.0, accuracy

    # ═══════════════════════════════════════════════════════
    #  CHECKPOINT MANAGEMENT
    # ═══════════════════════════════════════════════════════

    def _checkpoint_path(self, ticker: str) -> str:
        safe_ticker = ticker.replace("/", "_").replace("\\", "_")
        return os.path.join(self.checkpoint_dir, f"finsent_{safe_ticker}.pt")

    def _meta_path(self, ticker: str) -> str:
        safe_ticker = ticker.replace("/", "_").replace("\\", "_")
        return os.path.join(self.checkpoint_dir, f"finsent_{safe_ticker}_meta.json")

    def _best_state_path(self, ticker: str) -> str:
        safe_ticker = ticker.replace("/", "_").replace("\\", "_")
        return os.path.join(self.checkpoint_dir, f"finsent_{safe_ticker}_best.pt")

    def _save_best_state(self, ticker: str):
        """Save current model state as the best so far."""
        torch.save(self.model.state_dict(), self._best_state_path(ticker))

    def _restore_best_state(self, ticker: str):
        """Restore the best model state."""
        path = self._best_state_path(ticker)
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"[{ticker}] Restored best model state")

    def _save_checkpoint(self, ticker: str, market: str, report: Dict):
        """Save model checkpoint + training metadata."""
        ckpt_path = self._checkpoint_path(ticker)
        meta_path = self._meta_path(ticker)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "ticker": ticker,
                "market": market,
                "trained_at": datetime.utcnow().isoformat(),
            },
            ckpt_path,
        )

        meta = {
            "ticker": ticker,
            "market": market,
            "trained_at": datetime.utcnow().isoformat(),
            **report,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        try:
            self._loaded_ticker = ticker
            self._loaded_ckpt_path = ckpt_path
            self._loaded_ckpt_mtime = os.path.getmtime(ckpt_path)
        except Exception:
            self._loaded_ticker = ticker
            self._loaded_ckpt_path = ckpt_path
            self._loaded_ckpt_mtime = None

        logger.info(f"[{ticker}] Checkpoint saved to {ckpt_path}")

    def _load_checkpoint(self, ticker: str) -> Optional[Dict]:
        """Load model checkpoint if it exists."""
        ckpt_path = self._checkpoint_path(ticker)
        if not os.path.exists(ckpt_path):
            return None

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self._loaded_ticker = ticker
            self._loaded_ckpt_path = ckpt_path
            self._loaded_ckpt_mtime = self.get_checkpoint_mtime(ticker)
            logger.info(f"[{ticker}] Loaded checkpoint from {ckpt_path}")
            return ckpt
        except Exception as e:
            logger.warning(f"[{ticker}] Failed to load checkpoint: {e}")
            return None

    def get_checkpoint_meta(self, ticker: str) -> Optional[Dict]:
        """Get training metadata for a checkpoint."""
        meta_path = self._meta_path(ticker)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f)
        return None
