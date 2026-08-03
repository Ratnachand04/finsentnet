"""
Training loop for FinSentNet.
===============================

Production-grade training loop with:
  - Mixed precision support (FP16)
  - Gradient clipping
  - Early stopping on financial metrics (Sharpe, not loss)
  - Checkpoint saving/loading
  - Comprehensive metric logging
  - Validation Sharpe computation
  
Design principle: Monitor FINANCIAL metrics for model selection.
A model with lower loss but worse Sharpe is NOT a better model.
"""

import os
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict

from finsent.models.finsent_net import FinSentNet
from finsent.training.losses import FinSentLoss
from finsent.training.schedulers import CosineWarmupScheduler
from finsent.training.calibration import TemperatureScaler
from finsent.utils.logging_utils import setup_logger


class EarlyStopping:
    """Early stopping based on validation financial metric.
    
    Monitors validation Sharpe ratio (or other metric).
    Stops if no improvement for `patience` epochs.
    Saves best model checkpoint.
    """
    
    def __init__(
        self,
        patience: int = 15,
        mode: str = "max",
        delta: float = 0.0,
        checkpoint_path: str = "checkpoints/best_model.pt",
    ):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
        
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    def step(
        self,
        score: float,
        model: nn.Module,
        epoch: int,
    ) -> bool:
        """Check if training should stop.
        
        Returns: True if should stop, False otherwise.
        """
        improved = (
            score > self.best_score + self.delta if self.mode == "max"
            else score < self.best_score - self.delta
        )
        
        if improved:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class Trainer:
    """FinSentNet training manager.
    
    Handles the complete training lifecycle:
    1. Model initialization
    2. Training loop with validation
    3. Metric computation (accuracy, Sharpe, IC)
    4. Early stopping on financial metrics
    5. Post-training temperature calibration
    6. Best model restoration
    """
    
    def __init__(
        self,
        model: FinSentNet,
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = setup_logger("trainer")
        
        # Training config
        tcfg = config["training"]
        
        # Loss function
        self.criterion = FinSentLoss(
            direction_weight=tcfg["direction_loss_weight"],
            confidence_weight=tcfg["confidence_loss_weight"],
            focal_gamma=tcfg["focal_loss_gamma"],
            focal_alpha=tcfg["focal_loss_alpha"],
            label_smoothing=tcfg["label_smoothing"],
        )
        
        # Optimizer
        if tcfg["optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=tcfg["learning_rate"],
                weight_decay=tcfg["weight_decay"],
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=tcfg["learning_rate"],
                weight_decay=tcfg["weight_decay"],
            )
        
        # Scheduler
        if tcfg["scheduler"] == "cosine_warmup":
            self.scheduler = CosineWarmupScheduler(
                self.optimizer,
                warmup_epochs=tcfg["warmup_epochs"],
                total_epochs=tcfg["epochs"],
            )
        else:
            self.scheduler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=tcfg["patience"],
            mode="max",  # maximize Sharpe
        )
        
        # Gradient clipping
        self.grad_clip = tcfg["gradient_clip_norm"]
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        self.use_amp = device.type == "cuda"
        
        # History
        self.history = defaultdict(list)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns dict of average metrics for the epoch.
        """
        self.model.train()
        
        epoch_losses = defaultdict(list)
        correct = 0
        total = 0
        
        for batch in train_loader:
            price = batch["price"].to(self.device)
            text_ids = batch["text_ids"].to(self.device)
            text_mask = batch["text_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(price, text_ids, text_mask)
                    loss_dict = self.criterion(outputs, labels)
                
                self.scaler.scale(loss_dict["total_loss"]).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(price, text_ids, text_mask)
                loss_dict = self.criterion(outputs, labels)
                
                loss_dict["total_loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            # Track metrics
            for k, v in loss_dict.items():
                epoch_losses[k].append(v.item())
            
            predicted = outputs["direction_logits"].argmax(dim=-1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        metrics = {k: np.mean(v) for k, v in epoch_losses.items()}
        metrics["accuracy"] = correct / max(total, 1)
        
        return metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Validate and compute financial metrics.
        
        Computes:
        - Classification metrics (accuracy, per-class accuracy)
        - Financial metrics (Sharpe ratio, Information Coefficient)
        """
        self.model.eval()
        
        epoch_losses = defaultdict(list)
        all_predictions = []
        all_labels = []
        all_confidences = []
        all_forward_returns = []
        
        for batch in val_loader:
            price = batch["price"].to(self.device)
            text_ids = batch["text_ids"].to(self.device)
            text_mask = batch["text_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            fwd_ret = batch["forward_return"]
            
            outputs = self.model(price, text_ids, text_mask)
            loss_dict = self.criterion(outputs, labels)
            
            for k, v in loss_dict.items():
                epoch_losses[k].append(v.item())
            
            predicted = outputs["direction_logits"].argmax(dim=-1)
            confidence = outputs["confidence"].squeeze(-1)
            
            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())
            all_forward_returns.append(fwd_ret.numpy())
        
        # Aggregate
        preds = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        confs = np.concatenate(all_confidences)
        fwd_rets = np.concatenate(all_forward_returns)
        
        # Classification metrics
        accuracy = np.mean(preds == labels)
        
        # Per-class accuracy
        for cls, name in [(0, "down"), (1, "neutral"), (2, "up")]:
            mask = labels == cls
            if mask.sum() > 0:
                epoch_losses[f"acc_{name}"] = [(preds[mask] == cls).mean()]
        
        # ─── Financial Metrics ────────────────────────────────────
        
        # Strategy returns: go long if predict Up, short if Down, flat if Neutral
        strategy_returns = np.zeros_like(fwd_rets)
        strategy_returns[preds == 2] = fwd_rets[preds == 2]    # Long
        strategy_returns[preds == 0] = -fwd_rets[preds == 0]   # Short
        # Neutral: 0 return (no position)
        
        # Scale by confidence for position sizing
        confidence_scaled_returns = strategy_returns * confs
        
        # Sharpe Ratio (annualized)
        if len(strategy_returns) > 1 and np.std(strategy_returns) > 1e-8:
            sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Confidence-scaled Sharpe
        if len(confidence_scaled_returns) > 1 and np.std(confidence_scaled_returns) > 1e-8:
            sharpe_conf = (np.mean(confidence_scaled_returns) /
                          np.std(confidence_scaled_returns) * np.sqrt(252))
        else:
            sharpe_conf = 0.0
        
        # Information Coefficient (rank correlation between prediction and return)
        from scipy.stats import spearmanr
        # Convert predictions to numeric score: Down=-1, Neutral=0, Up=1
        pred_scores = preds.astype(float) - 1.0
        valid_mask = ~np.isnan(fwd_rets)
        if valid_mask.sum() > 10:
            ic, ic_pvalue = spearmanr(pred_scores[valid_mask], fwd_rets[valid_mask])
        else:
            ic, ic_pvalue = 0.0, 1.0
        
        # Compile metrics
        metrics = {k: np.mean(v) for k, v in epoch_losses.items()}
        metrics["accuracy"] = accuracy
        metrics["sharpe"] = sharpe
        metrics["sharpe_conf_scaled"] = sharpe_conf
        metrics["information_coefficient"] = ic
        metrics["ic_pvalue"] = ic_pvalue
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ) -> Dict[str, list]:
        """Full training loop.
        
        Returns training history dict.
        """
        if epochs is None:
            epochs = self.config["training"]["epochs"]
        
        self.logger.info("=" * 60)
        self.logger.info("Starting FinSentNet Training")
        self.logger.info(f"  Epochs: {epochs}")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info("=" * 60)
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # ─── Train ────────────────────────────────────────────
            train_metrics = self.train_epoch(train_loader)
            
            # ─── Validate ─────────────────────────────────────────
            val_metrics = self.validate(val_loader)
            
            # ─── Scheduler Step ───────────────────────────────────
            if self.scheduler is not None:
                self.scheduler.step()
            
            # ─── Log ──────────────────────────────────────────────
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]
            
            self.logger.info(
                f"Epoch {epoch:03d}/{epochs} ({epoch_time:.1f}s) | "
                f"LR: {lr:.2e} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.3f} | "
                f"Val Sharpe: {val_metrics['sharpe']:.3f} | "
                f"Val IC: {val_metrics['information_coefficient']:.4f}"
            )
            
            # Store history
            for k, v in train_metrics.items():
                self.history[f"train_{k}"].append(v)
            for k, v in val_metrics.items():
                self.history[f"val_{k}"].append(v)
            self.history["lr"].append(lr)
            
            # ─── Early Stopping ───────────────────────────────────
            monitor_metric = val_metrics.get(
                self.config["training"]["monitor"].replace("val_", ""),
                val_metrics["sharpe"],
            )
            
            if self.early_stopping.step(monitor_metric, self.model, epoch):
                self.logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch: {self.early_stopping.best_epoch} "
                    f"(Sharpe: {self.early_stopping.best_score:.4f})"
                )
                break
        
        # Restore best model
        self.model.load_state_dict(
            torch.load(self.early_stopping.checkpoint_path, map_location=self.device)
        )
        self.logger.info(f"Restored best model from epoch {self.early_stopping.best_epoch}")
        
        return dict(self.history)
    
    def post_training_calibrate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """Post-training temperature calibration on validation set."""
        self.logger.info("Running post-training temperature calibration...")
        
        scaler = TemperatureScaler()
        results = scaler.calibrate(
            self.model, val_loader, self.device
        )
        
        # Update model's confidence head temperature
        with torch.no_grad():
            self.model.dual_head.confidence_head.temperature.copy_(
                scaler.temperature
            )
        
        self.logger.info(f"Calibration complete. New temperature: {results['optimal_temperature']:.3f}")
        return results
