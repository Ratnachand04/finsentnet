"""
Learning rate schedulers for FinSentNet.
=========================================

Implements cosine annealing with warmup — the standard choice for
training deep networks on financial data.

Financial motivation:
    Financial data is non-stationary. A learning rate that's too high
    causes the model to overfit to recent regime characteristics.
    Warmup prevents early training instability on normalized financial features.
    Cosine annealing provides a smooth decay that helps the model
    converge to a broader minimum (better generalization).
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """Cosine annealing with linear warmup.
    
    Learning rate schedule:
        Phase 1 (warmup): η = η_max * t / T_warmup  (linear increase)
        Phase 2 (cosine):  η = η_min + (η_max - η_min) * 0.5 * (1 + cos(π * t' / T_cosine))
        
    where t' = (t - T_warmup) / (T_total - T_warmup)
    
    This is the default scheduler in many successful financial ML papers.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        epoch = self.last_epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class ReduceOnPlateauWithWarmup:
    """Reduce on plateau scheduler with initial warmup phase.
    
    Uses validation Sharpe ratio (not loss) as the monitoring metric.
    If Sharpe doesn't improve for `patience` epochs, reduce LR.
    
    This is financially motivated: we care about risk-adjusted returns,
    not reconstruction loss.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        factor: float = 0.5,
        patience: int = 10,
        min_lr: float = 1e-7,
        mode: str = "max",  # "max" for Sharpe (higher is better)
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.wait = 0
        self.epoch = 0
    
    def step(self, metric: float = None) -> None:
        self.epoch += 1
        
        if self.epoch <= self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.epoch / self.warmup_epochs
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg["lr"] = base_lr * warmup_factor
            return
        
        if metric is None:
            return
        
        # Check improvement
        improved = (
            metric > self.best_metric if self.mode == "max"
            else metric < self.best_metric
        )
        
        if improved:
            self.best_metric = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                for pg in self.optimizer.param_groups:
                    new_lr = max(pg["lr"] * self.factor, self.min_lr)
                    pg["lr"] = new_lr
                self.wait = 0
                print(f"[Scheduler] Reducing LR to {new_lr:.2e}")
    
    def get_lr(self) -> list:
        return [pg["lr"] for pg in self.optimizer.param_groups]
