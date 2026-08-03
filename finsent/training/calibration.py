"""
Temperature scaling for post-hoc calibration.
===============================================

After training, calibrate the model's confidence outputs using
a simple temperature parameter optimized on the validation set.

This is a post-training step that dramatically improves calibration
without changing the model's discriminative performance.

Reference: Guo et al., "On Calibration of Modern Neural Networks", 2017

Financial impact:
    Before calibration: model says 80% confident → actually correct 60% of time
    After calibration:  model says 80% confident → actually correct ~80% of time
    
    This translates directly to better position sizing:
        - Kelly criterion requires accurate P(win) estimates
        - Overconfident ≈ overbetting ≈ ruin risk
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import DataLoader


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for model calibration.
    
    Calibrated logits = raw logits / T
    
    where T is optimized on the validation set via NLL minimization.
    
    This is a single-parameter optimization that preserves
    the model's classification accuracy while improving
    probability calibration.
    """
    
    def __init__(self, init_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.tensor(init_temperature, dtype=torch.float32)
        )
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature.
        
        Args:
            logits: (batch, n_classes) raw model logits
        Returns:
            calibrated_logits: (batch, n_classes)
        """
        return logits / self.temperature.clamp(min=0.01)
    
    def calibrate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 200,
    ) -> Dict[str, float]:
        """Optimize temperature on validation set.
        
        Uses LBFGS optimizer (second-order, fast convergence for 1 parameter).
        Minimizes NLL on validation set.
        
        Returns:
            dict with optimal temperature and calibration metrics
        """
        model.eval()
        self.to(device)
        
        # Collect all logits and labels from validation set
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                price = batch["price"].to(device)
                text_ids = batch["text_ids"].to(device)
                text_mask = batch["text_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(price, text_ids, text_mask)
                all_logits.append(outputs["direction_logits"])
                all_labels.append(labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Optimize temperature
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits)
            loss = nll_criterion(scaled_logits, all_labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Compute calibration metrics
        with torch.no_grad():
            scaled_logits = self.forward(all_logits)
            probs = F.softmax(scaled_logits, dim=-1)
            predicted = probs.argmax(dim=-1)
            accuracy = (predicted == all_labels).float().mean().item()
            
            ece_before = self._compute_ece(
                F.softmax(all_logits, dim=-1), all_labels
            )
            ece_after = self._compute_ece(probs, all_labels)
        
        results = {
            "optimal_temperature": self.temperature.item(),
            "accuracy": accuracy,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "ece_reduction": (ece_before - ece_after) / ece_before * 100,
        }
        
        print(f"[Calibration] Temperature: {results['optimal_temperature']:.3f}")
        print(f"[Calibration] ECE: {ece_before:.4f} → {ece_after:.4f} "
              f"({results['ece_reduction']:.1f}% reduction)")
        
        return results
    
    @staticmethod
    def _compute_ece(
        probs: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 15,
    ) -> float:
        """Expected Calibration Error."""
        confidences, predictions = probs.max(dim=-1)
        correct = predictions.eq(labels).float()
        
        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = correct[mask].mean()
                bin_confidence = confidences[mask].mean()
                bin_weight = mask.float().sum() / len(labels)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece.item()
