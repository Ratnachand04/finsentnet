"""
Custom loss functions for FinSentNet.
======================================

Implements:
  - Focal Loss: Handles class imbalance without oversampling
  - Confidence Calibration Loss: Ensures calibrated probabilities
  - Combined Multi-Task Loss: Weighted combination of direction + confidence
  
Financial Motivation:
    Standard cross-entropy treats all misclassifications equally.
    In trading, misclassifying a strong up move as down is far more costly
    than misclassifying a slight up as neutral.
    
    Focal loss down-weights easy examples (clear trends) and up-weights
    hard examples (regime transitions, inflection points). This forces
    the model to focus on the samples that matter most for profitability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al., 2017) — adapted for financial prediction.
    
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    
    where:
        p_t = model's predicted probability for the true class
        α_t = class weight (handles prior class imbalance)
        γ  = focusing parameter (γ=0 → standard CE, γ=2 recommended)
    
    The (1 - p_t)^γ term is the "focusing" factor:
        - When p_t → 1 (easy, correct): factor → 0, loss suppressed
        - When p_t → 0 (hard, wrong): factor → 1, loss amplified
    
    This is superior to class-weighted CE because it's adaptive
    per-sample, not just per-class.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # (n_classes,) class weights
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,   # (batch, n_classes)
        targets: torch.Tensor,  # (batch,) integer class labels
    ) -> torch.Tensor:
        """Compute focal loss."""
        n_classes = logits.size(-1)
        
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get probability of true class
        # p_t = probs[targets] for each sample
        targets_one_hot = F.one_hot(targets, n_classes).float()
        
        # Label smoothing: soften targets
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / n_classes
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + smooth
        
        # p_t for each class
        p_t = (probs * targets_one_hot).sum(dim=-1)  # (batch,)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Cross-entropy (numerically stable)
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none",
            label_smoothing=self.label_smoothing,
        )
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ConfidenceCalibrationLoss(nn.Module):
    """Loss for training calibrated confidence predictions.
    
    A well-calibrated model satisfies:
        P(correct | confidence = q) = q  for all q ∈ [0, 1]
    
    We train this by using the model's actual correctness as supervision:
        target_confidence = 1.0 if predicted class == true class, else 0.0
    
    Loss = BCE(predicted_confidence, target_confidence)
    
    Additionally, we add an Expected Calibration Error (ECE) penalty
    that bins predictions and penalizes deviations from perfect calibration.
    
    Financial importance:
        An overconfident model bets too large on uncertain predictions.
        A well-calibrated model correctly sizes positions proportional
        to its actual edge, maximizing risk-adjusted returns.
    """
    
    def __init__(self, n_bins: int = 15, ece_weight: float = 1.0):
        super().__init__()
        self.n_bins = n_bins
        self.ece_weight = ece_weight
        self.bce = nn.BCELoss(reduction="mean")
    
    def forward(
        self,
        confidence: torch.Tensor,      # (batch, 1) — predicted confidence
        direction_logits: torch.Tensor, # (batch, n_classes)
        targets: torch.Tensor,          # (batch,) — true labels
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
            confidence_loss: total calibration loss
            bce_loss: binary cross-entropy component
            ece: Expected Calibration Error (for monitoring)
        """
        confidence = confidence.squeeze(-1)  # (batch,)
        
        # Is the direction prediction correct?
        predicted_class = direction_logits.argmax(dim=-1)
        is_correct = (predicted_class == targets).float()
        
        # BCE: confidence should predict correctness
        bce_loss = self.bce(confidence, is_correct)
        
        # ECE: binned calibration error
        ece = self._compute_ece(confidence, is_correct)
        
        total = bce_loss + self.ece_weight * ece
        
        return {
            "confidence_loss": total,
            "bce_loss": bce_loss,
            "ece": ece,
        }
    
    def _compute_ece(
        self,
        confidence: torch.Tensor,
        correctness: torch.Tensor,
    ) -> torch.Tensor:
        """Expected Calibration Error.
        
        ECE = Σ_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
        
        where B_b is the set of predictions in bin b,
        acc(B_b) is average accuracy in the bin,
        conf(B_b) is average confidence in the bin.
        """
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=confidence.device)
        ece = torch.zeros(1, device=confidence.device)
        n = confidence.size(0)
        
        for i in range(self.n_bins):
            mask = (confidence > bin_boundaries[i]) & (confidence <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = correctness[mask].mean()
                bin_confidence = confidence[mask].mean()
                bin_weight = mask.float().sum() / n
                ece += bin_weight * torch.abs(bin_accuracy - bin_confidence)
        
        return ece.squeeze()


class FinSentLoss(nn.Module):
    """Combined multi-task loss for FinSentNet.
    
    Total Loss = w_d * FocalLoss(direction) + w_c * CalibrationLoss(confidence)
    
    The direction loss drives prediction accuracy.
    The confidence loss ensures position sizing is appropriate.
    
    Both are necessary: a model with good direction accuracy but
    poor calibration will size positions wrong and underperform
    a model with lower accuracy but better-calibrated confidence.
    """
    
    def __init__(
        self,
        direction_weight: float = 0.7,
        confidence_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: Optional[list] = None,
        label_smoothing: float = 0.05,
        ece_weight: float = 1.0,
    ):
        super().__init__()
        
        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight
        
        alpha_tensor = torch.tensor(focal_alpha, dtype=torch.float32) if focal_alpha else None
        
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            alpha=alpha_tensor,
            label_smoothing=label_smoothing,
        )
        
        self.calibration_loss = ConfidenceCalibrationLoss(
            ece_weight=ece_weight,
        )
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model output dict containing direction_logits, confidence
            targets: (batch,) true direction labels
        
        Returns:
            dict with total_loss and component losses for logging
        """
        # Direction loss
        dir_loss = self.focal_loss(outputs["direction_logits"], targets)
        
        # Confidence calibration loss
        cal_results = self.calibration_loss(
            outputs["confidence"],
            outputs["direction_logits"],
            targets,
        )
        
        # Weighted total
        total_loss = (
            self.direction_weight * dir_loss +
            self.confidence_weight * cal_results["confidence_loss"]
        )
        
        return {
            "total_loss": total_loss,
            "direction_loss": dir_loss,
            "confidence_loss": cal_results["confidence_loss"],
            "bce_loss": cal_results["bce_loss"],
            "ece": cal_results["ece"],
        }
