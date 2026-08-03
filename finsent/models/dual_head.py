"""
Dual-Head Output Architecture.
===============================

Two prediction heads from the fused representation:

Head 1 — Direction Prediction:
    Predicts {Down, Neutral, Up} with class probabilities.
    Uses focal loss to handle class imbalance.
    
Head 2 — Calibrated Confidence:
    Outputs a confidence score calibrated via temperature scaling.
    This is NOT just softmax probability — it's a separate learned
    estimate of prediction reliability.
    
    Used for position sizing: higher confidence → larger position.
    This implements a form of the Kelly Criterion where the model
    allocates capital proportional to its calibrated edge.

Mathematical Basis:
    Direction: p(y|x) = softmax(W_d · f + b_d)  
    Confidence: c(x) = σ(W_c · f / τ + b_c)  where τ is learned temperature
    
    Temperature scaling (Guo et al., 2017):
        Standard softmax outputs are typically overconfident.
        Dividing logits by temperature τ > 1 produces better-calibrated
        probabilities, where P(correct | confidence = q) ≈ q.
        
    This calibration is critical for profitable trading:
        An overconfident model takes oversized positions on uncertain bets,
        leading to catastrophic drawdowns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DirectionHead(nn.Module):
    """Direction prediction head.
    
    Predicts market direction: Down (0), Neutral (1), Up (2).
    
    Architecture:
        Fused repr → Dense → GELU → Dropout → Dense → Logits
    
    Label smoothing is applied during training (in loss function)
    to prevent overconfident predictions.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        n_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (batch, input_dim) — fused cross-modal representation
        Returns:
            logits: (batch, n_classes) — raw logits (pre-softmax)
        """
        return self.head(fused)


class ConfidenceHead(nn.Module):
    """Calibrated confidence prediction head.
    
    Outputs a scalar confidence score in [0, 1].
    
    Temperature scaling:
        Instead of using raw softmax probability as confidence
        (which is poorly calibrated), we learn a separate confidence
        estimate with a learnable temperature parameter.
        
        confidence = σ(logit / τ)
        
        where τ is initialized > 1 to start conservative.
    
    Financial use:
        confidence * kelly_fraction → position_size_pct
        
        High confidence (>0.7): Full Kelly allocation
        Medium confidence (0.4-0.7): Reduced allocation  
        Low confidence (<0.4): Skip or minimal position
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        temperature_init: float = 1.5,
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # single confidence logit
        )
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(
            torch.tensor(temperature_init, dtype=torch.float32)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (batch, input_dim)
        Returns:
            confidence: (batch, 1) — calibrated confidence in [0, 1]
        """
        logit = self.head(fused)
        # Temperature-scaled sigmoid
        confidence = torch.sigmoid(logit / self.temperature.clamp(min=0.1))
        return confidence


class DualHead(nn.Module):
    """Combined dual-head output module.
    
    Integrates direction prediction and confidence estimation
    into a single module with shared fused representation input.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        n_classes: int = 3,
        dropout: float = 0.2,
        temperature_init: float = 1.5,
    ):
        super().__init__()
        
        self.direction_head = DirectionHead(
            input_dim=input_dim,
            hidden_dim=256,
            n_classes=n_classes,
            dropout=dropout,
        )
        
        self.confidence_head = ConfidenceHead(
            input_dim=input_dim,
            hidden_dim=128,
            temperature_init=temperature_init,
        )
    
    def forward(
        self,
        fused: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused: (batch, input_dim) — fused cross-modal representation
        
        Returns:
            dict with:
                direction_logits: (batch, n_classes) — raw logits
                direction_probs: (batch, n_classes) — softmax probabilities
                confidence: (batch, 1) — calibrated confidence
                temperature: scalar — current temperature value
        """
        direction_logits = self.direction_head(fused)
        direction_probs = F.softmax(direction_logits, dim=-1)
        confidence = self.confidence_head(fused)
        
        return {
            "direction_logits": direction_logits,
            "direction_probs": direction_probs,
            "confidence": confidence,
            "temperature": self.confidence_head.temperature,
        }
