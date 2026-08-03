"""
FINSENT NET PRO — Dual-Head Output
Head A: Direction Classifier  →  P(UP), P(NEUTRAL), P(DOWN)
Head B: Return Regressor      →  predicted log return %
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class DualHeadOutput(nn.Module):
    """
    Two-head prediction module sitting on top of the fused representation.

    Head A — Direction Classifier
        FC(d_model → 256) → GELU → Dropout(0.4) → FC(256 → 3)
        Softmax at inference → P(UP), P(NEUTRAL), P(DOWN)

    Head B — Return Magnitude Regressor
        FC(d_model → 256) → GELU → Dropout(0.3) → FC(256 → 1)
        Linear output → predicted return as a fraction (e.g. 0.08 = +8 %)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_classes: int = 3,
        dropout_cls: float = 0.4,
        dropout_reg: float = 0.3,
    ):
        super().__init__()

        # Head A: Direction Classifier (UP / NEUTRAL / DOWN)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout_cls),
            nn.Linear(256, num_classes),
        )

        # Head B: Return Magnitude Regressor
        self.return_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout_reg),
            nn.Linear(256, 1),
        )

    def forward(self, fused_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused_vec: (batch, d_model) — output of CrossModalAttentionFusion
        Returns:
            dict with direction_logits, direction_probs, return_pred
        """
        direction_logits = self.direction_head(fused_vec)
        return_pred = self.return_head(fused_vec)

        return {
            "direction_logits": direction_logits,
            "direction_probs": F.softmax(direction_logits, dim=-1),
            "return_pred": return_pred,
        }
