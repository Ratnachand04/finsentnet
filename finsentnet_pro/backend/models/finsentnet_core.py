"""
FINSENT NET PRO — Complete PyTorch Architecture
Based on FINSENT Research Blueprint: Modules 1-9

Architecture:
  Text Branch:  FinBERT embeddings → TextCNN → 3-layer BiLSTM → Multi-Head Self-Attention
  Price Branch: Multi-scale Conv1D → Dilated CNN → LSTM Encoder
  Fusion:       Cross-Modal Attention (Sentiment Query × Price KV)
  Output:       Dual-Head (Direction Classifier + Return Regressor)

Each component is also available as a standalone module:
  - text_branch.py      (TextCNN, MultiHeadSelfAttention, TextBranch)
  - price_branch.py     (PriceBranch)
  - cross_modal_fusion.py (CrossModalAttentionFusion)
  - dual_head_output.py (DualHeadOutput)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

# Re-export individual components for convenience
from .text_branch import TextCNN, MultiHeadSelfAttention, TextBranch
from .price_branch import PriceBranch
from .cross_modal_fusion import CrossModalAttentionFusion
from .dual_head_output import DualHeadOutput


# ═══════════════════════════════════════════════════════════
#  FULL MODEL
# ═══════════════════════════════════════════════════════════


class FinSentNetCore(nn.Module):
    """
    Complete FINSENT Architecture.

    Pipeline:
        Text Input  → TextBranch  → 512-d sentiment vector
        Price Input → PriceBranch → 512-d price context + sequence
        Cross-Modal Attention Fusion → 512-d fused vector
        Dual-Head Output:
            Head A: FC → Softmax → P(UP), P(NEUTRAL), P(DOWN)
            Head B: FC → Linear  → predicted log return %
    """

    def __init__(
        self,
        price_feature_dim: int = 20,
        d_model: int = 512,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Dual encoder branches
        self.text_branch = TextBranch(output_dim=d_model, dropout=dropout)
        self.price_branch = PriceBranch(
            input_dim=price_feature_dim,
            output_dim=d_model,
            dropout=dropout,
        )

        # Cross-modal attention fusion (THE CORE INNOVATION)
        self.fusion = CrossModalAttentionFusion(d_model=d_model, dropout=dropout)

        # Dual-head output
        self.dual_head = DualHeadOutput(d_model=d_model, num_classes=num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier / Glorot initialization for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        text_tokens: torch.Tensor,
        price_sequence: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_tokens:   (batch, seq_len)       — tokenized news text
            price_sequence:(batch, 30, features)   — OHLCV + indicators
            text_mask:     (batch, seq_len)        — optional padding mask

        Returns dict with:
            'direction_logits': (batch, 3)
            'direction_probs':  (batch, 3)   — calibrated probabilities
            'return_pred':      (batch, 1)   — predicted return magnitude
            'cross_attention':  (batch, seq_len) — interpretable weights
            'text_attention':   attention from self-attention layer
            'sentiment_vec':    (batch, d_model)
            'fused_vec':        (batch, d_model)
        """
        # Encode text
        sentiment_vec, text_attn = self.text_branch(text_tokens, text_mask)

        # Encode price
        price_ctx, price_seq = self.price_branch(price_sequence)

        # Cross-modal fusion
        fused_vec, cross_attn = self.fusion(sentiment_vec, price_seq)

        # Dual-head predictions
        head_out = self.dual_head(fused_vec)

        return {
            "direction_logits": head_out["direction_logits"],
            "direction_probs": head_out["direction_probs"],
            "return_pred": head_out["return_pred"],
            "cross_attention": cross_attn,
            "text_attention": text_attn,
            "sentiment_vec": sentiment_vec,
            "fused_vec": fused_vec,
        }
