"""
FINSENT NET PRO — Cross-Modal Attention Fusion
THE CORE INNOVATION OF FINSENT.

Cross-modal attention where:
  Query (Q)  = Sentiment vector from text branch
               "What is the market feeling right now?"
  Key   (K)  = Price feature sequence from price branch
               "What patterns exist in the price history?"
  Value (V)  = Price feature sequence
               "What information should we retrieve?"

The model learns WHEN sentiment is more predictive (earnings days)
vs WHEN price action dominates (technical breakout days).

Gating mechanism:
  α   = sigmoid(Wg · [sent; price] + bg)
  Fused = α · cross_attended + (1-α) · sentiment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class CrossModalAttentionFusion(nn.Module):
    """
    Gated cross-modal attention fusion layer.

    Sentiment vector queries the price-history key-value store.
    A learned sigmoid gate blends the two modalities dynamically.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = np.sqrt(self.d_k)

        # Query from sentiment, Key / Value from price
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Learned gating: decides how much to trust sentiment vs price
        self.gate_layer = nn.Linear(d_model * 2, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        sentiment_vec: torch.Tensor,    # (batch, d_model) — text branch output
        price_sequence: torch.Tensor,    # (batch, seq_len, d_model) — price branch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            fused_vec: (batch, d_model) — gated fusion of sentiment + price
            cross_attention: (batch, seq_len) — interpretable attention weights
        """
        # Expand sentiment to sequence shape for attention
        Q = self.layer_norm_q(sentiment_vec.unsqueeze(1))   # (batch, 1, d_model)
        KV = self.layer_norm_kv(price_sequence)

        # Cross-attention: sentiment queries price history
        q = self.W_q(Q)
        k = self.W_k(KV)
        v = self.W_v(KV)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)           # (batch, 1, seq_len)
        attn_weights = self.dropout(attn_weights)

        cross_context = torch.matmul(attn_weights, v).squeeze(1)  # (batch, d_model)
        cross_context = self.W_o(cross_context)

        # Learned gating: how much to blend sentiment vs price attention
        gate_input = torch.cat([sentiment_vec, cross_context], dim=-1)
        alpha = torch.sigmoid(self.gate_layer(gate_input))

        # Gated fusion
        fused = alpha * cross_context + (1 - alpha) * sentiment_vec

        return fused, attn_weights.squeeze(1)
