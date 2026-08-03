"""
Cross-Modal Attention Fusion — The Core Innovation.
====================================================

This is where text (sentiment) and price (quantitative) representations
are fused via cross-attention, allowing each modality to attend to the other.

Architecture:
    Text Repr (batch, d_model) → unsqueeze → (batch, 1, d_model)
    Price Repr (batch, d_model) → unsqueeze → (batch, 1, d_model)
    
    Stack of Cross-Attention Layers:
        Layer i:
            text_attended = CrossAttn(Q=text, K=price, V=price) + text
            price_attended = CrossAttn(Q=price, K=text, V=text) + price
            text_fused = FFN(text_attended)
            price_fused = FFN(price_attended)
    
    Final: Concatenate → Project → Fused Representation

Why Cross-Modal Attention > Simple Concatenation:
    1. Concatenation treats modalities independently in the final layers.
       Cross-attention creates explicit information flow between them.
    
    2. The model learns WHICH aspects of sentiment are relevant to 
       WHICH price patterns. For example:
       - "Record earnings" + recent uptrend → strong continuation signal
       - "Record earnings" + extended rally → potential reversal signal
       
    3. Attention weights are interpretable: we can see which text tokens
       the model associates with which price features, providing 
       explainability for investment decisions.

Mathematical Framework:
    Let T ∈ ℝ^(d_model) be the text representation
    Let P ∈ ℝ^(d_model) be the price representation
    
    Cross-attention (text attending to price):
        T' = MultiHead(Q=T, K=P, V=P) + T  (residual)
        T'' = FFN(T') + T'                   (residual)
    
    Cross-attention (price attending to text):
        P' = MultiHead(Q=P, K=T, V=T) + P  (residual)
        P'' = FFN(P') + P'                   (residual)
    
    Final: F = MLP(T'' ⊕ P'')  where ⊕ is concatenation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from finsent.models.layers import MultiHeadAttention, GatedResidualNetwork


class CrossModalAttentionLayer(nn.Module):
    """Single cross-modal attention layer.
    
    Bidirectional cross-attention:
    - Text attends to Price
    - Price attends to Text
    
    Each direction has its own attention + FFN + residual + norm.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Text → Price cross-attention
        self.text_cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.text_attn_norm = nn.LayerNorm(d_model)
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )
        self.text_ffn_norm = nn.LayerNorm(d_model)
        
        # Price → Text cross-attention
        self.price_cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.price_attn_norm = nn.LayerNorm(d_model)
        self.price_ffn = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )
        self.price_ffn_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        text: torch.Tensor,     # (batch, 1, d_model)
        price: torch.Tensor,    # (batch, 1, d_model)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_out: (batch, 1, d_model) — text enhanced by price context
            price_out: (batch, 1, d_model) — price enhanced by text context
            t2p_attn: text-to-price attention weights
            p2t_attn: price-to-text attention weights
        """
        # Text attends to Price
        t2p, t2p_attn = self.text_cross_attn(
            query=text, key=price, value=price
        )
        text_out = self.text_attn_norm(text + t2p)
        text_out = self.text_ffn_norm(text_out + self.text_ffn(text_out))
        
        # Price attends to Text
        p2t, p2t_attn = self.price_cross_attn(
            query=price, key=text, value=text
        )
        price_out = self.price_attn_norm(price + p2t)
        price_out = self.price_ffn_norm(price_out + self.price_ffn(price_out))
        
        return text_out, price_out, t2p_attn, p2t_attn


class CrossModalFusion(nn.Module):
    """Multi-layer cross-modal attention fusion module.
    
    Stacks multiple CrossModalAttentionLayers to allow
    progressively deeper inter-modal information flow.
    
    The final output is a fused representation that captures
    the interaction between sentiment and price dynamics.
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Stack of cross-attention layers
        self.layers = nn.ModuleList([
            CrossModalAttentionLayer(
                d_model=d_model,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Modal importance gating
        # Learns how much to weight text vs price in the final representation
        self.modal_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1),
        )
        
        # Final projection
        self.fusion_proj = GatedResidualNetwork(
            input_dim=d_model,
            hidden_dim=d_model * 2,
            output_dim=d_model,
            dropout=dropout,
        )
    
    def forward(
        self,
        text_repr: torch.Tensor,   # (batch, d_model)
        price_repr: torch.Tensor,  # (batch, d_model)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            text_repr: Text branch output, (batch, d_model)
            price_repr: Price branch output, (batch, d_model)
        
        Returns:
            fused: (batch, d_model) — fused cross-modal representation
            attention_info: dict with attention weights for interpretability
        """
        batch_size = text_repr.size(0)
        
        # Reshape for attention: (batch, d_model) → (batch, 1, d_model)
        text = text_repr.unsqueeze(1)
        price = price_repr.unsqueeze(1)
        
        # Store attention weights from each layer
        attention_info = {"t2p_attn": [], "p2t_attn": []}
        
        # Cross-modal attention layers
        for layer in self.layers:
            text, price, t2p_attn, p2t_attn = layer(text, price)
            attention_info["t2p_attn"].append(t2p_attn)
            attention_info["p2t_attn"].append(p2t_attn)
        
        # Squeeze back: (batch, 1, d_model) → (batch, d_model)
        text = text.squeeze(1)
        price = price.squeeze(1)
        
        # ─── Modal Importance Gating ──────────────────────────────
        # Learn adaptive weighting between modalities
        # This allows the model to rely more on price during volatile periods
        # and more on sentiment during earnings seasons, etc.
        combined = torch.cat([text, price], dim=-1)  # (batch, 2*d_model)
        gate_weights = self.modal_gate(combined)      # (batch, 2)
        
        # Weighted combination
        text_weight = gate_weights[:, 0:1]   # (batch, 1)
        price_weight = gate_weights[:, 1:2]  # (batch, 1)
        
        gated = text_weight * text + price_weight * price  # (batch, d_model)
        
        attention_info["modal_weights"] = gate_weights.detach()
        
        # ─── Final Projection ─────────────────────────────────────
        fused = self.fusion_proj(gated)  # (batch, d_model)
        
        return fused, attention_info
