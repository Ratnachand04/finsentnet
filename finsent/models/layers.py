"""
Custom neural network layers for FinSentNet.
=============================================

Low-level implementations for full control and understanding.
No black boxes — every operation is mathematically transparent.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VariationalDropout(nn.Module):
    """Variational / Locked Dropout (Gal & Ghahramani, 2016).
    
    Applies the SAME dropout mask across all time steps in a sequence.
    Standard dropout samples a new mask per timestep, which allows the
    RNN to "cheat" by relying on specific timesteps. Locked dropout
    forces the model to learn representations robust to missing features
    at ALL positions simultaneously.
    
    Critical for BiLSTM regularization in financial text processing
    where every token position carries potential signal.
    """
    
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply locked dropout.
        
        Args:
            x: (batch, seq_len, features) — the mask is fixed across seq_len
        Returns:
            (batch, seq_len, features) with locked dropout applied
        """
        if not self.training or self.dropout == 0.0:
            return x
        
        # Create a single mask for the feature dimension, broadcast across seq_len
        # Shape: (batch, 1, features) — same mask for every timestep
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1.0 - self.dropout)
        mask = mask / (1.0 - self.dropout)  # scale to maintain expected values
        
        return x * mask


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention from "Attention Is All You Need" (Vaswani 2017).
    
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    The √d_k scaling prevents dot products from growing too large
    in high dimensions, which would push softmax into saturated regions
    where gradients vanish.
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        query: torch.Tensor,    # (batch, n_heads, seq_q, d_k)
        key: torch.Tensor,      # (batch, n_heads, seq_k, d_k)
        value: torch.Tensor,    # (batch, n_heads, seq_k, d_v)
        mask: Optional[torch.Tensor] = None,  # (batch, 1, 1, seq_k) or broadcastable
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (batch, n_heads, seq_q, d_v)
            attention_weights: (batch, n_heads, seq_q, seq_k)
        """
        d_k = query.size(-1)
        
        # QK^T / √d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask (for padding or causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Softmax + Dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention from scratch.
    
    Instead of a single attention function, project Q, K, V into
    h different subspaces, compute attention in parallel, then
    concatenate and project back.
    
    This allows the model to jointly attend to information from
    different representation subspaces at different positions.
    
    Financial intuition: Different heads can learn to attend to:
      - Price momentum signals
      - Volatility regime indicators  
      - Sentiment-price divergences
      - Cross-asset correlations
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # Linear projections for Q, K, V (combined for efficiency)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(dropout=dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization (optimal for attention layers)."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(
        self,
        query: torch.Tensor,    # (batch, seq_q, d_model)
        key: torch.Tensor,      # (batch, seq_k, d_model)
        value: torch.Tensor,    # (batch, seq_k, d_model)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (batch, seq_q, d_model)
            attention_weights: (batch, num_heads, seq_q, seq_k)
        """
        batch_size = query.size(0)
        
        # Project and reshape to (batch, num_heads, seq, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        
        # Attention
        if mask is not None:
            mask = mask.unsqueeze(1)  # broadcast across heads
        
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_o(attn_output)
        
        return output, attn_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017).
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    For financial time series: position encodes temporal distance,
    allowing the model to learn recency effects (more recent data
    is typically more informative for prediction).
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) from Temporal Fusion Transformer.
    
    Provides adaptive non-linear processing with skip connections.
    The gating mechanism (GLU) allows the network to suppress 
    unnecessary information — critical for noisy financial data.
    
    η = LayerNorm(x + GLU(W₁·ELU(W₂·x + b)))
    
    Financial intuition: The gate learns to suppress noise in 
    irrelevant features while amplifying informative signals.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.output_dim = output_dim or input_dim
        
        # Primary path
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_proj = None
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, self.output_dim * 2)  # *2 for GLU
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # Skip connection projection if dims differ
        if input_dim != self.output_dim:
            self.skip_proj = nn.Linear(input_dim, self.output_dim)
        else:
            self.skip_proj = None
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Skip connection
        skip = self.skip_proj(x) if self.skip_proj is not None else x
        
        # Primary path
        hidden = F.elu(self.fc1(x))
        
        if self.context_proj is not None and context is not None:
            hidden = hidden + self.context_proj(context)
        
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        
        # Gated Linear Unit: GLU(a, b) = a ⊙ σ(b)
        gate_input, gate = hidden.chunk(2, dim=-1)
        hidden = gate_input * torch.sigmoid(gate)
        
        # Residual + LayerNorm
        output = self.layer_norm(skip + hidden)
        return output


class TemporalConvBlock(nn.Module):
    """Causal temporal convolution block.
    
    Uses causal padding to ensure output at time t only depends
    on inputs at times ≤ t (no look-ahead).
    
    Architecture: Conv1D → BatchNorm → GELU → Dropout → Residual
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Causal padding: pad only on the left side
        self.causal_padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # we handle padding manually
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if channels differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time_steps)
        Returns:
            (batch, out_channels, time_steps)
        """
        # Causal pad (left only)
        padded = F.pad(x, (self.causal_padding, 0))
        
        out = self.conv(padded)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # Residual connection
        out = out + self.residual(x)
        return out
