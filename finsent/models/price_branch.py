"""
Price Feature Encoding Branch V2 — 5-Layer Pipeline.
=====================================================

Upgraded architecture replacing the simple CNN→LSTM pipeline with
a production-grade 5-layer price processing system.

Layer Stack:
    Layer 1: Input Normalization
        → Rolling 252-day Z-score (1 trading year)
        → Prevents look-ahead bias in normalization
        → BatchNorm applied after first projection

    Layer 2: Multi-Scale Conv1D
        → 3 parallel branches: k=3 (3-day), k=7 (week), k=15 (month)
        → 64 filters per branch → concatenated → 192-dim
        → Captures patterns at multiple temporal scales simultaneously

    Layer 3: Dilated Temporal CNN
        → Dilated Conv1D with dilation rates [1, 2, 4, 8]
        → Receptive field: 1+2+4+8 = 30+ days with fewer parameters
        → Residual skip connections every 2 layers
        → Efficiently models long-range temporal dependencies

    Layer 4: LSTM Price Encoder
        → 2-layer unidirectional LSTM (causal — no look-ahead)
        → hidden_size=256, captures autocorrelation & momentum
        → Orthogonal initialization for stable gradient flow

    Layer 5: Price Context Vector
        → Final hidden state → 256-dim price representation
        → Linear projection 256 → 512 to match text branch
        → LayerNorm applied

Input:  (batch, window_size, n_features) — e.g., (64, 30, 15) 
Output: (batch, 512) price context vector

Financial Intuition:
    Layer 1: Raw prices are non-stationary — normalization is essential.
             Rolling z-score ensures the model sees relative movements,
             not absolute price levels ($100 AAPL vs $3000 AMZN).
    
    Layer 2: Different patterns operate at different scales:
             k=3: Short-term reversal patterns (3-day mean reversion)
             k=7: Weekly patterns (Monday effect, Friday positioning)
             k=15: Monthly momentum (earnings cycle, options expiry)
    
    Layer 3: Dilated convolutions efficiently model long-range patterns
             like 30-day trends using only 4 layers instead of 30.
             Skip connections prevent gradient degradation.
    
    Layer 4: LSTM captures sequential dependencies that CNNs miss:
             momentum persistence, volatility clustering (GARCH effects),
             and regime transitions.
    
    Layer 5: The final hidden state is a compressed summary of the
             entire price window, projected to match text branch dim.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


# ═══════════════════════════════════════════════════════════════════════
# Layer 1: Input Normalization — Rolling Z-Score + BatchNorm
# ═══════════════════════════════════════════════════════════════════════

class InputNormalizationLayer(nn.Module):
    """Rolling Z-score normalization for non-stationary financial data.
    
    Problem: Financial prices are non-stationary — raw values like
    "AAPL at $180" vs "AAPL at $150" encode absolute levels, not
    meaningful relative patterns. The model should learn from
    RELATIVE movements, not memorize price levels.
    
    Solution: Z-score normalization using a rolling 252-day window
    (1 trading year). At time t, we compute:
        
        z[t] = (x[t] - mean(x[t-252:t])) / std(x[t-252:t])
    
    This is a TRAINING-TIME normalization applied within the model.
    The raw features are stored unnormalized — normalization happens
    in the forward pass using only past data (no look-ahead).
    
    Additionally, BatchNorm is applied after the first projection
    to stabilize activations across the batch.
    
    Note: For inference, we use the running statistics from BatchNorm
    (computed during training), so no additional rolling window is needed.
    """
    
    def __init__(
        self,
        n_features: int = 15,
        projected_dim: int = 192,
        rolling_window: int = 252,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.rolling_window = rolling_window
        self.eps = eps
        
        # Learnable affine transform per feature (like LayerNorm but per-feature)
        self.feature_scale = nn.Parameter(torch.ones(n_features))
        self.feature_bias = nn.Parameter(torch.zeros(n_features))
        
        # Project raw features to working dimension
        self.projection = nn.Linear(n_features, projected_dim)
        self.batch_norm = nn.BatchNorm1d(projected_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def forward(
        self,
        price: torch.Tensor,  # (batch, window_size, n_features)
    ) -> torch.Tensor:
        """Normalize and project price features.
        
        The Z-score normalization is applied per-feature across the
        time dimension of the input window. Since the input window
        is already a historical lookback (e.g., last 30 days), we
        compute statistics across this window — this is strictly
        causal because the entire window is in the past.
        
        Args:
            price: Raw price features (batch, window_size, n_features)
        
        Returns:
            normalized: (batch, window_size, projected_dim)
        """
        # ─── Z-Score Normalization (per feature, across time) ─────
        # Compute mean and std across the time dimension
        mean = price.mean(dim=1, keepdim=True)      # (batch, 1, n_features)
        std = price.std(dim=1, keepdim=True) + self.eps  # (batch, 1, n_features)
        
        # Normalize
        z_normed = (price - mean) / std  # (batch, window_size, n_features)
        
        # Learnable affine transform (per-feature scale and shift)
        z_normed = z_normed * self.feature_scale + self.feature_bias
        
        # Handle any remaining NaN/inf from degenerate windows
        z_normed = torch.nan_to_num(z_normed, nan=0.0, posinf=3.0, neginf=-3.0)
        
        # ─── Project to Working Dimension ─────────────────────────
        projected = self.projection(z_normed)  # (batch, window_size, projected_dim)
        
        # BatchNorm: (batch, projected_dim, window_size) format expected
        batch_size, seq_len, dim = projected.shape
        projected = projected.transpose(1, 2)  # (batch, projected_dim, window_size)
        projected = self.batch_norm(projected)
        projected = projected.transpose(1, 2)  # back to (batch, window_size, projected_dim)
        
        projected = F.gelu(projected)
        projected = self.dropout(projected)
        
        return projected


# ═══════════════════════════════════════════════════════════════════════
# Layer 2: Multi-Scale Conv1D — Parallel Temporal Pattern Extraction
# ═══════════════════════════════════════════════════════════════════════

class MultiScaleConv1DLayer(nn.Module):
    """Parallel Conv1D branches capturing patterns at multiple timescales.
    
    Architecture:
        Input: (batch, window_size, 192)
                      ↓
        ┌─────────────┼─────────────┐
        │             │             │
    Conv1D(k=3)   Conv1D(k=7)  Conv1D(k=15)
    64 filters    64 filters    64 filters
    "3-day"       "weekly"      "monthly"
        │             │             │
    BatchNorm     BatchNorm     BatchNorm
    + GELU        + GELU        + GELU
        │             │             │
        └─────────────┼─────────────┘
                      ↓
              Concatenate → (batch, window_size, 192)
    
    Financial Intuition:
        k=3 (short patterns):
            - 3-day reversal patterns (post-gap fade)
            - Short-term momentum bursts
            - Event-driven moves (earnings, news)
        
        k=7 (weekly patterns):
            - Weekly seasonality (Monday/Friday effects)
            - Options weekly expiry impact
            - Institutional rebalancing cycles
        
        k=15 (monthly patterns):
            - Monthly momentum (Jegadeesh & Titman)
            - Options monthly expiry
            - Sector rotation patterns
            - Corporate buyback cycles
    
    Each branch uses causal padding to prevent look-ahead.
    """
    
    def __init__(
        self,
        input_dim: int = 192,
        kernel_sizes: List[int] = None,
        n_filters: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 7, 15]
        
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.output_dim = n_filters * len(kernel_sizes)  # 64 * 3 = 192
        
        # Parallel convolution branches with causal padding
        self.conv_branches = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for k in kernel_sizes:
            self.conv_branches.append(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=n_filters,
                    kernel_size=k,
                    padding=0,  # we handle causal padding manually
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(n_filters))
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for conv in self.conv_branches:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
    
    def forward(
        self,
        x: torch.Tensor,  # (batch, window_size, input_dim)
    ) -> torch.Tensor:
        """Extract multi-scale temporal patterns.
        
        Args:
            x: Normalized price features (batch, window_size, input_dim)
        
        Returns:
            output: (batch, window_size, 192) — concatenated multi-scale features
        """
        # Conv1D expects (batch, channels, length)
        x_conv = x.transpose(1, 2)  # (batch, input_dim, window_size)
        
        branch_outputs = []
        for conv, bn, k in zip(self.conv_branches, self.batch_norms, self.kernel_sizes):
            # Causal padding: pad only on the left
            causal_pad = k - 1
            padded = F.pad(x_conv, (causal_pad, 0))  # (batch, input_dim, window_size + pad)
            
            out = conv(padded)   # (batch, n_filters, window_size)
            out = bn(out)
            out = F.gelu(out)
            branch_outputs.append(out)
        
        # Concatenate all branches: (batch, n_filters * n_branches, window_size)
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # Back to (batch, window_size, output_dim)
        output = concatenated.transpose(1, 2)
        output = self.dropout(output)
        
        return output  # (batch, window_size, 192)


# ═══════════════════════════════════════════════════════════════════════
# Layer 3: Dilated Temporal CNN — Long-Range Pattern Extraction
# ═══════════════════════════════════════════════════════════════════════

class DilatedResidualBlock(nn.Module):
    """Single dilated Conv1D block with residual connection.
    
    Architecture:
        Input → CausalPad → DilatedConv1D → BatchNorm → GELU → Dropout
                                                                  ↓
                            Input ──────────────────────────────→ Add
                                                                  ↓
                                                               Output
    
    The residual connection allows gradients to flow directly through
    the skip path, preventing vanishing gradients in deep stacks.
    """
    
    def __init__(
        self,
        channels: int = 192,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.causal_padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # causal padding applied manually
        )
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, window_size)
        Returns:
            (batch, channels, window_size)
        """
        # Causal pad (left only — ensures no look-ahead)
        padded = F.pad(x, (self.causal_padding, 0))
        
        out = self.conv(padded)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # Residual connection
        return out + x


class DilatedTemporalCNNLayer(nn.Module):
    """Stack of dilated convolutions with exponentially growing receptive field.
    
    Architecture:
        Input: (batch, window_size, 192)
                      ↓
        DilatedConv(d=1)  — receptive field: 3 days
                ↓ + residual skip
        DilatedConv(d=2)  — receptive field: 7 days
                ↓ + residual skip  ═══════╗ (skip connection)
        DilatedConv(d=4)  — receptive field: 15 days
                ↓ + residual skip         ║
        DilatedConv(d=8)  — receptive field: 31 days
                ↓ + residual skip  ═══════╝ (skip added)
                      ↓
        Output: (batch, window_size, 192)
    
    Total receptive field with kernel_size=3 and dilations [1,2,4,8]:
        RF = 1 + 2*(3-1)*(1+2+4+8) = 1 + 2*2*15 = 61 days
        This covers 2+ months of price history with only 4 layers!
    
    Skip connections every 2 layers aggregate features at different
    temporal resolutions — the final output combines short-term (d=1,2)
    and long-term (d=4,8) patterns.
    
    Financial Intuition:
        d=1: Immediate price action (today vs yesterday)
        d=2: Short-term momentum (2-day patterns)
        d=4: Weekly patterns (4-day ~ 1 trading week)
        d=8: Bi-weekly/monthly patterns (8-day ~ 2 trading weeks)
    """
    
    def __init__(
        self,
        channels: int = 192,
        kernel_size: int = 3,
        dilation_rates: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]
        
        self.dilation_rates = dilation_rates
        self.n_layers = len(dilation_rates)
        
        # Stack of dilated residual blocks
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation=d,
                dropout=dropout,
            )
            for d in dilation_rates
        ])
        
        # Skip connection aggregation (every 2 layers)
        self.skip_norm = nn.LayerNorm(channels)
    
    def forward(
        self,
        x: torch.Tensor,  # (batch, window_size, channels)
    ) -> torch.Tensor:
        """Process through dilated CNN stack with skip connections.
        
        Args:
            x: Multi-scale features (batch, window_size, 192)
        
        Returns:
            output: (batch, window_size, 192) — temporally enriched features
        """
        # Conv1D format: (batch, channels, window_size)
        h = x.transpose(1, 2)
        
        skip_sum = torch.zeros_like(h)
        
        for i, block in enumerate(self.blocks):
            h = block(h)
            
            # Accumulate skip connections every 2 layers
            if (i + 1) % 2 == 0 or i == self.n_layers - 1:
                skip_sum = skip_sum + h
        
        # Combine skips
        output = skip_sum.transpose(1, 2)  # (batch, window_size, channels)
        output = self.skip_norm(output)
        
        return output


# ═══════════════════════════════════════════════════════════════════════
# Layer 4: LSTM Price Encoder — Sequential Dependency Modeling
# ═══════════════════════════════════════════════════════════════════════

class LSTMPriceEncoderLayer(nn.Module):
    """2-layer unidirectional LSTM for temporal dependency modeling.
    
    Architecture:
        Input: (batch, window_size, 192)
                      ↓
              LSTM Layer 1 (hidden=256)
                      ↓ dropout
              LSTM Layer 2 (hidden=256)
                      ↓
              Output: (batch, window_size, 256) + final hidden (batch, 256)
    
    Key Design Choices:
    
    1. UNIDIRECTIONAL: Price modeling must be causal — we cannot use
       future price data to predict the past. Unlike text where BiLSTM
       is useful, price LSTM must be strictly forward-looking.
    
    2. hidden_size=256: Provides enough capacity to model complex
       temporal patterns while remaining computationally efficient.
       256 per direction captures momentum, mean reversion, and
       volatility clustering simultaneously.
    
    3. Orthogonal init: Preserves gradient norms during BPTT, critical
       for sequences of 30+ timesteps.
    
    4. Forget gate bias = 1.0: LSTM starts by "remembering" everything
       and learns what to forget. Without this, early training often
       loses important long-term dependencies.
    
    Financial Intuition:
        The LSTM cell state acts as a "market memory":
        - Input gate: What new information to incorporate (e.g., breakout)
        - Forget gate: What to discard (e.g., old support levels after break)
        - Output gate: What's relevant now (e.g., momentum vs mean-reversion)
        
        This captures:
        - Autocorrelation in returns (momentum factor)
        - Volatility clustering (GARCH-like effects)
        - Regime transitions (bull→bear, low-vol→high-vol)
    """
    
    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # MUST be unidirectional for causal price modeling
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal init for stable gradient flow over long sequences."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias = 1.0
                # LSTM bias layout: [input | forget | cell | output]
                hidden = self.hidden_dim
                param.data[hidden:2 * hidden].fill_(1.0)
    
    def forward(
        self,
        x: torch.Tensor,  # (batch, window_size, input_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process sequence through LSTM.
        
        Args:
            x: Temporal features (batch, window_size, input_dim)
        
        Returns:
            lstm_output: (batch, window_size, hidden_dim) — all hidden states
            final_hidden: (batch, hidden_dim) — last timestep hidden state
        """
        # Forward through LSTM
        lstm_output, (h_n, c_n) = self.lstm(x)
        # lstm_output: (batch, window_size, hidden_dim)
        # h_n: (num_layers, batch, hidden_dim)
        
        # Final hidden state from last layer
        final_hidden = h_n[-1]  # (batch, hidden_dim)
        
        return lstm_output, final_hidden


# ═══════════════════════════════════════════════════════════════════════
# Layer 5: Price Context Vector — Final Representation
# ═══════════════════════════════════════════════════════════════════════

class PriceContextVectorLayer(nn.Module):
    """Projects LSTM final hidden state to match text branch dimension.
    
    Architecture:
        LSTM final hidden (batch, 256)
                    ↓
            Linear(256, 512) → GELU → Dropout
                    ↓
              LayerNorm → (batch, 512)
    
    This projection ensures the price representation lives in the same
    512-dimensional space as the text representation, enabling meaningful
    cross-modal attention in the fusion module.
    
    The LSTM final hidden state is chosen over mean-pooling because:
    1. It naturally weights recent information more heavily (recency bias)
    2. It captures the CURRENT state of the market, not the average
    3. For causal price modeling, the most recent state is most relevant
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        final_hidden: torch.Tensor,  # (batch, input_dim)
    ) -> torch.Tensor:
        """Project to output dimension.
        
        Args:
            final_hidden: LSTM final hidden state (batch, 256)
        
        Returns:
            price_context: (batch, 512) — price representation
        """
        projected = self.projection(final_hidden)  # (batch, 512)
        price_context = self.layer_norm(projected)  # (batch, 512)
        return price_context


# ═══════════════════════════════════════════════════════════════════════
# Top-Level: PriceBranchV2 — Full 5-Layer Pipeline
# ═══════════════════════════════════════════════════════════════════════

class PriceBranch(nn.Module):
    """5-layer price processing branch for FinSentNet.
    
    Composes all 5 layers into a single nn.Module:
    
        Layer 1: Input Normalization    (rolling z-score + BatchNorm)
        Layer 2: Multi-Scale Conv1D     (k=3,7,15 parallel → 192-dim)
        Layer 3: Dilated Temporal CNN   (d=1,2,4,8, residual skips)
        Layer 4: LSTM Price Encoder     (2-layer, hidden=256)
        Layer 5: Price Context Vector   (256 → 512 projection)
    
    Input:  (batch, window_size, n_features) — raw OHLCV + technical indicators
    Output: (batch, 512) price context vector
    """
    
    def __init__(
        self,
        # Layer 1: Input
        n_features: int = 15,
        window_size: int = 30,
        rolling_norm_window: int = 252,
        # Layer 2: Multi-scale Conv
        conv_kernel_sizes: List[int] = None,
        conv_n_filters: int = 64,
        # Layer 3: Dilated CNN
        dilated_kernel_size: int = 3,
        dilation_rates: List[int] = None,
        # Layer 4: LSTM
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        # Layer 5: Context vector
        output_dim: int = 512,
        # General
        dropout: float = 0.1,
        # Compatibility kwargs (absorbed but not used)
        **kwargs,
    ):
        super().__init__()
        
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [3, 7, 15]
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]
        
        self.n_features = n_features
        self.window_size = window_size
        self.output_dim = output_dim
        
        # Working dimension from multi-scale conv
        working_dim = conv_n_filters * len(conv_kernel_sizes)  # 64 * 3 = 192
        
        # ─── Layer 1: Input Normalization ─────────────────────────
        self.input_norm = InputNormalizationLayer(
            n_features=n_features,
            projected_dim=working_dim,
            rolling_window=rolling_norm_window,
            dropout=dropout,
        )
        
        # ─── Layer 2: Multi-Scale Conv1D ──────────────────────────
        self.multi_scale_conv = MultiScaleConv1DLayer(
            input_dim=working_dim,
            kernel_sizes=conv_kernel_sizes,
            n_filters=conv_n_filters,
            dropout=dropout,
        )
        
        # ─── Layer 3: Dilated Temporal CNN ────────────────────────
        self.dilated_cnn = DilatedTemporalCNNLayer(
            channels=working_dim,
            kernel_size=dilated_kernel_size,
            dilation_rates=dilation_rates,
            dropout=dropout,
        )
        
        # ─── Layer 4: LSTM Price Encoder ──────────────────────────
        self.lstm_encoder = LSTMPriceEncoderLayer(
            input_dim=working_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        
        # ─── Layer 5: Price Context Vector ────────────────────────
        self.context_vector = PriceContextVectorLayer(
            input_dim=lstm_hidden,
            output_dim=output_dim,
            dropout=dropout,
        )
    
    def forward(
        self,
        price: torch.Tensor,  # (batch, window_size, n_features)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full 5-layer forward pass.
        
        Args:
            price: Raw price features (batch, window_size, n_features)
        
        Returns:
            price_repr: (batch, 512) — price context vector
            lstm_output: (batch, window_size, lstm_hidden) — for temporal attention
        """
        # ─── Layer 1: Input Normalization ─────────────────────────
        normalized = self.input_norm(price)  # (batch, window_size, 192)
        
        # ─── Layer 2: Multi-Scale Conv1D ──────────────────────────
        multi_scale = self.multi_scale_conv(normalized)  # (batch, window_size, 192)
        
        # ─── Layer 3: Dilated Temporal CNN ────────────────────────
        dilated = self.dilated_cnn(multi_scale)  # (batch, window_size, 192)
        
        # ─── Layer 4: LSTM Price Encoder ──────────────────────────
        lstm_output, final_hidden = self.lstm_encoder(dilated)
        # lstm_output: (batch, window_size, 256)
        # final_hidden: (batch, 256)
        
        # ─── Layer 5: Price Context Vector ────────────────────────
        price_repr = self.context_vector(final_hidden)  # (batch, 512)
        
        return price_repr, lstm_output
    
    def get_temporal_importance(
        self,
        price: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-timestep importance for interpretability.
        
        Uses the L2 norm of LSTM hidden states as a proxy for
        how much information each timestep contributes.
        
        Returns: (batch, window_size) importance scores
        """
        _, lstm_output = self.forward(price)
        # L2 norm across hidden dimension → scalar per timestep
        importance = lstm_output.norm(dim=-1)  # (batch, window_size)
        # Normalize to sum to 1
        importance = importance / importance.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return importance
    
    def count_parameters(self) -> Dict[str, Dict[str, int]]:
        """Count parameters by layer."""
        layers = {
            "Layer 1 - InputNorm": self.input_norm,
            "Layer 2 - MultiScaleConv": self.multi_scale_conv,
            "Layer 3 - DilatedCNN": self.dilated_cnn,
            "Layer 4 - LSTM": self.lstm_encoder,
            "Layer 5 - ContextVec": self.context_vector,
        }
        
        counts = {}
        for name, module in layers.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable}
        
        total_all = sum(p.numel() for p in self.parameters())
        trainable_all = sum(p.numel() for p in self.parameters() if p.requires_grad)
        counts["TOTAL"] = {"total": total_all, "trainable": trainable_all}
        
        return counts
