"""
FINSENT NET PRO — Price Branch
Multi-scale Conv1D → Dilated CNN → LSTM Encoder → price context vector.
Captures short / medium / long-range patterns in OHLCV + Technical Indicators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PriceBranch(nn.Module):
    """
    Price Feature Encoder.

    Architecture:
        Multi-scale Conv1D (kernel 3, 7, 15) →
        Dilated CNN (dilation 1, 2, 4, 8) with residual →
        2-layer LSTM →
        512-dim price context vector

    Input:  (batch, seq_len=30, features=20+)  OHLCV + Technical Indicators
    Output: price_context  (batch, output_dim)  — final representation
            price_sequence (batch, seq_len, output_dim)  — for cross-attention KV
    """

    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 256,
        output_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Multi-scale Conv1D branches (short / medium / long patterns)
        self.conv_3 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv_7 = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)
        self.conv_15 = nn.Conv1d(input_dim, 64, kernel_size=15, padding=7)
        self.bn_multiscale = nn.BatchNorm1d(192)

        # Dilated CNN for longer-range patterns (receptive field × 2^n)
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(192, 192, kernel_size=3, padding=d, dilation=d)
            for d in [1, 2, 4, 8]
        ])
        self.dilated_bns = nn.ModuleList([nn.BatchNorm1d(192) for _ in range(4)])
        self.dilated_res = nn.Conv1d(192, 192, kernel_size=1)  # Skip connection

        # LSTM temporal encoder
        self.lstm = nn.LSTM(
            input_size=192,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        # Project to match text branch dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            price_context:  (batch, output_dim) — final price representation
            price_sequence: (batch, seq_len, output_dim) — for cross-attention
        """
        # (batch, features, seq_len) for Conv1d
        x_conv = x.transpose(1, 2)

        # Multi-scale convolutions
        c3 = F.gelu(self.conv_3(x_conv))
        c7 = F.gelu(self.conv_7(x_conv))
        c15 = F.gelu(self.conv_15(x_conv))
        multi_scale = self.bn_multiscale(torch.cat([c3, c7, c15], dim=1))

        # Dilated CNN with residual
        dilated_out = multi_scale
        for conv, bn in zip(self.dilated_convs, self.dilated_bns):
            dilated_out = F.gelu(bn(conv(dilated_out)))
        dilated_out = dilated_out + self.dilated_res(multi_scale)

        # Back to (batch, seq_len, features) for LSTM
        lstm_input = dilated_out.transpose(1, 2)
        lstm_out, (h_n, _) = self.lstm(lstm_input)

        # Final hidden state as price context
        price_context = self.projection(h_n[-1])

        # Full sequence projected for cross-attention KV
        price_sequence = self.projection(self.dropout(lstm_out))

        return price_context, price_sequence
