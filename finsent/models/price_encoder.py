"""Causal price encoder: multi-scale convolution, dilated TCN, unidirectional GRU.

Shrunk from the V2 branch (multi-scale conv at 64 channels, TCN, two LSTM layers of 256,
projected to 512) to roughly 190k parameters. The justification is the one that belongs
in the paper: with about 880k training rows at an average label uniqueness near 0.25,
there are only ~220k effectively independent samples. A branch with millions of
parameters is not learning a market, it is memorising a sample.

Three properties, each asserted by a test rather than claimed here:

* **Causality.** Every convolution is left-padded and right-trimmed, and the recurrence
  is unidirectional. Output at ``t`` cannot see input after ``t``.
* **Full coverage.** The dilated stack has receptive field
  ``1 + 2(k-1) * sum(dilations) = 61 > L = 60``, so the deepest layer sees the whole
  window. At the V2 lookback of 30 those layers read mostly zero padding, which is a
  silent defect: capacity was allocated to positions that carried no information.
* **Determinism.** No dropout at inference; no data-dependent control flow.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

__all__ = ["CausalConv1d", "TCNBlock", "PriceEncoder"]


class CausalConv1d(nn.Module):
    """1-D convolution that cannot see the future.

    Left-pads by ``(kernel_size - 1) * dilation`` and trims the same amount from the
    right, so position ``t`` of the output is a function of inputs ``<= t`` only.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, C, L)`` -> ``(B, C_out, L)``."""
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual dilated-convolution block with a channel LayerNorm.

    Uses LayerNorm rather than weight normalisation: the parameter count is then exactly
    predictable from the specification, and ``torch.nn.utils.weight_norm`` is deprecated
    in favour of the parametrization API, which would make checkpoints version-fragile.
    """

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, C, L)``."""
        h = self.drop(self.act(self.conv1(x)))
        h = self.conv2(h)
        out = x + h
        # LayerNorm over the channel axis, so transpose in and back out.
        return self.norm(out.transpose(1, 2)).transpose(1, 2)


class PriceEncoder(nn.Module):
    """``(B, L, F)`` -> sequence ``(B, L, d)`` and summary vector ``(B, d)``."""

    def __init__(
        self,
        n_features: int = 12,
        conv_kernels: Sequence[int] = (3, 7, 15),
        conv_channels: int = 32,
        tcn_channels: int = 64,
        tcn_kernel: int = 3,
        tcn_dilations: Sequence[int] = (1, 2, 4, 8),
        tcn_dropout: float = 0.1,
        gru_hidden: int = 128,
        gru_layers: int = 1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.tcn_kernel = tcn_kernel
        self.tcn_dilations = tuple(tcn_dilations)

        self.input_norm = nn.LayerNorm(n_features)
        self.multi_scale = nn.ModuleList(
            [CausalConv1d(n_features, conv_channels, k) for k in conv_kernels]
        )
        self.mixer = nn.Conv1d(conv_channels * len(conv_kernels), tcn_channels, kernel_size=1)
        self.act = nn.GELU()
        self.tcn = nn.ModuleList(
            [
                TCNBlock(tcn_channels, tcn_kernel, d, tcn_dropout)
                for d in self.tcn_dilations
            ]
        )
        self.gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.out_norm = nn.LayerNorm(gru_hidden)
        self.output_dim = gru_hidden

    @property
    def receptive_field(self) -> int:
        """Timesteps visible to the deepest dilated layer."""
        return 1 + 2 * (self.tcn_kernel - 1) * sum(self.tcn_dilations)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x
            ``(B, L, F)`` causal, cross-sectionally ranked features.

        Returns
        -------
        ``(sequence, summary)`` with shapes ``(B, L, d)`` and ``(B, d)``. The sequence is
        what the fusion block attends over; the summary is the price-only representation
        used by the modal gate and by the price-only ablation.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, L, F); got {tuple(x.shape)}")

        h = self.input_norm(x).transpose(1, 2)               # (B, F, L)
        h = torch.cat([conv(h) for conv in self.multi_scale], dim=1)
        h = self.act(self.mixer(self.act(h)))
        for block in self.tcn:
            h = block(h)

        seq, _ = self.gru(h.transpose(1, 2))                 # (B, L, d)
        return seq, self.out_norm(seq[:, -1, :])
