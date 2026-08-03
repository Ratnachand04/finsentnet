"""Gated cross-modal fusion — the architectural identity of this work.

One block, four heads, width 128, replacing V2's three stacked blocks of eight heads at
width 512. The mechanism is unchanged and is the part worth keeping:

    c = MHA(Q = s, K = H, V = H)          news vector selects the relevant price history
    g = sigmoid(W_g [s ; p ; c])          elementwise, per-dimension modal gate
    f = LayerNorm(g * c + (1-g) * p)

The interpretation to state in the paper: the news representation *queries* the price
path, so an earnings-beat headline can attend to the post-gap consolidation rather than
to the sixty-day trend. The gate then decides, per dimension, how much of that
cross-modal read to trust against the price-only representation.

Modality dropout
----------------
During training each modality is independently replaced by its null representation with
probability ``p_drop`` (never both at once). This is a genuine contribution rather than a
trick, and it buys three things:

1. the model cannot collapse onto one modality, which is the usual failure of gated
   fusion when one input is much stronger;
2. graceful degradation when the news feed fails in deployment -- V2 named this as a
   deployment risk and had no answer for it;
3. a *test-time* single-modality ablation that requires no retraining, which is a much
   sharper test of "does fusion matter" than removing the block and retraining a
   different model.

The gate is returned as a diagnostic on every forward pass. That is deliberate: the
claim "the gate learns when to listen to news" is an assertion until it is regressed
against news volume, realised volatility and earnings dates, and it cannot be regressed
unless it is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["FusionOutput", "GatedCrossModalFusion"]


@dataclass
class FusionOutput:
    """Fused representation plus everything the interpretability section needs."""

    fused: torch.Tensor          # (B, d)
    gate: torch.Tensor           # (B, d) elementwise gate in (0, 1)
    cross: torch.Tensor          # (B, d) cross-attention read of the price sequence
    attn_weights: torch.Tensor   # (B, heads, L) attention over the price window

    @property
    def gate_mean(self) -> torch.Tensor:
        """Scalar 'attention to news' per sample; the series plotted in Figure F8."""
        return self.gate.mean(dim=-1)

    @property
    def gate_std(self) -> torch.Tensor:
        return self.gate.std(dim=-1)


class GatedCrossModalFusion(nn.Module):
    """Single-block text-queries-price fusion with an elementwise modal gate."""

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        modality_dropout_text: float = 0.15,
        modality_dropout_price: float = 0.15,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.p_drop_text = modality_dropout_text
        self.p_drop_price = modality_dropout_price

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Linear(3 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _apply_modality_dropout(
        self,
        text: torch.Tensor,
        price: torch.Tensor,
        null_text: torch.Tensor,
        force: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drop at most one modality per sample.

        ``force`` overrides the random draw and is how the evaluation-time ablation
        asks for "price only" or "text only" from an already-trained model.
        """
        b = text.shape[0]
        if force == "price_only":
            return null_text, price
        if force == "text_only":
            return text, torch.zeros_like(price)
        if not self.training or (self.p_drop_text <= 0 and self.p_drop_price <= 0):
            return text, price

        u = torch.rand(b, device=text.device)
        drop_text = u < self.p_drop_text
        drop_price = (u >= self.p_drop_text) & (u < self.p_drop_text + self.p_drop_price)

        text = torch.where(drop_text.unsqueeze(1), null_text, text)
        price = torch.where(drop_price.unsqueeze(1), torch.zeros_like(price), price)
        return text, price

    def forward(
        self,
        text: torch.Tensor,
        price_summary: torch.Tensor,
        price_sequence: torch.Tensor,
        null_text: torch.Tensor | None = None,
        force_modality: str | None = None,
    ) -> FusionOutput:
        """
        Parameters
        ----------
        text
            ``(B, d)`` sentiment context from :class:`~finsent.models.text_encoder`.
        price_summary
            ``(B, d)`` price-only representation.
        price_sequence
            ``(B, L, d)`` price hidden states, attended over by the text query.
        null_text
            ``(B, d)`` learned no-news vector; required for modality dropout.
        force_modality
            ``None``, ``"price_only"`` or ``"text_only"`` -- evaluation-time ablation.
        """
        if null_text is None:
            null_text = torch.zeros_like(text)

        text, price_summary = self._apply_modality_dropout(
            text, price_summary, null_text, force_modality
        )

        query = text.unsqueeze(1)                                  # (B, 1, d)
        cross, weights = self.attn(
            query, price_sequence, price_sequence, need_weights=True, average_attn_weights=False
        )
        cross = cross.squeeze(1)                                   # (B, d)
        weights = weights.squeeze(2)                               # (B, heads, L)

        g = torch.sigmoid(self.gate(torch.cat([text, price_summary, cross], dim=-1)))
        fused = self.norm(g * cross + (1.0 - g) * price_summary)

        return FusionOutput(fused=fused, gate=g, cross=cross, attn_weights=weights)
