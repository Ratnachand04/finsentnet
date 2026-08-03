"""FinSentNet-C: the assembled model.

    price window (B, 60, 12) ------> PriceEncoder ------> (B, 60, 128), (B, 128)
                                                                |         |
    cached FinBERT (B, 8, 768) ----> TextEncoder -----> (B, 128)|         |
                                                            |   |         |
                                          GatedCrossModalFusion ----------+
                                                            |
                                                     PredictionHeads
                                                            |
                                    logits (B,3) | mu (B,) | logvar (B,)

About 448k trainable parameters against roughly 220k effectively independent training
samples (880k rows at an average label uniqueness near 0.25). That ratio is the capacity
argument the paper makes, and it is roughly twenty times better than V2's.

Frozen FinBERT (110M) never enters the training graph: its ``[CLS]`` vectors are cached
offline by ``finsent.data.embed_cache``. That is what makes the walk-forward sweep
feasible on a single consumer GPU, and it is also what guarantees that the text path is
identical at training and inference -- there is only one code path, so the V2 situation
where the deployed system silently substituted a lexicon score cannot recur.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from finsent.models.gated_fusion import FusionOutput, GatedCrossModalFusion
from finsent.models.heads import HeadOutput, PredictionHeads
from finsent.models.price_encoder import PriceEncoder
from finsent.models.text_encoder import TextEncoder

__all__ = ["ModelOutput", "FinSentNetC"]


@dataclass
class ModelOutput:
    """Predictions plus the diagnostics the interpretability section is built on."""

    heads: HeadOutput
    fusion: FusionOutput
    text_attention: torch.Tensor    # (B, K) which headline drove the signal
    text_context: torch.Tensor      # (B, d)
    price_summary: torch.Tensor     # (B, d)

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            **self.heads.as_dict(),
            "gate_mean": self.fusion.gate_mean,
            "gate_std": self.fusion.gate_std,
            "text_attn": self.text_attention,
            "cross_attn": self.fusion.attn_weights,
        }


class FinSentNetC(nn.Module):
    """Calibration-aware cross-modal model for equity direction."""

    def __init__(
        self,
        n_features: int = 12,
        d_model: int = 128,
        conv_kernels: tuple[int, ...] = (3, 7, 15),
        conv_channels: int = 32,
        tcn_channels: int = 64,
        tcn_kernel: int = 3,
        tcn_dilations: tuple[int, ...] = (1, 2, 4, 8),
        tcn_dropout: float = 0.1,
        text_emb_dim: int = 768,
        recency_dim: int = 16,
        source_emb_dim: int = 8,
        attn_hidden: int = 64,
        n_sources: int = 32,
        fusion_heads: int = 4,
        fusion_dropout: float = 0.1,
        modality_dropout_text: float = 0.15,
        modality_dropout_price: float = 0.15,
        head_hidden: int = 128,
        head_dropout: float = 0.1,
        logvar_clamp: tuple[float, float] = (-10.0, 2.0),
        heteroscedastic: bool = True,
        use_text: bool = True,
    ) -> None:
        super().__init__()
        self.use_text = use_text

        self.price_encoder = PriceEncoder(
            n_features=n_features,
            conv_kernels=conv_kernels,
            conv_channels=conv_channels,
            tcn_channels=tcn_channels,
            tcn_kernel=tcn_kernel,
            tcn_dilations=tcn_dilations,
            tcn_dropout=tcn_dropout,
            gru_hidden=d_model,
        )
        self.text_encoder = TextEncoder(
            emb_dim=text_emb_dim,
            proj_dim=d_model,
            recency_dim=recency_dim,
            source_emb_dim=source_emb_dim,
            attn_hidden=attn_hidden,
            n_sources=n_sources,
        )
        self.fusion = GatedCrossModalFusion(
            d_model=d_model,
            n_heads=fusion_heads,
            dropout=fusion_dropout,
            modality_dropout_text=modality_dropout_text,
            modality_dropout_price=modality_dropout_price,
        )
        self.heads = PredictionHeads(
            d_model=d_model,
            hidden=head_hidden,
            n_classes=3,
            dropout=head_dropout,
            logvar_clamp=logvar_clamp,
            heteroscedastic=heteroscedastic,
        )

    # -- construction ----------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg) -> "FinSentNetC":
        """Build from a :class:`finsent.config.Config`. No constant is hard-coded here."""
        m, d = cfg.model, cfg.data
        return cls(
            n_features=d.n_features,
            d_model=m.d_model,
            conv_kernels=m.conv_kernels,
            conv_channels=m.conv_channels,
            tcn_channels=m.tcn_channels,
            tcn_kernel=m.tcn_kernel,
            tcn_dilations=m.tcn_dilations,
            tcn_dropout=m.tcn_dropout,
            text_emb_dim=m.text_emb_dim,
            recency_dim=m.recency_dim,
            source_emb_dim=m.source_emb_dim,
            attn_hidden=m.attn_hidden,
            n_sources=m.n_sources,
            fusion_heads=m.fusion_heads,
            modality_dropout_text=m.modality_dropout_text,
            modality_dropout_price=m.modality_dropout_price,
            head_hidden=m.head_hidden,
            head_dropout=m.head_dropout,
            logvar_clamp=m.logvar_clamp,
            heteroscedastic=m.heteroscedastic,
        )

    # -- inference -------------------------------------------------------------------
    def forward(
        self,
        price: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        lag_hours: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
        force_modality: str | None = None,
    ) -> ModelOutput:
        """
        Parameters
        ----------
        price
            ``(B, L, F)`` causal cross-sectionally ranked features.
        text_embeddings
            ``(B, K, 768)`` cached FinBERT vectors. ``None`` means no news at all, which
            is a legitimate and common state, not an error.
        text_mask
            ``(B, K)`` True where a headline exists.
        force_modality
            ``"price_only"`` or ``"text_only"`` for the evaluation-time ablation. Because
            modality dropout trained the model to handle exactly these states, the
            ablation needs no retraining and measures the same network.
        """
        seq, summary = self.price_encoder(price)
        b = price.shape[0]

        if text_embeddings is None or not self.use_text:
            text_ctx = self.text_encoder.null_vector(b, price.device, summary.dtype)
            attn = torch.zeros(b, 1, device=price.device, dtype=summary.dtype)
        else:
            if text_mask is None:
                text_mask = torch.ones(
                    text_embeddings.shape[:2], dtype=torch.bool, device=price.device
                )
            text_ctx, attn = self.text_encoder(
                text_embeddings, text_mask, lag_hours, source_ids
            )

        fused = self.fusion(
            text=text_ctx,
            price_summary=summary,
            price_sequence=seq,
            null_text=self.text_encoder.null_vector(b, price.device, summary.dtype),
            force_modality=force_modality if self.use_text else "price_only",
        )
        return ModelOutput(
            heads=self.heads(fused.fused),
            fusion=fused,
            text_attention=attn,
            text_context=text_ctx,
            price_summary=summary,
        )

    # -- reporting -------------------------------------------------------------------
    def parameter_counts(self) -> dict[str, int]:
        """Trainable parameters per module — the paper's Table 2, generated not typed."""
        groups = {
            "price.multi_scale_conv": self.price_encoder.multi_scale,
            "price.mixer": self.price_encoder.mixer,
            "price.tcn": self.price_encoder.tcn,
            "price.gru": self.price_encoder.gru,
            "price.norms": nn.ModuleList(
                [self.price_encoder.input_norm, self.price_encoder.out_norm]
            ),
            "text.projection": nn.ModuleList(
                [self.text_encoder.proj, self.text_encoder.proj_norm]
            ),
            "text.attention_pooling": nn.ModuleList(
                [
                    self.text_encoder.score_hidden,
                    self.text_encoder.score_out,
                    self.text_encoder.mlp,
                    self.text_encoder.out_norm,
                ]
            ),
            "text.embeddings": self.text_encoder.source_emb,
            "fusion.attention": self.fusion.attn,
            "fusion.gate": nn.ModuleList([self.fusion.gate, self.fusion.norm]),
            "heads": self.heads,
        }
        counts = {
            name: sum(p.numel() for p in mod.parameters() if p.requires_grad)
            for name, mod in groups.items()
        }
        counts["text.null_embedding"] = int(self.text_encoder.null_embedding.numel())
        counts["TOTAL"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return counts

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        counts = self.parameter_counts()
        lines = [f"FinSentNet-C  |  {counts['TOTAL']:,d} trainable parameters"]
        lines.append(
            f"  price receptive field: {self.price_encoder.receptive_field} timesteps"
        )
        for name, n in counts.items():
            if name == "TOTAL":
                continue
            lines.append(f"    {name:<26s} {n:>9,d}  ({100.0 * n / counts['TOTAL']:5.1f}%)")
        return "\n".join(lines)
