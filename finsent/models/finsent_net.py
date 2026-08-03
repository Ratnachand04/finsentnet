"""
FinSentNet — Full Model Assembly (V2).
========================================

Assembles all branches into the complete FinSentNet architecture
with the upgraded 6-layer FinBERT text processing pipeline:

    ┌─────────────────────────┐       ┌─────────────────────────┐
    │      News Text          │       │      Price OHLCV        │
    │   (batch, seq_len)      │       │  (batch, window, feat)  │
    └───────────┬─────────────┘       └───────────┬─────────────┘
                │                                 │
    ┌───────────▼─────────────┐       ┌───────────▼─────────────┐
    │  Text Branch V2         │       │  Price Branch           │
    │  FinBERT → TextCNN →    │       │  CNN → LSTM →           │
    │  BiLSTM → Attn → Repr  │       │  Temporal Attention     │
    └───────────┬─────────────┘       └───────────┬─────────────┘
                │  (batch, 512)                   │  (batch, 512)
                │                                 │
    ┌───────────▼─────────────────────────────────▼─────────────┐
    │          Cross-Modal Attention Fusion (d_model=512)        │
    │      Bidirectional cross-attention × 3 layers              │
    │      + learned modal importance gating                     │
    └─────────────────────────┬─────────────────────────────────┘
                              │  (batch, 512)
    ┌─────────────────────────▼─────────────────────────────────┐
    │                  Dual-Head Output                          │
    │   ┌──────────────────┐      ┌──────────────────────────┐ │
    │   │  Direction        │      │  Confidence              │ │
    │   │  (↑ / — / ↓)     │      │  (temperature-calibrated)│ │
    │   └──────────────────┘      └──────────────────────────┘ │
    └───────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from finsent.models.text_branch_v2 import TextBranchV2
from finsent.models.price_branch import PriceBranch
from finsent.models.fusion import CrossModalFusion
from finsent.models.dual_head import DualHead


class FinSentNet(nn.Module):
    """Complete FinSentNet model (V2 — with FinBERT text pipeline).
    
    Combines the 6-layer FinBERT text processing branch with
    CNN-LSTM price encoding through cross-modal attention fusion
    for financial prediction.
    
    Key upgrade from V1:
        - Text branch now uses FinBERT (768-dim) instead of word embeddings
        - TextCNN extracts multi-scale n-gram patterns
        - 3-layer BiLSTM with variational dropout
        - Global d_model = 512 (was 256)
    
    Total parameters: ~120M (FinBERT ~110M mostly frozen, ~20M trainable)
    """
    
    def __init__(
        self,
        # Text branch (V2)
        finbert_model: str = "yiyanghkust/finbert-pretrain",
        finbert_embedding_dim: int = 768,
        finbert_frozen_layers: int = 2,
        finbert_local_path: Optional[str] = None,
        cnn_filter_sizes: list = None,
        cnn_num_filters: int = 128,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 3,
        lstm_dropout: float = 0.3,
        text_attention_heads: int = 8,
        text_attention_dropout: float = 0.1,
        text_dropout: float = 0.3,
        max_seq_length: int = 512,
        # Price branch (V2)
        n_price_features: int = 15,
        price_window: int = 30,
        price_conv_kernels: list = None,
        price_conv_filters: int = 64,
        price_dilation_rates: list = None,
        price_lstm_hidden: int = 256,
        price_lstm_layers: int = 2,
        price_dropout: float = 0.2,
        # Fusion
        d_model: int = 512,
        fusion_heads: int = 8,
        fusion_layers: int = 3,
        fusion_ff_dim: int = 1024,
        fusion_dropout: float = 0.1,
        # Dual head
        n_classes: int = 3,
        temperature_init: float = 1.5,
    ):
        super().__init__()
        
        if cnn_filter_sizes is None:
            cnn_filter_sizes = [2, 3, 4, 5]
        if price_conv_kernels is None:
            price_conv_kernels = [3, 7, 15]
        if price_dilation_rates is None:
            price_dilation_rates = [1, 2, 4, 8]
        
        # ─── Text Branch V2 (6-Layer FinBERT Pipeline) ────────────
        self.text_branch = TextBranchV2(
            finbert_model=finbert_model,
            finbert_embedding_dim=finbert_embedding_dim,
            num_fine_tune_layers=finbert_frozen_layers,
            finbert_local_path=finbert_local_path,
            cnn_filter_sizes=cnn_filter_sizes,
            cnn_num_filters=cnn_num_filters,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_dropout,
            attention_heads=text_attention_heads,
            attention_dropout=text_attention_dropout,
            output_dim=d_model,
            dropout=text_dropout,
            max_seq_length=max_seq_length,
        )
        
        # ─── Price Branch V2 (5-Layer Pipeline) ───────────────────
        self.price_branch = PriceBranch(
            n_features=n_price_features,
            window_size=price_window,
            conv_kernel_sizes=price_conv_kernels,
            conv_n_filters=price_conv_filters,
            dilation_rates=price_dilation_rates,
            lstm_hidden=price_lstm_hidden,
            lstm_layers=price_lstm_layers,
            output_dim=d_model,
            dropout=price_dropout,
        )
        
        # ─── Cross-Modal Fusion ───────────────────────────────────
        self.fusion = CrossModalFusion(
            d_model=d_model,
            num_heads=fusion_heads,
            num_layers=fusion_layers,
            feedforward_dim=fusion_ff_dim,
            dropout=fusion_dropout,
        )
        
        # ─── Dual-Head Output ─────────────────────────────────────
        self.dual_head = DualHead(
            input_dim=d_model,
            n_classes=n_classes,
            dropout=0.2,
            temperature_init=temperature_init,
        )
    
    def forward(
        self,
        price: torch.Tensor,              # (batch, window_size, n_features)
        text_ids: torch.Tensor,            # (batch, max_seq_length)
        text_mask: torch.Tensor,           # (batch, max_seq_length)
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass through FinSentNet V2.
        
        Returns dict with:
            direction_logits: (batch, n_classes)
            direction_probs: (batch, n_classes)
            confidence: (batch, 1)
            temperature: scalar
            text_repr: (batch, d_model)
            price_repr: (batch, d_model)
            fused_repr: (batch, d_model)
            attention_info: dict of cross-modal attention weights
            text_attention: text self-attention weights (from Layer 5)
            temporal_attention: price temporal attention weights
        """
        # ─── Branch Encoding ──────────────────────────────────────
        text_repr, text_attn = self.text_branch(text_ids, text_mask)
        price_repr, temporal_attn = self.price_branch(price)
        
        # ─── Cross-Modal Fusion ───────────────────────────────────
        fused_repr, attention_info = self.fusion(text_repr, price_repr)
        
        # ─── Dual-Head Prediction ─────────────────────────────────
        outputs = self.dual_head(fused_repr)
        
        # Augment with intermediate representations
        outputs["text_repr"] = text_repr
        outputs["price_repr"] = price_repr
        outputs["fused_repr"] = fused_repr
        outputs["attention_info"] = attention_info
        outputs["text_attention"] = text_attn
        outputs["temporal_attention"] = temporal_attn
        
        return outputs
    
    def count_parameters(self) -> Dict[str, dict]:
        """Count parameters by module."""
        param_counts = {}
        for name, module in [
            ("text_branch", self.text_branch),
            ("price_branch", self.price_branch),
            ("fusion", self.fusion),
            ("dual_head", self.dual_head),
        ]:
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_counts[name] = {"total": total, "trainable": trainable}
        
        total_all = sum(v["total"] for v in param_counts.values())
        trainable_all = sum(v["trainable"] for v in param_counts.values())
        param_counts["TOTAL"] = {"total": total_all, "trainable": trainable_all}
        
        return param_counts
    
    def print_architecture(self) -> None:
        """Print architecture summary."""
        print("\n" + "=" * 70)
        print("FinSentNet V2 Architecture Summary")
        print("=" * 70)
        
        counts = self.count_parameters()
        for name, info in counts.items():
            total = info["total"]
            trainable = info["trainable"]
            frozen = total - trainable
            if total > 1e6:
                print(
                    f"  {name:20s}: {total/1e6:7.2f}M params "
                    f"({trainable/1e6:.2f}M trainable, {frozen/1e6:.2f}M frozen)"
                )
            else:
                print(
                    f"  {name:20s}: {total:>10,d} params "
                    f"({trainable:,d} trainable, {frozen:,d} frozen)"
                )
        
        print("=" * 70)
        
        # Also print text branch layer breakdown
        print("\n  Text Branch Layer Breakdown:")
        print("  " + "-" * 50)
        text_counts = self.text_branch.count_parameters()
        for name, info in text_counts.items():
            total = info["total"]
            trainable = info["trainable"]
            if total > 1e6:
                print(f"    {name:25s}: {total/1e6:7.2f}M ({trainable/1e6:.2f}M trainable)")
            else:
                print(f"    {name:25s}: {total:>10,d} ({trainable:,d} trainable)")
        print("=" * 70)
    
    @classmethod
    def from_config(cls, config: dict) -> "FinSentNet":
        """Construct model from config dict."""
        tc = config["text_branch"]
        pc = config["price_branch"]
        fc = config["fusion"]
        dc = config["dual_head"]
        
        return cls(
            # Text branch V2
            finbert_model=tc.get("finbert_model", "yiyanghkust/finbert-pretrain"),
            finbert_embedding_dim=tc.get("finbert_embedding_dim", 768),
            finbert_frozen_layers=tc.get("finbert_frozen_layers", 2),
            cnn_filter_sizes=tc.get("cnn_filter_sizes", [2, 3, 4, 5]),
            cnn_num_filters=tc.get("cnn_num_filters", 128),
            lstm_hidden_dim=tc.get("lstm_hidden_dim", 256),
            lstm_num_layers=tc.get("lstm_num_layers", 3),
            lstm_dropout=tc.get("lstm_dropout", 0.3),
            text_attention_heads=tc.get("attention_heads", 8),
            text_attention_dropout=tc.get("attention_dropout", 0.1),
            text_dropout=tc.get("dropout", 0.3),
            max_seq_length=tc.get("max_seq_length", 512),
            # Price branch V2
            n_price_features=config["data"]["price_features"],
            price_window=config["data"]["price_window"],
            price_conv_kernels=pc.get("conv_kernel_sizes", [3, 7, 15]),
            price_conv_filters=pc.get("conv_n_filters", 64),
            price_dilation_rates=pc.get("dilation_rates", [1, 2, 4, 8]),
            price_lstm_hidden=pc.get("lstm_hidden", 256),
            price_lstm_layers=pc.get("lstm_layers", 2),
            price_dropout=pc.get("lstm_dropout", 0.2),
            # Fusion
            d_model=fc["d_model"],
            fusion_heads=fc["num_heads"],
            fusion_layers=fc["num_layers"],
            fusion_ff_dim=fc["feedforward_dim"],
            fusion_dropout=fc["dropout"],
            # Dual head
            n_classes=dc["direction_classes"],
            temperature_init=dc["temperature_init"],
        )
