"""
FINSENT NET PRO — Text Branch

Layered architecture (requested spec):
  1) FinBERT WordPiece tokenizer
  2) FinBERT contextual embeddings (768d), with only last 2 transformer layers trainable
  3) TextCNN with kernel sizes [2,3,4,5], 128 filters each -> 512d
  4) 3-layer stacked BiLSTM (256 hidden per direction -> 512d), variational dropout=0.3
  5) Multi-head self-attention (8 heads, d_model=512, d_k=d_v=64) with positional encoding
  6) Sentence representation = CLS token + masked mean pool, then LayerNorm
"""

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None


logger = logging.getLogger("finsent.models.text_branch")


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding added before self-attention."""

    def __init__(self, d_model: int = 512, max_len: int = 512):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class VariationalDropout(nn.Module):
    """Locked (variational) dropout mask shared across time steps."""

    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = x.new_empty((x.size(0), 1, x.size(2))).bernoulli_(keep_prob)
        mask = mask.div_(keep_prob)
        return x * mask


class TextCNN(nn.Module):
    """Parallel Conv1D blocks for n-gram pattern extraction."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_filters: int = 128,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4, 5]

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            activated = F.relu(conv(x))
            pooled.append(activated.max(dim=2).values)
        return self.dropout(torch.cat(pooled, dim=1))


class MultiHeadSelfAttention(nn.Module):
    """8-head scaled dot-product self-attention with residual connection."""

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.pre_norm(x)
        batch_size, seq_len, _ = x.shape

        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context) + residual, attn


class TextBranch(nn.Module):
    """Financial text encoder with FinBERT + TextCNN + BiLSTM + Self-Attention."""

    def __init__(
        self,
        vocab_size: int = 30522,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_lstm_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.3,
        finbert_model_name: str = "ProsusAI/finbert",
        use_finbert: bool = True,
        finetune_last_n_layers: int = 2,
        max_position_embeddings: int = 512,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.use_finbert = bool(use_finbert)
        self.finbert_model_name = finbert_model_name

        # Layer 1: FinBERT WordPiece tokenizer
        self.tokenizer = None
        if self.use_finbert and AutoTokenizer is not None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
            except Exception as exc:
                logger.warning("Tokenizer load failed for %s: %s", finbert_model_name, exc)

        # Layer 2: Contextual FinBERT embeddings (fallback to nn.Embedding)
        self.finbert = None
        if self.use_finbert and AutoModel is not None:
            try:
                self.finbert = AutoModel.from_pretrained(finbert_model_name)
                self._configure_finbert_finetuning(finetune_last_n_layers)
            except Exception as exc:
                logger.warning("FinBERT load failed for %s: %s", finbert_model_name, exc)
                self.finbert = None

        if self.finbert is None:
            self.use_finbert = False
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.embedding.weight[0].zero_()
        else:
            self.embedding = None

        # Layer 3: TextCNN with [2,3,4,5], 128 filters each -> 512d
        self.text_cnn = TextCNN(
            embed_dim=embed_dim,
            num_filters=128,
            kernel_sizes=[2, 3, 4, 5],
            dropout=dropout,
        )

        # Layer 4: 3-layer stacked BiLSTM, hidden 256 per direction
        self.bilstm_layers = nn.ModuleList()
        lstm_in_dim = embed_dim
        for _ in range(num_lstm_layers):
            self.bilstm_layers.append(
                nn.LSTM(
                    input_size=lstm_in_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            lstm_in_dim = hidden_dim * 2

        self.variational_dropout = VariationalDropout(p=0.3)

        # Layer 5: Positional encoding + 8-head self-attention at d_model=512
        self.positional_encoding = PositionalEncoding(
            d_model=output_dim,
            max_len=max_position_embeddings,
        )
        self.self_attention = MultiHeadSelfAttention(
            d_model=output_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # Layer 6: CLS + mean pooled sentence representation
        self.output_norm = nn.LayerNorm(output_dim)
        self.input_dropout = nn.Dropout(dropout)

        self._init_lstm_weights(hidden_dim)

    def _configure_finbert_finetuning(self, finetune_last_n_layers: int) -> None:
        """Freeze early transformer layers and unfreeze the last N layers."""
        if self.finbert is None:
            return

        for param in self.finbert.parameters():
            param.requires_grad = False

        encoder_layers = getattr(getattr(self.finbert, "encoder", None), "layer", None)
        if encoder_layers is None:
            return

        n = max(0, int(finetune_last_n_layers))
        if n == 0:
            return

        for layer in encoder_layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

    def _init_lstm_weights(self, hidden_dim: int) -> None:
        for lstm in self.bilstm_layers:
            for name, param in lstm.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    param.data[hidden_dim:2 * hidden_dim].fill_(1.0)

    @torch.no_grad()
    def tokenize_texts(
        self,
        texts: List[str],
        max_length: int = 128,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Layer 1 utility: tokenize raw strings into input_ids + attention_mask."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer unavailable. Install transformers and load FinBERT tokenizer.")

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded["input_ids"], encoded["attention_mask"]

    def _embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.finbert is not None:
            outputs = self.finbert(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        else:
            attention_mask = attention_mask.long()

        # Guard against fully-masked sequences which would make attention softmax undefined.
        empty_rows = attention_mask.sum(dim=1) == 0
        if empty_rows.any():
            attention_mask = attention_mask.clone()
            attention_mask[empty_rows, 0] = 1

        # Layer 2: contextual token embeddings
        embeddings = self.input_dropout(self._embed(input_ids, attention_mask))

        # Layer 3: local n-gram sentiment features (512d)
        cnn_features = self.text_cnn(embeddings)

        # Layer 4: stacked BiLSTM with locked dropout between layers
        lstm_out = embeddings
        for layer_idx, lstm in enumerate(self.bilstm_layers):
            lstm_out, _ = lstm(lstm_out)
            if layer_idx < len(self.bilstm_layers) - 1:
                lstm_out = self.variational_dropout(lstm_out)

        # Inject CNN summary into the CLS stream before attention.
        lstm_out = lstm_out.clone()
        lstm_out[:, 0, :] = lstm_out[:, 0, :] + cnn_features

        # Layer 5: positional encoding + self-attention
        encoded = self.positional_encoding(lstm_out)
        attn_mask = attention_mask[:, None, None, :]
        attended, attn_weights = self.self_attention(encoded, mask=attn_mask)

        # Layer 6: CLS + masked mean pooling -> 512d sentence vector + LayerNorm
        cls_repr = attended[:, 0, :]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        pooled_mean = (attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        sentence_vec = self.output_norm(cls_repr + pooled_mean)

        return sentence_vec, attn_weights
