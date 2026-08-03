"""
Text Processing Branch — BiLSTM with Multi-Head Self-Attention.
==================================================================

Architecture:
    Word Embedding → Positional Encoding → BiLSTM → Multi-Head Self-Attention → 
    Weighted Pool → GRN → Text Representation

Input: (batch, max_seq_length) integer token IDs
Output: (batch, output_dim) text feature vector

Financial Intuition:
    BiLSTM captures bidirectional context in financial language.
    This is crucial because negation changes meaning completely:
        "not profitable" vs "profitable not expected to decline"
    
    The self-attention mechanism learns which words in headlines
    are most predictive of price movement. In financial text,
    this often highlights:
    - Sentiment words: "beats", "disappoints", "warns"
    - Magnitude words: "significantly", "slightly", "record"
    - Entity references: company names, sector terms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from finsent.models.layers import (
    MultiHeadAttention,
    PositionalEncoding,
    GatedResidualNetwork,
)


class TextBranch(nn.Module):
    """BiLSTM + Multi-Head Attention text encoder.
    
    Design choices:
    1. Word-level embeddings (not subword) — financial vocabulary is stable
    2. BiLSTM for sequential understanding — captures word order effects
    3. Multi-head self-attention — learns which tokens matter most for prediction
    4. Attention-weighted pooling — better than mean/max for variable-length inputs
    5. GRN output projection — adaptive gating suppresses noise
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.3,
        bidirectional: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        lstm_output_dim = hidden_dim * self.num_directions
        
        # ─── Embedding Layer ──────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout=dropout)
        self.embed_dropout = nn.Dropout(dropout)
        
        # ─── BiLSTM ──────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Project LSTM output to attention dimension
        self.lstm_proj = nn.Linear(lstm_output_dim, output_dim)
        
        # ─── Multi-Head Self-Attention ────────────────────────────
        self.self_attention = MultiHeadAttention(
            d_model=output_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(output_dim)
        
        # ─── Attention Pooling ────────────────────────────────────
        # Learnable query vector for attention-weighted pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, output_dim))
        nn.init.xavier_uniform_(self.pool_query)
        
        # ─── Output GRN ──────────────────────────────────────────
        self.output_grn = GatedResidualNetwork(
            input_dim=output_dim,
            hidden_dim=output_dim * 2,
            output_dim=output_dim,
            dropout=dropout,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Embedding: normal with small scale
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.embedding.padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[self.embedding.padding_idx])
        
        # LSTM: orthogonal initialization (reduces vanishing gradients)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1.0 (prevents early forgetting)
                hidden = self.hidden_dim
                param.data[hidden:2 * hidden].fill_(1.0)
    
    def forward(
        self,
        text_ids: torch.Tensor,       # (batch, seq_len)
        text_mask: torch.Tensor,      # (batch, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_ids: Token integer IDs, (batch, seq_len)
            text_mask: Padding mask, 1.0 for real tokens, 0.0 for padding
        
        Returns:
            text_repr: (batch, output_dim) — aggregated text representation
            attention_weights: (batch, num_heads, 1, seq_len) — for visualization
        """
        batch_size, seq_len = text_ids.shape
        
        # ─── Embedding + Positional Encoding ──────────────────────
        embedded = self.embedding(text_ids)          # (batch, seq_len, embed_dim)
        embedded = self.pos_encoder(embedded)
        embedded = self.embed_dropout(embedded)
        
        # ─── BiLSTM ──────────────────────────────────────────────
        # Pack padded sequences for efficiency
        lengths = text_mask.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        
        # Project to attention dimension
        projected = self.lstm_proj(lstm_out)  # (batch, seq_len, output_dim)
        
        # ─── Self-Attention ──────────────────────────────────────
        # Create attention mask: (batch, 1, seq_len) — MHA adds the head dim internally
        attn_mask = text_mask.unsqueeze(1)
        
        # Self-attention over LSTM outputs
        attended, _ = self.self_attention(
            query=projected,
            key=projected,
            value=projected,
            mask=attn_mask,
        )
        attended = self.attn_norm(attended + projected)  # residual
        
        # ─── Attention-Weighted Pooling ──────────────────────────
        # Use learnable query to attend over sequence
        pool_query = self.pool_query.expand(batch_size, -1, -1)
        pooled, pool_weights = self.self_attention(
            query=pool_query,           # (batch, 1, d_model)
            key=attended,               # (batch, seq_len, d_model)
            value=attended,             # (batch, seq_len, d_model)
            mask=attn_mask,
        )
        pooled = pooled.squeeze(1)      # (batch, output_dim)
        
        # ─── Output GRN ──────────────────────────────────────────
        text_repr = self.output_grn(pooled)  # (batch, output_dim)
        
        return text_repr, pool_weights
    
    def get_word_importance(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-word importance scores for interpretability.
        
        Returns: (batch, seq_len) importance scores summed across heads.
        """
        _, attn_weights = self.forward(text_ids, text_mask)
        # Average across heads, squeeze query dim
        importance = attn_weights.mean(dim=1).squeeze(1)  # (batch, seq_len)
        return importance
