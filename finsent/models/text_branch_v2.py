"""
Text Processing Branch V2 — FinBERT-Powered 6-Layer Pipeline.
================================================================

Upgraded architecture replacing the simple word-embedding + BiLSTM
with a production-grade, 6-layer deep NLP pipeline optimized for
financial sentiment extraction.

Layer Stack:
    Layer 1: FinBERT WordPiece Tokenizer
        → Handles financial vocabulary (EBITDA, Fed pivot, basis points)
        → Produces input_ids + attention_mask

    Layer 2: FinBERT Contextual Embeddings (768-dim)
        → Pre-trained on 10-K/Q filings (FinancialBERT)
        → Last 2 transformer layers fine-tuned; earlier layers frozen

    Layer 3: TextCNN — Parallel Multi-Scale Conv1D
        → Filter sizes [2,3,4,5] capture bi/tri/quad/penta-gram patterns
        → 128 filters per size → 512-dim after max-over-time pooling

    Layer 4: Bidirectional LSTM — Sequential Context Encoding
        → 3-layer stacked BiLSTM, hidden=256/direction → 512-dim
        → Variational dropout=0.3 (same mask across timesteps)
        → Orthogonal weight initialization

    Layer 5: Multi-Head Self-Attention — Intra-Article Importance
        → 8 heads, d_model=512, d_k=d_v=64
        → Positional encoding injected before attention
        → Learns which words/n-grams are most predictive

    Layer 6: Sentence Representation — Final Aggregation
        → CLS-token representation (from FinBERT, projected 768→512)
        → Mean-pool of attention outputs
        → Concatenation → Linear → LayerNorm → 512-dim sentence vector

Input:  (batch, max_seq_length) integer token IDs + attention mask
Output: (batch, 512) sentence-level feature vector

Financial Intuition:
    Each layer addresses a different aspect of financial text understanding:
    - FinBERT captures domain-specific semantics ("EBITDA" ≠ random token)
    - TextCNN extracts local n-gram patterns ("beats expectations", "missed revenue")
    - BiLSTM models long-range dependencies ("despite strong Q1, full-year guidance cut")
    - Self-attention learns global word importance across the entire article
    - CLS+MeanPool combines global document understanding with detailed token info
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

from finsent.models.layers import (
    MultiHeadAttention,
    PositionalEncoding,
    GatedResidualNetwork,
    VariationalDropout,
)


# ═══════════════════════════════════════════════════════════════════════
# Layer 1: FinBERT WordPiece Tokenizer
# ═══════════════════════════════════════════════════════════════════════

class FinBERTTokenizerWrapper:
    """Wrapper around HuggingFace FinBERT WordPiece tokenizer.
    
    Handles financial-domain vocabulary including terms like:
        - 'EBITDA', 'EPS', 'P/E ratio', 'basis points'
        - 'Fed pivot', 'short squeeze', 'dead cat bounce'
        - 'inverted yield curve', 'quantitative tightening'
    
    WordPiece tokenization breaks unknown words into known subwords:
        'outperformance' → ['out', '##perform', '##ance']
    
    This is NOT an nn.Module — it's a preprocessing utility.
    Use it before the forward pass to convert raw text → token IDs.
    """
    
    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-pretrain",
        max_length: int = 512,
        local_path: Optional[str] = None,
    ):
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library required for FinBERT tokenizer. "
                "Install with: pip install transformers>=4.30.0"
            )
        
        load_path = local_path if local_path else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.max_length = max_length
        self.model_name = model_name
    
    def __call__(
        self,
        texts: List[str],
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of financial texts.
        
        Args:
            texts: List of raw text strings (headlines, articles, etc.)
            return_tensors: Output format ("pt" for PyTorch tensors)
        
        Returns:
            Dict with:
                input_ids: (batch, seq_len) — WordPiece token IDs
                attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding
                token_type_ids: (batch, seq_len) — segment IDs (0 for single sequence)
        """
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )
        return encoded
    
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size
    
    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
    
    @property
    def cls_token_id(self) -> int:
        return self.tokenizer.cls_token_id
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Convert token IDs back to text (for debugging/visualization)."""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════════
# Layer 2: FinBERT Contextual Embeddings (768-dim)
# ═══════════════════════════════════════════════════════════════════════

class FinBERTEmbeddingLayer(nn.Module):
    """FinBERT contextual embedding layer.
    
    Loads a pre-trained FinBERT model and produces 768-dimensional
    contextual embeddings. The model was fine-tuned on financial text
    (10-K/Q filings, earnings calls, financial news).
    
    Freezing Strategy:
        - All embedding layers:        FROZEN
        - Transformer layers 0 to -3:  FROZEN  
        - Transformer layers -2 to -1: TRAINABLE (fine-tuned)
        - Pooler layer:                FROZEN
    
    This freezing strategy preserves the rich financial language
    understanding in early layers while allowing the last 2 layers
    to adapt to our specific sentiment classification task.
    
    Memory: ~110M params total, ~14M trainable (last 2 layers).
    """
    
    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-pretrain",
        num_fine_tune_layers: int = 2,
        local_path: Optional[str] = None,
    ):
        super().__init__()
        
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers library required for FinBERT embeddings. "
                "Install with: pip install transformers>=4.30.0"
            )
        
        load_path = local_path if local_path else model_name
        self.finbert = AutoModel.from_pretrained(load_path)
        self.embedding_dim = self.finbert.config.hidden_size  # 768
        self.num_fine_tune_layers = num_fine_tune_layers
        
        # ─── Freeze all parameters first ──────────────────────────
        for param in self.finbert.parameters():
            param.requires_grad = False
        
        # ─── Unfreeze last N transformer layers ───────────────────
        # FinBERT (BERT-based) has encoder.layer[0..11]
        total_layers = len(self.finbert.encoder.layer)
        fine_tune_start = total_layers - num_fine_tune_layers
        
        for i in range(fine_tune_start, total_layers):
            for param in self.finbert.encoder.layer[i].parameters():
                param.requires_grad = True
    
    def forward(
        self,
        input_ids: torch.Tensor,        # (batch, seq_len)
        attention_mask: torch.Tensor,    # (batch, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract contextual embeddings from FinBERT.
        
        Args:
            input_ids: WordPiece token IDs from tokenizer
            attention_mask: 1.0 for real tokens, 0.0 for padding
        
        Returns:
            sequence_output: (batch, seq_len, 768) — per-token embeddings
            cls_embedding: (batch, 768) — CLS token representation
        """
        outputs = self.finbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        sequence_output = outputs.last_hidden_state   # (batch, seq_len, 768)
        cls_embedding = sequence_output[:, 0, :]      # (batch, 768) — CLS is always position 0
        
        return sequence_output, cls_embedding
    
    def count_params(self) -> Dict[str, int]:
        """Count frozen vs trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}


# ═══════════════════════════════════════════════════════════════════════
# Layer 3: TextCNN — Parallel Multi-Scale Convolutions
# ═══════════════════════════════════════════════════════════════════════

class TextCNNLayer(nn.Module):
    """Parallel Conv1D with multiple filter sizes for n-gram extraction.
    
    Inspired by "Convolutional Neural Networks for Sentence Classification"
    (Yoon Kim, 2014), adapted for financial text.
    
    Architecture:
        Input (batch, seq_len, 768) → projection → (batch, seq_len, 512)
                                        ↓
        ┌─────────────────────────────────────────────────┐
        │  Conv1D(k=2) → ReLU → MaxPool → (batch, 128)   │
        │  Conv1D(k=3) → ReLU → MaxPool → (batch, 128)   │
        │  Conv1D(k=4) → ReLU → MaxPool → (batch, 128)   │
        │  Conv1D(k=5) → ReLU → MaxPool → (batch, 128)   │
        └─────────────────────────────────────────────────┘
                                        ↓
                            Concatenate → (batch, 512)
                                        ↓
                            BatchNorm → Dropout → (batch, 512)
    
    Financial Intuition:
        k=2: "beat expectations", "missed revenue", "rate hike"
        k=3: "above street estimates", "cut full-year guidance"
        k=4: "better than expected results", "raised quarterly dividend payment"
        k=5: "significantly exceeded wall street consensus estimates"
    
    Why not just use BERT? TextCNN captures explicit local patterns that
    complement BERT's global contextual understanding. The combination
    gives the model both fine-grained n-gram features AND broad context.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        filter_sizes: List[int] = None,
        num_filters: int = 128,
        output_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        if filter_sizes is None:
            filter_sizes = [2, 3, 4, 5]
        
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        
        # Project FinBERT 768-dim to working dimension
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Parallel convolution branches
        # Conv1D expects (batch, channels, seq_len), so we transpose
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=output_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=0,  # no padding — we handle variable length via max-pool
            )
            for k in filter_sizes
        ])
        
        # Verify output dimension
        total_filters = num_filters * len(filter_sizes)
        assert total_filters == output_dim, (
            f"num_filters({num_filters}) × len(filter_sizes)({len(filter_sizes)}) "
            f"= {total_filters} must equal output_dim({output_dim})"
        )
        
        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming initialization for Conv layers (optimal for ReLU)."""
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
    
    def forward(
        self,
        embeddings: torch.Tensor,       # (batch, seq_len, 768)
        attention_mask: torch.Tensor,    # (batch, seq_len)
    ) -> torch.Tensor:
        """Extract multi-scale n-gram features.
        
        Args:
            embeddings: FinBERT contextual embeddings (batch, seq_len, 768)
            attention_mask: 1.0 for real tokens, 0.0 for padding
        
        Returns:
            cnn_features: (batch, 512) — concatenated max-pooled features
        """
        # Project to working dimension
        x = self.input_proj(embeddings)     # (batch, seq_len, 512)
        
        # Transpose for Conv1D: (batch, 512, seq_len)
        x = x.transpose(1, 2)
        
        # Mask padding positions (set to large negative before max-pool)
        # Expand mask: (batch, seq_len) → (batch, 1, seq_len)
        mask_expanded = attention_mask.unsqueeze(1).float()
        x = x * mask_expanded  # zero out padding positions
        
        # Apply each conv filter and max-pool
        conv_outputs = []
        for conv in self.convs:
            # conv output: (batch, num_filters, seq_len - kernel_size + 1)
            c = F.relu(conv(x))
            
            # Max-over-time pooling: take the strongest activation per filter
            # This captures the most prominent n-gram pattern regardless of position
            c = c.max(dim=2)[0]  # (batch, num_filters)
            conv_outputs.append(c)
        
        # Concatenate all filter outputs: (batch, num_filters * len(filter_sizes))
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch, 512)
        
        # Normalize and regularize
        output = self.batch_norm(concatenated)
        output = self.dropout(output)
        
        return output


# ═══════════════════════════════════════════════════════════════════════
# Layer 4: Bidirectional LSTM — Sequential Context Encoding
# ═══════════════════════════════════════════════════════════════════════

class StackedBiLSTMLayer(nn.Module):
    """3-layer stacked Bidirectional LSTM with variational dropout.
    
    Architecture:
        Input (batch, seq_len, 512)
            ↓
        BiLSTM Layer 1: 512 → 512 (256 per direction)
            ↓ variational dropout (p=0.3)
        BiLSTM Layer 2: 512 → 512 (256 per direction)
            ↓ variational dropout (p=0.3)
        BiLSTM Layer 3: 512 → 512 (256 per direction)
            ↓
        Output (batch, seq_len, 512)
    
    Key Design Choices:
    
    1. Variational Dropout: Same dropout mask across all timesteps
       (Gal & Ghahramani, 2016). Standard dropout in RNNs applies
       different masks per timestep, which is theoretically unsound
       and empirically worse for sequence modeling.
    
    2. Orthogonal Initialization: For recurrent weight matrices (Wₕₕ).
       Orthogonal matrices preserve gradient norms during backpropagation
       through time, significantly reducing vanishing/exploding gradients
       in deep RNNs (Saxe et al., 2014).
    
    3. Forget Gate Bias = 1.0: Initializes the forget gate to "remember"
       by default (Jozefowicz et al., 2015). Without this, LSTMs tend
       to forget important early information before training converges.
    
    4. 3 Layers: Empirically optimal for NLP tasks. More layers add
       diminishing returns and training instability. Each layer increases
       the receptive field over the sequence.
    
    Financial Intuition:
        Forward LSTM:  Reads "Despite strong Q1 results..." and builds expectation
        Backward LSTM: Reads "...guidance was cut for full year" and propagates disappointment
        Together: The model understands that the positive Q1 is NEGATED by the guidance cut
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_dim * self.num_directions  # 512
        
        # Build stacked LSTM layers manually for variational dropout
        # We can't use nn.LSTM's built-in dropout because it's NOT variational
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else self.output_dim
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=bidirectional,
                    dropout=0.0,  # we apply variational dropout manually
                )
            )
        
        # Variational dropout between layers (NOT after the last layer)
        self.var_dropouts = nn.ModuleList([
            VariationalDropout(dropout=dropout)
            for _ in range(num_layers - 1)
        ])
        
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal init for recurrent weights, Xavier for input weights."""
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if "weight_ih" in name:
                    # Input-hidden weights: Xavier uniform
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    # Hidden-hidden weights: Orthogonal (preserves gradient norms)
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1.0
                    # LSTM bias layout: [input_gate | forget_gate | cell_gate | output_gate]
                    # Each gate has hidden_dim biases
                    hidden = self.hidden_dim
                    param.data[hidden:2 * hidden].fill_(1.0)
    
    def forward(
        self,
        x: torch.Tensor,                # (batch, seq_len, input_dim)
        attention_mask: torch.Tensor,    # (batch, seq_len)
    ) -> torch.Tensor:
        """Process sequence through stacked BiLSTM with variational dropout.
        
        Args:
            x: Input features (batch, seq_len, input_dim)
            attention_mask: 1.0 for real tokens, 0.0 for padding
        
        Returns:
            output: (batch, seq_len, 512) — BiLSTM output at each position
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute actual sequence lengths for packing
        lengths = attention_mask.sum(dim=1).long().clamp(min=1)
        
        current = x
        for i, lstm in enumerate(self.lstm_layers):
            # Pack padded sequences for computational efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                current, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Forward through this LSTM layer
            packed_out, _ = lstm(packed)
            
            # Unpack
            current, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=seq_len
            )
            # current: (batch, seq_len, hidden_dim * num_directions)
            
            # Apply variational dropout between layers (not after last)
            if i < self.num_layers - 1:
                current = self.var_dropouts[i](current)
        
        return current  # (batch, seq_len, 512)


# ═══════════════════════════════════════════════════════════════════════
# Layer 5: Multi-Head Self-Attention — Intra-Article Word Importance
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadSelfAttentionLayer(nn.Module):
    """Multi-head self-attention with positional encoding.
    
    Architecture:
        Input (batch, seq_len, 512)
            ↓
        Positional Encoding (sinusoidal, added to input)
            ↓
        Multi-Head Self-Attention (8 heads, d_k=d_v=64)
            ↓
        Residual Connection + LayerNorm
            ↓
        Feed-Forward Network (512 → 2048 → 512)
            ↓
        Residual Connection + LayerNorm
            ↓
        Output (batch, seq_len, 512)
    
    Configuration:
        - 8 attention heads
        - d_model = 512
        - d_k = d_v = 512 / 8 = 64
        - Feed-forward expansion factor = 4× (standard transformer)
    
    Why add positional encoding HERE (after BiLSTM)?
        The BiLSTM already captures sequential order, but self-attention
        is order-invariant. Adding positional encoding ensures the
        attention mechanism can use positional information when computing
        importance weights. The BiLSTM output contains positional info
        implicitly, but explicit encoding strengthens it.
    
    Financial Intuition:
        Self-attention learns which words in a financial article are
        most important for prediction. For example:
        - High attention on "cut", "warned", "downgraded" → bearish signal
        - High attention on "beat", "raised", "upgraded" → bullish signal
        - Cross-attention between "despite" and the following clause
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_expansion: int = 4,
        max_seq_len: int = 512,
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 64
        
        # Positional encoding (sinusoidal)
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout,
        )
        
        # Multi-head self-attention (reuses implementation from layers.py)
        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Feed-forward network (standard transformer FFN)
        ff_dim = d_model * ff_expansion  # 2048
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,                # (batch, seq_len, d_model)
        attention_mask: torch.Tensor,    # (batch, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention with positional encoding.
        
        Args:
            x: Input features (batch, seq_len, 512)
            attention_mask: 1.0 for real tokens, 0.0 for padding
        
        Returns:
            output: (batch, seq_len, 512) — attention-enhanced features
            attn_weights: (batch, num_heads, seq_len, seq_len) — attention maps
        """
        # ─── Add Positional Encoding ──────────────────────────────
        x_pos = self.pos_encoder(x)
        
        # ─── Self-Attention ───────────────────────────────────────
        # Create attention mask: (batch, 1, seq_len) for broadcasting
        attn_mask = attention_mask.unsqueeze(1)
        
        attended, attn_weights = self.self_attention(
            query=x_pos,
            key=x_pos,
            value=x_pos,
            mask=attn_mask,
        )
        
        # Residual + LayerNorm (Pre-LN variant for stability)
        x = self.attn_norm(x + self.attn_dropout(attended))
        
        # ─── Feed-Forward Network ─────────────────────────────────
        ff_out = self.ffn(x)
        x = self.ffn_norm(x + ff_out)
        
        return x, attn_weights


# ═══════════════════════════════════════════════════════════════════════
# Layer 6: Sentence Representation — CLS + MeanPool Aggregation
# ═══════════════════════════════════════════════════════════════════════

class SentenceRepresentationLayer(nn.Module):
    """Combines CLS-token and mean-pooled representations.
    
    Architecture:
        CLS embedding (768-dim from FinBERT)
            ↓ Linear projection (768 → 512)
            ↓
        ┌──────────────────────────────────────┐
        │  CLS representation (batch, 512)     │
        │  +                                   │
        │  Mean-pool of attention outputs      │
        │  (batch, seq_len, 512) → (batch,512) │
        └──────────────────────────────────────┘
                    ↓ Concatenate → (batch, 1024)
                    ↓ Linear → (batch, 512)
                    ↓ LayerNorm
                    ↓
                Output: (batch, 512)
    
    Why BOTH CLS and Mean-Pool?
    
    CLS token: Captures the global document-level understanding from
    FinBERT's pre-training. It was explicitly trained via NSP (Next
    Sentence Prediction) to encode document-level semantics.
    
    Mean pool: Captures the average information from ALL tokens after
    attention reweighting. This preserves detailed token-level info
    that CLS alone might compress away.
    
    Together: The model gets both the "forest" (CLS) and the "trees"
    (mean-pool), which is empirically stronger than either alone,
    especially for sentiment classification.
    """
    
    def __init__(
        self,
        finbert_dim: int = 768,
        attention_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # Project CLS embedding from FinBERT dimension to working dimension
        self.cls_proj = nn.Sequential(
            nn.Linear(finbert_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Combine CLS + mean-pool → output
        # Input: concat(cls_projected, mean_pooled) = 2 × output_dim
        self.fusion = nn.Sequential(
            nn.Linear(output_dim + attention_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier init for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        cls_embedding: torch.Tensor,     # (batch, 768) — from FinBERT
        attention_output: torch.Tensor,  # (batch, seq_len, 512) — from Layer 5
        attention_mask: torch.Tensor,    # (batch, seq_len)
    ) -> torch.Tensor:
        """Create sentence-level representation.
        
        Args:
            cls_embedding: CLS token from FinBERT (batch, 768)
            attention_output: Self-attention output (batch, seq_len, 512)
            attention_mask: 1.0 for real tokens, 0.0 for padding
        
        Returns:
            sentence_repr: (batch, 512) — final sentence representation
        """
        # ─── CLS Projection ──────────────────────────────────────
        cls_projected = self.cls_proj(cls_embedding)  # (batch, 512)
        
        # ─── Mean Pooling (mask-aware) ────────────────────────────
        # Only average over non-padding positions
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        
        # Sum of real token embeddings / count of real tokens
        sum_embeddings = (attention_output * mask_expanded).sum(dim=1)  # (batch, 512)
        token_counts = mask_expanded.sum(dim=1).clamp(min=1e-9)        # (batch, 1)
        mean_pooled = sum_embeddings / token_counts                     # (batch, 512)
        
        # ─── Concatenate and Project ──────────────────────────────
        combined = torch.cat([cls_projected, mean_pooled], dim=-1)  # (batch, 1024)
        fused = self.fusion(combined)                                # (batch, 512)
        
        # ─── LayerNorm ────────────────────────────────────────────
        sentence_repr = self.layer_norm(fused)  # (batch, 512)
        
        return sentence_repr


# ═══════════════════════════════════════════════════════════════════════
# Top-Level: TextBranchV2 — Full 6-Layer Pipeline
# ═══════════════════════════════════════════════════════════════════════

class TextBranchV2(nn.Module):
    """FinBERT-powered text processing branch with 6-layer deep pipeline.
    
    Composes all 6 layers into a single nn.Module:
    
        Layer 1: FinBERT WordPiece Tokenizer   (preprocessing, not in forward)
        Layer 2: FinBERT Embeddings            (768-dim contextual)
        Layer 3: TextCNN                       (512-dim n-gram features)
        Layer 4: Stacked BiLSTM                (512-dim sequential context)
        Layer 5: Multi-Head Self-Attention      (512-dim with importance weights)
        Layer 6: Sentence Representation        (512-dim final vector)
    
    The forward pass expects pre-tokenized input (input_ids + attention_mask).
    Use FinBERTTokenizerWrapper for tokenization in the data pipeline.
    
    Input:  (batch, seq_len) token IDs + attention mask
    Output: (batch, 512) sentence vector + attention weights
    """
    
    def __init__(
        self,
        # Layer 2: FinBERT
        finbert_model: str = "yiyanghkust/finbert-pretrain",
        finbert_embedding_dim: int = 768,
        num_fine_tune_layers: int = 2,
        finbert_local_path: Optional[str] = None,
        # Layer 3: TextCNN
        cnn_filter_sizes: List[int] = None,
        cnn_num_filters: int = 128,
        # Layer 4: BiLSTM
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 3,
        lstm_dropout: float = 0.3,
        # Layer 5: Attention
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        # Layer 6: Output
        output_dim: int = 512,
        # General
        dropout: float = 0.3,
        max_seq_length: int = 512,
    ):
        super().__init__()
        
        if cnn_filter_sizes is None:
            cnn_filter_sizes = [2, 3, 4, 5]
        
        self.output_dim = output_dim
        cnn_output_dim = cnn_num_filters * len(cnn_filter_sizes)  # 128 * 4 = 512
        lstm_output_dim = lstm_hidden_dim * 2  # bidirectional → 512
        
        # Validate dimensions flow correctly
        assert cnn_output_dim == output_dim, (
            f"TextCNN output ({cnn_output_dim}) must equal output_dim ({output_dim}). "
            f"Adjust cnn_num_filters or cnn_filter_sizes."
        )
        assert lstm_output_dim == output_dim, (
            f"BiLSTM output ({lstm_output_dim}) must equal output_dim ({output_dim}). "
            f"Adjust lstm_hidden_dim."
        )
        
        # ─── Layer 2: FinBERT Embeddings ──────────────────────────
        self.finbert_embeddings = FinBERTEmbeddingLayer(
            model_name=finbert_model,
            num_fine_tune_layers=num_fine_tune_layers,
            local_path=finbert_local_path,
        )
        
        # ─── Layer 3: TextCNN ─────────────────────────────────────
        self.text_cnn = TextCNNLayer(
            input_dim=finbert_embedding_dim,
            filter_sizes=cnn_filter_sizes,
            num_filters=cnn_num_filters,
            output_dim=cnn_output_dim,
            dropout=dropout,
        )
        
        # ─── Layer 4: Stacked BiLSTM ─────────────────────────────
        # TextCNN produces a single 512-dim vector (max-pooled).
        # But BiLSTM needs a SEQUENCE input.
        # Solution: We also pass the projected FinBERT sequence through BiLSTM,
        # then combine with TextCNN features.
        self.finbert_to_lstm_proj = nn.Linear(finbert_embedding_dim, output_dim)
        
        self.bilstm = StackedBiLSTMLayer(
            input_dim=output_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            bidirectional=True,
        )
        
        # ─── Layer 5: Multi-Head Self-Attention ───────────────────
        self.self_attention = MultiHeadSelfAttentionLayer(
            d_model=output_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            max_seq_len=max_seq_length,
        )
        
        # ─── Layer 6: Sentence Representation ────────────────────
        self.sentence_repr = SentenceRepresentationLayer(
            finbert_dim=finbert_embedding_dim,
            attention_dim=output_dim,
            output_dim=output_dim,
            dropout=dropout,
        )
        
        # ─── TextCNN Residual Gate ────────────────────────────────
        # Gated combination of TextCNN features with final representation
        # This ensures n-gram features aren't lost through the deep pipeline
        self.cnn_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid(),
        )
        self.final_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,         # (batch, seq_len) — from tokenizer
        attention_mask: torch.Tensor,     # (batch, seq_len) — from tokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full 6-layer forward pass.
        
        Args:
            input_ids: WordPiece token IDs (batch, seq_len)
            attention_mask: 1.0 for real tokens, 0.0 for padding (batch, seq_len)
        
        Returns:
            text_repr: (batch, 512) — final sentence representation
            attn_weights: (batch, num_heads, seq_len, seq_len) — attention maps
        """
        # ─── Layer 2: FinBERT Embeddings ──────────────────────────
        # (batch, seq_len, 768), (batch, 768)
        sequence_embeddings, cls_embedding = self.finbert_embeddings(
            input_ids, attention_mask
        )
        
        # ─── Layer 3: TextCNN ─────────────────────────────────────
        # Extract n-gram features via parallel convolutions
        cnn_features = self.text_cnn(sequence_embeddings, attention_mask)  # (batch, 512)
        
        # ─── Layer 4: Stacked BiLSTM ─────────────────────────────
        # Project FinBERT embeddings to BiLSTM input dimension
        lstm_input = self.finbert_to_lstm_proj(sequence_embeddings)  # (batch, seq_len, 512)
        lstm_output = self.bilstm(lstm_input, attention_mask)         # (batch, seq_len, 512)
        
        # ─── Layer 5: Multi-Head Self-Attention ───────────────────
        attn_output, attn_weights = self.self_attention(
            lstm_output, attention_mask
        )
        # attn_output: (batch, seq_len, 512)
        # attn_weights: (batch, num_heads, seq_len, seq_len)
        
        # ─── Layer 6: Sentence Representation ────────────────────
        sentence_vector = self.sentence_repr(
            cls_embedding, attn_output, attention_mask
        )
        # sentence_vector: (batch, 512)
        
        # ─── Gated Residual from TextCNN ─────────────────────────
        # Don't lose the explicit n-gram features from Layer 3
        gate_input = torch.cat([sentence_vector, cnn_features], dim=-1)  # (batch, 1024)
        gate = self.cnn_gate(gate_input)                                  # (batch, 512)
        
        # Gated combination: gate controls how much CNN info flows through
        text_repr = gate * cnn_features + (1 - gate) * sentence_vector
        text_repr = self.final_norm(text_repr)  # (batch, 512)
        
        return text_repr, attn_weights
    
    def get_word_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-token importance scores for interpretability.
        
        Returns: (batch, seq_len) importance scores averaged across heads.
        """
        _, attn_weights = self.forward(input_ids, attention_mask)
        # Average across heads and sum across query positions
        # attn_weights: (batch, num_heads, seq_len, seq_len)
        importance = attn_weights.mean(dim=1).mean(dim=1)  # (batch, seq_len)
        return importance
    
    def count_parameters(self) -> Dict[str, Dict[str, int]]:
        """Count parameters by layer for architecture analysis."""
        layers = {
            "Layer 2 - FinBERT": self.finbert_embeddings,
            "Layer 3 - TextCNN": self.text_cnn,
            "Layer 4 - BiLSTM": self.bilstm,
            "Layer 5 - Attention": self.self_attention,
            "Layer 6 - SentRepr": self.sentence_repr,
        }
        
        counts = {}
        for name, module in layers.items():
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            counts[name] = {"total": total, "trainable": trainable}
        
        # Add projection and gating layers
        other_total = sum(
            p.numel() for n, p in self.named_parameters()
            if not any(layer_name.lower().replace(" ", "_").replace("-", "_") in n for layer_name in layers)
        )
        other_trainable = sum(
            p.numel() for n, p in self.named_parameters()
            if p.requires_grad and not any(
                layer_name.lower().replace(" ", "_").replace("-", "_") in n for layer_name in layers
            )
        )
        
        total_all = sum(p.numel() for p in self.parameters())
        trainable_all = sum(p.numel() for p in self.parameters() if p.requires_grad)
        counts["TOTAL"] = {"total": total_all, "trainable": trainable_all}
        
        return counts
