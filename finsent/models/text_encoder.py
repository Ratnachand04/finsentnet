"""Text encoder: frozen cached embeddings and salience-weighted attention pooling.

What was removed, and why
-------------------------
The V2 branch ran FinBERT and then stacked a TextCNN, a three-layer BiLSTM, an eight-head
self-attention block, a sinusoidal positional encoding and a three-way gated combination
on top of it. On headlines averaging about twelve words that is several million
parameters of pure overfitting surface: FinBERT has already contextualised the tokens,
and a positional encoding applied *after* a recurrent layer is self-contradictory, since
the recurrence has already encoded order.

Worse, the deployed V2 path did not run FinBERT at all. It computed a lexicon sentiment
score and mapped it deterministically to a fabricated token id, so the branch described
in the manuscript was not the branch that executed. That bridge is deleted, and this
module runs the same computation at training and at inference, by construction: there is
only one code path.

What replaces it
----------------
FinBERT runs **offline, once**, and its ``[CLS]`` vectors are cached to disk. Training
sees only the cache, which is what makes 400 names over nine years feasible on a single
consumer GPU. Online, the ``K`` most recent cached vectors for a (ticker, day) are pooled
by attention over three signals: the content vector, a recency encoding, and a learned
source embedding. About 126k trainable parameters, down from several million.

The attention weights are returned, which gives per-prediction interpretability for
free: *which headline drove today's signal* is a more useful artifact than token-level
attention over a twelve-word sentence, and it costs nothing.

No-news days
------------
Most ticker-days carry no relevant headline. Those receive a learned ``s_null`` vector
rather than being dropped, because dropping them would be a selection bias: news days
are systematically higher-volatility.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["recency_encoding", "TextEncoder"]


def recency_encoding(hours: torch.Tensor, dim: int = 16) -> torch.Tensor:
    """Fixed Fourier features of ``log(1 + hours)``; no learned parameters.

    Log spacing matters: the difference between a 24-hour-old and a 30-hour-old headline
    is meaningful, while the difference between 90 and 96 hours is not. Linear features
    would spend most of their resolution where it cannot matter.
    """
    if dim % 2 != 0:
        raise ValueError("recency dim must be even")
    x = torch.log1p(hours.clamp(min=0.0)).unsqueeze(-1)
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=hours.device, dtype=x.dtype) * (-math.log(100.0) / half)
    )
    scaled = x * freqs
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class TextEncoder(nn.Module):
    """``(B, K, 768)`` cached embeddings -> ``(B, d)`` sentiment context."""

    def __init__(
        self,
        emb_dim: int = 768,
        proj_dim: int = 128,
        recency_dim: int = 16,
        source_emb_dim: int = 8,
        attn_hidden: int = 64,
        n_sources: int = 32,
    ) -> None:
        super().__init__()
        self.proj_dim = proj_dim
        self.recency_dim = recency_dim

        self.proj = nn.Linear(emb_dim, proj_dim)
        self.proj_norm = nn.LayerNorm(proj_dim)

        # Index 0 is reserved for padding / unknown source.
        self.source_emb = nn.Embedding(n_sources + 1, source_emb_dim, padding_idx=0)

        self.score_hidden = nn.Linear(proj_dim + recency_dim + source_emb_dim, attn_hidden)
        self.score_out = nn.Linear(attn_hidden, 1, bias=False)

        self.mlp = nn.Linear(proj_dim, proj_dim)
        self.out_norm = nn.LayerNorm(proj_dim)

        # Learned representation of "this ticker had no news today".
        self.null_embedding = nn.Parameter(torch.zeros(proj_dim))
        nn.init.normal_(self.null_embedding, std=0.02)

        self.output_dim = proj_dim

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        lag_hours: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        embeddings
            ``(B, K, emb_dim)`` cached FinBERT vectors, zero-padded.
        mask
            ``(B, K)`` boolean, True where a real headline exists.
        lag_hours
            ``(B, K)`` hours from publication to the decision open.
        source_ids
            ``(B, K)`` integer source ids, 0 for padding.

        Returns
        -------
        ``(s, attn)`` with shapes ``(B, d)`` and ``(B, K)``. Rows with no headlines
        receive ``null_embedding`` and a uniform-zero attention vector.
        """
        if embeddings.dim() != 3:
            raise ValueError(f"expected (B, K, E); got {tuple(embeddings.shape)}")

        b, k, _ = embeddings.shape
        mask = mask.bool()

        h = self.proj_norm(self.proj(embeddings))            # (B, K, d)

        lag = lag_hours if lag_hours is not None else torch.zeros(b, k, device=h.device)
        rec = recency_encoding(lag.to(h.dtype), self.recency_dim)

        src = source_ids if source_ids is not None else torch.zeros(
            b, k, dtype=torch.long, device=h.device
        )
        src_vec = self.source_emb(src.clamp(min=0, max=self.source_emb.num_embeddings - 1))

        logits = self.score_out(torch.tanh(self.score_hidden(
            torch.cat([h, rec, src_vec], dim=-1)
        ))).squeeze(-1)                                      # (B, K)

        logits = logits.masked_fill(~mask, float("-inf"))
        has_news = mask.any(dim=1)

        # Softmax over an all-masked row is undefined; compute it safely, then discard.
        safe_logits = torch.where(
            has_news.unsqueeze(1), logits, torch.zeros_like(logits)
        )
        attn = torch.softmax(safe_logits, dim=1)
        attn = attn * mask.to(attn.dtype)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp(min=1e-9)
        attn = torch.where(has_news.unsqueeze(1), attn, torch.zeros_like(attn))

        pooled = torch.bmm(attn.unsqueeze(1), h).squeeze(1)  # (B, d)
        s = self.out_norm(self.mlp(pooled))

        s = torch.where(has_news.unsqueeze(1), s, self.null_embedding.expand(b, -1))
        return s, attn

    def null_vector(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """The learned no-news representation, broadcast to a batch.

        Used by modality dropout and by the forced-single-modality ablation, so that
        "no text" at evaluation means exactly what it meant during training.
        """
        v = self.null_embedding
        if dtype is not None:
            v = v.to(dtype)
        if device is not None:
            v = v.to(device)
        return v.expand(batch_size, -1)
