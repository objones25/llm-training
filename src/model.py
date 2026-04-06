"""GPT-style decoder-only transformer.

Architecture
------------
    token_embedding + position_embedding
    → N × TransformerBlock (pre-norm, causal self-attention + GELU FFN)
    → LayerNorm
    → lm_head (weight-tied to token_embedding)

Public API
----------
    GPT(cfg: TrainConfig) → nn.Module
        forward(idx: LongTensor[B, T]) → FloatTensor[B, T, vocab_size]

Internal classes (not exported):
    CausalSelfAttention
    FeedForward
    TransformerBlock
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TrainConfig


# ── Attention ─────────────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Uses a combined QKV projection for efficiency. The causal mask is
    registered as a buffer so it moves to the correct device automatically.

    Parameters
    ----------
    cfg : TrainConfig
        Must satisfy ``cfg.d_model % cfg.n_heads == 0``.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, (
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
        )
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        # Combined Q, K, V projection — avoids three separate matmuls
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        # No mask buffer needed: F.scaled_dot_product_attention handles the
        # causal mask internally when is_causal=True.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor[B, T, d_model]

        Returns
        -------
        FloatTensor[B, T, d_model]
        """
        B, T, C = x.shape

        # QKV projection then split
        qkv = self.qkv(x)  # (B, T, 3*d_model)
        q, k, v = qkv.split(C, dim=-1)  # each (B, T, d_model)

        # Reshape into heads: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask.
        # PyTorch 2.0+ dispatches to FlashAttention-2 on CUDA automatically.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)
        return self.out_proj(out)


# ── Feed-forward ──────────────────────────────────────────────────────────────


class FeedForward(nn.Module):
    """Two-layer MLP with GELU activation.

    Expands d_model → d_ff → d_model. No bias, consistent with modern GPT
    practice and the weight-decay grouping contract in CONTRIBUTING.md.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor[B, T, d_model]

        Returns
        -------
        FloatTensor[B, T, d_model]
        """
        return self.fc2(self.act(self.fc1(x)))


# ── Transformer block ─────────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → Attention → residual,
    then LayerNorm → FFN → residual.

    Pre-norm (applied before the sublayer rather than after) is standard in
    GPT-2 and later models. It improves gradient flow depth without needing
    careful init tuning.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor[B, T, d_model]

        Returns
        -------
        FloatTensor[B, T, d_model]
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


# ── GPT ───────────────────────────────────────────────────────────────────────


class GPT(nn.Module):
    """Decoder-only GPT transformer.

    Input
    -----
    idx : LongTensor[B, T]  — token IDs, T ≤ cfg.seq_len

    Output
    ------
    logits : FloatTensor[B, T, vocab_size]

    Notes
    -----
    - ``lm_head.weight`` is tied to ``token_embedding.weight`` (GPT-2 style).
      Because of this, ``named_parameters()`` deduplicates the tied tensor and
      only yields it under ``"token_embedding.weight"``.
    - Embedding modules are named with ``"embedding"`` in their attribute name
      so the CONTRIBUTING.md guard correctly excludes them from N:
          ``n_params = sum(p.numel() for name, p in model.named_parameters()
                           if "embedding" not in name)``
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share token embedding weights with the output projection.
        # This halves the embedding parameter count and empirically helps
        # language model training (see Press & Wolf, 2017).
        self.lm_head.weight = self.token_embedding.weight

        # Store non-embedding parameter count as an attribute.
        # The "embedding" filter correctly excludes token_embedding.weight
        # (yielded under that name) and position_embedding.weight.
        # lm_head.weight is the same tensor — deduplicated, not double-counted.
        # Surfaced by train.py so the model itself does not print.
        self.n_params: int = sum(
            p.numel()
            for name, p in self.named_parameters()
            if "embedding" not in name
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        idx : LongTensor[B, T]
            Token IDs. T must be ≤ cfg.seq_len.

        Returns
        -------
        logits : FloatTensor[B, T, vocab_size]
        """
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)
