"""GPT-style decoder-only transformer with RoPE positional encoding.

Architecture
------------
    token_embedding
    → N × TransformerBlock (pre-norm, causal self-attention + GELU FFN)
      (RoPE applied to Q and K inside each attention layer)
    → RMSNorm
    → lm_head (weight-tied to token_embedding)

Public API
----------
    GPT(cfg: TrainConfig) → nn.Module
        forward(idx: LongTensor[B, T], kv_cache=None) → FloatTensor[B, T, vocab_size]

    KV-cache inference (two-phase)
        cache = KVCache.empty(...)
        logits = model(seed_tokens, kv_cache=cache)   # prefill
        logits = model(next_token,  kv_cache=cache)   # generate (repeated)

Internal classes (not exported):
    RMSNorm
    CausalSelfAttention
    FeedForward
    TransformerBlock
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TrainConfig
from src.kv_cache import KVCache, LayerKVCache


# ── RoPE helpers ──────────────────────────────────────────────────────────────


def _precompute_rope_cos_sin(
    head_dim: int, seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for Rotary Position Embeddings.

    Uses the standard theta_i = 1 / (10000 ^ (2i / head_dim)) formula.

    Returns
    -------
    cos, sin : FloatTensor[seq_len, head_dim]
        Broadcast-ready tables indexed by absolute position.
    """
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    half = head_dim // 2
    # Frequencies: shape [half]
    theta = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    # Positions: shape [seq_len]
    positions = torch.arange(seq_len, device=device).float()
    # Outer product → [seq_len, half]
    freqs = torch.outer(positions, theta)
    # Duplicate each frequency for both sin/cos halves → [seq_len, head_dim]
    freqs = torch.cat([freqs, freqs], dim=-1)
    return freqs.cos(), freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension by negating the first half.

    For a vector split into [a, b], returns [-b, a].
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to query or key tensors.

    Parameters
    ----------
    x   : FloatTensor[B, n_heads, T, head_dim]
    cos : FloatTensor[T, head_dim]  (slice for the relevant positions)
    sin : FloatTensor[T, head_dim]

    Returns
    -------
    FloatTensor[B, n_heads, T, head_dim]
    """
    # Broadcast cos/sin over batch and head dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# ── RMSNorm ───────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Drop-in replacement for LayerNorm that omits mean subtraction and the
    bias term, keeping only the learnable scale (gamma).  Slightly faster than
    LayerNorm and equivalent in practice for transformer pre-norm use.

    Reference: Zhang & Sennrich, 2019 — "Root Mean Square Layer Normalization"
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# ── Attention ─────────────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Rotary Position Embeddings (RoPE).

    Uses a combined QKV projection for efficiency. RoPE is applied to Q and K
    before attention, replacing learned absolute position embeddings. Cosine
    and sine buffers are precomputed at init and registered so they move to
    the correct device automatically.

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

        # Precompute RoPE tables once; register as buffers so they follow the
        # module to the correct device via .to() / .cuda() / etc.
        cos, sin = _precompute_rope_cos_sin(
            self.head_dim, cfg.seq_len, device=torch.device("cpu")
        )
        self.register_buffer("rope_cos", cos)  # [seq_len, head_dim]
        self.register_buffer("rope_sin", sin)  # [seq_len, head_dim]

    def forward(
        self,
        x: torch.Tensor,
        layer_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor[B, T, d_model]
        layer_cache : LayerKVCache or None
            When provided, K/V for new tokens are appended to the cache and
            attention is computed over the full cached history.  Pass None
            during training (standard causal attention, no cache).

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

        # Determine position offset for RoPE.
        # Read cache length BEFORE extending — new tokens start at pos_offset.
        pos_offset = layer_cache.k.shape[2] if layer_cache is not None else 0

        # Apply RoPE to Q and to the new K tokens only.
        # Cached K tokens already have RoPE baked in from when they were first
        # processed; only the freshly projected K slice is rotated.
        cos = self.rope_cos[pos_offset : pos_offset + T]  # [T, head_dim]
        sin = self.rope_sin[pos_offset : pos_offset + T]  # [T, head_dim]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if layer_cache is not None:
            # Extend the cache with the RoPE-encoded new K/V tensors.
            k = torch.cat([layer_cache.k, k], dim=2)  # (B, n_heads, T_cached+T, head_dim)
            v = torch.cat([layer_cache.v, v], dim=2)
            layer_cache.k = k
            layer_cache.v = v
            # Prefill (T > 1): causal mask needed within the new sequence.
            # Generation (T == 1): query is already at the last position —
            # it can attend to the entire cache without a mask.
            is_causal = T > 1
        else:
            is_causal = True

        # Scaled dot-product attention.
        # PyTorch 2.0+ dispatches to FlashAttention-2 on CUDA automatically.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
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
    """Pre-norm transformer block: RMSNorm → Attention → residual,
    then RMSNorm → FFN → residual.

    Pre-norm (applied before the sublayer rather than after) is standard in
    GPT-2 and later models. It improves gradient flow depth without needing
    careful init tuning.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = RMSNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(
        self,
        x: torch.Tensor,
        layer_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor[B, T, d_model]
        layer_cache : LayerKVCache or None
            Passed through to ``CausalSelfAttention.forward``.

        Returns
        -------
        FloatTensor[B, T, d_model]
        """
        x = x + self.attn(self.ln_1(x), layer_cache=layer_cache)
        x = x + self.ff(self.ln_2(x))
        return x


# ── GPT ───────────────────────────────────────────────────────────────────────


class GPT(nn.Module):
    """Decoder-only GPT transformer with RoPE positional encoding.

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
    - Positional information is injected via RoPE inside each attention layer;
      there is no separate ``position_embedding`` module.
    - The ``"embedding"`` name filter used for n_params correctly excludes only
      ``token_embedding.weight`` (and lm_head, which is tied to it).
    """

    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self._n_layers = cfg.n_layers
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share token embedding weights with the output projection.
        # This halves the embedding parameter count and empirically helps
        # language model training (see Press & Wolf, 2017).
        self.lm_head.weight = self.token_embedding.weight

        # GPT-2 style initialization — keeps initial logit magnitudes small so
        # step-0 loss ≈ ln(vocab_size) rather than hundreds.
        self.apply(self._init_weights)
        # Scale down residual projections: each block adds two contributions to
        # the residual stream (attn out_proj + ff fc2), so depth amplifies
        # variance by 2*n_layers.  Dividing std by sqrt(2*n_layers) keeps the
        # residual stream variance independent of depth (see GPT-2 paper §2.3).
        residual_std = 0.02 / (2 * cfg.n_layers) ** 0.5
        for name, p in self.named_parameters():
            if name.endswith(("attn.out_proj.weight", "ff.fc2.weight")):
                nn.init.normal_(p, mean=0.0, std=residual_std)

        # Store non-embedding parameter count as an attribute.
        # The "embedding" filter correctly excludes token_embedding.weight
        # (yielded under that name); lm_head.weight is tied — not double-counted.
        # Surfaced by train.py so the model itself does not print.
        self.n_params: int = sum(
            p.numel()
            for name, p in self.named_parameters()
            if "embedding" not in name
        )

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 style weight initialization.

        - Embeddings and all Linear weights: N(0, 0.02)
        - Linear biases: zeros

        Residual projections (out_proj, fc2) are scaled down separately after
        apply() returns, in __init__, using a named-parameter pass.
        """
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        idx : LongTensor[B, T]
            Token IDs.  T must be ≤ cfg.seq_len.  During cached generation
            this is typically a single token (T == 1).
        kv_cache : KVCache or None
            When provided, RoPE position offsets inside each attention layer
            are computed from the current cache length automatically.  The
            cache is updated in place at each attention layer.  Pass None
            during training (no overhead, identical behavior).

        Returns
        -------
        logits : FloatTensor[B, T, vocab_size]
        """
        B, T = idx.shape

        x = self.token_embedding(idx)
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache.layers[i] if kv_cache is not None else None
            x = block(x, layer_cache=layer_cache)
        x = self.ln_f(x)
        return self.lm_head(x)
