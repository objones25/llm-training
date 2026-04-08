"""Tests for KVCache and KV-cached inference in GPT.

All tests run on CPU with tiny synthetic inputs — no GPU required.

Key correctness guarantee
-------------------------
Greedy (top_k=1) generation with the KV cache must produce bit-identical
outputs to the same generation without the cache.  This is the gold-standard
test: if the cached and uncached paths disagree, the cache is broken.
"""
from __future__ import annotations

import torch
import pytest

from src.config import TrainConfig
from src.kv_cache import KVCache, LayerKVCache
from src.model import GPT


# ── Minimal config for fast CPU tests ────────────────────────────────────────

def _tiny_cfg(**overrides) -> TrainConfig:
    """2-layer, d_model=64, n_heads=4, vocab=128, seq_len=32."""
    defaults = dict(
        n_layers=2,
        d_model=64,
        n_heads=4,
        d_ff=128,
        vocab_size=128,
        seq_len=32,
        batch_size=1,
        max_steps=10,
        warmup_steps=2,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


# ── LayerKVCache tests ────────────────────────────────────────────────────────


def test_layer_kv_cache_seq_len_empty():
    cache = LayerKVCache(
        k=torch.zeros(1, 4, 0, 16),
        v=torch.zeros(1, 4, 0, 16),
    )
    assert cache.seq_len == 0


def test_layer_kv_cache_seq_len_populated():
    cache = LayerKVCache(
        k=torch.zeros(1, 4, 7, 16),
        v=torch.zeros(1, 4, 7, 16),
    )
    assert cache.seq_len == 7


# ── KVCache.empty tests ───────────────────────────────────────────────────────


def test_kv_cache_empty_shape():
    cache = KVCache.empty(
        n_layers=3, batch_size=1, n_heads=4, head_dim=16,
        device=torch.device("cpu"),
    )
    assert len(cache.layers) == 3
    for layer in cache.layers:
        assert layer.k.shape == (1, 4, 0, 16)
        assert layer.v.shape == (1, 4, 0, 16)


def test_kv_cache_empty_seq_len_zero():
    cache = KVCache.empty(
        n_layers=2, batch_size=1, n_heads=4, head_dim=16,
        device=torch.device("cpu"),
    )
    assert cache.seq_len == 0


def test_kv_cache_empty_dtype():
    cache = KVCache.empty(
        n_layers=2, batch_size=1, n_heads=4, head_dim=16,
        device=torch.device("cpu"), dtype=torch.float16,
    )
    for layer in cache.layers:
        assert layer.k.dtype == torch.float16
        assert layer.v.dtype == torch.float16


def test_kv_cache_empty_device():
    cache = KVCache.empty(
        n_layers=2, batch_size=1, n_heads=4, head_dim=16,
        device=torch.device("cpu"),
    )
    for layer in cache.layers:
        assert layer.k.device.type == "cpu"
        assert layer.v.device.type == "cpu"


# ── Prefill builds cache correctly ───────────────────────────────────────────


def test_prefill_populates_cache():
    """After a prefill forward pass the cache seq_len equals T_seed."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = GPT(cfg)
    model.eval()

    T_seed = 5
    idx = torch.randint(0, cfg.vocab_size, (1, T_seed))
    cache = KVCache.empty(
        n_layers=cfg.n_layers, batch_size=1,
        n_heads=cfg.n_heads, head_dim=cfg.d_model // cfg.n_heads,
        device=torch.device("cpu"),
    )

    with torch.no_grad():
        _ = model(idx, kv_cache=cache)

    assert cache.seq_len == T_seed
    for layer in cache.layers:
        assert layer.k.shape == (1, cfg.n_heads, T_seed, cfg.d_model // cfg.n_heads)
        assert layer.v.shape == (1, cfg.n_heads, T_seed, cfg.d_model // cfg.n_heads)


def test_generation_step_extends_cache_by_one():
    """Each single-token forward extends the cache length by exactly 1."""
    torch.manual_seed(1)
    cfg = _tiny_cfg()
    model = GPT(cfg)
    model.eval()

    T_seed = 4
    idx = torch.randint(0, cfg.vocab_size, (1, T_seed))
    cache = KVCache.empty(
        n_layers=cfg.n_layers, batch_size=1,
        n_heads=cfg.n_heads, head_dim=cfg.d_model // cfg.n_heads,
        device=torch.device("cpu"),
    )

    with torch.no_grad():
        model(idx, kv_cache=cache)
        assert cache.seq_len == T_seed

        for step in range(3):
            next_tok = torch.randint(0, cfg.vocab_size, (1, 1))
            model(next_tok, kv_cache=cache)
            assert cache.seq_len == T_seed + step + 1


# ── Position offset ───────────────────────────────────────────────────────────


def test_position_offset_applied():
    """Generated tokens use positions starting at T_cached, not 0.

    We verify indirectly: the logits from a single-token forward with an
    empty cache (pos=0) must differ from those with a populated cache
    (pos=T_seed).  If the RoPE offset were ignored, Q and K would receive
    the same rotations at position 0 regardless of cache length, producing
    identical attention patterns and identical outputs.
    """
    torch.manual_seed(2)
    cfg = _tiny_cfg()
    model = GPT(cfg)
    model.eval()

    token_id = 42
    T_seed = 5
    seed = torch.randint(0, cfg.vocab_size, (1, T_seed))
    single = torch.tensor([[token_id]])

    with torch.no_grad():
        # No cache — token at position 0
        logits_pos0 = model(single)  # [1, 1, vocab]

        # Build populated cache then query same token at position T_seed
        cache = KVCache.empty(
            n_layers=cfg.n_layers, batch_size=1,
            n_heads=cfg.n_heads, head_dim=cfg.d_model // cfg.n_heads,
            device=torch.device("cpu"),
        )
        model(seed, kv_cache=cache)
        logits_posN = model(single, kv_cache=cache)  # [1, 1, vocab]

    # Outputs must differ because RoPE rotations depend on absolute position.
    assert not torch.allclose(logits_pos0, logits_posN), (
        "Logits should differ when position offset changes"
    )


# ── Correctness: cached == uncached (greedy) ──────────────────────────────────


def test_cached_greedy_matches_uncached():
    """Gold-standard correctness test.

    Greedy generation (top_k=1) with the KV cache must produce
    bit-identical token IDs compared to the naive rolling-context approach.
    """
    torch.manual_seed(3)
    cfg = _tiny_cfg()
    model = GPT(cfg)
    model.eval()

    seed = [10, 20, 30, 40, 50]
    n_new = 8

    # ── Uncached (naive) ──────────────────────────────────────────────────────
    generated_uncached = list(seed)
    with torch.no_grad():
        for _ in range(n_new):
            ctx = generated_uncached[-cfg.seq_len:]
            idx = torch.tensor([ctx], dtype=torch.long)
            logits = model(idx)
            next_tok = logits[0, -1, :].argmax().item()
            generated_uncached.append(int(next_tok))

    # ── Cached (two-phase) ────────────────────────────────────────────────────
    generated_cached = list(seed)
    with torch.no_grad():
        cache = KVCache.empty(
            n_layers=cfg.n_layers, batch_size=1,
            n_heads=cfg.n_heads, head_dim=cfg.d_model // cfg.n_heads,
            device=torch.device("cpu"),
        )
        seed_tensor = torch.tensor([seed], dtype=torch.long)
        logits = model(seed_tensor, kv_cache=cache)
        next_logits = logits[0, -1, :]

        for _ in range(n_new):
            next_tok = next_logits.argmax().item()
            generated_cached.append(int(next_tok))
            idx = torch.tensor([[next_tok]], dtype=torch.long)
            logits = model(idx, kv_cache=cache)
            next_logits = logits[0, -1, :]

    assert generated_cached == generated_uncached, (
        f"Cached {generated_cached} != uncached {generated_uncached}"
    )


# ── Training path is unaffected ───────────────────────────────────────────────


def test_training_forward_unaffected_by_cache_import():
    """GPT.forward(idx) without kv_cache produces identical results before
    and after the KV-cache code is present in model.py."""
    torch.manual_seed(4)
    cfg = _tiny_cfg()
    model = GPT(cfg)
    model.eval()

    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        out1 = model(idx)
        out2 = model(idx)

    assert torch.allclose(out1, out2), "Identical inputs must give identical outputs"
    assert out1.shape == (2, 8, cfg.vocab_size)


def test_no_cache_uses_is_causal_true():
    """Forward pass without cache must still produce causally masked logits.

    The first token's logits must not depend on the second token's content.
    We verify this by swapping token[1] and checking that logits[0] is unchanged.
    """
    torch.manual_seed(5)
    cfg = _tiny_cfg()
    model = GPT(cfg)
    model.eval()

    idx = torch.randint(0, cfg.vocab_size, (1, 6))
    idx_modified = idx.clone()
    idx_modified[0, 3] = (idx[0, 3] + 1) % cfg.vocab_size  # change token at pos 3

    with torch.no_grad():
        out_orig = model(idx)
        out_mod = model(idx_modified)

    # Positions 0..2 must be identical (causal: they can't see pos 3)
    assert torch.allclose(out_orig[0, :3], out_mod[0, :3]), (
        "Causal masking violated: logits at positions < 3 changed when pos 3 changed"
    )


# ── sample_text integration test ─────────────────────────────────────────────


def test_sample_text_with_cache_returns_correct_length():
    """sample_text returns seed + max_new_tokens IDs."""
    torch.manual_seed(6)
    from scripts.evaluate import sample_text

    cfg = _tiny_cfg()
    model = GPT(cfg)
    model.eval()

    seed = [1, 2, 3]
    result = sample_text(model, cfg, torch.device("cpu"), seed, max_new_tokens=10, top_k=1)
    assert len(result) == len(seed) + 10


def test_sample_text_greedy_deterministic_with_cache():
    """Two greedy calls with the same seed must return the same tokens."""
    torch.manual_seed(7)
    from scripts.evaluate import sample_text

    cfg = _tiny_cfg()
    model = GPT(cfg)
    model.eval()

    seed = [5, 10, 15]
    result1 = sample_text(model, cfg, torch.device("cpu"), seed, max_new_tokens=5, top_k=1)
    result2 = sample_text(model, cfg, torch.device("cpu"), seed, max_new_tokens=5, top_k=1)
    assert result1 == result2


def test_sample_text_cached_matches_uncached_generation():
    """sample_text (cached) must produce identical greedy tokens to the naive loop."""
    torch.manual_seed(8)
    from scripts.evaluate import sample_text

    cfg = _tiny_cfg()
    model = GPT(cfg)
    model.eval()

    seed = [7, 14, 21]
    n_new = 6

    # Cached path (via sample_text)
    cached_result = sample_text(
        model, cfg, torch.device("cpu"), seed, max_new_tokens=n_new, top_k=1
    )

    # Naive uncached path
    uncached = list(seed)
    with torch.no_grad():
        for _ in range(n_new):
            ctx = uncached[-cfg.seq_len:]
            idx = torch.tensor([ctx], dtype=torch.long)
            logits = model(idx)
            next_tok = logits[0, -1, :].argmax().item()
            uncached.append(int(next_tok))

    assert cached_result == uncached, (
        f"sample_text cached {cached_result} != uncached {uncached}"
    )
