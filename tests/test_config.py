"""Tests for src/config.py — TrainConfig.__post_init__ validation.

All invalid-config cases must raise ValueError at construction time (fail fast),
not deep inside a forward pass or optimizer step.
"""

from __future__ import annotations

import pytest

from src.config import TrainConfig


def _valid_kwargs() -> dict:
    """Minimal valid config suitable for tests."""
    return dict(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
        batch_size=2,
        max_steps=10,
        warmup_steps=2,
        grad_clip=1.0,
        weight_decay=0.1,
    )


def test_valid_config_constructs() -> None:
    """A valid config must not raise."""
    cfg = TrainConfig(**_valid_kwargs())
    assert cfg.vocab_size == 256


def test_d_model_not_divisible_by_n_heads() -> None:
    kw = _valid_kwargs()
    kw["d_model"] = 64
    kw["n_heads"] = 3  # 64 % 3 != 0
    with pytest.raises(ValueError, match="d_model"):
        TrainConfig(**kw)


def test_warmup_steps_equals_max_steps() -> None:
    kw = _valid_kwargs()
    kw["warmup_steps"] = kw["max_steps"]
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainConfig(**kw)


def test_warmup_steps_exceeds_max_steps() -> None:
    kw = _valid_kwargs()
    kw["warmup_steps"] = kw["max_steps"] + 1
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainConfig(**kw)


def test_negative_warmup_steps() -> None:
    kw = _valid_kwargs()
    kw["warmup_steps"] = -1
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainConfig(**kw)


def test_zero_batch_size() -> None:
    kw = _valid_kwargs()
    kw["batch_size"] = 0
    with pytest.raises(ValueError, match="batch_size"):
        TrainConfig(**kw)


def test_zero_max_steps() -> None:
    kw = _valid_kwargs()
    kw["max_steps"] = 0
    with pytest.raises(ValueError, match="max_steps"):
        TrainConfig(**kw)


def test_zero_vocab_size() -> None:
    kw = _valid_kwargs()
    kw["vocab_size"] = 0
    with pytest.raises(ValueError, match="vocab_size"):
        TrainConfig(**kw)


def test_zero_seq_len() -> None:
    kw = _valid_kwargs()
    kw["seq_len"] = 0
    with pytest.raises(ValueError, match="seq_len"):
        TrainConfig(**kw)


def test_zero_grad_clip() -> None:
    kw = _valid_kwargs()
    kw["grad_clip"] = 0.0
    with pytest.raises(ValueError, match="grad_clip"):
        TrainConfig(**kw)


def test_negative_weight_decay() -> None:
    kw = _valid_kwargs()
    kw["weight_decay"] = -0.1
    with pytest.raises(ValueError, match="weight_decay"):
        TrainConfig(**kw)


def test_zero_weight_decay_allowed() -> None:
    """weight_decay=0 is valid (no regularization)."""
    kw = _valid_kwargs()
    kw["weight_decay"] = 0.0
    cfg = TrainConfig(**kw)
    assert cfg.weight_decay == 0.0


def test_zero_warmup_steps_allowed() -> None:
    """warmup_steps=0 is valid (no warmup ramp)."""
    kw = _valid_kwargs()
    kw["warmup_steps"] = 0
    cfg = TrainConfig(**kw)
    assert cfg.warmup_steps == 0


def test_adamw_defaults() -> None:
    """Explicit AdamW defaults must match canonical values."""
    cfg = TrainConfig(**_valid_kwargs())
    assert cfg.adamw_betas == (0.9, 0.999)
    assert cfg.adamw_eps == 1e-8


def test_compile_amp_defaults_off() -> None:
    """use_compile and use_amp must default to False."""
    cfg = TrainConfig(**_valid_kwargs())
    assert cfg.use_compile is False
    assert cfg.use_amp is False


def test_lr_mult_defaults() -> None:
    """ln_lr_mult and embed_lr_mult must have the expected defaults."""
    cfg = TrainConfig(**_valid_kwargs())
    assert cfg.ln_lr_mult == 3.0
    assert cfg.embed_lr_mult == 0.1


def test_zero_ln_lr_mult_raises() -> None:
    """ln_lr_mult=0 must raise ValueError."""
    kw = _valid_kwargs()
    kw["ln_lr_mult"] = 0.0
    with pytest.raises(ValueError, match="ln_lr_mult"):
        TrainConfig(**kw)


def test_negative_ln_lr_mult_raises() -> None:
    """ln_lr_mult < 0 must raise ValueError."""
    kw = _valid_kwargs()
    kw["ln_lr_mult"] = -1.0
    with pytest.raises(ValueError, match="ln_lr_mult"):
        TrainConfig(**kw)


def test_zero_embed_lr_mult_raises() -> None:
    """embed_lr_mult=0 must raise ValueError."""
    kw = _valid_kwargs()
    kw["embed_lr_mult"] = 0.0
    with pytest.raises(ValueError, match="embed_lr_mult"):
        TrainConfig(**kw)


def test_negative_embed_lr_mult_raises() -> None:
    """embed_lr_mult < 0 must raise ValueError."""
    kw = _valid_kwargs()
    kw["embed_lr_mult"] = -0.5
    with pytest.raises(ValueError, match="embed_lr_mult"):
        TrainConfig(**kw)
