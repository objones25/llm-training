"""Tests for src/model.py — GPT transformer.

All tests run on CPU with synthetic data. No network access.
Minimum viable shape: batch=2, seq_len=16, vocab=256 (rules 5, 13).
Every test that touches randomness sets its own seed (rule 15).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.config import TrainConfig
from src.model import GPT


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> TrainConfig:
    """Minimal config for fast CPU tests."""
    return TrainConfig(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
    )


@pytest.fixture
def model(cfg: TrainConfig) -> GPT:
    torch.manual_seed(42)
    return GPT(cfg)


# ── Shape / dtype contracts ───────────────────────────────────────────────────


def test_forward_pass_shape(model: GPT, cfg: TrainConfig) -> None:
    """Logits must be (B, T, vocab_size) with float32 dtype (rule 13)."""
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits = model(idx)
    assert logits.shape == (2, cfg.seq_len, cfg.vocab_size)
    assert logits.dtype == torch.float32


# ── Value contracts ───────────────────────────────────────────────────────────


def test_loss_finite_nonneg(model: GPT, cfg: TrainConfig) -> None:
    """Cross-entropy loss must be finite and non-negative (rule 6)."""
    torch.manual_seed(1)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_loss_monotonically_decreasing(cfg: TrainConfig) -> None:
    """Loss must decrease over 3 gradient steps on a fixed synthetic batch (rule 6)."""
    torch.manual_seed(42)
    m = GPT(cfg)
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-2)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))

    losses: list[float] = []
    for _ in range(3):
        optimizer.zero_grad()
        logits = m(idx)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[0] > losses[1] > losses[2], (
        f"Loss not monotonically decreasing: {losses}"
    )


def test_gradients_no_nan(model: GPT, cfg: TrainConfig) -> None:
    """Every named parameter must have non-None, non-zero, finite gradients (rule 10)."""
    torch.manual_seed(2)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))

    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"
        assert not torch.all(param.grad == 0), f"Gradient is all zeros for {name}"
        assert torch.isfinite(param.grad).all(), f"Gradient has nan/inf for {name}"


# ── Structural contracts (rule 21) ────────────────────────────────────────────


def test_weight_tying(model: GPT) -> None:
    """lm_head.weight must be the same tensor object as token_embedding.weight (rule 21)."""
    assert model.lm_head.weight is model.token_embedding.weight


def test_init_prints_param_count(cfg: TrainConfig, capsys: pytest.CaptureFixture) -> None:
    """__init__ must print the non-embedding param count (CONTRIBUTING.md guard)."""
    torch.manual_seed(3)
    GPT(cfg)
    captured = capsys.readouterr()
    assert "model non_embedding_params=" in captured.out


def test_non_embedding_param_count_value(cfg: TrainConfig, capsys: pytest.CaptureFixture) -> None:
    """Non-embedding N must match the analytic formula (rule 13).

    Per block:
        qkv:      3 * d_model^2
        out_proj: d_model^2
        fc1:      d_model * d_ff
        fc2:      d_ff * d_model
        ln_1:     d_model (weight) + d_model (bias)
        ln_2:     d_model (weight) + d_model (bias)

    Plus final LayerNorm ln_f: d_model (weight) + d_model (bias).
    lm_head.weight is weight-tied — not double-counted.
    """
    torch.manual_seed(4)
    GPT(cfg)
    captured = capsys.readouterr()

    expected = cfg.n_layers * (
        3 * cfg.d_model ** 2          # qkv
        + cfg.d_model ** 2            # out_proj
        + 2 * cfg.d_model * cfg.d_ff  # fc1 + fc2
        + 4 * cfg.d_model             # ln_1 weight+bias, ln_2 weight+bias
    ) + 2 * cfg.d_model               # ln_f weight+bias

    assert f"model non_embedding_params={expected:,}" in captured.out, (
        f"Expected {expected:,} non-embedding params. Got: {captured.out.strip()}"
    )


# ── Inference correctness ─────────────────────────────────────────────────────


def test_eval_deterministic(model: GPT, cfg: TrainConfig) -> None:
    """model.eval() must produce identical outputs on two identical inputs (rule 22)."""
    model.eval()
    torch.manual_seed(5)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    with torch.no_grad():
        out1 = model(idx)
        out2 = model(idx)
    assert torch.equal(out1, out2)


def test_causal_mask(model: GPT, cfg: TrainConfig) -> None:
    """Output at position t must not depend on tokens at positions > t."""
    model.eval()
    torch.manual_seed(6)
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len))

    with torch.no_grad():
        logits_orig = model(idx).clone()

    # Corrupt the second half of the sequence
    split = cfg.seq_len // 2
    idx_corrupted = idx.clone()
    idx_corrupted[:, split:] = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len - split))

    with torch.no_grad():
        logits_corrupted = model(idx_corrupted)

    # Positions before the corruption point must be unaffected
    assert torch.allclose(
        logits_orig[:, :split, :],
        logits_corrupted[:, :split, :],
        atol=1e-5,
    ), "Causal mask violated: future tokens affected earlier positions"


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_short_sequence(model: GPT, cfg: TrainConfig) -> None:
    """Forward pass must work for T=1 (T < seq_len), causal mask sliced correctly."""
    torch.manual_seed(7)
    idx = torch.randint(0, cfg.vocab_size, (2, 1))
    logits = model(idx)
    assert logits.shape == (2, 1, cfg.vocab_size)


def test_n_heads_assertion(cfg: TrainConfig) -> None:
    """GPT must raise AssertionError when d_model is not divisible by n_heads."""
    bad_cfg = TrainConfig(
        vocab_size=cfg.vocab_size,
        n_layers=cfg.n_layers,
        d_model=64,
        n_heads=3,  # 64 % 3 != 0
        d_ff=cfg.d_ff,
        seq_len=cfg.seq_len,
    )
    with pytest.raises(AssertionError):
        GPT(bad_cfg)
