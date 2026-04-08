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
    optimizer = torch.optim.Adam(m.parameters(), lr=3e-3)
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

    assert (
        losses[0] > losses[1] > losses[2]
    ), f"Loss not monotonically decreasing: {losses}"


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


def test_init_stores_param_count(cfg: TrainConfig) -> None:
    """GPT.__init__ must store non-embedding param count as model.n_params (rule 21)."""
    torch.manual_seed(3)
    model = GPT(cfg)
    assert hasattr(model, "n_params"), "GPT must expose n_params attribute"
    assert isinstance(model.n_params, int)
    assert model.n_params > 0


def test_non_embedding_param_count_value(cfg: TrainConfig) -> None:
    """Non-embedding N must match the analytic formula (rule 13).

    Per block:
        qkv:      3 * d_model^2
        out_proj: d_model^2
        fc1:      d_model * d_ff
        fc2:      d_ff * d_model
        ln_1:     d_model (weight only — RMSNorm has no bias)
        ln_2:     d_model (weight only — RMSNorm has no bias)

    Plus final RMSNorm ln_f: d_model (weight only).
    lm_head.weight is weight-tied — not double-counted.
    """
    torch.manual_seed(4)
    model = GPT(cfg)

    expected = (
        cfg.n_layers
        * (
            3 * cfg.d_model**2  # qkv
            + cfg.d_model**2  # out_proj
            + 2 * cfg.d_model * cfg.d_ff  # fc1 + fc2
            + 2 * cfg.d_model  # ln_1 weight, ln_2 weight (no bias)
        )
        + cfg.d_model
    )  # ln_f weight (no bias)

    assert (
        model.n_params == expected
    ), f"Expected {expected:,} non-embedding params, got {model.n_params:,}"


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
    idx_corrupted[:, split:] = torch.randint(
        0, cfg.vocab_size, (1, cfg.seq_len - split)
    )

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


def test_init_loss_near_uniform(cfg: TrainConfig) -> None:
    """Step-0 loss must be within 5% of ln(vocab_size).

    GPT-2 style initialization (std=0.02 + residual scaling) should produce
    near-uniform output distributions at init. If this fails, the model is
    confidently wrong before training starts, wasting the entire warmup phase.
    """
    import math

    torch.manual_seed(0)
    m = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (2, cfg.seq_len))
    logits = m(idx)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) / expected < 0.05, (
        f"Initial loss {loss.item():.4f} is not within 5% of "
        f"ln(vocab_size)={expected:.4f}. Check weight initialization."
    )


def test_n_heads_assertion(cfg: TrainConfig) -> None:
    """ValueError must be raised when d_model is not divisible by n_heads.

    __post_init__ in TrainConfig validates this at construction time, so the
    error is raised before GPT() is ever called.
    """
    with pytest.raises(ValueError, match="d_model"):
        TrainConfig(
            vocab_size=cfg.vocab_size,
            n_layers=cfg.n_layers,
            d_model=64,
            n_heads=3,  # 64 % 3 != 0
            d_ff=cfg.d_ff,
            seq_len=cfg.seq_len,
        )
