"""Tests for src/scheduler.py.

All tests run on CPU with a tiny synthetic optimizer — no GPU, no data.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.optim.lr_scheduler as lr_sched

from src.config import TrainConfig
from src.scheduler import make_scheduler


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> TrainConfig:
    return TrainConfig(
        learning_rate=1e-3,
        warmup_steps=10,
        max_steps=100,
        # Keep other params minimal
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        vocab_size=256,
    )


@pytest.fixture
def optimizer(cfg: TrainConfig) -> torch.optim.Optimizer:
    param = torch.nn.Parameter(torch.zeros(4))
    return torch.optim.AdamW([param], lr=cfg.learning_rate)


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def _step_to(scheduler: lr_sched.LambdaLR, optimizer: torch.optim.Optimizer, step: int) -> float:
    """Advance scheduler to *step* and return the LR at that step."""
    for _ in range(step):
        scheduler.step()
    return _get_lr(optimizer)


# ── Return type contract ──────────────────────────────────────────────────────


def test_returns_lambda_lr(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
    """make_scheduler must return a LambdaLR instance."""
    scheduler = make_scheduler(optimizer, cfg)
    assert isinstance(scheduler, lr_sched.LambdaLR)


# ── Schedule shape contracts (CLAUDE.md rule 14) ──────────────────────────────


def test_lr_at_step_0_is_zero(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
    """LR at step 0 must be 0 (warmup starts from 0)."""
    make_scheduler(optimizer, cfg)
    # LambdaLR initialises with last_epoch=-1 and calls step() once on
    # construction, which sets last_epoch=0 and applies lambda(0).
    assert _get_lr(optimizer) == pytest.approx(0.0, abs=1e-9)


def test_lr_peaks_at_warmup_end(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
    """LR at step warmup_steps must equal cfg.learning_rate (the peak)."""
    scheduler = make_scheduler(optimizer, cfg)
    lr = _step_to(scheduler, optimizer, cfg.warmup_steps)
    assert lr == pytest.approx(cfg.learning_rate, rel=1e-6)


def test_lr_near_zero_at_max_steps(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
    """LR at step max_steps must be 0 (cos(π) = -1 → factor = 0)."""
    scheduler = make_scheduler(optimizer, cfg)
    lr = _step_to(scheduler, optimizer, cfg.max_steps)
    assert lr == pytest.approx(0.0, abs=1e-9)


def test_warmup_is_monotone_increasing(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
    """LR must increase strictly at every step during the warmup phase."""
    scheduler = make_scheduler(optimizer, cfg)
    lrs = []
    for _ in range(cfg.warmup_steps + 1):
        lrs.append(_get_lr(optimizer))
        scheduler.step()
    # Exclude step 0 (LR=0) when checking increases
    for i in range(1, len(lrs) - 1):
        assert lrs[i] > lrs[i - 1], f"LR not increasing at warmup step {i}"


def test_cosine_is_monotone_decreasing(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
    """LR must decrease strictly at every step after warmup ends."""
    scheduler = make_scheduler(optimizer, cfg)
    # Advance through warmup first
    for _ in range(cfg.warmup_steps):
        scheduler.step()
    # Collect cosine phase LRs
    lrs = []
    for _ in range(cfg.max_steps - cfg.warmup_steps + 1):
        lrs.append(_get_lr(optimizer))
        scheduler.step()
    for i in range(1, len(lrs)):
        assert lrs[i] < lrs[i - 1], f"LR not decreasing at cosine step {i}"


def test_mid_warmup_lr_is_proportional(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
    """LR at the midpoint of warmup must equal half the peak LR."""
    scheduler = make_scheduler(optimizer, cfg)
    mid = cfg.warmup_steps // 2
    lr = _step_to(scheduler, optimizer, mid)
    expected = cfg.learning_rate * mid / cfg.warmup_steps
    assert lr == pytest.approx(expected, rel=1e-6)


def test_mid_cosine_lr_is_half_peak(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
    """LR at the cosine midpoint (halfway between warmup end and max_steps) ≈ 0.5 * peak."""
    scheduler = make_scheduler(optimizer, cfg)
    cosine_steps = cfg.max_steps - cfg.warmup_steps
    mid_cosine = cfg.warmup_steps + cosine_steps // 2
    lr = _step_to(scheduler, optimizer, mid_cosine)
    # At cosine midpoint progress=0.5: 0.5*(1+cos(π*0.5)) = 0.5*(1+0) = 0.5
    expected = cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * 0.5))
    assert lr == pytest.approx(expected, rel=1e-4)


# ── Guard (CONTRIBUTING.md) ───────────────────────────────────────────────────


def test_raises_if_warmup_equals_max_steps(cfg: TrainConfig) -> None:
    """ValueError when warmup_steps == max_steps (no cosine phase).

    TrainConfig.__post_init__ validates this eagerly, so the error is raised
    at config construction time rather than inside make_scheduler.
    """
    with pytest.raises(ValueError):
        TrainConfig(
            warmup_steps=cfg.max_steps,
            max_steps=cfg.max_steps,
            n_layers=2, d_model=64, n_heads=2, d_ff=128, vocab_size=256,
        )


def test_raises_if_warmup_exceeds_max_steps(cfg: TrainConfig) -> None:
    """ValueError when warmup_steps > max_steps.

    TrainConfig.__post_init__ validates this eagerly, so the error is raised
    at config construction time rather than inside make_scheduler.
    """
    with pytest.raises(ValueError):
        TrainConfig(
            warmup_steps=cfg.max_steps + 1,
            max_steps=cfg.max_steps,
            n_layers=2, d_model=64, n_heads=2, d_ff=128, vocab_size=256,
        )


# ── State save / load (CLAUDE.md rule 11) ─────────────────────────────────────


def test_scheduler_state_save_load(optimizer: torch.optim.Optimizer, cfg: TrainConfig) -> None:
    """Saving and loading state_dict must reproduce identical LR on the next step."""
    torch.manual_seed(42)
    scheduler = make_scheduler(optimizer, cfg)

    # Advance partway through training
    mid = cfg.warmup_steps + (cfg.max_steps - cfg.warmup_steps) // 3
    for _ in range(mid):
        scheduler.step()

    # Capture state
    sched_state = scheduler.state_dict()
    opt_state = optimizer.state_dict()
    lr_before = _get_lr(optimizer)

    # One more step on original
    scheduler.step()
    lr_after_original = _get_lr(optimizer)

    # Restore and replay the same step
    param2 = torch.nn.Parameter(torch.zeros(4))
    opt2 = torch.optim.AdamW([param2], lr=cfg.learning_rate)
    sched2 = make_scheduler(opt2, cfg)
    opt2.load_state_dict(opt_state)
    sched2.load_state_dict(sched_state)

    assert _get_lr(opt2) == pytest.approx(lr_before, rel=1e-9)
    sched2.step()
    assert _get_lr(opt2) == pytest.approx(lr_after_original, rel=1e-9)
