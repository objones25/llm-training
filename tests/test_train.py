"""Tests for src/train.py.

All tests inject a synthetic cycling token_stream and a tiny 2-layer model
on CPU.  No GPU, no HuggingFace network calls, and no real dataset.

Rules covered
-------------
 6  Loss monotonically decreases over ≥3 steps on a fixed synthetic batch.
10  Gradients are non-None, non-zero, and finite after one backward pass.
18  Smoke: 10 steps on synthetic data, final loss < initial loss.
23  RuntimeError is raised (not silently ignored) when loss is nan/inf.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import torch
import torch.nn.functional

from src.config import TrainConfig
from src.model import GPT
from src.train import train


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cfg(tmp_path: Path, **overrides) -> TrainConfig:
    """Return a tiny TrainConfig suitable for unit tests."""
    defaults: dict = dict(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
        batch_size=2,
        max_steps=10,
        warmup_steps=2,
        learning_rate=1e-3,
        weight_decay=0.1,
        grad_clip=1.0,
        checkpoint_every=1000,   # no checkpoint by default
        plot_every=1000,          # no plots by default
        grad_log_every=5,
        weight_log_every=5,
        grad_norm_warn_threshold=100.0,
        checkpoint_dir=str(tmp_path / "ckpts"),
        plot_dir=str(tmp_path / "plots"),
        log_file=str(tmp_path / "train.log"),
        device="cpu",
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _token_stream() -> list[int]:
    """10 240 tokens cycling through 0..255 — deterministic, no seed needed."""
    return list(range(256)) * 40


def _fixed_batch_stream(cfg: TrainConfig) -> itertools.cycle:
    """Cycle exactly one batch worth of tokens so every step is identical."""
    tokens_per_batch = cfg.batch_size * (cfg.seq_len + 1)  # 2 * 17 = 34
    return itertools.cycle(range(tokens_per_batch))


def _parse_step_losses(stdout: str) -> list[float]:
    losses = []
    for line in stdout.splitlines():
        if not line.startswith("step="):
            continue
        kv = dict(pair.split("=") for pair in line.split())
        losses.append(float(kv["loss"]))
    return losses


# ── Return type ───────────────────────────────────────────────────────────────


def test_train_returns_gpt(tmp_path: Path) -> None:
    """train() must return a GPT instance."""
    cfg = _cfg(tmp_path)
    torch.manual_seed(0)
    result = train(cfg, token_stream=_token_stream())
    assert isinstance(result, GPT)


# ── Smoke test (rule 18) ──────────────────────────────────────────────────────


def test_smoke_10_steps_loss_decreases(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Rule 18: 10-step smoke run on synthetic data must end with loss < initial."""
    cfg = _cfg(tmp_path, max_steps=10)
    torch.manual_seed(0)
    train(cfg, token_stream=_token_stream())
    losses = _parse_step_losses(capsys.readouterr().err)
    assert len(losses) == 10, f"Expected 10 logged steps, got {len(losses)}"
    assert losses[-1] < losses[0], (
        f"Final loss {losses[-1]:.4f} not less than initial {losses[0]:.4f}"
    )


# ── Monotone decrease on fixed batch (rule 6) ────────────────────────────────


def test_loss_monotone_fixed_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Rule 6: loss must decrease monotonically over ≥3 steps on a fixed batch.

    warmup_steps=0 gives full LR from step 0 (no warmup plateau). A cycling
    stream of exactly one batch worth of tokens ensures identical inputs/targets
    at every step so any decrease is purely due to gradient descent.

    learning_rate=1e-3 is chosen deliberately: with GPT-2 style initialization
    the model starts near ln(vocab_size) ≈ 5.5, so lr=1e-2 overshoots and
    breaks monotonicity. 1e-3 stays comfortably in the convergent regime.
    """
    cfg = _cfg(
        tmp_path,
        max_steps=5,
        warmup_steps=0,
        learning_rate=1e-3,
    )
    torch.manual_seed(0)
    train(cfg, token_stream=_fixed_batch_stream(cfg))
    losses = _parse_step_losses(capsys.readouterr().err)
    assert len(losses) >= 3
    assert losses[0] > losses[1] > losses[2], (
        f"Expected strict decrease for first 3 steps, got {losses[:3]}"
    )


# ── Gradient sanity (rule 10) ─────────────────────────────────────────────────


def test_gradients_nonzero_no_nan(tmp_path: Path) -> None:
    """Rule 10: after one backward pass, every named parameter must have a
    non-None, non-zero, finite gradient."""
    cfg = _cfg(tmp_path, max_steps=1, warmup_steps=0)
    torch.manual_seed(0)
    # Gradients remain populated after the loop — zero_grad() is only called
    # at the *start* of each iteration, never at the end.
    model = train(cfg, token_stream=_token_stream())
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for '{name}'"
        assert p.grad.abs().sum().item() > 0.0, f"Zero gradient for '{name}'"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient for '{name}'"


# ── NaN loss raises RuntimeError (rule 23) ────────────────────────────────────


def test_nan_loss_raises_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 23: a nan loss must raise RuntimeError immediately, not be silently
    continued."""
    monkeypatch.setattr(
        torch.nn.functional,
        "cross_entropy",
        lambda *args, **kwargs: torch.tensor(float("nan")),
    )
    cfg = _cfg(tmp_path)
    torch.manual_seed(0)
    with pytest.raises(RuntimeError, match="nan"):
        train(cfg, token_stream=_token_stream())


# ── Checkpointing cadence ─────────────────────────────────────────────────────


def test_checkpoint_created_at_interval(tmp_path: Path) -> None:
    """A checkpoint file must be created after every checkpoint_every steps."""
    cfg = _cfg(tmp_path, max_steps=6, checkpoint_every=3)
    torch.manual_seed(0)
    train(cfg, token_stream=_token_stream())
    ckpt_dir = Path(cfg.checkpoint_dir)
    assert (ckpt_dir / "checkpoint_0000003.pt").exists(), (
        "Expected checkpoint at step 3"
    )


# ── Plots cadence ─────────────────────────────────────────────────────────────


def test_plots_created(tmp_path: Path) -> None:
    """All six plot files must exist and be non-empty after at least one
    plot_every cadence step."""
    cfg = _cfg(
        tmp_path,
        max_steps=5,
        plot_every=1,
        grad_log_every=1,
        weight_log_every=1,
    )
    torch.manual_seed(0)
    train(cfg, token_stream=_token_stream())
    plot_dir = Path(cfg.plot_dir)
    for fname in (
        "loss.png",
        "lr.png",
        "grad_norm.png",
        "grad_heatmap.png",
        "weight_norm.png",
        "grad_hist.png",
    ):
        p = plot_dir / fname
        assert p.exists(), f"Missing plot file: {fname}"
        assert p.stat().st_size > 0, f"Empty plot file: {fname}"
