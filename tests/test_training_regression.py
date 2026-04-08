"""Training regression check — 100 steps on a tiny synthetic dataset.

Validates that the training loop produces a measurable loss decrease after
100 steps. Catches silent regressions in the optimizer, scheduler, or loss
computation. Marked as 'slow' so it runs only in the dedicated CI workflow
(training-check.yml) and not in the default test suite.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.config import TrainConfig
from src.train import train


def _cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
        batch_size=2,
        max_steps=100,
        warmup_steps=10,
        learning_rate=1e-3,
        weight_decay=0.1,
        grad_clip=1.0,
        checkpoint_every=10_000,
        plot_every=10_000,
        grad_log_every=50,
        weight_log_every=50,
        grad_norm_warn_threshold=100.0,
        checkpoint_dir=str(tmp_path / "ckpts"),
        plot_dir=str(tmp_path / "plots"),
        log_file=str(tmp_path / "train.log"),
        device="cpu",
    )


def _token_stream() -> list[int]:
    """Deterministic cycling token stream — 25 600 tokens."""
    return list(range(256)) * 100


@pytest.mark.slow
def test_training_regression_loss_decreases(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """100-step training run must end with final loss strictly less than initial loss."""
    cfg = _cfg(tmp_path)
    torch.manual_seed(42)
    train(cfg, token_stream=_token_stream())

    err = capsys.readouterr().err
    losses: list[float] = []
    for line in err.splitlines():
        if not line.startswith("step="):
            continue
        kv = dict(pair.split("=") for pair in line.split())
        losses.append(float(kv["loss"]))

    assert len(losses) == 100, f"Expected 100 logged steps, got {len(losses)}"
    assert (
        losses[-1] < losses[0]
    ), f"Regression: final loss {losses[-1]:.4f} not less than initial {losses[0]:.4f}"
    # Sanity bound: after 100 steps loss should be below random-guessing ceiling
    random_loss = math.log(cfg.vocab_size)  # ln(256) ≈ 5.55
    assert losses[-1] < random_loss + 0.5, (
        f"Loss {losses[-1]:.4f} exceeds random-guessing bound "
        f"{random_loss + 0.5:.4f} — optimizer may be broken"
    )
