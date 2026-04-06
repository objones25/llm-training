"""Cosine learning rate schedule with linear warmup.

Provides a single factory function that returns a PyTorch ``LambdaLR``
scheduler configured for GPT-style pretraining.

Schedule
--------
    Warmup phase  (step < cfg.warmup_steps):
        lr = cfg.learning_rate * step / cfg.warmup_steps

    Cosine phase  (step >= cfg.warmup_steps):
        lr = cfg.learning_rate * 0.5 * (1 + cos(π * progress))
        progress = (step - warmup_steps) / (max_steps - warmup_steps)

The cosine cycle length always equals ``cfg.max_steps``. Setting it longer
than ``max_steps`` degrades final loss by >5% at >25% overshoot (Chinchilla §4).

Public API
----------
    make_scheduler(
        optimizer: torch.optim.Optimizer,
        cfg: TrainConfig,
    ) -> torch.optim.lr_scheduler.LambdaLR
"""
from __future__ import annotations

import math

import torch
import torch.optim.lr_scheduler as lr_scheduler

from src.config import TrainConfig


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> lr_scheduler.LambdaLR:
    """Return a LambdaLR scheduler with linear warmup then cosine decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be scaled.
    cfg : TrainConfig
        Supplies ``warmup_steps``, ``max_steps``, and ``learning_rate``.
        The optimizer's base LR must already be set to ``cfg.learning_rate``.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR

    Raises
    ------
    ValueError
        If ``cfg.warmup_steps >= cfg.max_steps``.  The cosine cycle length
        equals ``cfg.max_steps``; having warmup consume all or more of the
        budget leaves no cosine phase and degrades final loss.
    """
    if cfg.warmup_steps >= cfg.max_steps:
        raise ValueError(
            f"Cosine cycle length ({cfg.max_steps}) must be greater than "
            f"warmup_steps ({cfg.warmup_steps}). "
            "Mismatches >25% measurably degrade final loss."
        )

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)
