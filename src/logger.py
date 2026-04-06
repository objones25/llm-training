"""Gradient and weight norm logging for LM pretraining.

Provides ``GradientLogger``, a stateless logger called by ``train.py`` every
step.  All formatting and output lives here -- ``train.py`` never calls
``print`` directly.

Output contracts (from CLAUDE.md)
----------------------------------
Every step -- single line to stdout::

    step=100 loss=3.4821 lr=0.000287 grad_norm=1.2341 grad_norm_min=0.0012 grad_norm_max=4.3210

Six fields, ``key=value`` format, space-separated.  If any layer's gradient
norm exceeds ``cfg.grad_norm_warn_threshold``, an additional WARNING line
follows immediately::

    WARNING step=100 layer=transformer.block.5.attn.q_proj grad_norm=14.3201 exceeds threshold=10.0

Every ``cfg.grad_log_every`` steps -- per-layer gradient norms::

    grad step=100 layer=transformer.block.0.attn.q_proj norm=0.3821

Every ``cfg.weight_log_every`` steps -- per-layer weight norms::

    weight step=500 layer=transformer.block.0.attn.q_proj norm=1.2341

Public API
----------
    GradientLogger(cfg: TrainConfig)
        .log_step(step, loss, lr, model)
        .log_layers(step, model)
"""
from __future__ import annotations

import math

import torch.nn as nn

from src.config import TrainConfig


class GradientLogger:
    """Logs per-step summaries and per-layer gradient/weight norms to stdout."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        model: nn.Module,
    ) -> None:
        """Emit the per-step summary line, plus WARNING lines for any layer
        whose gradient norm exceeds cfg.grad_norm_warn_threshold.

        Parameters
        ----------
        step : int
        loss : float
        lr : float
        model : nn.Module
            Must have gradients populated (i.e. called after loss.backward()).
        """
        layer_norms: dict[str, float] = {
            name: p.grad.norm().item()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

        if layer_norms:
            norms = list(layer_norms.values())
            grad_norm = math.sqrt(sum(n * n for n in norms))
            grad_norm_min = min(norms)
            grad_norm_max = max(norms)
        else:
            grad_norm = grad_norm_min = grad_norm_max = 0.0

        print(
            f"step={step} loss={loss:.4f} lr={lr:.6f} "
            f"grad_norm={grad_norm:.4f} "
            f"grad_norm_min={grad_norm_min:.4f} "
            f"grad_norm_max={grad_norm_max:.4f}"
        )

        threshold = self.cfg.grad_norm_warn_threshold
        for name, norm in layer_norms.items():
            if norm > threshold:
                print(
                    f"WARNING step={step} layer={name} "
                    f"grad_norm={norm:.4f} exceeds threshold={threshold}"
                )

    def log_layers(
        self,
        step: int,
        model: nn.Module,
    ) -> None:
        """Emit per-layer gradient norms and/or weight norms at configured cadences.

        Parameters
        ----------
        step : int
        model : nn.Module
        """
        if step % self.cfg.grad_log_every == 0:
            for name, p in model.named_parameters():
                if p.grad is not None:
                    print(f"grad step={step} layer={name} norm={p.grad.norm().item():.4f}")

        if step % self.cfg.weight_log_every == 0:
            for name, p in model.named_parameters():
                print(f"weight step={step} layer={name} norm={p.norm().item():.4f}")
