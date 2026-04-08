"""Gradient and training-step logger.

All training output routes through this module.

Console (INFO and above)
    One terse summary line per step, plus WARNING lines when a layer
    gradient norm exceeds the configured threshold.

File (DEBUG and above)
    Everything — per-layer gradient norms at ``grad_log_every`` cadence,
    per-layer weight norms at ``weight_log_every`` cadence, the step
    summary, and WARNING lines.

Setup
-----
Call ``configure_logging(cfg)`` once from ``train.py`` before the training
loop.  Tests do *not* call ``configure_logging``; pytest's ``caplog``
fixture captures records through normal propagation to the root logger.
"""

from __future__ import annotations

import logging
import math

import torch.nn as nn

from src.config import TrainConfig

_log = logging.getLogger("llm_training")


def configure_logging(cfg: TrainConfig) -> None:
    """Attach console and file handlers.  Call once before training starts."""
    _log.setLevel(logging.DEBUG)
    _log.propagate = False  # Prevent double-printing via root logger.

    fmt = logging.Formatter("%(message)s")

    # Console: terse — step summaries and warnings only.
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    _log.addHandler(console)

    # File: verbose — everything including per-layer grad/weight lines.
    if cfg.log_file:
        fh = logging.FileHandler(cfg.log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        _log.addHandler(fh)


class GradientLogger:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        layer_norms: dict[str, float],
    ) -> None:
        """Emit one terse summary line (INFO — console + file).

        Parameters
        ----------
        layer_norms:
            Pre-clip per-layer gradient norms keyed by parameter name.
            Must be captured *before* ``clip_grad_norm_()`` is called so
            the logged values reflect true gradient magnitudes.

        The summary line includes ``grad_norm_max_layer`` — the name of the
        single layer contributing the highest norm — so spikes can be
        attributed without waiting for the per-layer cadence.

        If the total gradient norm exceeds ``cfg.grad_norm_spike_threshold``,
        a full per-layer breakdown is written immediately at DEBUG level
        (prefix ``spike``), regardless of ``grad_log_every`` cadence.
        """
        if layer_norms:
            norms = list(layer_norms.values())
            grad_norm = math.sqrt(sum(n * n for n in norms))
            grad_norm_min = min(norms)
            grad_norm_max = max(norms)
            max_layer = max(layer_norms, key=layer_norms.__getitem__)
        else:
            grad_norm = grad_norm_min = grad_norm_max = 0.0
            max_layer = "none"

        _log.info(
            f"step={step} loss={loss:.4f} lr={lr:.6f} "
            f"grad_norm={grad_norm:.4f} "
            f"grad_norm_min={grad_norm_min:.4f} "
            f"grad_norm_max={grad_norm_max:.4f} "
            f"grad_norm_max_layer={max_layer}"
        )

        # Per-layer WARNING for individual layers exceeding their threshold.
        warn_threshold = self.cfg.grad_norm_warn_threshold
        for name, norm in layer_norms.items():
            if norm > warn_threshold:
                _log.warning(
                    f"WARNING step={step} layer={name} "
                    f"grad_norm={norm:.4f} exceeds threshold={warn_threshold}"
                )

        # Spike dump: when total norm crosses the spike threshold, write every
        # layer's norm immediately so the culprit is visible at the exact step,
        # not just at the next grad_log_every boundary.
        if grad_norm > self.cfg.grad_norm_spike_threshold:
            for name, norm in layer_norms.items():
                _log.debug(f"spike step={step} layer={name} norm={norm:.4f}")

    def log_val(self, step: int, val_loss: float) -> None:
        """Emit one validation-loss line (INFO — console + file).

        Format: ``val step=N val_loss=X.XXXX``
        """
        _log.info(f"val step={step} val_loss={val_loss:.4f}")

    def log_layers(
        self,
        step: int,
        layer_norms: dict[str, float],
        model: nn.Module,
    ) -> None:
        """Emit per-layer grad and weight lines (DEBUG — file only).

        Parameters
        ----------
        layer_norms:
            Pre-clip per-layer gradient norms (same dict passed to
            ``log_step``).  Only parameters present in this dict get a
            grad line.
        model:
            Live model — iterated for weight norms only.
        """
        if step % self.cfg.grad_log_every == 0:
            for name, norm in layer_norms.items():
                _log.debug(f"grad step={step} layer={name} norm={norm:.4f}")

        if step % self.cfg.weight_log_every == 0:
            for name, p in model.named_parameters():
                _log.debug(
                    f"weight step={step} layer={name} norm={p.norm().item():.4f}"
                )
