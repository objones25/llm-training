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
        """
        if layer_norms:
            norms = list(layer_norms.values())
            grad_norm = math.sqrt(sum(n * n for n in norms))
            grad_norm_min = min(norms)
            grad_norm_max = max(norms)
        else:
            grad_norm = grad_norm_min = grad_norm_max = 0.0

        _log.info(
            f"step={step} loss={loss:.4f} lr={lr:.6f} "
            f"grad_norm={grad_norm:.4f} "
            f"grad_norm_min={grad_norm_min:.4f} "
            f"grad_norm_max={grad_norm_max:.4f}"
        )

        threshold = self.cfg.grad_norm_warn_threshold
        for name, norm in layer_norms.items():
            if norm > threshold:
                _log.warning(
                    f"WARNING step={step} layer={name} "
                    f"grad_norm={norm:.4f} exceeds threshold={threshold}"
                )

    def log_layers(
        self,
        step: int,
        layer_norms: dict[str, float],
        model: object,
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
            for name, p in model.named_parameters():  # type: ignore[union-attr]
                _log.debug(f"weight step={step} layer={name} norm={p.norm().item():.4f}")
