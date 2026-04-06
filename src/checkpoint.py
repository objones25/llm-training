"""Model, optimizer, and scheduler checkpoint serialization.

Saves and restores the full training state needed to resume a run:
model weights, optimizer state (momentum/variance buffers), scheduler state
(LR step counter), and the current step count.

Checkpoint file format (single ``torch.save`` dict):

    {
        "step":              int,
        "model_state":       model.state_dict(),
        "optimizer_state":   optimizer.state_dict(),
        "scheduler_state":   scheduler.state_dict(),   # added in v2
        "cfg":               TrainConfig,
    }

Files are named ``checkpoint_{step:07d}.pt`` and written inside
``cfg.checkpoint_dir``, which is created automatically if absent.

Public API
----------
    save_checkpoint(
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        step:      int,
        cfg:       TrainConfig,
        scheduler: LRScheduler | None = None,
    ) -> Path

    load_checkpoint(
        path:      Path | str,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler | None = None,
    ) -> int
"""
from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

from src.config import TrainConfig


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
    scheduler: LRScheduler | None = None,
) -> Path:
    """Save model, optimizer, and scheduler state to disk.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    step : int
        Current training step, embedded in the filename and stored in the dict.
    cfg : TrainConfig
        Supplies ``checkpoint_dir``.  The directory is created if absent.
    scheduler : LRScheduler | None
        When provided, its state_dict is stored in the checkpoint so that
        the LR schedule can be resumed exactly.  Pass ``None`` to omit
        (backward-compatible with v1 checkpoints on load).

    Returns
    -------
    Path
        Absolute path of the file that was written.
    """
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_{step:07d}.pt"
    payload: dict = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg,
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler | None = None,
) -> int:
    """Restore model, optimizer, and scheduler state from a checkpoint file.

    Tensors are always loaded to CPU first (``map_location="cpu"``).  Move
    the model to its target device *after* calling this function.

    Parameters
    ----------
    path : Path | str
        Path to the ``.pt`` file produced by :func:`save_checkpoint`.
    model : nn.Module
        Model whose weights will be overwritten in-place.
    optimizer : torch.optim.Optimizer
        Optimizer whose state will be overwritten in-place.
    scheduler : LRScheduler | None
        When provided and the checkpoint contains ``scheduler_state``, its
        state is restored so the LR schedule resumes correctly.  If the
        checkpoint pre-dates scheduler serialization, a warning is emitted
        and the scheduler is left at its initial state.

    Returns
    -------
    int
        The training step at which the checkpoint was saved.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # Register TrainConfig as a safe global so weights_only=True can deserialise
    # the dataclass stored in the checkpoint dict (PyTorch 2.4+ API).
    torch.serialization.add_safe_globals([TrainConfig])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None:
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        else:
            warnings.warn(
                "Checkpoint does not contain scheduler_state (pre-v2 format). "
                "Scheduler will start from its initial state — LR trajectory "
                "will not match the original run.",
                UserWarning,
                stacklevel=2,
            )
    return ckpt["step"]
