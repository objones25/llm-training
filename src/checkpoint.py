"""Model and optimizer checkpoint serialization.

Saves and restores the full training state needed to resume a run:
model weights, optimizer state (momentum/variance buffers), and the current
step count.

Checkpoint file format (single ``torch.save`` dict):

    {
        "step":            int,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg":             TrainConfig,
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
    ) -> Path

    load_checkpoint(
        path:      Path | str,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> int
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.config import TrainConfig


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: TrainConfig,
) -> Path:
    """Save model and optimizer state to disk.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer
    step : int
        Current training step, embedded in the filename and stored in the dict.
    cfg : TrainConfig
        Supplies ``checkpoint_dir``.  The directory is created if absent.

    Returns
    -------
    Path
        Absolute path of the file that was written.
    """
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg,
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer state from a checkpoint file.

    Parameters
    ----------
    path : Path | str
        Path to the ``.pt`` file produced by :func:`save_checkpoint`.
    model : nn.Module
        Model whose weights will be overwritten in-place.
    optimizer : torch.optim.Optimizer
        Optimizer whose state will be overwritten in-place.

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
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"]
