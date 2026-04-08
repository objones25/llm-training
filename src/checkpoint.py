"""Model, optimizer, and scheduler checkpoint serialization.

Saves and restores the full training state needed to resume a run:
model weights, optimizer state (momentum/variance buffers), scheduler state
(LR step counter), and the current step count.

Checkpoint file format (single ``torch.save`` dict):

    {
        "step":               int,
        "model_state":        model.state_dict(),
        "optimizer_state":    optimizer.state_dict() | (sd1, sd2),
        "optimizer_is_tuple": bool,
        "scheduler_state":    scheduler.state_dict(),   # optional
        "cfg":                TrainConfig,
    }

Two save modes are supported:

    Numbered  — ``checkpoint_{step:07d}.pt``, written on demand (legacy behavior
                retained for compatibility).
    Best      — ``best.pt``, overwritten whenever val loss improves.  Pass
                ``save_as_best=True`` to activate this mode.

Tuple optimizer support (Muon + AdamW):
    When ``optimizer`` is a ``(Muon, AdamW)`` tuple, both state_dicts are saved
    as a tuple.  ``load_checkpoint`` requires the same type of optimizer to be
    passed on load; a mismatch raises ``ValueError``.

Public API
----------
    save_checkpoint(
        model:        nn.Module,
        optimizer:    torch.optim.Optimizer | tuple,
        step:         int,
        cfg:          TrainConfig,
        scheduler:    LRScheduler | tuple[LRScheduler, LRScheduler] | None = None,
        save_as_best: bool = False,
    ) -> Path

    load_checkpoint(
        path:      Path | str,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer | tuple,
        scheduler: LRScheduler | tuple[LRScheduler, LRScheduler] | None = None,
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
    optimizer: torch.optim.Optimizer | tuple,
    step: int,
    cfg: TrainConfig,
    scheduler: LRScheduler | tuple[LRScheduler, LRScheduler] | None = None,
    save_as_best: bool = False,
) -> Path:
    """Save model, optimizer, and scheduler state to disk.

    Parameters
    ----------
    model : nn.Module
    optimizer : torch.optim.Optimizer | tuple
        Either a single optimizer or a ``(Muon, AdamW)`` tuple returned by
        ``make_optimizer`` when ``cfg.use_muon=True``.
    step : int
        Current training step, embedded in the filename and stored in the dict.
    cfg : TrainConfig
        Supplies ``checkpoint_dir``.  The directory is created if absent.
    scheduler : LRScheduler | tuple[LRScheduler, LRScheduler] | None
        When provided, its state_dict is stored in the checkpoint so the LR
        schedule can be resumed exactly.  Pass a tuple when using dual
        schedulers (Muon + AdamW).
    save_as_best : bool
        When True, save to ``checkpoint_dir/best.pt`` (overwriting any
        existing file).  When False (default), save to the numbered filename
        ``checkpoint_{step:07d}.pt``.

    Returns
    -------
    Path
        Absolute path of the file that was written.
    """
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if save_as_best:
        path = ckpt_dir / "best.pt"
    else:
        path = ckpt_dir / f"checkpoint_{step:07d}.pt"

    optimizer_is_tuple = isinstance(optimizer, tuple)
    if optimizer_is_tuple:
        optimizer_state = tuple(opt.state_dict() for opt in optimizer)
    else:
        optimizer_state = optimizer.state_dict()

    payload: dict = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer_state,
        "optimizer_is_tuple": optimizer_is_tuple,
        "cfg": cfg,
    }

    if scheduler is not None:
        if isinstance(scheduler, tuple):
            payload["scheduler_state"] = tuple(s.state_dict() for s in scheduler)
            payload["scheduler_is_tuple"] = True
        else:
            payload["scheduler_state"] = scheduler.state_dict()
            payload["scheduler_is_tuple"] = False

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: Path | str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | tuple,
    scheduler: LRScheduler | tuple[LRScheduler, LRScheduler] | None = None,
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
    optimizer : torch.optim.Optimizer | tuple
        Optimizer whose state will be overwritten in-place.  Must be a tuple
        if and only if the checkpoint was saved with a tuple optimizer.
    scheduler : LRScheduler | tuple[LRScheduler, LRScheduler] | None
        When provided and the checkpoint contains ``scheduler_state``, its
        state is restored so the LR schedule resumes correctly.

    Returns
    -------
    int
        The training step at which the checkpoint was saved.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the optimizer type (single vs tuple) does not match what was saved.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Register TrainConfig as a safe global so weights_only=True can deserialise
    # the dataclass stored in the checkpoint dict (PyTorch 2.4+ API).
    torch.serialization.add_safe_globals([TrainConfig])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    model.load_state_dict(ckpt["model_state"])

    saved_is_tuple = ckpt.get("optimizer_is_tuple", False)
    passed_is_tuple = isinstance(optimizer, tuple)
    if saved_is_tuple != passed_is_tuple:
        raise ValueError(
            f"Optimizer type mismatch: checkpoint has "
            f"{'tuple' if saved_is_tuple else 'single'} optimizer but "
            f"{'tuple' if passed_is_tuple else 'single'} was passed."
        )

    if passed_is_tuple:
        for opt, sd in zip(optimizer, ckpt["optimizer_state"]):
            opt.load_state_dict(sd)
    else:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    if scheduler is not None:
        if "scheduler_state" not in ckpt:
            warnings.warn(
                "Checkpoint does not contain scheduler_state (pre-v2 format). "
                "Scheduler will start from its initial state — LR trajectory "
                "will not match the original run.",
                UserWarning,
                stacklevel=2,
            )
        else:
            saved_sched_is_tuple = ckpt.get("scheduler_is_tuple", False)
            if isinstance(scheduler, tuple):
                if saved_sched_is_tuple:
                    for sched, sd in zip(scheduler, ckpt["scheduler_state"]):
                        sched.load_state_dict(sd)
                else:
                    # Single scheduler state saved; load into first scheduler only.
                    scheduler[0].load_state_dict(ckpt["scheduler_state"])
            else:
                if saved_sched_is_tuple:
                    scheduler.load_state_dict(ckpt["scheduler_state"][0])
                else:
                    scheduler.load_state_dict(ckpt["scheduler_state"])

    return ckpt["step"]
