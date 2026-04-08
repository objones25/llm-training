"""Matplotlib visualizations for LM pretraining diagnostics.

All plots use the Agg backend (headless -- no display required) and are saved
to disk by overwriting the existing file in place.  No plot logic lives in
``train.py``; ``train.py`` calls these functions at the configured cadence.

Backend note
------------
``matplotlib.use("Agg")`` must appear before any other matplotlib import.
This is enforced here at module level per CONTRIBUTING.md.

Public API
----------
    plot_loss(steps, losses, path, val_steps=None, val_losses=None)
    plot_lr(steps, lrs, path)
    plot_grad_norm(steps, grad_norms, grad_norm_mins, grad_norm_maxs, path)
    plot_grad_heatmap(steps, layer_names, norms_matrix, path)
    plot_grad_hist(norms, path)
    plot_weight_norm(steps, layer_names, norms_matrix, path)

All ``path`` arguments accept ``str`` or ``pathlib.Path``.
``norms_matrix`` is a 2-D sequence of shape ``(len(steps), len(layer_names))``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # must precede all other matplotlib imports

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pathlib import Path  # noqa: E402
from typing import Sequence  # noqa: E402


def plot_loss(
    steps: Sequence[int],
    losses: Sequence[float],
    path: Path | str,
    val_steps: Sequence[int] | None = None,
    val_losses: Sequence[float] | None = None,
) -> None:
    """Save loss-vs-step curve (log Y scale) to *path*.

    When *val_steps* and *val_losses* are provided, the validation loss is
    overlaid on the same axes so train/val divergence (overfitting) is
    immediately visible.

    Parameters
    ----------
    steps : Sequence[int]
    losses : Sequence[float]
    path : Path | str
        Destination file; created or overwritten in place.
    val_steps : Sequence[int] | None
        Steps at which validation loss was evaluated.  ``None`` omits the
        validation curve.
    val_losses : Sequence[float] | None
        Validation loss values corresponding to *val_steps*.
    """
    fig, ax = plt.subplots()
    ax.plot(steps, losses, label="train")
    if val_steps is not None and val_losses is not None and len(val_steps) > 0:
        ax.plot(val_steps, val_losses, label="val", linestyle="--")
        ax.legend()
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training loss")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_lr(
    steps: Sequence[int],
    lrs: Sequence[float],
    path: Path | str,
) -> None:
    """Save learning-rate-vs-step curve to *path*.

    Parameters
    ----------
    steps : Sequence[int]
    lrs : Sequence[float]
    path : Path | str
    """
    fig, ax = plt.subplots()
    ax.plot(steps, lrs)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning rate schedule")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_grad_norm(
    steps: Sequence[int],
    grad_norms: Sequence[float],
    grad_norm_mins: Sequence[float],
    grad_norm_maxs: Sequence[float],
    path: Path | str,
) -> None:
    """Save gradient norm over time (total, min, max with shaded band) to *path*.

    Parameters
    ----------
    steps : Sequence[int]
    grad_norms : Sequence[float]
        Total (L2) gradient norm per step.
    grad_norm_mins : Sequence[float]
        Per-step minimum layer gradient norm.
    grad_norm_maxs : Sequence[float]
        Per-step maximum layer gradient norm.
    path : Path | str
    """
    xs = list(steps)
    fig, ax = plt.subplots()
    ax.plot(xs, grad_norms, label="grad_norm")
    ax.plot(xs, grad_norm_mins, label="grad_norm_min", linestyle="--")
    ax.plot(xs, grad_norm_maxs, label="grad_norm_max", linestyle="--")
    ax.fill_between(xs, grad_norm_mins, grad_norm_maxs, alpha=0.2)
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm")
    ax.set_title("Gradient norm over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_grad_heatmap(
    steps: Sequence[int],
    layer_names: Sequence[str],
    norms_matrix: Sequence[Sequence[float]],
    path: Path | str,
) -> None:
    """Save per-layer gradient norm heatmap (log color scale) to *path*.

    Parameters
    ----------
    steps : Sequence[int]
        X-axis ticks.
    layer_names : Sequence[str]
        Y-axis labels (one per layer).
    norms_matrix : Sequence[Sequence[float]]
        Shape ``(len(steps), len(layer_names))``.  Values must be > 0;
        zeros are clipped to 1e-10 before applying log scaling.
    path : Path | str
    """
    from matplotlib.colors import LogNorm

    mat = np.array(norms_matrix, dtype=float).T  # (layers, steps)
    mat = np.clip(mat, 1e-10, None)

    fig, ax = plt.subplots(
        figsize=(max(4, len(steps) * 0.4 + 2), max(3, len(layer_names) * 0.3 + 1))
    )
    im = ax.imshow(mat, aspect="auto", norm=LogNorm(vmin=mat.min(), vmax=mat.max()))
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=7)
    ax.set_xlabel("step")
    ax.set_title("Per-layer gradient norm heatmap")
    fig.colorbar(im, ax=ax, label="norm (log scale)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_grad_hist(
    norms: Sequence[float],
    path: Path | str,
) -> None:
    """Save gradient norm distribution histogram (log X scale) to *path*.

    Parameters
    ----------
    norms : Sequence[float]
        All per-parameter gradient norms at the current step snapshot.
        Values are clipped to a minimum of 1e-10 before log scaling.
    path : Path | str
    """
    clipped = [max(v, 1e-10) for v in norms]
    fig, ax = plt.subplots()
    ax.hist(clipped, bins=20)
    ax.set_xscale("log")
    ax.set_xlabel("gradient norm (log scale)")
    ax.set_ylabel("count")
    ax.set_title("Gradient norm distribution")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_weight_norm(
    steps: Sequence[int],
    layer_names: Sequence[str],
    norms_matrix: Sequence[Sequence[float]],
    path: Path | str,
) -> None:
    """Save per-layer weight norm heatmap (log color scale) to *path*.

    Same format as :func:`plot_grad_heatmap` but for weight norms.

    Parameters
    ----------
    steps : Sequence[int]
    layer_names : Sequence[str]
    norms_matrix : Sequence[Sequence[float]]
        Shape ``(len(steps), len(layer_names))``.
    path : Path | str
    """
    from matplotlib.colors import LogNorm

    mat = np.array(norms_matrix, dtype=float).T  # (layers, steps)
    mat = np.clip(mat, 1e-10, None)

    fig, ax = plt.subplots(
        figsize=(max(4, len(steps) * 0.4 + 2), max(3, len(layer_names) * 0.3 + 1))
    )
    im = ax.imshow(mat, aspect="auto", norm=LogNorm(vmin=mat.min(), vmax=mat.max()))
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([str(s) for s in steps], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, fontsize=7)
    ax.set_xlabel("step")
    ax.set_title("Per-layer weight norm heatmap")
    fig.colorbar(im, ax=ax, label="norm (log scale)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
