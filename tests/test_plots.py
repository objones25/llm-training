"""Tests for src/plots.py.

All tests pass synthetic data directly to plot functions and assert that a
valid non-empty .png file is produced at the expected path (rule 26).
No training loop is run. All disk writes use tmp_path.
"""
from __future__ import annotations

from pathlib import Path

from src.plots import (
    plot_grad_heatmap,
    plot_grad_hist,
    plot_grad_norm,
    plot_loss,
    plot_lr,
    plot_weight_norm,
)


# ── Synthetic data helpers ────────────────────────────────────────────────────

STEPS = list(range(0, 50, 10))          # [0, 10, 20, 30, 40]
LOSSES = [5.0, 4.2, 3.8, 3.5, 3.3]
LRS = [0.0, 1e-4, 2e-4, 1.8e-4, 1.5e-4]
GRAD_NORMS = [1.2, 1.0, 0.9, 0.85, 0.8]
GRAD_MINS = [0.1, 0.09, 0.08, 0.07, 0.06]
GRAD_MAXS = [2.5, 2.2, 2.0, 1.9, 1.8]
LAYER_NAMES = ["block.0.attn", "block.0.ff", "block.1.attn", "block.1.ff"]
# norms_matrix[step_idx][layer_idx]
NORMS_MATRIX = [
    [0.5, 0.4, 0.6, 0.3],
    [0.45, 0.38, 0.55, 0.28],
    [0.4, 0.35, 0.5, 0.25],
    [0.38, 0.32, 0.48, 0.22],
    [0.35, 0.30, 0.45, 0.20],
]
HIST_NORMS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 0.4, 0.6, 0.9, 0.3]


def _assert_valid_png(path: Path) -> None:
    """Assert that path exists and is a non-empty file."""
    assert path.exists(), f"Expected file at {path} -- not found"
    assert path.stat().st_size > 0, f"File at {path} is empty"
    assert path.suffix == ".png", f"Expected .png suffix, got {path.suffix}"


# ── plot_loss ─────────────────────────────────────────────────────────────────


def test_plot_loss_creates_png(tmp_path: Path) -> None:
    """plot_loss must produce a non-empty .png at the given path."""
    out = tmp_path / "loss.png"
    plot_loss(STEPS, LOSSES, out)
    _assert_valid_png(out)


def test_plot_loss_overwrites_existing(tmp_path: Path) -> None:
    """Calling plot_loss twice must overwrite, not append to, the existing file."""
    out = tmp_path / "loss.png"
    plot_loss(STEPS, LOSSES, out)

    # Slightly different data to ensure the file actually changes.
    plot_loss(STEPS, [v + 0.1 for v in LOSSES], out)
    assert out.exists()
    # File must not be empty after overwrite.
    assert out.stat().st_size > 0


def test_plot_loss_accepts_string_path(tmp_path: Path) -> None:
    """plot_loss must accept a plain string as path."""
    out = str(tmp_path / "loss.png")
    plot_loss(STEPS, LOSSES, out)
    assert Path(out).exists()


def test_plot_loss_with_val_data(tmp_path: Path) -> None:
    """plot_loss must produce a valid .png when val_steps and val_losses are provided."""
    out = tmp_path / "loss.png"
    val_steps = STEPS[::2]   # [0, 20, 40]
    val_losses = [5.1, 3.9, 3.4]
    plot_loss(STEPS, LOSSES, out, val_steps=val_steps, val_losses=val_losses)
    _assert_valid_png(out)


def test_plot_loss_without_val_data_no_crash(tmp_path: Path) -> None:
    """plot_loss must not crash when val args are None (default)."""
    out = tmp_path / "loss.png"
    plot_loss(STEPS, LOSSES, out, val_steps=None, val_losses=None)
    _assert_valid_png(out)


def test_plot_loss_with_empty_val_lists(tmp_path: Path) -> None:
    """plot_loss must not crash when val lists are empty (no evals yet)."""
    out = tmp_path / "loss.png"
    plot_loss(STEPS, LOSSES, out, val_steps=[], val_losses=[])
    _assert_valid_png(out)


# ── plot_lr ───────────────────────────────────────────────────────────────────


def test_plot_lr_creates_png(tmp_path: Path) -> None:
    """plot_lr must produce a non-empty .png at the given path."""
    out = tmp_path / "lr.png"
    plot_lr(STEPS, LRS, out)
    _assert_valid_png(out)


# ── plot_grad_norm ────────────────────────────────────────────────────────────


def test_plot_grad_norm_creates_png(tmp_path: Path) -> None:
    """plot_grad_norm must produce a non-empty .png at the given path."""
    out = tmp_path / "grad_norm.png"
    plot_grad_norm(STEPS, GRAD_NORMS, GRAD_MINS, GRAD_MAXS, out)
    _assert_valid_png(out)


def test_plot_grad_norm_shaded_region(tmp_path: Path) -> None:
    """plot_grad_norm must not crash when min < max (shaded fill_between region)."""
    out = tmp_path / "grad_norm.png"
    # Exaggerated spread to exercise fill_between path.
    plot_grad_norm(STEPS, GRAD_NORMS, [0.01] * 5, [10.0] * 5, out)
    _assert_valid_png(out)


def test_plot_grad_norm_equal_min_max(tmp_path: Path) -> None:
    """plot_grad_norm must not crash when min == max (degenerate shaded region)."""
    out = tmp_path / "grad_norm.png"
    flat = [1.0] * 5
    plot_grad_norm(STEPS, flat, flat, flat, out)
    _assert_valid_png(out)


# ── plot_grad_heatmap ─────────────────────────────────────────────────────────


def test_plot_grad_heatmap_creates_png(tmp_path: Path) -> None:
    """plot_grad_heatmap must produce a non-empty .png at the given path."""
    out = tmp_path / "grad_heatmap.png"
    plot_grad_heatmap(STEPS, LAYER_NAMES, NORMS_MATRIX, out)
    _assert_valid_png(out)


def test_plot_grad_heatmap_single_step(tmp_path: Path) -> None:
    """plot_grad_heatmap must not crash with only one step (degenerate x-axis)."""
    out = tmp_path / "grad_heatmap.png"
    plot_grad_heatmap([0], LAYER_NAMES, [[0.5, 0.4, 0.6, 0.3]], out)
    _assert_valid_png(out)


def test_plot_grad_heatmap_single_layer(tmp_path: Path) -> None:
    """plot_grad_heatmap must not crash with only one layer."""
    out = tmp_path / "grad_heatmap.png"
    plot_grad_heatmap(STEPS, ["block.0.attn"], [[v] for v in [0.5, 0.4, 0.3, 0.25, 0.2]], out)
    _assert_valid_png(out)


# ── plot_grad_hist ────────────────────────────────────────────────────────────


def test_plot_grad_hist_creates_png(tmp_path: Path) -> None:
    """plot_grad_hist must produce a non-empty .png at the given path."""
    out = tmp_path / "grad_hist.png"
    plot_grad_hist(HIST_NORMS, out)
    _assert_valid_png(out)


def test_plot_grad_hist_single_value(tmp_path: Path) -> None:
    """plot_grad_hist must not crash when all norms are identical."""
    out = tmp_path / "grad_hist.png"
    plot_grad_hist([1.0] * 8, out)
    _assert_valid_png(out)


# ── plot_weight_norm ──────────────────────────────────────────────────────────


def test_plot_weight_norm_creates_png(tmp_path: Path) -> None:
    """plot_weight_norm must produce a non-empty .png at the given path."""
    out = tmp_path / "weight_norm.png"
    plot_weight_norm(STEPS, LAYER_NAMES, NORMS_MATRIX, out)
    _assert_valid_png(out)


def test_plot_weight_norm_single_step(tmp_path: Path) -> None:
    """plot_weight_norm must not crash with only one step."""
    out = tmp_path / "weight_norm.png"
    plot_weight_norm([0], LAYER_NAMES, [[1.0, 0.9, 1.1, 0.8]], out)
    _assert_valid_png(out)
