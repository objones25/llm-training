"""Tests for src/logger.py.

All tests use a tiny 2-layer GPT model on CPU with synthetic backward passes.
Stdout is captured via pytest's capsys fixture -- no disk I/O.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.config import TrainConfig
from src.logger import GradientLogger
from src.model import GPT


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> TrainConfig:
    return TrainConfig(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
        batch_size=2,
        grad_log_every=10,
        weight_log_every=20,
        grad_norm_warn_threshold=10.0,
    )


@pytest.fixture
def model(cfg: TrainConfig) -> GPT:
    torch.manual_seed(0)
    return GPT(cfg)


@pytest.fixture
def logger(cfg: TrainConfig) -> GradientLogger:
    return GradientLogger(cfg)


def _run_backward(model: GPT, cfg: TrainConfig) -> None:
    """Run one forward/backward pass to populate .grad on all parameters."""
    torch.manual_seed(42)
    idx = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()


# ── log_step: format and field contracts ─────────────────────────────────────


def test_log_step_contains_all_six_fields(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """Every call to log_step must emit all six required key=value fields."""
    _run_backward(model, cfg)
    logger.log_step(step=100, loss=3.4821, lr=0.000287, model=model)
    out = capsys.readouterr().out
    for field in ("step=", "loss=", "lr=", "grad_norm=", "grad_norm_min=", "grad_norm_max="):
        assert field in out, f"Missing field '{field}' in log_step output:\n{out}"


def test_log_step_format(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """log_step output must be a single line of space-separated key=value pairs."""
    _run_backward(model, cfg)
    logger.log_step(step=5, loss=2.1, lr=1e-4, model=model)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("step=")]
    assert len(lines) == 1, "Expected exactly one step= line"
    pairs = lines[0].split()
    for pair in pairs:
        assert "=" in pair, f"Token '{pair}' is not in key=value format"


def test_log_step_step_value(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """The step field must match the argument passed in."""
    _run_backward(model, cfg)
    logger.log_step(step=777, loss=1.0, lr=1e-3, model=model)
    out = capsys.readouterr().out
    assert "step=777" in out


def test_log_step_grad_norm_values_correct(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """grad_norm must be the L2 norm of all parameter gradients concatenated."""
    _run_backward(model, cfg)

    # Compute reference: L2 norm over all per-layer L2 norms (equivalent to
    # sqrt(sum of squared gradient elements across all parameters).
    layer_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    expected_total = math.sqrt(sum(n ** 2 for n in layer_norms))
    expected_min = min(layer_norms)
    expected_max = max(layer_norms)

    logger.log_step(step=1, loss=0.0, lr=0.0, model=model)
    out = capsys.readouterr().out
    step_line = next(l for l in out.splitlines() if l.startswith("step="))
    kv = dict(pair.split("=") for pair in step_line.split())

    assert abs(float(kv["grad_norm"]) - expected_total) < 1e-3
    assert abs(float(kv["grad_norm_min"]) - expected_min) < 1e-3
    assert abs(float(kv["grad_norm_max"]) - expected_max) < 1e-3


# ── log_step: WARNING contracts (rule 25) ────────────────────────────────────


def test_warning_emitted_when_norm_exceeds_threshold(
    model: GPT, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """Rule 25: WARNING line must be printed (not raised) when a layer grad norm
    exceeds cfg.grad_norm_warn_threshold.  Training must not be interrupted."""
    # Use a very low threshold so we are guaranteed to exceed it.
    low_cfg = TrainConfig(
        vocab_size=256, n_layers=2, d_model=64, n_heads=2, d_ff=128,
        seq_len=16, batch_size=2,
        grad_norm_warn_threshold=0.0,  # anything > 0 triggers warning
    )
    torch.manual_seed(0)
    low_model = GPT(low_cfg)
    _run_backward(low_model, low_cfg)

    low_logger = GradientLogger(low_cfg)
    # Must not raise:
    low_logger.log_step(step=10, loss=1.0, lr=1e-3, model=low_model)

    out = capsys.readouterr().out
    assert "WARNING" in out, "Expected at least one WARNING line in output"


def test_warning_format(
    model: GPT, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """WARNING line must contain step, layer, grad_norm, and threshold fields."""
    low_cfg = TrainConfig(
        vocab_size=256, n_layers=2, d_model=64, n_heads=2, d_ff=128,
        seq_len=16, batch_size=2,
        grad_norm_warn_threshold=0.0,
    )
    torch.manual_seed(0)
    low_model = GPT(low_cfg)
    _run_backward(low_model, low_cfg)

    GradientLogger(low_cfg).log_step(step=99, loss=1.0, lr=1e-3, model=low_model)
    out = capsys.readouterr().out
    warn_lines = [l for l in out.splitlines() if l.startswith("WARNING")]
    assert warn_lines, "No WARNING lines found"
    for line in warn_lines:
        assert "step=" in line
        assert "layer=" in line
        assert "grad_norm=" in line
        assert "threshold=" in line or "exceeds" in line


def test_no_warning_below_threshold(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """No WARNING line must appear when all layer norms are below threshold."""
    # cfg.grad_norm_warn_threshold=10.0 -- normal gradients are well below this.
    _run_backward(model, cfg)
    logger.log_step(step=1, loss=1.0, lr=1e-3, model=model)
    out = capsys.readouterr().out
    assert "WARNING" not in out, f"Unexpected WARNING in output:\n{out}"


# ── log_layers: per-layer lines (rule 24) ────────────────────────────────────


def test_log_layers_emits_grad_lines_for_every_param(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """Rule 24: log_layers must emit one 'grad' line per named parameter with a
    gradient, at the grad_log_every cadence."""
    _run_backward(model, cfg)
    # cfg.grad_log_every=10, so step=10 triggers grad lines
    logger.log_layers(step=10, model=model)
    out = capsys.readouterr().out
    grad_lines = [l for l in out.splitlines() if l.startswith("grad ")]

    params_with_grad = [
        name for name, p in model.named_parameters() if p.grad is not None
    ]
    assert len(grad_lines) == len(params_with_grad), (
        f"Expected {len(params_with_grad)} grad lines, got {len(grad_lines)}"
    )
    for name in params_with_grad:
        assert any(name in line for line in grad_lines), (
            f"No grad line found for parameter '{name}'"
        )


def test_log_layers_grad_cadence(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """Grad lines must only appear at steps that are multiples of grad_log_every."""
    _run_backward(model, cfg)
    # Step 5 is not a multiple of grad_log_every=10 -- no grad lines expected.
    logger.log_layers(step=5, model=model)
    out = capsys.readouterr().out
    grad_lines = [l for l in out.splitlines() if l.startswith("grad ")]
    assert grad_lines == [], f"Unexpected grad lines at non-cadence step:\n{out}"


def test_log_layers_weight_cadence(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """Weight lines must only appear at steps that are multiples of weight_log_every."""
    _run_backward(model, cfg)
    # step=10: multiple of grad_log_every=10, but NOT weight_log_every=20
    logger.log_layers(step=10, model=model)
    out = capsys.readouterr().out
    weight_lines = [l for l in out.splitlines() if l.startswith("weight ")]
    assert weight_lines == [], (
        f"Unexpected weight lines at non-weight-cadence step:\n{out}"
    )


def test_log_layers_emits_weight_lines_at_cadence(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """Weight lines must appear for every named parameter at weight_log_every cadence."""
    logger.log_layers(step=20, model=model)  # step=20 is multiple of weight_log_every=20
    out = capsys.readouterr().out
    weight_lines = [l for l in out.splitlines() if l.startswith("weight ")]

    all_params = list(model.named_parameters())
    assert len(weight_lines) == len(all_params), (
        f"Expected {len(all_params)} weight lines, got {len(weight_lines)}"
    )
    for name, _ in all_params:
        assert any(name in line for line in weight_lines), (
            f"No weight line found for parameter '{name}'"
        )


def test_log_layers_grad_line_format(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """Each grad line must match: 'grad step=N layer=<name> norm=<value>'."""
    _run_backward(model, cfg)
    logger.log_layers(step=10, model=model)
    out = capsys.readouterr().out
    grad_lines = [l for l in out.splitlines() if l.startswith("grad ")]
    for line in grad_lines:
        assert "step=" in line
        assert "layer=" in line
        assert "norm=" in line


def test_log_step_no_grads_emits_zero_norms(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """When no parameter has a gradient, grad_norm fields must all be 0.0."""
    # Do NOT run backward -- all .grad are None.
    logger.log_step(step=0, loss=5.0, lr=1e-3, model=model)
    out = capsys.readouterr().out
    step_line = next(l for l in out.splitlines() if l.startswith("step="))
    kv = dict(pair.split("=") for pair in step_line.split())
    assert float(kv["grad_norm"]) == 0.0
    assert float(kv["grad_norm_min"]) == 0.0
    assert float(kv["grad_norm_max"]) == 0.0


def test_log_layers_no_output_between_cadence_steps(
    model: GPT, logger: GradientLogger, cfg: TrainConfig, capsys: pytest.CaptureFixture
) -> None:
    """log_layers must produce no output at steps that are not on any cadence."""
    _run_backward(model, cfg)
    # step=7 is neither a multiple of 10 (grad) nor 20 (weight)
    logger.log_layers(step=7, model=model)
    out = capsys.readouterr().out
    assert out == "", f"Unexpected output at non-cadence step:\n{out}"
