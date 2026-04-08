"""Tests for src/logger.py.

All tests use a tiny 2-layer GPT model on CPU with synthetic backward passes.
Log records are captured via pytest's caplog fixture — no disk I/O and no
real file handlers (configure_logging is never called here).
"""

from __future__ import annotations

import logging
import math

import pytest
import torch
import torch.nn.functional as F

from src.config import TrainConfig
from src.logger import GradientLogger
from src.model import GPT

_LOGGER_NAME = "llm_training"


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


def _get_layer_norms(model: GPT) -> dict[str, float]:
    """Return pre-clip per-layer gradient norms for all parameters with .grad."""
    return {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


# ── log_step: format and field contracts ─────────────────────────────────────


def test_log_step_contains_required_fields(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every call to log_step must emit all required key=value fields."""
    _run_backward(model, cfg)
    layer_norms = _get_layer_norms(model)
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        logger.log_step(step=100, loss=3.4821, lr=0.000287, layer_norms=layer_norms)
    required = (
        "step=",
        "loss=",
        "lr=",
        "grad_norm=",
        "grad_norm_min=",
        "grad_norm_max=",
        "grad_norm_max_layer=",
    )
    for field in required:
        assert any(
            field in msg for msg in caplog.messages
        ), f"Missing field '{field}' in log_step output"


def test_log_step_format(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_step output must be a single line of space-separated key=value pairs."""
    _run_backward(model, cfg)
    layer_norms = _get_layer_norms(model)
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        logger.log_step(step=5, loss=2.1, lr=1e-4, layer_norms=layer_norms)
    step_msgs = [msg for msg in caplog.messages if msg.startswith("step=")]
    assert len(step_msgs) == 1, "Expected exactly one step= message"
    pairs = step_msgs[0].split()
    for pair in pairs:
        assert "=" in pair, f"Token '{pair}' is not in key=value format"


def test_log_step_step_value(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The step field must match the argument passed in."""
    _run_backward(model, cfg)
    layer_norms = _get_layer_norms(model)
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        logger.log_step(step=777, loss=1.0, lr=1e-3, layer_norms=layer_norms)
    assert any("step=777" in msg for msg in caplog.messages)


def test_log_step_grad_norm_values_correct(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """grad_norm must be the L2 norm of all parameter gradients concatenated."""
    _run_backward(model, cfg)
    layer_norms = _get_layer_norms(model)

    # Compute reference: L2 norm over all per-layer L2 norms (equivalent to
    # sqrt(sum of squared gradient elements across all parameters).
    norms = list(layer_norms.values())
    expected_total = math.sqrt(sum(n**2 for n in norms))
    expected_min = min(norms)
    expected_max = max(norms)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        logger.log_step(step=1, loss=0.0, lr=0.0, layer_norms=layer_norms)
    step_msg = next(msg for msg in caplog.messages if msg.startswith("step="))
    kv = dict(pair.split("=") for pair in step_msg.split())

    assert abs(float(kv["grad_norm"]) - expected_total) < 1e-3
    assert abs(float(kv["grad_norm_min"]) - expected_min) < 1e-3
    assert abs(float(kv["grad_norm_max"]) - expected_max) < 1e-3


# ── log_step: WARNING contracts (rule 25) ────────────────────────────────────


def test_warning_emitted_when_norm_exceeds_threshold(
    model: GPT, cfg: TrainConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """Rule 25: WARNING line must be logged (not raised) when a layer grad norm
    exceeds cfg.grad_norm_warn_threshold.  Training must not be interrupted."""
    # Use a very low threshold so we are guaranteed to exceed it.
    low_cfg = TrainConfig(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
        batch_size=2,
        grad_norm_warn_threshold=0.0,  # anything > 0 triggers warning
    )
    torch.manual_seed(0)
    low_model = GPT(low_cfg)
    _run_backward(low_model, low_cfg)
    layer_norms = _get_layer_norms(low_model)

    low_logger = GradientLogger(low_cfg)
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        # Must not raise:
        low_logger.log_step(step=10, loss=1.0, lr=1e-3, layer_norms=layer_norms)

    warning_msgs = [msg for msg in caplog.messages if msg.startswith("WARNING")]
    assert warning_msgs, "Expected at least one WARNING message"


def test_warning_format(
    model: GPT, cfg: TrainConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """WARNING message must contain step, layer, grad_norm, and threshold fields."""
    low_cfg = TrainConfig(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
        batch_size=2,
        grad_norm_warn_threshold=0.0,
    )
    torch.manual_seed(0)
    low_model = GPT(low_cfg)
    _run_backward(low_model, low_cfg)
    layer_norms = _get_layer_norms(low_model)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        GradientLogger(low_cfg).log_step(
            step=99, loss=1.0, lr=1e-3, layer_norms=layer_norms
        )
    warn_msgs = [msg for msg in caplog.messages if msg.startswith("WARNING")]
    assert warn_msgs, "No WARNING messages found"
    for msg in warn_msgs:
        assert "step=" in msg
        assert "layer=" in msg
        assert "grad_norm=" in msg
        assert "threshold=" in msg or "exceeds" in msg


def test_no_warning_below_threshold(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No WARNING must appear when all layer norms are below threshold."""
    # cfg.grad_norm_warn_threshold=10.0 — normal gradients are well below this.
    _run_backward(model, cfg)
    layer_norms = _get_layer_norms(model)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        logger.log_step(step=1, loss=1.0, lr=1e-3, layer_norms=layer_norms)
    warning_msgs = [msg for msg in caplog.messages if msg.startswith("WARNING")]
    assert warning_msgs == [], f"Unexpected WARNING messages: {warning_msgs}"


# ── log_layers: per-layer lines (rule 24) ────────────────────────────────────


def test_log_layers_emits_grad_lines_for_every_param(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Rule 24: log_layers must emit one 'grad' line per parameter with a
    gradient, at the grad_log_every cadence."""
    _run_backward(model, cfg)
    layer_norms = _get_layer_norms(model)
    # cfg.grad_log_every=10, so step=10 triggers grad lines
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        logger.log_layers(step=10, layer_norms=layer_norms, model=model)
    grad_msgs = [msg for msg in caplog.messages if msg.startswith("grad ")]

    params_with_grad = [
        name for name, p in model.named_parameters() if p.grad is not None
    ]
    assert len(grad_msgs) == len(
        params_with_grad
    ), f"Expected {len(params_with_grad)} grad messages, got {len(grad_msgs)}"
    for name in params_with_grad:
        assert any(
            name in msg for msg in grad_msgs
        ), f"No grad message found for parameter '{name}'"


def test_log_layers_grad_cadence(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Grad lines must only appear at steps that are multiples of grad_log_every."""
    _run_backward(model, cfg)
    layer_norms = _get_layer_norms(model)
    # Step 5 is not a multiple of grad_log_every=10 — no grad lines expected.
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        logger.log_layers(step=5, layer_norms=layer_norms, model=model)
    grad_msgs = [msg for msg in caplog.messages if msg.startswith("grad ")]
    assert grad_msgs == [], f"Unexpected grad messages at non-cadence step: {grad_msgs}"


def test_log_layers_weight_cadence(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Weight lines must only appear at steps that are multiples of weight_log_every."""
    _run_backward(model, cfg)
    layer_norms = _get_layer_norms(model)
    # step=10: multiple of grad_log_every=10, but NOT weight_log_every=20
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        logger.log_layers(step=10, layer_norms=layer_norms, model=model)
    weight_msgs = [msg for msg in caplog.messages if msg.startswith("weight ")]
    assert (
        weight_msgs == []
    ), f"Unexpected weight messages at non-weight-cadence step: {weight_msgs}"


def test_log_layers_emits_weight_lines_at_cadence(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Weight lines must appear for every named parameter at weight_log_every cadence."""
    # step=20 is multiple of weight_log_every=20; no backward needed for weight norms
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        logger.log_layers(step=20, layer_norms={}, model=model)
    weight_msgs = [msg for msg in caplog.messages if msg.startswith("weight ")]

    all_params = list(model.named_parameters())
    assert len(weight_msgs) == len(
        all_params
    ), f"Expected {len(all_params)} weight messages, got {len(weight_msgs)}"
    for name, _ in all_params:
        assert any(
            name in msg for msg in weight_msgs
        ), f"No weight message found for parameter '{name}'"


def test_log_layers_grad_line_format(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each grad message must match: 'grad step=N layer=<name> norm=<value>'."""
    _run_backward(model, cfg)
    layer_norms = _get_layer_norms(model)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        logger.log_layers(step=10, layer_norms=layer_norms, model=model)
    grad_msgs = [msg for msg in caplog.messages if msg.startswith("grad ")]
    for msg in grad_msgs:
        assert "step=" in msg
        assert "layer=" in msg
        assert "norm=" in msg


def test_log_step_no_grads_emits_zero_norms(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When layer_norms is empty, grad_norm fields must all be 0.0."""
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        logger.log_step(step=0, loss=5.0, lr=1e-3, layer_norms={})
    step_msg = next(msg for msg in caplog.messages if msg.startswith("step="))
    kv = dict(pair.split("=") for pair in step_msg.split())
    assert float(kv["grad_norm"]) == 0.0
    assert float(kv["grad_norm_min"]) == 0.0
    assert float(kv["grad_norm_max"]) == 0.0


def test_log_layers_no_output_between_cadence_steps(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """log_layers must produce no records at steps that are not on any cadence."""
    _run_backward(model, cfg)
    layer_norms = _get_layer_norms(model)
    # step=7 is neither a multiple of 10 (grad) nor 20 (weight)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        logger.log_layers(step=7, layer_norms=layer_norms, model=model)
    assert (
        caplog.records == []
    ), f"Unexpected log records at non-cadence step: {caplog.records}"


# ── log_val (validation loss) ─────────────────────────────────────────────────


def test_log_val_format(
    logger: GradientLogger, cfg: TrainConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """log_val must emit a single line: 'val step=N val_loss=X.XXXX'."""
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        logger.log_val(step=250, val_loss=4.1234)
    val_msgs = [msg for msg in caplog.messages if msg.startswith("val ")]
    assert len(val_msgs) == 1, f"Expected exactly one val message, got: {val_msgs}"
    kv = dict(pair.split("=") for pair in val_msgs[0].split() if "=" in pair)
    assert kv["step"] == "250"
    assert abs(float(kv["val_loss"]) - 4.1234) < 1e-3


def test_log_val_is_info_level(
    logger: GradientLogger, cfg: TrainConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """log_val must log at INFO level so it appears on the console."""
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        logger.log_val(step=0, val_loss=9.0)
    records = [r for r in caplog.records if r.message.startswith("val ")]
    assert records, "No val record found"
    assert records[0].levelno == logging.INFO


# ── grad_norm_max_layer and spike dump (new) ──────────────────────────────────


def test_log_step_max_layer_name_is_correct(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """grad_norm_max_layer in the step line must name the layer with the
    highest per-parameter gradient norm."""
    _run_backward(model, cfg)
    layer_norms = _get_layer_norms(model)
    expected_max_layer = max(layer_norms, key=layer_norms.__getitem__)

    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        logger.log_step(step=1, loss=1.0, lr=1e-3, layer_norms=layer_norms)

    step_msg = next(msg for msg in caplog.messages if msg.startswith("step="))
    assert (
        f"grad_norm_max_layer={expected_max_layer}" in step_msg
    ), f"Expected grad_norm_max_layer={expected_max_layer!r} in: {step_msg!r}"


def test_spike_dump_emitted_when_total_norm_exceeds_threshold(
    model: GPT, cfg: TrainConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """When total grad_norm exceeds grad_norm_spike_threshold, a 'spike' line
    must be emitted at DEBUG level for every layer in layer_norms."""
    # Set spike threshold to 0.0 so any non-zero gradient triggers it.
    spike_cfg = TrainConfig(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
        batch_size=2,
        grad_norm_spike_threshold=0.001,  # guaranteed to fire
    )
    torch.manual_seed(0)
    spike_model = GPT(spike_cfg)
    _run_backward(spike_model, spike_cfg)
    layer_norms = _get_layer_norms(spike_model)

    spike_logger = GradientLogger(spike_cfg)
    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        spike_logger.log_step(step=42, loss=1.0, lr=1e-3, layer_norms=layer_norms)

    spike_msgs = [msg for msg in caplog.messages if msg.startswith("spike ")]
    assert len(spike_msgs) == len(
        layer_norms
    ), f"Expected {len(layer_norms)} spike lines, got {len(spike_msgs)}"
    for name in layer_norms:
        assert any(
            f"layer={name}" in msg for msg in spike_msgs
        ), f"No spike line found for layer '{name}'"
    # Format: spike step=N layer=<name> norm=X.XXXX
    for msg in spike_msgs:
        assert "step=42" in msg
        assert "norm=" in msg


def test_no_spike_dump_below_threshold(
    model: GPT,
    logger: GradientLogger,
    cfg: TrainConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No spike lines must appear when total grad_norm is below the threshold."""
    # cfg.grad_norm_spike_threshold defaults to 3.0; set it very high
    high_cfg = TrainConfig(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
        batch_size=2,
        grad_norm_spike_threshold=1000.0,
    )
    torch.manual_seed(0)
    high_model = GPT(high_cfg)
    _run_backward(high_model, high_cfg)
    layer_norms = _get_layer_norms(high_model)

    with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
        GradientLogger(high_cfg).log_step(
            step=1, loss=1.0, lr=1e-3, layer_norms=layer_norms
        )

    spike_msgs = [msg for msg in caplog.messages if msg.startswith("spike ")]
    assert spike_msgs == [], f"Unexpected spike messages: {spike_msgs}"
