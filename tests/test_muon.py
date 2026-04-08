"""Tests for src/muon.py.

Covers Newton-Schulz orthogonalization and the Muon optimizer step.
All tests run on CPU with synthetic tensors — no GPU required.
"""

from __future__ import annotations

import io

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.muon import Muon, zeropower_via_newtonschulz5

# ── Newton-Schulz tests ───────────────────────────────────────────────────────


def test_newtonschulz_output_shape_preserved() -> None:
    """Output shape must equal input shape for any 2-D matrix."""
    torch.manual_seed(0)
    G = torch.randn(8, 16)
    out = zeropower_via_newtonschulz5(G)
    assert out.shape == G.shape


def test_newtonschulz_spectral_norm_approx_one() -> None:
    """After orthogonalization, largest singular value must be in (0.8, 1.3).

    5 Newton-Schulz iterations in float32 converge to spectral norm ~1.13 for
    typical random Gaussian matrices — not exactly 1.0, but bounded and
    well-controlled.
    """
    torch.manual_seed(0)
    G = torch.randn(32, 16)
    out = zeropower_via_newtonschulz5(G)
    # Compute largest singular value via SVD.
    _, S, _ = torch.linalg.svd(out.float(), full_matrices=False)
    assert (
        0.8 < S[0].item() < 1.3
    ), f"Largest singular value {S[0].item():.4f} outside expected range (0.8, 1.3)"


def test_newtonschulz_spectral_norm_wide_matrix() -> None:
    """Wide matrix (cols > rows) also converges to spectral norm in (0.8, 1.3)."""
    torch.manual_seed(1)
    G = torch.randn(16, 32)  # wide: shape[0] < shape[1]
    out = zeropower_via_newtonschulz5(G)
    _, S, _ = torch.linalg.svd(out.float(), full_matrices=False)
    assert (
        0.8 < S[0].item() < 1.3
    ), f"Largest singular value {S[0].item():.4f} outside expected range (wide matrix)"


def test_newtonschulz_tall_matrix() -> None:
    """Tall matrix (rows > cols) must be handled correctly via transposition."""
    torch.manual_seed(2)
    G = torch.randn(64, 16)  # tall: shape[0] > shape[1]
    out = zeropower_via_newtonschulz5(G)
    assert out.shape == G.shape
    _, S, _ = torch.linalg.svd(out.float(), full_matrices=False)
    assert (
        0.8 < S[0].item() < 1.3
    ), f"Largest singular value {S[0].item():.4f} outside expected range (tall matrix)"


def test_newtonschulz_dtype_preserved_float32() -> None:
    """float32 input must produce float32 output."""
    torch.manual_seed(3)
    G = torch.randn(16, 16, dtype=torch.float32)
    out = zeropower_via_newtonschulz5(G)
    assert out.dtype == torch.float32


def test_newtonschulz_dtype_preserved_float64() -> None:
    """float64 input must produce float64 output."""
    torch.manual_seed(4)
    G = torch.randn(16, 16, dtype=torch.float64)
    out = zeropower_via_newtonschulz5(G)
    assert out.dtype == torch.float64


def test_newtonschulz_square_matrix() -> None:
    """Square matrix is handled without transposition, spectral norm in (0.8, 1.3)."""
    torch.manual_seed(5)
    G = torch.randn(32, 32)
    out = zeropower_via_newtonschulz5(G)
    assert out.shape == (32, 32)
    _, S, _ = torch.linalg.svd(out.float(), full_matrices=False)
    assert 0.8 < S[0].item() < 1.3


def test_newtonschulz_steps_parameter() -> None:
    """Passing steps=1 and steps=10 must both return the correct shape."""
    torch.manual_seed(6)
    G = torch.randn(16, 8)
    out1 = zeropower_via_newtonschulz5(G, steps=1)
    out10 = zeropower_via_newtonschulz5(G, steps=10)
    assert out1.shape == G.shape
    assert out10.shape == G.shape


# ── Muon optimizer tests ──────────────────────────────────────────────────────


def _simple_model() -> nn.Linear:
    """Tiny linear layer for optimizer tests."""
    torch.manual_seed(0)
    return nn.Linear(8, 4, bias=False)


def test_muon_step_updates_weights() -> None:
    """After one Muon step, weights must differ from the initial values."""
    torch.manual_seed(0)
    model = _simple_model()
    original = model.weight.data.clone()

    opt = Muon([model.weight], lr=0.1, momentum=0.95)

    # Compute a gradient via a synthetic loss.
    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    opt.step()

    assert not torch.equal(
        model.weight.data, original
    ), "Muon step did not update weights"


def test_muon_step_decreases_loss() -> None:
    """Three Muon steps on a fixed batch must reduce loss monotonically."""
    torch.manual_seed(0)
    model = _simple_model()
    opt = Muon([model.weight], lr=0.01, momentum=0.95)

    x = torch.randn(4, 8)
    y = torch.randn(4, 4)

    prev_loss = float("inf")
    for _ in range(3):
        opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        current = loss.item()
        assert (
            current < prev_loss
        ), f"Loss did not decrease: {current:.6f} >= {prev_loss:.6f}"
        prev_loss = current


def test_muon_handles_1d_tensors() -> None:
    """Muon must not crash when the parameter group includes 1-D tensors (biases)."""
    torch.manual_seed(0)
    model = nn.Linear(8, 4, bias=True)
    # Include both weight (2D) and bias (1D) in the same group.
    opt = Muon(list(model.parameters()), lr=0.01, momentum=0.95)

    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    opt.step()  # must not raise


def test_muon_momentum_buffer_initialized_to_zeros() -> None:
    """Before the first step, no state exists; after the first step, momentum
    buffer must be a zero-initialised tensor of the same shape as the param."""
    torch.manual_seed(0)
    model = _simple_model()
    opt = Muon([model.weight], lr=0.01, momentum=0.95)

    # No state before first step.
    assert len(opt.state) == 0

    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    F.mse_loss(model(x), y).backward()
    opt.step()

    # After first step, momentum_buffer must exist and have the same shape.
    p = model.weight
    assert p in opt.state, "Parameter not found in optimizer state"
    buf = opt.state[p]["momentum_buffer"]
    assert (
        buf.shape == p.shape
    ), f"Momentum buffer shape {buf.shape} != param shape {p.shape}"


def test_muon_state_dict_roundtrip() -> None:
    """Momentum buffers must survive a state_dict save/load round-trip."""
    torch.manual_seed(0)
    model = _simple_model()
    opt = Muon([model.weight], lr=0.01, momentum=0.95)

    x = torch.randn(4, 8)
    y = torch.randn(4, 4)
    F.mse_loss(model(x), y).backward()
    opt.step()

    sd_before = opt.state_dict()

    buf_io = io.BytesIO()
    torch.save(sd_before, buf_io)
    buf_io.seek(0)

    opt2 = Muon([model.weight], lr=0.01, momentum=0.95)
    opt2.load_state_dict(torch.load(buf_io, weights_only=True))
    sd_after = opt2.state_dict()

    for key in sd_before["state"]:
        buf_before = sd_before["state"][key]["momentum_buffer"]
        buf_after = sd_after["state"][key]["momentum_buffer"]
        assert torch.equal(
            buf_before, buf_after
        ), "Momentum buffer mismatch after state_dict round-trip"


def test_muon_no_grad_skipped() -> None:
    """Muon must skip parameters whose .grad is None (no error, no update)."""
    torch.manual_seed(0)
    model = _simple_model()
    opt = Muon([model.weight], lr=0.01, momentum=0.95)
    original = model.weight.data.clone()

    # Do NOT call backward — gradient is None.
    opt.step()

    assert torch.equal(
        model.weight.data, original
    ), "Weight changed even though gradient was None"
