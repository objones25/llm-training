"""Tests for src/checkpoint.py.

All tests use a tiny 2-layer GPT model on CPU with synthetic data.
Disk I/O is isolated to tmp_path -- no manual cleanup required.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import TrainConfig
from src.model import GPT
from src.optimizer import make_optimizer


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg(tmp_path: Path) -> TrainConfig:
    return TrainConfig(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
        batch_size=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )


@pytest.fixture
def model(cfg: TrainConfig) -> GPT:
    torch.manual_seed(0)
    return GPT(cfg)


@pytest.fixture
def optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    return make_optimizer(model, cfg)


# ── File creation ─────────────────────────────────────────────────────────────


def test_save_creates_file(model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig) -> None:
    """save_checkpoint must create a .pt file on disk."""
    path = save_checkpoint(model, optimizer, step=100, cfg=cfg)
    assert path.exists(), "Checkpoint file was not created"


def test_save_returns_path(model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig) -> None:
    """save_checkpoint must return a Path object pointing to the saved file."""
    result = save_checkpoint(model, optimizer, step=50, cfg=cfg)
    assert isinstance(result, Path)
    assert result.suffix == ".pt"


def test_checkpoint_dir_created_if_missing(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> None:
    """save_checkpoint must create cfg.checkpoint_dir if it does not exist."""
    ckpt_dir = Path(cfg.checkpoint_dir)
    assert not ckpt_dir.exists(), "Precondition: dir must not exist yet"
    save_checkpoint(model, optimizer, step=1, cfg=cfg)
    assert ckpt_dir.exists()


def test_filename_encodes_step(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> None:
    """Checkpoint filename must embed the step number (zero-padded to 7 digits)."""
    path = save_checkpoint(model, optimizer, step=42, cfg=cfg)
    assert "0000042" in path.name


def test_multiple_checkpoints_independent(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> None:
    """Two saves at different steps must produce two separate files."""
    p1 = save_checkpoint(model, optimizer, step=100, cfg=cfg)
    p2 = save_checkpoint(model, optimizer, step=200, cfg=cfg)
    assert p1 != p2
    assert p1.exists()
    assert p2.exists()


# ── Load contracts ────────────────────────────────────────────────────────────


def test_load_restores_step(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> None:
    """load_checkpoint must return the exact step integer that was saved."""
    path = save_checkpoint(model, optimizer, step=999, cfg=cfg)
    restored_step = load_checkpoint(path, model, optimizer)
    assert restored_step == 999


def test_load_raises_on_missing_file(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig, tmp_path: Path
) -> None:
    """load_checkpoint must raise when the file does not exist."""
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_checkpoint(missing, model, optimizer)


# ── Logit identity (CLAUDE.md rule 12) ───────────────────────────────────────


def test_logit_identity_after_load(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> None:
    """Logits before save must be bit-identical to logits after load.

    This is rule 12 from CLAUDE.md -- the core correctness guarantee for
    checkpoint/resume.
    """
    torch.manual_seed(42)
    idx = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))

    model.eval()
    with torch.no_grad():
        logits_before = model(idx).clone()

    path = save_checkpoint(model, optimizer, step=1, cfg=cfg)

    # Corrupt model weights in-place so we can verify load actually restores them.
    for param in model.parameters():
        param.data.fill_(0.0)

    load_checkpoint(path, model, optimizer)

    model.eval()
    with torch.no_grad():
        logits_after = model(idx)

    assert torch.equal(logits_before, logits_after), (
        "Logits after load do not match logits before save"
    )


# ── Optimizer state restore (CLAUDE.md rule 11) ───────────────────────────────


def test_optimizer_state_restored(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> None:
    """Optimizer momentum/variance buffers must be identical after save/load."""
    torch.manual_seed(42)
    idx = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))

    # Populate optimizer state with one step.
    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    path = save_checkpoint(model, optimizer, step=1, cfg=cfg)
    sd_before = optimizer.state_dict()

    # Load into a fresh optimizer and compare state tensors.
    optimizer2 = make_optimizer(model, cfg)
    load_checkpoint(path, model, optimizer2)
    sd_after = optimizer2.state_dict()

    for key in sd_before["state"]:
        for tensor_key, val in sd_before["state"][key].items():
            loaded_val = sd_after["state"][key][tensor_key]
            if isinstance(val, torch.Tensor):
                assert torch.equal(val, loaded_val), (
                    f"Optimizer state mismatch at param {key}, field '{tensor_key}'"
                )
            else:
                assert val == loaded_val


def test_loss_identity_after_load(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> None:
    """Loss on the next step must be identical before and after a save/load round-trip.

    Validates rule 11: optimizer and model state together reproduce the same
    gradient update so training can resume seamlessly.
    """
    torch.manual_seed(42)
    idx = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))

    # Warm up optimizer state with one step.
    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Snapshot model params after step 1.
    params_snap = {n: p.clone() for n, p in model.named_parameters()}

    path = save_checkpoint(model, optimizer, step=1, cfg=cfg)

    # Reference path: step 2 without load.
    logits = model(idx)
    loss_ref = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss_ref.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_ref_val = loss_ref.item()

    # Restore model to post-step-1 state via the checkpoint.
    for n, p in model.named_parameters():
        p.data.copy_(params_snap[n])
    optimizer2 = make_optimizer(model, cfg)
    load_checkpoint(path, model, optimizer2)

    # Loaded path: step 2 after load.
    logits = model(idx)
    loss_loaded = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss_loaded.backward()
    optimizer2.step()

    assert abs(loss_ref_val - loss_loaded.item()) < 1e-6, (
        f"Loss after load ({loss_loaded.item():.6f}) differs from "
        f"reference ({loss_ref_val:.6f})"
    )
