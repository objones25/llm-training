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
from src.scheduler import make_scheduler


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


# ── Scheduler state (CONTRIBUTING.md rule 11) ─────────────────────────────────


def test_scheduler_state_round_trip(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> None:
    """Scheduler LR at the step after load must match the reference run.

    Validates that resuming from a checkpoint reproduces the exact LR trajectory.
    """
    scheduler = make_scheduler(optimizer, cfg)

    # Advance scheduler by 3 steps to move past step 0.
    for _ in range(3):
        scheduler.step()
    lr_before = optimizer.param_groups[0]["lr"]

    path = save_checkpoint(model, optimizer, step=3, cfg=cfg, scheduler=scheduler)

    # Restore into a fresh optimizer + scheduler.
    optimizer2 = make_optimizer(model, cfg)
    scheduler2 = make_scheduler(optimizer2, cfg)
    load_checkpoint(path, model, optimizer2, scheduler=scheduler2)

    # One more step on both — LRs must be identical.
    scheduler.step()
    scheduler2.step()
    lr_after = optimizer.param_groups[0]["lr"]
    lr_after2 = optimizer2.param_groups[0]["lr"]

    assert abs(lr_after - lr_after2) < 1e-9, (
        f"LR mismatch after scheduler round-trip: {lr_after} vs {lr_after2}"
    )
    _ = lr_before  # referenced only to confirm step > 0


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


# ── save_as_best tests ────────────────────────────────────────────────────────


def test_save_as_best_creates_best_pt(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> None:
    """save_as_best=True must create best.pt in checkpoint_dir."""
    path = save_checkpoint(model, optimizer, step=500, cfg=cfg, save_as_best=True)
    assert path.exists()
    assert path.name == "best.pt"


def test_save_as_best_filename(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> None:
    """Path returned when save_as_best=True must end in best.pt."""
    path = save_checkpoint(model, optimizer, step=1000, cfg=cfg, save_as_best=True)
    assert path.name == "best.pt"


def test_save_as_best_overwrites(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig
) -> None:
    """Two saves with save_as_best=True must produce exactly one file."""
    p1 = save_checkpoint(model, optimizer, step=100, cfg=cfg, save_as_best=True)
    p2 = save_checkpoint(model, optimizer, step=200, cfg=cfg, save_as_best=True)
    assert p1 == p2  # same path
    ckpt_dir = Path(cfg.checkpoint_dir)
    best_files = list(ckpt_dir.glob("best.pt"))
    assert len(best_files) == 1, "More than one best.pt file found"


# ── Tuple optimizer (Muon + AdamW) checkpoint tests ───────────────────────────


@pytest.fixture
def muon_cfg(tmp_path: pytest.TempPathFactory) -> TrainConfig:
    return TrainConfig(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
        batch_size=2,
        use_muon=True,
        checkpoint_dir=str(tmp_path / "muon_checkpoints"),
    )


@pytest.fixture
def muon_model(muon_cfg: TrainConfig) -> GPT:
    torch.manual_seed(0)
    return GPT(muon_cfg)


@pytest.fixture
def tuple_optimizer(muon_model: GPT, muon_cfg: TrainConfig):
    from src.optimizer import make_optimizer
    return make_optimizer(muon_model, muon_cfg)


def test_save_load_tuple_optimizer(
    muon_model: GPT, tuple_optimizer, muon_cfg: TrainConfig
) -> None:
    """save/load round-trip with a (Muon, AdamW) tuple optimizer must work."""
    # Run one step to populate optimizer state.
    torch.manual_seed(42)
    muon_opt, adamw_opt = tuple_optimizer
    idx = torch.randint(0, muon_cfg.vocab_size, (muon_cfg.batch_size, muon_cfg.seq_len))
    targets = torch.randint(0, muon_cfg.vocab_size, (muon_cfg.batch_size, muon_cfg.seq_len))
    import torch.nn.functional as F
    logits = muon_model(idx)
    loss = F.cross_entropy(logits.view(-1, muon_cfg.vocab_size), targets.view(-1))
    loss.backward()
    muon_opt.step()
    adamw_opt.step()
    muon_opt.zero_grad()
    adamw_opt.zero_grad()

    path = save_checkpoint(muon_model, tuple_optimizer, step=1, cfg=muon_cfg)
    assert path.exists()

    restored_step = load_checkpoint(path, muon_model, tuple_optimizer)
    assert restored_step == 1


def test_logit_identity_tuple_optimizer(
    muon_model: GPT, tuple_optimizer, muon_cfg: TrainConfig
) -> None:
    """Logits must be bit-identical after save/load with tuple optimizer."""
    torch.manual_seed(42)
    idx = torch.randint(0, muon_cfg.vocab_size, (muon_cfg.batch_size, muon_cfg.seq_len))

    muon_model.eval()
    with torch.no_grad():
        logits_before = muon_model(idx).clone()

    path = save_checkpoint(muon_model, tuple_optimizer, step=1, cfg=muon_cfg)

    for param in muon_model.parameters():
        param.data.fill_(0.0)

    load_checkpoint(path, muon_model, tuple_optimizer)

    muon_model.eval()
    with torch.no_grad():
        logits_after = muon_model(idx)

    assert torch.equal(logits_before, logits_after), (
        "Logits after load (tuple optimizer) do not match logits before save"
    )


def test_optimizer_type_mismatch_raises(
    model: GPT, optimizer: torch.optim.AdamW, cfg: TrainConfig,
    muon_model: GPT, tuple_optimizer, muon_cfg: TrainConfig,
) -> None:
    """Loading a single-optimizer checkpoint with a tuple optimizer must raise ValueError."""
    path = save_checkpoint(model, optimizer, step=1, cfg=cfg)
    with pytest.raises(ValueError, match="mismatch"):
        load_checkpoint(path, muon_model, tuple_optimizer)
