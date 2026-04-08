"""Tests for scripts/evaluate.py.

All tests use synthetic data and a tiny 2-layer GPT on CPU.
No real checkpoints, no disk val.bin, no GPU, no network calls.

Rules covered
-------------
 - find_latest_checkpoint: empty dir, single file, multiple files, missing dir
 - load_checkpoint_for_eval: round-trip (save -> load) preserves cfg and weights
 - compute_perplexity: finite, non-negative, raises on empty batches
 - sample_text: returns list of ints of correct length, seed tokens preserved
 - Smoke: save checkpoint -> load -> compute_perplexity
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from scripts.evaluate import (
    compute_perplexity,
    find_latest_checkpoint,
    load_checkpoint_for_eval,
    sample_text,
)
from src.checkpoint import save_checkpoint
from src.config import TrainConfig
from src.model import GPT
from src.optimizer import make_optimizer

# -- Helpers ------------------------------------------------------------------


def _cfg(**overrides) -> TrainConfig:
    defaults: dict = dict(
        vocab_size=256,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
        seq_len=16,
        batch_size=2,
        max_steps=100,
        warmup_steps=5,
        checkpoint_every=1000,
        plot_every=1000,
        device="cpu",
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _make_val_batches(
    cfg: TrainConfig, n: int = 3
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return *n* synthetic (inputs, targets) pairs on CPU."""
    torch.manual_seed(42)
    batches = []
    for _ in range(n):
        inputs = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
        targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
        batches.append((inputs, targets))
    return batches


def _save_fake_checkpoint(tmp_path: Path, step: int = 10):
    """Save a real checkpoint using save_checkpoint and return its path."""
    cfg = _cfg(checkpoint_dir=str(tmp_path / "ckpts"))
    torch.manual_seed(0)
    model = GPT(cfg)
    optimizer = make_optimizer(model, cfg)
    ckpt_path = save_checkpoint(model, optimizer, step, cfg)
    return ckpt_path, model, cfg


# -- find_latest_checkpoint ---------------------------------------------------


def test_find_latest_checkpoint_missing_dir(tmp_path: Path) -> None:
    """Returns None when the directory does not exist."""
    result = find_latest_checkpoint(tmp_path / "nonexistent")
    assert result is None


def test_find_latest_checkpoint_empty_dir(tmp_path: Path) -> None:
    """Returns None when the directory exists but has no .pt files."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    result = find_latest_checkpoint(ckpt_dir)
    assert result is None


def test_find_latest_checkpoint_single_file(tmp_path: Path) -> None:
    """Returns the only .pt file when exactly one checkpoint exists."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    only = ckpt_dir / "checkpoint_0000001.pt"
    only.touch()
    result = find_latest_checkpoint(ckpt_dir)
    assert result == only


def test_find_latest_checkpoint_multiple_files(tmp_path: Path) -> None:
    """Returns the highest-numbered checkpoint when multiple exist."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    for n in (1, 5, 3):
        (ckpt_dir / f"checkpoint_{n:07d}.pt").touch()
    result = find_latest_checkpoint(ckpt_dir)
    assert result == ckpt_dir / "checkpoint_0000005.pt"


def test_find_latest_checkpoint_ignores_non_pt_files(tmp_path: Path) -> None:
    """Non-.pt files must not be returned."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    (ckpt_dir / "checkpoint_0000001.txt").touch()
    result = find_latest_checkpoint(ckpt_dir)
    assert result is None


# -- load_checkpoint_for_eval -------------------------------------------------


def test_load_checkpoint_for_eval_raises_on_missing(tmp_path: Path) -> None:
    """FileNotFoundError must be raised when the .pt file does not exist."""
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint_for_eval(tmp_path / "ghost.pt")


def test_load_checkpoint_for_eval_returns_correct_types(tmp_path: Path) -> None:
    """Returns (GPT, TrainConfig, int) from a real checkpoint."""
    ckpt_path, _, _ = _save_fake_checkpoint(tmp_path)
    model, cfg, step = load_checkpoint_for_eval(ckpt_path)
    assert isinstance(model, GPT)
    assert isinstance(cfg, TrainConfig)
    assert isinstance(step, int)


def test_load_checkpoint_for_eval_step_matches(tmp_path: Path) -> None:
    """Returned step must equal the step saved in the checkpoint."""
    ckpt_path, _, _ = _save_fake_checkpoint(tmp_path, step=42)
    _, _, step = load_checkpoint_for_eval(ckpt_path)
    assert step == 42


def test_load_checkpoint_for_eval_weights_match(tmp_path: Path) -> None:
    """Loaded model weights must be identical to the saved model weights."""
    ckpt_path, original_model, cfg = _save_fake_checkpoint(tmp_path)
    loaded_model, _, _ = load_checkpoint_for_eval(ckpt_path)
    for (name, orig_p), (_, loaded_p) in zip(
        original_model.named_parameters(), loaded_model.named_parameters()
    ):
        assert torch.equal(
            orig_p.data, loaded_p.data
        ), f"Weight mismatch for parameter '{name}'"


# -- compute_perplexity -------------------------------------------------------


def test_compute_perplexity_raises_on_empty_batches() -> None:
    """ValueError must be raised when val_batches is empty."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    with pytest.raises(ValueError, match="empty"):
        compute_perplexity(model, [], cfg, torch.device("cpu"))


def test_compute_perplexity_returns_finite() -> None:
    """avg_loss and perplexity must both be finite."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    val_batches = _make_val_batches(cfg)
    avg_loss, ppl = compute_perplexity(model, val_batches, cfg, torch.device("cpu"))
    assert math.isfinite(avg_loss)
    assert math.isfinite(ppl)


def test_compute_perplexity_loss_non_negative() -> None:
    """avg_loss must be >= 0 (cross-entropy is always non-negative)."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    val_batches = _make_val_batches(cfg)
    avg_loss, _ = compute_perplexity(model, val_batches, cfg, torch.device("cpu"))
    assert avg_loss >= 0.0


def test_compute_perplexity_equals_exp_loss() -> None:
    """perplexity must equal exp(avg_loss) exactly."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    val_batches = _make_val_batches(cfg)
    avg_loss, ppl = compute_perplexity(model, val_batches, cfg, torch.device("cpu"))
    assert abs(ppl - math.exp(avg_loss)) < 1e-4


def test_compute_perplexity_sets_eval_mode() -> None:
    """Model must be in eval mode after compute_perplexity returns."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    model.train()
    val_batches = _make_val_batches(cfg)
    compute_perplexity(model, val_batches, cfg, torch.device("cpu"))
    assert not model.training, "Model must be in eval mode after compute_perplexity"


# -- sample_text --------------------------------------------------------------


def test_sample_text_returns_list_of_ints() -> None:
    """sample_text must return a list of integers."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    seed = [1, 2, 3]
    result = sample_text(model, cfg, torch.device("cpu"), seed, max_new_tokens=5)
    assert isinstance(result, list)
    assert all(isinstance(t, int) for t in result)


def test_sample_text_length() -> None:
    """Output length must equal len(seed_tokens) + max_new_tokens."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    seed = [1, 2, 3]
    max_new = 10
    result = sample_text(model, cfg, torch.device("cpu"), seed, max_new_tokens=max_new)
    assert len(result) == len(seed) + max_new


def test_sample_text_seed_preserved() -> None:
    """The first len(seed_tokens) elements must be the original seed tokens."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    seed = [7, 13, 42]
    result = sample_text(model, cfg, torch.device("cpu"), seed, max_new_tokens=5)
    assert result[: len(seed)] == seed


def test_sample_text_tokens_in_vocab() -> None:
    """All generated token IDs must be within [0, vocab_size)."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    seed = [0, 1]
    result = sample_text(model, cfg, torch.device("cpu"), seed, max_new_tokens=20)
    assert all(
        0 <= t < cfg.vocab_size for t in result
    ), "Generated token IDs must be within vocab bounds"


def test_sample_text_greedy_deterministic() -> None:
    """top_k=1 (greedy) must produce identical output on two identical calls."""
    cfg = _cfg()
    torch.manual_seed(0)
    model = GPT(cfg)
    seed = [5, 10]
    r1 = sample_text(model, cfg, torch.device("cpu"), seed, max_new_tokens=8, top_k=1)
    r2 = sample_text(model, cfg, torch.device("cpu"), seed, max_new_tokens=8, top_k=1)
    assert r1 == r2


# -- Smoke: end-to-end save -> load -> evaluate --------------------------------


def test_evaluate_smoke(tmp_path: Path) -> None:
    """End-to-end: save a checkpoint, load it for eval, compute perplexity.

    Must complete in <10s on CPU with a tiny model.
    """
    ckpt_path, _, orig_cfg = _save_fake_checkpoint(tmp_path, step=5)

    model, cfg, step = load_checkpoint_for_eval(ckpt_path)
    assert step == 5
    assert cfg.vocab_size == orig_cfg.vocab_size

    val_batches = _make_val_batches(cfg, n=4)
    avg_loss, ppl = compute_perplexity(model, val_batches, cfg, torch.device("cpu"))

    assert math.isfinite(avg_loss)
    assert ppl > 1.0, "Perplexity of a random LM must be > 1"
    assert ppl < 10_000.0, f"Suspiciously high perplexity: {ppl:.2f}"
