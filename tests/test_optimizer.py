"""Tests for src/optimizer.py.

All tests use the GPT model with a tiny synthetic config — no GPU, no data.
"""
from __future__ import annotations

import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TrainConfig
from src.model import GPT
from src.optimizer import make_optimizer


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
        learning_rate=1e-3,
        weight_decay=0.1,
    )


@pytest.fixture
def model(cfg: TrainConfig) -> GPT:
    return GPT(cfg)


# ── Return type ───────────────────────────────────────────────────────────────


def test_returns_adamw(model: GPT, cfg: TrainConfig) -> None:
    """make_optimizer must return a torch.optim.AdamW instance."""
    opt = make_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)


# ── Param group structure ─────────────────────────────────────────────────────


def test_two_param_groups(model: GPT, cfg: TrainConfig) -> None:
    """Optimizer must have exactly two parameter groups (decay and no-decay)."""
    opt = make_optimizer(model, cfg)
    assert len(opt.param_groups) == 2


def test_decay_group_has_correct_weight_decay(model: GPT, cfg: TrainConfig) -> None:
    """The decay group must use cfg.weight_decay."""
    opt = make_optimizer(model, cfg)
    decay_groups = [g for g in opt.param_groups if g["weight_decay"] != 0.0]
    assert len(decay_groups) == 1
    assert decay_groups[0]["weight_decay"] == cfg.weight_decay


def test_no_decay_group_has_zero_weight_decay(model: GPT, cfg: TrainConfig) -> None:
    """The no-decay group must have weight_decay == 0.0."""
    opt = make_optimizer(model, cfg)
    no_decay_groups = [g for g in opt.param_groups if g["weight_decay"] == 0.0]
    assert len(no_decay_groups) == 1
    assert no_decay_groups[0]["weight_decay"] == 0.0


def test_learning_rate_set_from_cfg(model: GPT, cfg: TrainConfig) -> None:
    """All param groups must use cfg.learning_rate."""
    opt = make_optimizer(model, cfg)
    for group in opt.param_groups:
        assert group["lr"] == cfg.learning_rate


# ── Param assignment correctness ──────────────────────────────────────────────


def test_layernorm_params_in_no_decay_group(model: GPT, cfg: TrainConfig) -> None:
    """Every parameter belonging to a LayerNorm module must be in no-decay group."""
    opt = make_optimizer(model, cfg)
    no_decay_ids = {
        id(p)
        for g in opt.param_groups
        if g["weight_decay"] == 0.0
        for p in g["params"]
    }
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                assert id(param) in no_decay_ids, (
                    "LayerNorm parameter not found in no-decay group"
                )


def test_bias_params_in_no_decay_group(model: GPT, cfg: TrainConfig) -> None:
    """Any parameter with 'bias' in its name must be in the no-decay group."""
    opt = make_optimizer(model, cfg)
    no_decay_ids = {
        id(p)
        for g in opt.param_groups
        if g["weight_decay"] == 0.0
        for p in g["params"]
    }
    for name, param in model.named_parameters():
        if "bias" in name:
            assert id(param) in no_decay_ids, (
                f"Bias parameter '{name}' not found in no-decay group"
            )


def test_non_bias_non_norm_params_in_decay_group(model: GPT, cfg: TrainConfig) -> None:
    """Parameters that are neither LayerNorm nor bias must be in the decay group."""
    opt = make_optimizer(model, cfg)
    decay_ids = {
        id(p)
        for g in opt.param_groups
        if g["weight_decay"] != 0.0
        for p in g["params"]
    }
    ln_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                ln_ids.add(id(param))

    for name, param in model.named_parameters():
        if "bias" not in name and id(param) not in ln_ids:
            assert id(param) in decay_ids, (
                f"Non-bias, non-LayerNorm parameter '{name}' not found in decay group"
            )


def test_all_params_covered_exactly_once(model: GPT, cfg: TrainConfig) -> None:
    """Every trainable model parameter must appear in exactly one group."""
    opt = make_optimizer(model, cfg)
    all_ids_in_groups = [id(p) for g in opt.param_groups for p in g["params"]]
    trainable_ids = [id(p) for p in model.parameters() if p.requires_grad]

    # Same count (no duplicates, no missing)
    assert len(all_ids_in_groups) == len(trainable_ids), (
        f"Group param count ({len(all_ids_in_groups)}) != model param count ({len(trainable_ids)})"
    )
    assert sorted(all_ids_in_groups) == sorted(trainable_ids), (
        "Optimizer param groups do not exactly match the model's trainable parameters"
    )


def test_frozen_params_excluded_from_groups(model: GPT, cfg: TrainConfig) -> None:
    """Parameters with requires_grad=False must not appear in any param group."""
    # Freeze the first block's attention weights
    first_qkv = model.blocks[0].attn.qkv
    first_qkv.weight.requires_grad_(False)

    opt = make_optimizer(model, cfg)
    group_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(first_qkv.weight) not in group_ids, (
        "Frozen parameter must not be included in any optimizer group"
    )

    # Restore for other tests.
    first_qkv.weight.requires_grad_(True)


# ── State save / load ─────────────────────────────────────────────────────────


def test_state_save_load_identity(model: GPT, cfg: TrainConfig) -> None:
    """Optimizer state_dict must survive a save/load round-trip exactly.

    After loading, momentum buffers (exp_avg, exp_avg_sq) and step counts must
    be bit-identical to the original.
    """
    torch.manual_seed(42)
    opt = make_optimizer(model, cfg)

    # Populate optimizer state with one forward/backward/step pass.
    idx = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()

    # Capture state before save.
    sd_before = opt.state_dict()

    # Round-trip through BytesIO.
    buf = io.BytesIO()
    torch.save(sd_before, buf)
    buf.seek(0)

    opt2 = make_optimizer(model, cfg)
    opt2.load_state_dict(torch.load(buf, weights_only=True))
    sd_after = opt2.state_dict()

    # Param group hyperparameters must match.
    for g1, g2 in zip(sd_before["param_groups"], sd_after["param_groups"]):
        assert g1["weight_decay"] == g2["weight_decay"]
        assert g1["lr"] == g2["lr"]

    # Internal state tensors (step, exp_avg, exp_avg_sq) must be identical.
    for key in sd_before["state"]:
        for tensor_key, val in sd_before["state"][key].items():
            loaded_val = sd_after["state"][key][tensor_key]
            if isinstance(val, torch.Tensor):
                assert torch.equal(val, loaded_val), (
                    f"State mismatch at param {key}, field '{tensor_key}'"
                )
            else:
                assert val == loaded_val, (
                    f"State mismatch at param {key}, field '{tensor_key}'"
                )
