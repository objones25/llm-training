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
        ln_lr_mult=3.0,
        embed_lr_mult=0.1,
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


def test_three_param_groups(model: GPT, cfg: TrainConfig) -> None:
    """Optimizer must have exactly three parameter groups (ln, embed, matrix)."""
    opt = make_optimizer(model, cfg)
    assert len(opt.param_groups) == 3


def test_exactly_one_weight_decay_group(model: GPT, cfg: TrainConfig) -> None:
    """Exactly one group (matrix) must have non-zero weight decay."""
    opt = make_optimizer(model, cfg)
    decay_groups = [g for g in opt.param_groups if g["weight_decay"] != 0.0]
    assert len(decay_groups) == 1
    assert decay_groups[0]["weight_decay"] == cfg.weight_decay


def test_two_no_decay_groups(model: GPT, cfg: TrainConfig) -> None:
    """Exactly two groups (ln, embed) must have weight_decay == 0.0."""
    opt = make_optimizer(model, cfg)
    no_decay_groups = [g for g in opt.param_groups if g["weight_decay"] == 0.0]
    assert len(no_decay_groups) == 2


# ── Per-group learning rates ──────────────────────────────────────────────────


def test_matrix_group_uses_base_lr(model: GPT, cfg: TrainConfig) -> None:
    """The matrix (weight decay) group must use cfg.learning_rate exactly."""
    opt = make_optimizer(model, cfg)
    decay_group = next(g for g in opt.param_groups if g["weight_decay"] != 0.0)
    assert decay_group["lr"] == cfg.learning_rate


def test_ln_group_uses_multiplied_lr(model: GPT, cfg: TrainConfig) -> None:
    """The LayerNorm group LR must be base_lr × ln_lr_mult."""
    opt = make_optimizer(model, cfg)
    # The ln group contains LayerNorm weight/bias — identify it by checking
    # that one of its params belongs to a LayerNorm module.
    ln_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters():
                ln_param_ids.add(id(p))

    ln_group = next(
        g for g in opt.param_groups
        if any(id(p) in ln_param_ids for p in g["params"])
    )
    assert ln_group["lr"] == pytest.approx(cfg.learning_rate * cfg.ln_lr_mult)


def test_embed_group_uses_multiplied_lr(model: GPT, cfg: TrainConfig) -> None:
    """The embedding group LR must be base_lr × embed_lr_mult."""
    opt = make_optimizer(model, cfg)
    embed_param_ids: set[int] = {
        id(p) for name, p in model.named_parameters() if "embedding" in name
    }
    embed_group = next(
        g for g in opt.param_groups
        if any(id(p) in embed_param_ids for p in g["params"])
    )
    assert embed_group["lr"] == pytest.approx(cfg.learning_rate * cfg.embed_lr_mult)


# ── Param assignment correctness ──────────────────────────────────────────────


def test_layernorm_params_in_no_decay_group(model: GPT, cfg: TrainConfig) -> None:
    """Every parameter belonging to a LayerNorm module must be in a no-decay group."""
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
                    "LayerNorm parameter not found in a no-decay group"
                )


def test_embedding_params_in_no_decay_group(model: GPT, cfg: TrainConfig) -> None:
    """Embedding parameters must be in a no-decay group."""
    opt = make_optimizer(model, cfg)
    no_decay_ids = {
        id(p)
        for g in opt.param_groups
        if g["weight_decay"] == 0.0
        for p in g["params"]
    }
    for name, param in model.named_parameters():
        if "embedding" in name:
            assert id(param) in no_decay_ids, (
                f"Embedding parameter '{name}' not found in a no-decay group"
            )


def test_embedding_params_not_in_matrix_group(model: GPT, cfg: TrainConfig) -> None:
    """Embedding parameters must not appear in the weight-decay (matrix) group."""
    opt = make_optimizer(model, cfg)
    matrix_group = next(g for g in opt.param_groups if g["weight_decay"] != 0.0)
    matrix_ids = {id(p) for p in matrix_group["params"]}
    for name, param in model.named_parameters():
        if "embedding" in name:
            assert id(param) not in matrix_ids, (
                f"Embedding parameter '{name}' must not be in the matrix (decay) group"
            )


def test_weight_matrix_params_in_decay_group(model: GPT, cfg: TrainConfig) -> None:
    """Non-embedding, non-LayerNorm weight matrices must be in the decay group."""
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
        if "embedding" not in name and id(param) not in ln_ids:
            assert id(param) in decay_ids, (
                f"Weight matrix '{name}' not found in the decay group"
            )


def test_bias_params_have_no_weight_decay(model: GPT, cfg: TrainConfig) -> None:
    """Any parameter with 'bias' in its name must not be in a weight-decay group."""
    opt = make_optimizer(model, cfg)
    bias_ids = {id(p) for n, p in model.named_parameters() if "bias" in n}
    for group in opt.param_groups:
        if group["weight_decay"] != 0.0:
            group_ids = {id(p) for p in group["params"]}
            for bias_id in bias_ids:
                assert bias_id not in group_ids, (
                    "Bias parameter must not appear in a weight-decay group"
                )


def test_all_params_covered_exactly_once(model: GPT, cfg: TrainConfig) -> None:
    """Every trainable model parameter must appear in exactly one group."""
    opt = make_optimizer(model, cfg)
    all_ids_in_groups = [id(p) for g in opt.param_groups for p in g["params"]]
    trainable_ids = [id(p) for p in model.parameters() if p.requires_grad]

    assert len(all_ids_in_groups) == len(trainable_ids), (
        f"Group param count ({len(all_ids_in_groups)}) != model param count ({len(trainable_ids)})"
    )
    assert sorted(all_ids_in_groups) == sorted(trainable_ids), (
        "Optimizer param groups do not exactly match the model's trainable parameters"
    )


def test_frozen_params_excluded_from_groups(model: GPT, cfg: TrainConfig) -> None:
    """Parameters with requires_grad=False must not appear in any param group."""
    first_qkv = model.blocks[0].attn.qkv
    first_qkv.weight.requires_grad_(False)

    opt = make_optimizer(model, cfg)
    group_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(first_qkv.weight) not in group_ids, (
        "Frozen parameter must not be included in any optimizer group"
    )

    first_qkv.weight.requires_grad_(True)


# ── State save / load ─────────────────────────────────────────────────────────


def test_state_save_load_identity(model: GPT, cfg: TrainConfig) -> None:
    """Optimizer state_dict must survive a save/load round-trip exactly.

    After loading, momentum buffers (exp_avg, exp_avg_sq) and step counts must
    be bit-identical to the original.
    """
    torch.manual_seed(42)
    opt = make_optimizer(model, cfg)

    idx = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    logits = model(idx)
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()

    sd_before = opt.state_dict()

    buf = io.BytesIO()
    torch.save(sd_before, buf)
    buf.seek(0)

    opt2 = make_optimizer(model, cfg)
    opt2.load_state_dict(torch.load(buf, weights_only=True))
    sd_after = opt2.state_dict()

    for g1, g2 in zip(sd_before["param_groups"], sd_after["param_groups"]):
        assert g1["weight_decay"] == g2["weight_decay"]
        assert g1["lr"] == g2["lr"]

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
