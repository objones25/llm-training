"""AdamW optimizer with three explicit parameter groups.

Separates model parameters into three groups before constructing the optimizer:

    ln group      — LayerNorm parameters (identified by module type).
                    LR = base_lr × cfg.ln_lr_mult.  Weight decay = 0.
                    LN gradients are bounded by construction (normalization caps
                    the signal), so the small-gradient mode in the distribution
                    is structural.  A higher LR compensates for the smaller
                    effective gradient magnitude.

    embed group   — Embedding parameters (token + position embeddings).
                    LR = base_lr × cfg.embed_lr_mult.  Weight decay = 0.
                    Embeddings benefit from a lower LR; decaying their weights
                    is not standard practice.

    matrix group  — All remaining weight matrices (QKV, out_proj, FF, lm_head).
                    LR = base_lr.  Weight decay = cfg.weight_decay.

LayerNorm parameters are identified by inspecting module types (not by name
substring matching) because the GPT model names its LayerNorm modules as
``ln_1``, ``ln_2``, and ``ln_f`` — none of which contain "norm".

Public API
----------
    make_optimizer(
        model: nn.Module,
        cfg: TrainConfig,
    ) -> torch.optim.AdamW
"""
from __future__ import annotations

import torch.nn as nn
import torch.optim

from src.config import TrainConfig


def make_optimizer(
    model: nn.Module,
    cfg: TrainConfig,
) -> torch.optim.AdamW:
    """Return an AdamW optimizer with three parameter groups.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be optimized.
    cfg : TrainConfig
        Supplies ``learning_rate``, ``weight_decay``, ``ln_lr_mult``, and
        ``embed_lr_mult``.

    Returns
    -------
    torch.optim.AdamW
        Three param groups:
        - ln:     lr = learning_rate × ln_lr_mult,    weight_decay = 0
        - embed:  lr = learning_rate × embed_lr_mult, weight_decay = 0
        - matrix: lr = learning_rate,                 weight_decay = cfg.weight_decay
    """
    # Collect LayerNorm parameter IDs via type inspection.
    # Name-based substring matching ("norm") would miss the GPT model's
    # ln_1/ln_2/ln_f naming convention.
    ln_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                ln_ids.add(id(param))

    # Collect embedding parameter IDs by name.
    embed_ids: set[int] = set()
    for name, param in model.named_parameters():
        if "embedding" in name:
            embed_ids.add(id(param))

    ln_params: list[nn.Parameter] = []
    embed_params: list[nn.Parameter] = []
    matrix_params: list[nn.Parameter] = []

    for _name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in ln_ids:
            ln_params.append(param)
        elif id(param) in embed_ids:
            embed_params.append(param)
        else:
            matrix_params.append(param)

    base_lr = cfg.learning_rate
    param_groups = [
        {"params": ln_params,     "lr": base_lr * cfg.ln_lr_mult,    "weight_decay": 0.0},
        {"params": embed_params,  "lr": base_lr * cfg.embed_lr_mult, "weight_decay": 0.0},
        {"params": matrix_params, "lr": base_lr,                     "weight_decay": cfg.weight_decay},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=cfg.adamw_betas,
        eps=cfg.adamw_eps,
    )
