"""AdamW optimizer with explicit weight-decay parameter groups.

Separates model parameters into two groups before constructing the optimizer:

    decay group    — all trainable parameters that are not LayerNorm parameters
                     and do not have "bias" in their name.  Weight decay applied.
    no-decay group — LayerNorm parameters (identified by module type) and any
                     bias parameters.  Weight decay set to 0.0.

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
    """Return an AdamW optimizer with separate decay and no-decay param groups.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be optimized.
    cfg : TrainConfig
        Supplies ``learning_rate`` and ``weight_decay``.

    Returns
    -------
    torch.optim.AdamW
        Two param groups:
        - decay:    weight_decay = cfg.weight_decay  (non-norm, non-bias params)
        - no_decay: weight_decay = 0.0               (LayerNorm + bias params)
    """
    # Collect parameter IDs belonging to LayerNorm modules via type inspection.
    # Name-based substring matching ("norm") would miss the GPT model's ln_1/ln_2/ln_f
    # naming convention, so we inspect module types instead.
    no_decay_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                no_decay_ids.add(id(param))

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in no_decay_ids or "bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=cfg.learning_rate)
