"""Optimizer construction with three explicit parameter groups.

Separates model parameters into three groups:

    ln group      — RMSNorm parameters (identified by module type).
                    LR = base_lr × cfg.ln_lr_mult.  Weight decay = 0.
                    Norm gradients are bounded by construction (normalization caps
                    the signal), so the small-gradient mode in the distribution
                    is structural.  A higher LR compensates for the smaller
                    effective gradient magnitude.

    embed group   — Embedding parameters (token + position embeddings).
                    LR = base_lr × cfg.embed_lr_mult.  Weight decay = 0.
                    Embeddings benefit from a lower LR; decaying their weights
                    is not standard practice.

    matrix group  — All remaining weight matrices (QKV, out_proj, FF, lm_head).
                    LR = base_lr.  Weight decay = cfg.weight_decay.

When ``cfg.use_muon`` is True, the matrix group uses the Muon optimizer
(Newton-Schulz orthogonalization on gradient updates) and the ln+embed groups
use AdamW.  The function then returns a ``(Muon, AdamW)`` tuple.

When ``cfg.use_muon`` is False (default), a single AdamW optimizer is returned
with all three param groups.

RMSNorm parameters are identified by inspecting module types (not by name
substring matching) because the GPT model names its RMSNorm modules as
``ln_1``, ``ln_2``, and ``ln_f`` — none of which contain "norm".

Public API
----------
    make_optimizer(
        model: nn.Module,
        cfg: TrainConfig,
    ) -> torch.optim.AdamW | tuple[Muon, torch.optim.AdamW]
"""

from __future__ import annotations

import torch.nn as nn
import torch.optim

from src.config import TrainConfig
from src.model import RMSNorm
from src.muon import Muon


def make_optimizer(
    model: nn.Module,
    cfg: TrainConfig,
) -> torch.optim.AdamW | tuple[Muon, torch.optim.AdamW]:
    """Return an optimizer (or optimizer pair) for the model.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters will be optimized.
    cfg : TrainConfig
        Supplies ``learning_rate``, ``weight_decay``, ``ln_lr_mult``,
        ``embed_lr_mult``, and ``use_muon``.

    Returns
    -------
    torch.optim.AdamW
        When ``cfg.use_muon`` is False: a single AdamW with three param groups
        (ln, embed, matrix).

    tuple[Muon, torch.optim.AdamW]
        When ``cfg.use_muon`` is True: ``(muon_opt, adamw_opt)`` where
        ``muon_opt`` holds the matrix group and ``adamw_opt`` holds ln+embed.
    """
    # Collect RMSNorm parameter IDs via type inspection.
    # Name-based substring matching would miss the GPT model's
    # ln_1/ln_2/ln_f naming convention.
    ln_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, RMSNorm):
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

    if cfg.use_muon:
        muon_opt = Muon(matrix_params, lr=base_lr)
        adamw_opt = torch.optim.AdamW(
            [
                {
                    "params": ln_params,
                    "lr": base_lr * cfg.ln_lr_mult,
                    "weight_decay": 0.0,
                },
                {
                    "params": embed_params,
                    "lr": base_lr * cfg.embed_lr_mult,
                    "weight_decay": 0.0,
                },
            ],
            lr=base_lr,
            betas=cfg.adamw_betas,
            eps=cfg.adamw_eps,
        )
        return (muon_opt, adamw_opt)

    param_groups = [
        {"params": ln_params, "lr": base_lr * cfg.ln_lr_mult, "weight_decay": 0.0},
        {
            "params": embed_params,
            "lr": base_lr * cfg.embed_lr_mult,
            "weight_decay": 0.0,
        },
        {"params": matrix_params, "lr": base_lr, "weight_decay": cfg.weight_decay},
    ]
    return torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=cfg.adamw_betas,
        eps=cfg.adamw_eps,
    )
