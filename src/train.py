"""Training loop for LM pretraining.

Orchestrates the full training pipeline: model construction, optimizer,
LR schedule, gradient clipping, NaN detection, logging, plotting, and
checkpointing.  No ``print`` calls live here — all output goes through
``GradientLogger``.

NaN guard (CONTRIBUTING.md, rule 23)
-------------------------------------
If ``loss`` is not finite at any step, a ``RuntimeError`` is raised
immediately.  Training is not silently continued or soft-stopped.

Public API
----------
    train(
        cfg:          TrainConfig,
        model:        GPT | None = None,
        token_stream: Iterable[int] | None = None,
    ) -> GPT
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn.functional as F

from src.checkpoint import save_checkpoint
from src.config import TrainConfig
from src.dataloader import make_batches
from src.logger import GradientLogger
from src.model import GPT
from src.optimizer import make_optimizer
from src.plots import (
    plot_grad_heatmap,
    plot_grad_hist,
    plot_grad_norm,
    plot_loss,
    plot_lr,
    plot_weight_norm,
)
from src.scheduler import make_scheduler


def train(
    cfg: TrainConfig,
    model: GPT | None = None,
    token_stream: Iterable[int] | None = None,
) -> GPT:
    """Run the pretraining loop for up to ``cfg.max_steps`` steps.

    Parameters
    ----------
    cfg : TrainConfig
    model : GPT | None
        Pre-built model.  Constructed from *cfg* when ``None``.
    token_stream : Iterable[int] | None
        Pre-tokenized integer token IDs.  When ``None``, loads documents
        from HuggingFace (``cfg.dataset_name``) and tokenizes with a
        ``BPETokenizer`` loaded from ``'tokenizer.model'`` in the current
        working directory.

    Returns
    -------
    GPT
        The trained model (weights updated in-place; returned for convenience).

    Raises
    ------
    RuntimeError
        If ``loss`` is ``nan`` or ``inf`` at any step.
    """
    plot_dir = Path(cfg.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)

    if model is None:
        model = GPT(cfg)
    model = model.to(device)

    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    logger = GradientLogger(cfg)

    if token_stream is None:
        from src.dataset import stream_documents
        from src.tokenizer import BPETokenizer

        _tokenizer = BPETokenizer.load("tokenizer.model")
        _docs = stream_documents(cfg)
        token_stream = (
            tok_id
            for doc in _docs
            for tok_id in _tokenizer.encode(doc)
        )

    batches = make_batches(token_stream, cfg)

    # ── Accumulated data for plots ────────────────────────────────────────────
    steps_list: list[int] = []
    losses_list: list[float] = []
    lrs_list: list[float] = []
    grad_norms_list: list[float] = []
    grad_norm_mins_list: list[float] = []
    grad_norm_maxs_list: list[float] = []

    layer_names: list[str] | None = None
    grad_heatmap_steps: list[int] = []
    grad_heatmap_rows: list[list[float]] = []
    weight_heatmap_steps: list[int] = []
    weight_heatmap_rows: list[list[float]] = []
    latest_grad_norms: list[float] = []

    for step, (inputs, targets) in enumerate(batches):
        if step >= cfg.max_steps:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        model.train()
        optimizer.zero_grad()

        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss is {loss.item()} at step {step}. Aborting."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # Capture the LR used for this step before the scheduler advances it.
        current_lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()

        # Gradients are still populated here (zero_grad not called until
        # the next iteration).
        logger.log_step(step, loss_val, current_lr, model)
        logger.log_layers(step, model)

        # ── Per-step scalar tracking ──────────────────────────────────────────
        steps_list.append(step)
        losses_list.append(loss_val)
        lrs_list.append(current_lr)

        layer_norms_dict: dict[str, float] = {
            name: p.grad.norm().item()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        if layer_norms_dict:
            norms = list(layer_norms_dict.values())
            grad_norm_total = math.sqrt(sum(n * n for n in norms))
            grad_norm_min = min(norms)
            grad_norm_max = max(norms)
            latest_grad_norms = norms
        else:
            grad_norm_total = grad_norm_min = grad_norm_max = 0.0
            latest_grad_norms = []

        grad_norms_list.append(grad_norm_total)
        grad_norm_mins_list.append(grad_norm_min)
        grad_norm_maxs_list.append(grad_norm_max)

        # ── Heatmap data at configured cadences ───────────────────────────────
        if step % cfg.grad_log_every == 0 and layer_norms_dict:
            if layer_names is None:
                layer_names = list(layer_norms_dict.keys())
            grad_heatmap_steps.append(step)
            grad_heatmap_rows.append(
                [layer_norms_dict.get(n, 0.0) for n in layer_names]
            )

        if step % cfg.weight_log_every == 0:
            weight_norms_dict: dict[str, float] = {
                name: p.norm().item()
                for name, p in model.named_parameters()
            }
            if layer_names is None:
                layer_names = list(weight_norms_dict.keys())
            weight_heatmap_steps.append(step)
            weight_heatmap_rows.append(
                [weight_norms_dict.get(n, 0.0) for n in layer_names]
            )

        # ── Save / overwrite plots ─────────────────────────────────────────────
        if step % cfg.plot_every == 0:
            plot_loss(steps_list, losses_list, plot_dir / "loss.png")
            plot_lr(steps_list, lrs_list, plot_dir / "lr.png")
            plot_grad_norm(
                steps_list,
                grad_norms_list,
                grad_norm_mins_list,
                grad_norm_maxs_list,
                plot_dir / "grad_norm.png",
            )
            if grad_heatmap_steps:
                plot_grad_heatmap(
                    grad_heatmap_steps,
                    layer_names or [],
                    grad_heatmap_rows,
                    plot_dir / "grad_heatmap.png",
                )
            if weight_heatmap_steps:
                plot_weight_norm(
                    weight_heatmap_steps,
                    layer_names or [],
                    weight_heatmap_rows,
                    plot_dir / "weight_norm.png",
                )
            if latest_grad_norms:
                plot_grad_hist(latest_grad_norms, plot_dir / "grad_hist.png")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (step + 1) % cfg.checkpoint_every == 0:
            save_checkpoint(model, optimizer, step + 1, cfg)

    return model
