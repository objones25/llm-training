"""Training loop for LM pretraining.

Orchestrates the full training pipeline: model construction, optimizer,
LR schedule, gradient clipping, NaN detection, logging, plotting, and
checkpointing.  No ``print`` calls live here — all output goes through
``GradientLogger``.

NaN guard (CONTRIBUTING.md, rule 23)
-------------------------------------
If ``loss`` is not finite at any step, a ``RuntimeError`` is raised
immediately.  Training is not silently continued or soft-stopped.

Optimizer modes
---------------
When ``cfg.use_muon`` is False (default), a single AdamW optimizer handles
all parameter groups.

When ``cfg.use_muon`` is True, ``make_optimizer`` returns a ``(Muon, AdamW)``
tuple.  The training loop creates two independent LambdaLR schedulers (one per
optimizer) using the same cosine-warmup lambda so LR trajectories stay in sync.

Checkpointing
-------------
A single ``best.pt`` file is saved (overwriting) whenever the validation loss
improves.  Numbered checkpoints are not written.

Public API
----------
    train(
        cfg:             TrainConfig,
        model:           GPT | None = None,
        token_stream:    Iterable[int] | None = None,
        val_token_stream: Iterable[int] | None = None,
    ) -> GPT
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F

from src.checkpoint import save_checkpoint
from src.config import TrainConfig
from src.dataloader import make_batches
from src.logger import GradientLogger, _log, configure_logging
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
    val_token_stream: Iterable[int] | None = None,
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
    val_token_stream : Iterable[int] | None
        Pre-tokenized validation token IDs.  Consumed eagerly at startup into
        fixed batches; evaluated every ``cfg.val_every`` steps when provided.
        When ``None`` or ``cfg.val_every == 0``, validation is skipped.

    Returns
    -------
    GPT
        The trained model (weights updated in-place; returned for convenience).

    Raises
    ------
    RuntimeError
        If ``loss`` is ``nan`` or ``inf`` at any step.
    """
    configure_logging(cfg)

    plot_dir = Path(cfg.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device)

    if model is None:
        model = GPT(cfg)
    model = model.to(device)

    if cfg.use_compile:
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]

    optimizer_result = make_optimizer(model, cfg)
    _use_muon = isinstance(optimizer_result, tuple)

    if isinstance(optimizer_result, tuple):
        muon_opt, adamw_opt = optimizer_result
        muon_scheduler = make_scheduler(muon_opt, cfg)
        adamw_scheduler = make_scheduler(adamw_opt, cfg)
        # Represent the "primary" optimizer for LR logging (AdamW embed group).
        _primary_opt = adamw_opt
        _schedulers = (muon_scheduler, adamw_scheduler)
    else:
        optimizer = cast(torch.optim.AdamW, optimizer_result)
        scheduler = make_scheduler(optimizer, cfg)
        _primary_opt = optimizer
        _schedulers = None

    logger = GradientLogger(cfg)

    # Surface the non-embedding parameter count (removed from GPT.__init__ print).
    if hasattr(model, "n_params"):
        _log.info(f"model non_embedding_params={model.n_params:,}")

    # AMP is only supported on CUDA; MPS does not implement GradScaler.
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    if token_stream is None:
        from src.dataset import stream_documents
        from src.tokenizer import BPETokenizer

        _tokenizer = BPETokenizer.load("tokenizer.model")
        _docs = stream_documents(cfg)
        token_stream = (tok_id for doc in _docs for tok_id in _tokenizer.encode(doc))

    batches = make_batches(token_stream, cfg)

    # ── Validation batches — consumed eagerly so the same data is reused ─────
    _val_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    if val_token_stream is not None and cfg.val_every > 0:
        _val_batches = list(make_batches(val_token_stream, cfg))
        _log.info(f"val_batches loaded: {len(_val_batches)} batches")

    # ── Accumulated data for plots ────────────────────────────────────────────
    steps_list: list[int] = []
    losses_list: list[float] = []
    lrs_list: list[float] = []
    grad_norms_list: list[float] = []
    grad_norm_mins_list: list[float] = []
    grad_norm_maxs_list: list[float] = []
    val_steps_list: list[int] = []
    val_losses_list: list[float] = []

    layer_names: list[str] | None = None
    grad_heatmap_steps: list[int] = []
    grad_heatmap_rows: list[list[float]] = []
    weight_heatmap_steps: list[int] = []
    weight_heatmap_rows: list[list[float]] = []
    latest_grad_norms: list[float] = []

    # ── Early stopping / best checkpoint state ────────────────────────────────
    _best_val_loss: float = float("inf")
    _patience_counter: int = 0
    _early_stopped: bool = False

    # model.train() is set once before the loop — not repeated per step.
    model.train()

    step = -1
    for step, (inputs, targets) in enumerate(batches):
        if step >= cfg.max_steps:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        if _use_muon:
            muon_opt.zero_grad()
            adamw_opt.zero_grad()
        else:
            optimizer.zero_grad()

        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, cfg.vocab_size), targets.reshape(-1)
                )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss is {loss.item()} at step {step}. Aborting.")

            scaler.scale(loss).backward()

            if _use_muon:
                # Unscale both optimizers so clip_grad_norm_ and logged norms
                # reflect true gradient magnitudes, not AMP-scaled values.
                scaler.unscale_(muon_opt)
                scaler.unscale_(adamw_opt)
            else:
                scaler.unscale_(optimizer)

            # Capture pre-clip per-layer norms BEFORE clip_grad_norm_ fires.
            layer_norms_dict: dict[str, float] = {
                name: p.grad.norm().item()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            grad_norm_total = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            ).item()

            # Log the matrix-group LR (Muon) — it's what 95% of the model uses.
            current_lr = muon_opt.param_groups[0]["lr"] if _use_muon else _primary_opt.param_groups[0]["lr"]
            if _use_muon:
                scaler.step(adamw_opt)
                muon_opt.step()
            else:
                scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size), targets.reshape(-1)
            )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss is {loss.item()} at step {step}. Aborting.")

            loss.backward()

            # Capture pre-clip per-layer norms BEFORE clip_grad_norm_ fires.
            layer_norms_dict = {
                name: p.grad.norm().item()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            # clip_grad_norm_ returns the total pre-clip norm — reuse it directly.
            grad_norm_total = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            ).item()

            # Log the matrix-group LR (Muon) — it's what 95% of the model uses.
            current_lr = muon_opt.param_groups[0]["lr"] if _use_muon else _primary_opt.param_groups[0]["lr"]
            if _use_muon:
                muon_opt.step()
                adamw_opt.step()
            else:
                optimizer.step()

        if _use_muon:
            muon_scheduler.step()
            adamw_scheduler.step()
        else:
            scheduler.step()

        loss_val = loss.item()

        # Gradients are still populated here (zero_grad not called until
        # the next iteration).
        logger.log_step(step, loss_val, current_lr, layer_norms_dict)
        logger.log_layers(step, layer_norms_dict, model)

        # ── Per-step scalar tracking ──────────────────────────────────────────
        steps_list.append(step)
        losses_list.append(loss_val)
        lrs_list.append(current_lr)

        if layer_norms_dict:
            norms = list(layer_norms_dict.values())
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
                name: p.norm().item() for name, p in model.named_parameters()
            }
            if layer_names is None:
                layer_names = list(weight_norms_dict.keys())
            weight_heatmap_steps.append(step)
            weight_heatmap_rows.append(
                [weight_norms_dict.get(n, 0.0) for n in layer_names]
            )

        # ── Validation loss ───────────────────────────────────────────────────
        if _val_batches and cfg.val_every > 0 and step % cfg.val_every == 0:
            with torch.no_grad():
                model.eval()
                val_loss_total = 0.0
                n_val = min(cfg.val_batches, len(_val_batches))
                for val_inputs, val_targets in _val_batches[:n_val]:
                    val_logits = model(val_inputs.to(device))
                    val_loss_total += F.cross_entropy(
                        val_logits.reshape(-1, cfg.vocab_size),
                        val_targets.to(device).reshape(-1),
                    ).item()
                model.train()
            val_loss_avg = val_loss_total / n_val
            logger.log_val(step, val_loss_avg)
            val_steps_list.append(step)
            val_losses_list.append(val_loss_avg)

            # ── Best checkpoint + early stopping ──────────────────────────────
            if val_loss_avg < _best_val_loss:
                _best_val_loss = val_loss_avg
                _patience_counter = 0
                # Save best.pt whenever val loss improves.
                _sched_to_save = _schedulers if _use_muon else scheduler
                save_checkpoint(
                    model,
                    optimizer_result,
                    step,
                    cfg,
                    scheduler=_sched_to_save,
                    save_as_best=True,
                )
            else:
                if cfg.early_stopping_patience > 0:
                    _patience_counter += 1
                    if _patience_counter >= cfg.early_stopping_patience:
                        _log.info(
                            f"early_stopping triggered at step={step} "
                            f"best_val_loss={_best_val_loss:.4f} "
                            f"patience={cfg.early_stopping_patience}"
                        )
                        _early_stopped = True
                        break

        # ── Save / overwrite plots ─────────────────────────────────────────────
        if step % cfg.plot_every == 0:
            plot_loss(
                steps_list,
                losses_list,
                plot_dir / "loss.png",
                val_steps=val_steps_list or None,
                val_losses=val_losses_list or None,
            )
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

    # ── Token stream exhaustion guard ──────────────────────────────────────────
    steps_completed = step + 1 if step >= 0 else 0
    if not _early_stopped and steps_completed < cfg.max_steps:
        warnings.warn(
            f"Token stream exhausted after {steps_completed} steps; "
            f"requested {cfg.max_steps}. Training was cut short.",
            UserWarning,
            stacklevel=2,
        )

    return model
