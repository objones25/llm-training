# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build a small LLM from scratch for learning purposes using `HuggingFaceFW/fineweb-edu` (sample-10BT). The project is built **incrementally** — each component is implemented and tested before moving to the next. See `CONTRIBUTING.md` for the required workflow order and all 28 testing rules.

---

## Module Map

All source code lives under `src/`. One module per concern — no exceptions.

```text
src/
  config.py
  tokenizer.py
  model.py
  dataset.py
  dataloader.py
  scheduler.py
  optimizer.py
  muon.py
  checkpoint.py
  logger.py
  kv_cache.py
  plots.py
  train.py
tests/
  test_tokenizer.py
  test_model.py
  test_dataset.py
  test_dataloader.py
  test_scheduler.py
  test_optimizer.py
  test_muon.py
  test_checkpoint.py
  test_logger.py
  test_plots.py
  test_train.py
  test_kv_cache.py
  test_config.py
  test_training_regression.py
  test_evaluate.py
```

---

## Config Contract

All hyperparameters must be defined in `src/config.py` as a dataclass. Nothing is hardcoded in `train.py` or anywhere else. See `src/config.py` for full field definitions and defaults.

### Config Validation Rules

`TrainConfig.__post_init__` validates all parameters at construction time:

- `vocab_size`, `batch_size`, `seq_len`, `max_steps`, `val_batches` must be positive integers
- `grad_clip`, `grad_norm_spike_threshold` must be positive
- `weight_decay` must be non-negative (zero is allowed, disables weight decay)
- `warmup_steps`, `val_every`, `early_stopping_patience` must be non-negative
- `warmup_steps` must be strictly less than `max_steps`
- `d_model` must be divisible by `n_heads` (required for multi-head attention)
- `ln_lr_mult`, `embed_lr_mult` must be strictly positive

When testing invalid configs, construct them **inside** `pytest.raises(ValueError)` blocks — never before. If constructed outside the block, the exception is raised immediately and the test fails.

---

## Logging Contract

All logging logic lives in `src/logger.py`. Output routes through Python's `logging` module via a named logger `"llm_training"`.

### Setup

`train.py` calls `configure_logging(cfg)` once before the training loop. This attaches:

- **Console (StreamHandler, INFO)** — one terse summary line per step + WARNING lines
- **File (FileHandler, DEBUG)** — everything: per-layer grad/weight norms, step summary, WARNINGs

`GradientLogger` is instantiated after `configure_logging`. Tests do **not** call `configure_logging`; pytest's `caplog` fixture captures records via normal propagation to the root logger.

### `GradientLogger` API

```python
logger.log_step(step, loss, lr, layer_norms)   # layer_norms: pre-clip dict[str, float]
logger.log_val(step, val_loss)                  # validation loss checkpoint
logger.log_layers(step, layer_norms, model)     # grad lines use dict; weight lines use model
```

### Every step — single line to console + file

```text
step=100 loss=3.4821 lr=0.000287 grad_norm=1.2341 grad_norm_min=0.0012 grad_norm_max=4.3210 grad_norm_max_layer=transformer.block.5.ff.w2
```

All seven fields are required every step. Format is `key=value` pairs separated by spaces.

If any single layer's gradient norm exceeds `grad_norm_warn_threshold`, emit an additional WARNING line immediately after:

```text
WARNING step=100 layer=transformer.block.5.attn.q_proj grad_norm=14.3201 exceeds threshold=10.0
```

Training continues — this is observability, not a hard stop.

If the total gradient norm exceeds `grad_norm_spike_threshold`, emit a full per-layer breakdown at DEBUG level immediately (file only, not console):

```text
spike step=100 layer=transformer.block.0.attn.q_proj norm=0.3821
spike step=100 layer=transformer.block.5.attn.q_proj norm=14.3201
```

### Every `grad_log_every` steps — per-layer gradient norms (pre-clip)

```text
grad step=100 layer=transformer.block.0.attn.q_proj norm=0.3821
grad step=100 layer=transformer.block.5.ff.w2       norm=0.0003
```

### Every `weight_log_every` steps — per-layer weight norms

```text
weight step=500 layer=transformer.block.0.attn.q_proj norm=1.2341
```

### At `val_every` steps — validation loss

```text
val step=250 val_loss=3.1842
```

---

## Plots Contract

All plots are produced by `src/plots.py` and saved to `plot_dir`. No plot logic lives in `train.py` or `logger.py`. All plots use `matplotlib.use("Agg")`. Updated every `plot_every` steps by overwriting in place.

- `loss.png` — training step vs. loss (log scale); confirm loss trending down, catch spikes early
- `lr.png` — training step vs. LR; verify warmup ramp and cosine decay shape
- `grad_norm.png` — total norm + min/max band over time; first signal of instability
- `grad_heatmap.png` — per-layer norm heatmap (step × layer, log-scale color); primary diagnostic for vanishing/exploding gradients by layer
- `grad_hist.png` — distribution of per-parameter gradient norms at most recent `grad_log_every` step; distinguish healthy vs. bimodal/collapsed distributions
- `weight_norm.png` — per-layer weight norm heatmap; detect silent weight growth, verify weight decay

---

## Commands

**Always use `uv run`.** Never call `python` or `pytest` directly — they will use the wrong interpreter and miss project dependencies.

```bash
uv sync
uv run python -m pytest -x --tb=short
uv run python -m pytest tests/test_tokenizer.py -x --tb=short
uv run python -m pytest tests/test_model.py::test_forward_pass_shape -x --tb=short
uv run python -m pytest --cov=src --cov-report=term-missing -x --tb=short
```

See `docs/TRAINING.md` for cloud pod setup, checkpoint evaluation, pre-training checklist, and full training operations.

---

## Optimizer Parameter Groups

`optimizer.py` splits parameters into **three explicit groups**:

- **ln group** — RMSNorm parameters (identified by `isinstance(module, RMSNorm)`, not name substring). `lr = learning_rate × ln_lr_mult` (default 3×). No weight decay. Rationale: norm gradients are bounded by construction (~10⁻²); higher LR compensates.
- **embed group** — Embedding parameters (name contains `"embedding"`). `lr = learning_rate × embed_lr_mult` (default 0.1×). No weight decay.
- **matrix group** — All remaining weight matrices (QKV, out_proj, FF, lm_head). `lr = learning_rate`. Weight decay = `cfg.weight_decay`.

LN params must be identified by `isinstance(module, RMSNorm)` — name-based matching would miss `ln_1`/`ln_2`/`ln_f` naming used in this model.

When `use_muon=True`: matrix group uses Muon (`src/muon.py`), ln+embed groups use AdamW. `make_optimizer` returns `(Muon, AdamW)` tuple. `train.py` creates two independent LambdaLR schedulers (one per optimizer) so LR trajectories stay in sync. Logged `lr` comes from `muon_opt.param_groups[0]["lr"]` (matrix group), not the AdamW ln-group (which is 3× higher due to `ln_lr_mult`).

**Critical AMP+Muon bug:** When `use_amp=True`, `scaler.unscale_()` must be called on **both** `muon_opt` and `adamw_opt` before `clip_grad_norm_()`. If only AdamW is unscaled, Muon's gradients remain AMP-scaled (~×16384). `clip_grad_norm_` across all params then sees millions instead of single digits, producing a clip ratio of ~1/1,800,000 that crushes AdamW (ln/embed) gradients to zero — those parameters stop learning entirely. Symptom: loss barely moves and logged grad norms stay in the millions for hundreds of steps.

For full architectural rationale, scaling laws, and hyperparameter derivations, see `deep_dive.md`.

---

## Library Policy

- Always verify API signatures with Context7 or official docs before using. Do not assume from memory.
- Record any API mistakes as a memory entry to avoid repetition.
- Never use deprecated APIs. Check docs for the version pinned in `pyproject.toml`.

### Pinned major versions

| Library      | Minimum version |
| ------------ | --------------- |
| torch        | 2.2             |
| datasets     | 2.18            |
| transformers | 4.40            |
| tiktoken     | 0.6             |
| matplotlib   | 3.8             |

---

## Implementation Details & Contracts

### Checkpoint Strategy

Training saves a single `best.pt` file (overwriting) whenever validation loss improves. Numbered `checkpoint_{step:07d}.pt` files are **not** written. The `checkpoint_every` config field is retained for backwards compatibility but is unused by the training loop.

`save_checkpoint()` accepts a `save_as_best=True` flag to write to `checkpoint_dir/best.pt`. Pass `save_as_best=False` (default) for numbered checkpoints (legacy, not used by `train.py`).

When using Muon (`use_muon=True`), the `optimizer` argument is a `(Muon, AdamW)` tuple and `scheduler` is a `(LambdaLR, LambdaLR)` tuple. Both are stored and restored correctly by `save_checkpoint`/`load_checkpoint`.

`save_checkpoint()` and `load_checkpoint()` **require** passing the scheduler parameter to preserve LR trajectory:

```python
# Saving (single optimizer)
save_checkpoint(model, optimizer, step, cfg, scheduler=scheduler, save_as_best=True)

# Saving (Muon + AdamW tuple)
save_checkpoint(model, (muon_opt, adamw_opt), step, cfg,
                scheduler=(muon_sched, adamw_sched), save_as_best=True)

# Loading
load_checkpoint(path, model, optimizer, scheduler=scheduler)
```

If you resume training without passing `scheduler=`, the scheduler starts from its initial state and LR will diverge from the original run — defeating the purpose of resuming. This is a **required pattern**, not optional.

### Attention Mechanism and KV Cache

`CausalSelfAttention` in `src/model.py` uses `torch.nn.functional.scaled_dot_product_attention()` with `is_causal=True` during training. This enables:

- FlashAttention-2 on CUDA (hardware-accelerated, memory-efficient)
- Correct causal masking on all backends
- Deterministic behavior (no custom mask buffer juggling)

During generation with KV cache, `is_causal` is set to `True` only when processing more than one token (prefill phase); single-token generation sets `is_causal=False` since the cache contains full history.

KV cache is defined in `src/kv_cache.py`:

- `LayerKVCache` dataclass: stores accumulated K and V tensors `[B, n_heads, T_cached, head_dim]` for one layer
- `KVCache` dataclass: list of `LayerKVCache` objects, one per transformer layer
- `KVCache.empty()` classmethod: creates zero-length cache ready for prefill
- `kv_cache.seq_len` property: returns current cached sequence length

The model's forward pass accepts an optional `kv_cache` parameter:

- `kv_cache=None` (training): standard transformer, O(T²) attention
- `kv_cache` provided (generation): two-phase prefill+generate, O(T) per token

### Gradient Norm Logging — Pre-Clip Timing

All gradient norms logged (both total and per-layer) are computed **before** `torch.nn.utils.clip_grad_norm_()` is applied. This is intentional and critical:

- Pre-clip norms reveal the true state of gradients
- Post-clip norms would hide the magnitude of instability that was clipped away
- You cannot diagnose vanishing or exploding gradients from post-clip values

The training loop captures per-layer norms in a dictionary before clipping, then clips, then logs.

### Model Parameter Count Attribute

`GPT.__init__()` computes non-embedding parameter count and stores it as `self.n_params` (integer attribute). The `train.py` function surfaces this on startup:

```python
if hasattr(model, "n_params"):
    print(f"model non_embedding_params={model.n_params:,}")
```

Do **not** print this inside `GPT.__init__()` — let the training loop decide when and how to display it.

### Device and Optimization Flags

- `use_compile: bool` — opt-in `torch.compile()` with "reduce-overhead" mode. Adds 30-60s startup cost.
- `use_amp: bool` — automatic mixed precision. Only works on CUDA; gracefully disabled on CPU/MPS.

### Training Loop Structure

Key ordering:

1. `model.train()` is called **once** before the loop, not per-step
2. When `use_amp=True` and `use_muon=True`: unscale **both** optimizers before norm capture
3. Gradient norms are captured **before** `clip_grad_norm_()`
4. Total norm is obtained directly from the return value of `clip_grad_norm_()`
5. Loss NaN check happens immediately after loss computation, before backward
6. Scheduler step is called after optimizer step (standard PyTorch pattern)
7. Logged `lr` is from `muon_opt.param_groups[0]["lr"]` in Muon mode (matrix group)

---

## Reference

`deep_dive.md` — synthesis of Kaplan, Chinchilla, Sardana (2401.00448), Besiroglu (2404.10102),
and Reconciling (2406.12907) scaling law papers. Consult before making any architecture or
hyperparameter decision.
