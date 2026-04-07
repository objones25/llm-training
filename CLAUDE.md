# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Build a small LLM from scratch for learning purposes using `HuggingFaceFW/fineweb-edu` (sample-10BT). The project is built **incrementally** — each component is implemented and tested before moving to the next. See `CONTRIBUTING.md` for the required workflow order.

---

## Module Map

All source code lives under `src/`. One module per concern — no exceptions.

```text
src/
  config.py         # Single source of truth for all hyperparameters
  tokenizer.py      # BPE tokenizer, encode/decode
  model.py          # Transformer definition
  dataset.py        # HuggingFace dataset loading and streaming
  dataloader.py     # Batching, padding, token ID validation
  scheduler.py      # Cosine LR schedule with warmup
  optimizer.py      # AdamW setup and weight decay grouping
  checkpoint.py     # Save and load model + optimizer state
  logger.py         # GradientLogger — per-step and per-layer logging
  kv_cache.py       # KVCache and LayerKVCache for efficient generation
  plots.py          # All matplotlib visualizations
  train.py          # Training loop, nan detection, calls logger and plots
tests/
  test_tokenizer.py
  test_model.py
  test_dataset.py
  test_dataloader.py
  test_scheduler.py
  test_optimizer.py
  test_checkpoint.py
  test_logger.py
  test_plots.py
  test_train.py
  test_kv_cache.py
  test_config.py
```

---

## Config Contract

All hyperparameters must be defined in `src/config.py` as a dataclass. Nothing is hardcoded in `train.py` or anywhere else.

```python
@dataclass
class TrainConfig:
    # Model
    vocab_size: int = 8192
    n_layers: int = 12
    d_model: int = 2048
    n_heads: int = 32
    d_ff: int = 8192

    # Training
    max_steps: int = 20_000
    batch_size: int = 32
    seq_len: int = 512
    learning_rate: float = 1.5e-4
    warmup_steps: int = 1_500
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    # Optimizer
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8
    ln_lr_mult: float = 3.0      # LR multiplier for LayerNorm params (no weight decay)
    embed_lr_mult: float = 0.1   # LR multiplier for embedding params (no weight decay)

    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_config: str = "sample-10BT"   # subset name, second arg to load_dataset
    dataset_split: str = "train"

    # Device and optimization
    device: str = "cpu"        # "mps" for Apple Silicon, "cuda" for NVIDIA GPU
    use_compile: bool = False  # torch.compile — opt-in; adds 30-60s cold-start overhead
    use_amp: bool = False      # Automatic mixed precision — CUDA only

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 1_000

    # Validation
    val_every: int = 250              # steps between val-loss evaluations; 0 disables
    val_batches: int = 20             # number of val batches to average per evaluation
    early_stopping_patience: int = 0   # val evals without improvement before stopping; 0 disables

    # Observability
    grad_log_every: int = 100         # cadence for per-layer gradient norm breakdown
    weight_log_every: int = 500       # cadence for per-layer weight norm logging
    plot_every: int = 500             # cadence for saving plots to disk
    grad_norm_warn_threshold: float = 10.0  # per-layer threshold; emits WARNING line
    grad_norm_spike_threshold: float = 2.5 # total-norm threshold; dumps all layers immediately
    plot_dir: str = "plots"
    log_file: str = "train.log"            # verbose DEBUG log; "" disables file logging

    def __post_init__(self) -> None:
        """Validate configuration parameters eagerly at construction time.

        All validation failures raise ValueError immediately, preventing
        invalid configs from silently poisoning downstream logic.
        """
        # (See src/config.py for full validation rules)
```

Tests must instantiate `TrainConfig` explicitly with any overrides they need. Never rely on defaults silently.

### Config Validation Rules

`TrainConfig.__post_init__` validates all parameters at construction time:

- `vocab_size`, `batch_size`, `seq_len`, `max_steps`, `val_batches` must be positive integers
- `grad_clip`, `grad_norm_spike_threshold` must be positive
- `weight_decay` must be non-negative (zero is allowed, disables weight decay)
- `warmup_steps`, `val_every`, `early_stopping_patience` must be non-negative
- `warmup_steps` must be strictly less than `max_steps`
- `d_model` must be divisible by `n_heads` (required for multi-head attention)
- `ln_lr_mult`, `embed_lr_mult` must be strictly positive

When testing invalid configs, construct them **inside** `pytest.raises(ValueError)` blocks — never before. If constructed outside the block, the exception is raised immediately and the test fails. See rule 28 below for the correct pattern.

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

`layer_norms` must be captured from `model.named_parameters()` **before** `clip_grad_norm_()` fires. `train.py` builds this dict, clips, then passes the pre-clip dict to the logger.

### Every step — single line to console + file

```text
step=100 loss=3.4821 lr=0.000287 grad_norm=1.2341 grad_norm_min=0.0012 grad_norm_max=4.3210 grad_norm_max_layer=transformer.block.5.ff.w2
```

All seven fields are required every step. Format is `key=value` pairs separated by spaces. The `grad_norm_max_layer` field identifies which layer contributed the maximum gradient norm — this enables spike attribution without waiting for the per-layer breakdown cadence.

**Important:** All per-layer gradient norms logged here are captured **before** `torch.nn.utils.clip_grad_norm_()` is applied. This is intentional — you must see the true gradient magnitudes before clipping to diagnose instability.

If any single layer's gradient norm exceeds `grad_norm_warn_threshold`, emit an additional
WARNING line immediately after:

```text
WARNING step=100 layer=transformer.block.5.attn.q_proj grad_norm=14.3201 exceeds threshold=10.0
```

Training continues — this is observability, not a hard stop.

If the total gradient norm exceeds `grad_norm_spike_threshold`, emit a full per-layer breakdown at DEBUG level immediately:

```text
spike step=100 layer=transformer.block.0.attn.q_proj norm=0.3821
spike step=100 layer=transformer.block.5.attn.q_proj norm=14.3201
spike step=100 layer=transformer.block.5.ff.w2 norm=0.0003
```

These spike logs are written to file only (DEBUG level), not console, and are unfiltered — every layer is logged regardless of `grad_log_every` cadence. This enables rapid diagnosis of total norm spikes.

### Every `grad_log_every` steps — per-layer gradient norms

```text
grad step=100 layer=transformer.block.0.attn.q_proj norm=0.3821
grad step=100 layer=transformer.block.0.attn.v_proj norm=0.3714
grad step=100 layer=transformer.block.5.attn.q_proj norm=4.2910
grad step=100 layer=transformer.block.5.ff.w2       norm=0.0003
```

These norms are also **pre-clip**. Compare these values over time to detect vanishing or exploding gradients by layer.

### Every `weight_log_every` steps — per-layer weight norms

```text
weight step=500 layer=transformer.block.0.attn.q_proj norm=1.2341
weight step=500 layer=transformer.block.5.ff.w2       norm=0.9821
```

### At `val_every` steps — validation loss

```text
val step=250 val_loss=3.1842
```

When `val_every > 0` and validation data is available, this line is emitted at INFO level (console and file) every `val_every` training steps. Used for early stopping and convergence tracking.

---

## Plots Contract

All plots are produced by `src/plots.py` and saved to `plot_dir`. No plot logic lives in
`train.py` or `logger.py`. All plots use `matplotlib.use("Agg")` — no display or GUI required.

Plots are updated every `plot_every` steps by overwriting the existing file in place. This
means you can inspect the latest state at any point during a run without waiting for it to finish.

### Loss curve — `loss.png`

- X: training step. Y: loss (log scale by default)
- Single line. Updated every `plot_every` steps.
- Purpose: confirm loss is trending down and catch spikes early.

### Learning rate curve — `lr.png`

- X: training step. Y: learning rate value
- Purpose: visually verify warmup ramp and cosine decay shape.

### Gradient norm over time — `grad_norm.png`

- X: training step
- Three lines: total `grad_norm`, `grad_norm_min`, `grad_norm_max`
- Shaded region between min and max
- Purpose: first signal of instability — spikes here precede loss spikes.

### Per-layer gradient norm heatmap — `grad_heatmap.png`

- X: training step (sampled at `grad_log_every` cadence). Y: layer name. Color: norm (log scale)
- Purpose: **primary diagnostic for vanishing/exploding gradients by layer**. A layer going
  dark (near zero) or saturating (bright) tells you exactly where the problem is.

### Gradient distribution histogram — `grad_hist.png`

- Distribution of all per-parameter gradient norms at the most recent `grad_log_every` step
- X: norm value (log scale). Y: count
- Redrawn each `plot_every` steps — current snapshot, not history
- Purpose: distinguish a healthy bell-shaped distribution from bimodal or collapsed distributions.

### Weight norm heatmap — `weight_norm.png`

- Same format as `grad_heatmap.png` but for weight norms over training steps
- Purpose: detect silent weight growth that precedes loss spikes, and verify weight decay
  is working correctly across all layers.

---

## Commands

**Always use `uv run` to execute Python scripts and tools in this project.** Never call `python` or `pytest` directly — they will use the wrong interpreter and miss project dependencies.

```bash
# Install dependencies
uv sync

# Run all tests
uv run python -m pytest -x --tb=short

# Run a single test file
uv run python -m pytest tests/test_tokenizer.py -x --tb=short

# Run a specific test
uv run python -m pytest tests/test_model.py::test_forward_pass_shape -x --tb=short

# Run tests with coverage
uv run python -m pytest --cov=src --cov-report=term-missing -x --tb=short
```

---

## Pre-Training Checklist

Run this before any full training job. Do not skip steps.

- [ ] **Init sanity:** confirm step-0 loss ≈ `log(vocab_size)`. If not, the model is not
      initialized correctly — debug before proceeding.
- [ ] **Compute budget:** verify `C = 6 × N × B × S` matches your expected FLOP budget.
- [ ] **Schedule length:** confirm cosine cycle length equals `max_steps` exactly.
      Overestimating by >25% measurably degrades final loss.
- [ ] **Loss target:** run the parametric formula below at your chosen `N` and `D`. If final
      loss exceeds the prediction by >5%, treat it as a signal to debug data quality or
      optimizer settings before scaling up.
- [ ] **Plot directory** exists and is writable before training starts.

### Parametric Loss Target (Epoch AI corrected Chinchilla)

```text
L(N, D) = 1.8172 + 482.01 / N^0.3478 + 2085.43 / D^0.3658
```

Use this — not the original rounded Chinchilla values — when estimating expected loss.
`N` is **non-embedding** parameter count only.

---

## Testing Rules (Non-Negotiable)

All 28 rules below must be followed for every component. See `CONTRIBUTING.md` for when
each test gets written.

1. Every training component has a corresponding `test_` prefixed test file.
2. Tests run without a GPU — CPU tensors with tiny synthetic inputs only.
3. All code is importable `.py` modules — no notebooks or interactive code.
4. Each test is hermetic — no shared mutable state, no inter-test file dependencies.
5. Model forward passes use minimum viable shape: `batch=2, seq_len=16, vocab=256`.
6. Loss must be finite, non-negative, and **monotonically decreasing** over ≥3 gradient steps on a fixed synthetic batch.
7. Dataset pipeline tests mock HuggingFace `datasets` — never hit the network in tests.
8. Tokenizer tests assert round-trip fidelity: `decode(encode(text)) == text` for fixed strings.
9. DataLoader tests assert no batch contains padding-only sequences and token IDs are within vocab bounds.
10. Training step tests assert gradients are non-None, non-zero, and contain **no `nan` or `inf` values** for every named parameter after backward.
11. Optimizer and scheduler state must be saveable/loadable with exact match on a subsequent step's loss.
12. Checkpoint tests assert identical logits before save and after load. Scheduler state must round-trip exactly when passed to `save_checkpoint` and `load_checkpoint`.
13. Every function with a numeric output has a test asserting output shape and dtype explicitly.
14. Cosine schedule tests assert: LR at step 0 == warmup start, peaks at configured max, near-zero at `max_steps`.
15. Tests touching randomness set a fixed seed **in the test body**, not in a fixture.
16. No test may take longer than 10 seconds — mock expensive parts.
17. Data preprocessing tests assert deterministic output given the same input and seed.
18. A smoke test exists for a full mini training loop: 10 steps, 2-layer model, synthetic data, final loss < initial loss.
19. Baseline CI command: `pytest -x --tb=short`.
20. Tests write to disk only via `tmp_path` from pytest fixtures — no manual cleanup.
21. Any function that modifies model weights (init, weight tying, parameter freezing) must have a test asserting the parameter tensor values directly — not just that a forward pass runs.
22. A test must assert that `model.eval()` produces identical outputs on two identical inputs — catches dropout left active during inference.
23. A test must assert that the training loop raises explicitly (not silently continues) when loss is `nan`.
24. A test must assert that `GradientLogger.log_layers()` emits per-layer lines for every named parameter, at the correct step cadence, using a fixed synthetic model. Tests use `caplog` — not `capsys` — because output routes through the `logging` module.
25. A test must assert that a `WARNING` line is emitted (not an exception) when any layer's gradient norm exceeds `grad_norm_warn_threshold`.
26. Plot tests must assert that each plot function produces a valid, non-empty `.png` file at the expected path — use synthetic data passed directly to the plot function, do not run a training loop.
27. `TrainConfig` must have a dedicated test file (`test_config.py`) with comprehensive coverage of all `__post_init__` validation rules. Each invalid config must be constructed **inside** `pytest.raises(ValueError)`, never before it.
28. Model initialization must store the non-embedding parameter count as `model.n_params` (an attribute). Tests must assert `model.n_params` exists and is equal to the manually computed count. Do not print parameter counts in `__init__` — let `train.py` surface this value.

---

## Architecture Guidance (from `deep_dive.md`)

### Scaling decisions

- **Depth vs. width barely matters** — choose for engineering reasons. Performance depends
  on total non-embedding parameter count `N`, not shape. A (6-layer, 4288-dim) model reaches
  within 3% of a (48-layer, 1600-dim) model at the same N.
- **Exclude embedding parameters from N** when reporting or comparing scale. Mixing embedding
  vs. non-embedding N is the primary reason Kaplan and Chinchilla appear to disagree on exponents.
- **Compute budget formula:** `C = 6NBS` — 6 FLOPs per token per parameter for training;
  2 for inference.

### Optimizer

- Use **AdamW**, not Adam. AdamW trains worse for most of the run but ends better.
  Decoupled weight decay matters especially at scale.
- `optimizer.py` splits parameters into **three explicit groups**:
  - **ln group** — LayerNorm parameters (identified by module type, not name).
    `lr = learning_rate × ln_lr_mult` (default 3×). No weight decay.
    Rationale: LN gradients are bounded by construction (~10⁻²); higher LR compensates.
  - **embed group** — Embedding parameters (token + position).
    `lr = learning_rate × embed_lr_mult` (default 0.1×). No weight decay.
  - **matrix group** — All remaining weight matrices (QKV, out_proj, FF, lm_head).
    `lr = learning_rate`. Weight decay = `cfg.weight_decay`.
- Do **not** apply weight decay to biases or LayerNorm parameters.
- LN params are identified by `isinstance(module, nn.LayerNorm)` — name-based matching
  would miss `ln_1`/`ln_2`/`ln_f` naming used in this model.

### Learning rate schedule

- Use **cosine decay with warmup**.
- Cosine cycle length must match `max_steps` exactly. Overestimating by >25% degrades final loss.
- Set `max_steps` before training starts and do not change it mid-run.

### Batch size

- Critical batch size grows as loss decreases: `B_crit ∝ L^(-1/0.21)`.
- For this scale, start small. The formula matters mainly near convergence.

### Token allocation (Chinchilla-optimal)

- Scale model size `N` and token count `D` equally as compute increases.
- For inference-heavy use, prefer **smaller model + more tokens** over a larger compute-optimal model.
- Do **not** train to full convergence — stop ~10% above the converged loss.

**Dataset scale (measured):** `train.bin` = 11.66B tokens (23.3 GB), `val.bin` = 117.9M tokens. Total 11.78B tokens.

**Chinchilla-optimal for this dataset:** `N_optimal = 11.78B / 20 ≈ 589M non-embedding params`. The current 18.9M-param model consumed only 328M of 11.78B available tokens (2.8%). A compute-optimal next run uses `n_layers=12, d_model=2048, n_heads=32, d_ff=8192` (604M params) trained on the full dataset — approximately 12 GPU-hours on an H100.

---

## Library Policy

- Always verify API signatures with Context7 or official docs before using.
  Do not assume from memory.
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

---

## Implementation Details & Contracts

### Checkpoint State Preservation

`save_checkpoint()` and `load_checkpoint()` in `src/checkpoint.py` now **require** passing the scheduler parameter to preserve LR trajectory:

```python
# Saving
save_checkpoint(model, optimizer, step, cfg, scheduler=scheduler)

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

### Embedding Output Scaling

The embeddings (token + positional) are scaled by `sqrt(d_model)` in `GPT.forward()` before being passed to the first transformer block. This is the GPT-3 style initialization:

```python
x = (token_emb + pos_emb) * math.sqrt(self.d_model)
```

The scaling factor is pre-computed in `__init__` as `self._embed_scale` to avoid recomputation in the forward pass. This stabilizes training with embeddings of the same magnitude as the transformer blocks' hidden states.

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
2. Gradient norms are captured **before** `clip_grad_norm_()`
3. Total norm is obtained directly from the return value of `clip_grad_norm_()`
4. Loss NaN check happens immediately after loss computation, before backward
5. Scheduler step is called after optimizer step (standard PyTorch pattern)

---

## Reference

`deep_dive.md` — synthesis of Kaplan, Chinchilla, Sardana (2401.00448), Besiroglu (2404.10102),
and Reconciling (2406.12907) scaling law papers. Consult before making any architecture or
hyperparameter decision.
