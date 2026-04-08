# CONTRIBUTING.md

This document defines the workflow every contributor must follow. The rules here are not
suggestions — they exist to keep the project debuggable at every stage of development.

---

## The Non-Negotiable Order

Every component must be built in this exact sequence:

```text
1. Define the interface (function signatures + docstrings, no implementation)
2. Write the tests (all applicable rules from CLAUDE.md must be covered — they should all fail)
3. Implement until tests pass
4. Open a PR
```

**Never write implementation before tests.** If you find yourself writing tests after the
fact to cover code you already wrote, the PR will be rejected.

---

## Component Build Order

Build components in this sequence. Do not start a component until all tests for the
previous one pass.

```text
1.  config.py         — TrainConfig dataclass, no logic
2.  tokenizer.py      — encode, decode, round-trip fidelity
3.  model.py          — Transformer forward pass, shape/dtype contracts
4.  kv_cache.py       — LayerKVCache and KVCache dataclasses for generation
5.  dataset.py        — streaming loader, HuggingFace mock in tests
6.  dataloader.py     — batching, padding, vocab bounds checks
7.  scheduler.py      — cosine LR with warmup, schedule length contract
8.  optimizer.py      — AdamW with correct weight decay grouping
9.  muon.py           — Muon optimizer with Newton-Schulz orthogonalization
10. checkpoint.py     — save/load model + optimizer, logit identity check
11. logger.py         — GradientLogger, per-step and per-layer output contracts
12. plots.py          — all matplotlib plots, file output contracts
13. train.py          — training loop, logging contract, nan detection
14. smoke test        — full mini loop: 10 steps, 2-layer, synthetic data
```

The rationale: each layer depends on the one above it. A bug in `model.py` caught before
`train.py` exists takes minutes to fix. The same bug found during a training run takes hours.
`logger.py` and `plots.py` are built before `train.py` so the training loop is observable
from step one — never add observability after the fact.

---

## Before You Write Any Code

Read the relevant section of `deep_dive.md` first. For each component, the minimum required
reading is:

| Component       | Minimum reading                                         |
| --------------- | ------------------------------------------------------- |
| `model.py`      | Sections 2, 3 (architecture, FLOPs accounting)          |
| `kv_cache.py`   | Attention Mechanism and KV Cache section in `CLAUDE.md` |
| `optimizer.py`  | Section 4 (AdamW finding)                               |
| `scheduler.py`  | Section 4 (cosine schedule constraint)                  |
| `dataloader.py` | Section 8 (critical batch size)                         |
| `logger.py`     | Logging Contract in `CLAUDE.md`                         |
| `plots.py`      | Plots Contract in `CLAUDE.md`                           |
| `train.py`      | Sections 6, 7 (compute-optimal, inference scaling)      |
| `config.py`     | Pre-Training Checklist in `CLAUDE.md`                   |

If you make an architecture or hyperparameter decision that isn't grounded in `deep_dive.md`,
document your reasoning explicitly in a code comment.

---

## Writing Tests

### Structure

Each test file must follow this layout:

```python
# tests/test_<component>.py

import pytest
import torch
from src.config import TrainConfig

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    # Always construct config explicitly with test-appropriate values.
    # Never rely on production defaults silently.
    return TrainConfig(n_layers=2, d_model=64, n_heads=2, d_ff=128, vocab_size=256)


# ── Shape / dtype contracts ───────────────────────────────────────────────────

def test_output_shape(cfg):
    ...


# ── Value contracts ───────────────────────────────────────────────────────────

def test_loss_decreasing(cfg):
    ...


# ── Failure / edge cases ──────────────────────────────────────────────────────

def test_raises_on_nan_loss(cfg):
    ...
```

### Test checklist before submitting

- [ ] No test hits the network (mock `datasets` for anything that loads data)
- [ ] No test takes longer than 10 seconds
- [ ] Every test that touches randomness sets its own seed with `torch.manual_seed(42)` in the test body
- [ ] Every numeric output is checked for both shape **and** dtype
- [ ] Gradients are checked for non-None, non-zero, **and** absence of `nan`/`inf`
- [ ] Disk writes use `tmp_path` only
- [ ] Plot tests pass synthetic data directly to plot functions — no training loop
- [ ] KV cache tests verify: (1) correct cache shapes after prefill, (2) cached generation output matches uncached greedy sampling, (3) seq_len property returns correct value

---

## Failing Early: Required Guards in Implementation

Every component must fail loudly rather than silently degrade. The following guards are
required — their presence will be checked in PR review.

### `train.py` — nan detection

```python
if not torch.isfinite(loss):
    raise RuntimeError(f"Loss is {loss.item()} at step {step}. Aborting.")
```

This must appear inside the training loop, every step, before the optimizer step.
Do not wrap it in a flag that can be disabled.

### `model.py` — non-embedding parameter count

The model's `__init__` must compute and store the non-embedding parameter count as an attribute:

```python
self.n_params = sum(
    p.numel() for name, p in self.named_parameters()
    if "embedding" not in name
)
```

**Do not print inside `__init__`**. Let `train.py` surface this value on startup by checking `hasattr(model, "n_params")` and printing when training begins. This keeps `__init__` free of side effects and testable in isolation.

### `scheduler.py` — schedule length guard

```python
if cosine_steps != max_steps:
    raise ValueError(
        f"Cosine cycle length ({cosine_steps}) must equal max_steps ({max_steps}). "
        "Mismatches >25% measurably degrade final loss."
    )
```

### `optimizer.py` — weight decay grouping

Weight decay must never be applied to RMSNorm or embedding parameters. Implemented as
three explicit parameter groups identified by **module type** (not name substring):

```python
# RMSNorm params — identified by type, not name.
# The GPT model names these ln_1/ln_2/ln_f — none contain "norm".
ln_ids: set[int] = set()
for module in model.modules():
    if isinstance(module, RMSNorm):
        for param in module.parameters():
            ln_ids.add(id(param))

# Embedding params — identified by name substring.
embed_ids: set[int] = set()
for name, param in model.named_parameters():
    if "embedding" in name:
        embed_ids.add(id(param))

# ln: lr × ln_lr_mult, no weight decay
# embed: lr × embed_lr_mult, no weight decay
# matrix (all others): base lr, weight_decay applied
```

### `plots.py` — Agg backend

Must be set at the top of the file, before any other matplotlib import:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Never use the default backend. Plots must work on a headless SSH session.

### `config.py` — Eager Validation in `__post_init__`

`TrainConfig.__post_init__` validates parameters immediately at construction time and raises `ValueError` if any check fails. This is intentional — it forces invalid configs to fail fast before they can poison downstream logic.

**Critical testing rule:** When testing invalid configs, construct them **inside** `pytest.raises(ValueError)` blocks:

```python
# CORRECT ✓
def test_negative_batch_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        cfg = TrainConfig(batch_size=-1)

# WRONG ✗ — raises before entering the with block
def test_negative_batch_size():
    cfg = TrainConfig(batch_size=-1)  # Exception here, before pytest.raises
    with pytest.raises(ValueError):
        ...  # Never reached
```

If you construct an invalid config outside the `pytest.raises` context, the exception is raised during construction and your test fails — pytest never gets to catch it.

---

## Gradient Health Interpretation Guide

Use this table when reading `grad_heatmap.png`, `grad_norm.png`, and `grad_hist.png`.
All of these are diagnosable from plots alone — you should not need to add extra
instrumentation after the fact.

| Pattern                                                              | What it means                                  | Where to look first                                              |
| -------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------- |
| `grad_norm_max` >> `grad_norm_min` every step                        | One layer dominating all others                | Final projection layer and embeddings                            |
| Per-layer norms decaying monotonically toward layer 0 in the heatmap | Vanishing gradients through depth              | Residual connections, weight init scale                          |
| Per-layer norms spiking at one specific block                        | Exploding in that block                        | Attention logit scale, missing QK normalization                  |
| Weight norm growing while grad norm shrinks                          | Silent weight growth — AdamW misconfigured     | Weight decay too low, check parameter grouping in `optimizer.py` |
| All norms collapse to near-zero mid-run                              | Dead model                                     | LR too high early in run — check warmup length                   |
| `grad_hist.png` bimodal distribution                                 | Two populations of layers behaving differently | Compare early vs. late layers in the heatmap                     |
| `grad_hist.png` distribution shifted hard left                       | Gradients universally too small                | Init scale too small, or residual scale missing                  |
| WARNING lines appear but loss is stable                              | Occasional spikes being absorbed by grad clip  | Normal at this scale — monitor but do not panic                  |
| WARNING lines appear and loss is rising                              | Gradient explosion preceding loss spike        | Reduce LR, check for missing LayerNorm                           |

The heatmap is the most information-dense diagnostic. Read it left-to-right (time) and
top-to-bottom (depth). Healthy training shows roughly uniform color across both axes with
a slight brightening over time as the model learns.

---

## PR Requirements

A PR will not be merged unless:

1. `uv run black src/ scripts/ tests/` passes with no reformatting needed.
2. `uv run ruff check src/ scripts/ tests/` passes with zero violations.
3. `uv run mypy src/ scripts/` passes with zero errors.
4. `pytest -x --tb=short` passes with zero failures and zero warnings.
5. `pytest --cov=src --cov-report=term-missing` shows ≥90% line coverage for the changed module.
6. All 28 testing rules in `CONTRIBUTING.md > Testing Rules` are satisfied for the new component.
7. The Pre-Training Checklist in `CLAUDE.md` is updated if any hyperparameter defaults changed.
8. Any new dependency is pinned in `pyproject.toml` and added to the Library Policy table in `CLAUDE.md`.

Run the full pre-commit check in one line:

```bash
uv run black src/ scripts/ tests/ && uv run ruff check src/ scripts/ tests/ && uv run mypy src/ scripts/ && uv run python -m pytest -x --tb=short
```

---

## Common Mistakes to Avoid

| Mistake                                   | Consequence                                   | Fix                                              |
| ----------------------------------------- | --------------------------------------------- | ------------------------------------------------ |
| Using `Adam` instead of `AdamW`           | Worse final loss, especially at scale         | Always use `AdamW`                               |
| Setting cosine length > `max_steps`       | Measurable loss degradation                   | Enforce the guard in `scheduler.py`              |
| Comparing `N` including embeddings        | Misleading scale comparisons                  | Always report non-embedding N                    |
| Not checking `nan` gradients              | Silent corruption of all subsequent steps     | Rule 10: check for `nan` and `inf` explicitly    |
| Hardcoding hyperparameters in `train.py`  | Untestable, non-reproducible runs             | Everything goes in `TrainConfig`                 |
| Training to full convergence              | Wastes compute                                | Stop ~10% above converged loss                   |
| Writing tests after implementation        | Tests rationalize code rather than specify it | Interface → tests → implementation, always       |
| Putting plot logic in `train.py`          | Untestable, bloated training loop             | All plots live in `plots.py`                     |
| Putting log formatting in `train.py`      | Same problem                                  | All logging lives in `logger.py`                 |
| Using a display backend for matplotlib    | Crashes on headless SSH                       | Always set `matplotlib.use("Agg")` first         |
| Not verifying cached == uncached sampling | Silent mismatch in generation quality         | Test greedy KV cache output matches uncached run |
| Forgetting `is_causal` adapts for cache   | Incorrect attention mask during generation    | Set `is_causal=(T>1)` when cache is provided     |

---

## Regression Prevention Rules

These rules exist because past code reviews fixed instances of each issue. Do not introduce regressions.

### Rule: Always pass `scheduler=` to checkpoint functions

`save_checkpoint()` and `load_checkpoint()` in `src/checkpoint.py` both accept an optional `scheduler` parameter. **Always pass it when training.**

```python
# Correct
save_checkpoint(model, optimizer, step, cfg, scheduler=scheduler)
load_checkpoint(path, model, optimizer, scheduler=scheduler)

# Wrong — scheduler state is lost, LR trajectory breaks on resume
save_checkpoint(model, optimizer, step, cfg)  # scheduler omitted
```

Without this, resuming a run loses the exact LR schedule — the scheduler resets to initial state and training diverges.

**Test coverage:** Rule 12 requires checkpoint tests to verify scheduler state round-trips exactly.

### Rule: Capture gradient norms before clipping

Per-layer gradient norms must be captured **before** `torch.nn.utils.clip_grad_norm_()` is applied. This is the only way to see the true gradient magnitudes.

```python
# Correct
layer_norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()

# Wrong — captures post-clip norms, hides explosion/vanishing signals
torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
layer_norms = {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}
```

Post-clip norms are useless for diagnostics because they don't tell you what the clipping removed.

### Rule: Call `model.train()` before the training loop, not inside it

```python
# Correct
model.train()
for step, batch in enumerate(batches):
    ...

# Wrong — inefficient and unclear intent
for step, batch in enumerate(batches):
    model.train()
    ...
```

Calling `model.train()` once is cheaper, clearer, and prevents accidental mode switches mid-loop.

### Rule: Construct invalid configs inside `pytest.raises()` blocks

Because `TrainConfig.__post_init__` validates eagerly, invalid configs raise `ValueError` at construction time — before any test code can run.

```python
# Correct
def test_warmup_exceeds_max_steps():
    with pytest.raises(ValueError, match="must be less than"):
        cfg = TrainConfig(max_steps=100, warmup_steps=150)  # Inside the block

# Wrong — exception raised before pytest.raises is entered
def test_warmup_exceeds_max_steps():
    cfg = TrainConfig(max_steps=100, warmup_steps=150)  # Exception here
    with pytest.raises(ValueError):
        pass  # Never reached
```

### Rule: Store `model.n_params` as an attribute, not printed in `__init__`

```python
# Correct (in GPT.__init__)
self.n_params = sum(p.numel() for name, p in self.named_parameters() if "embedding" not in name)

# Wrong — side effect in __init__
print(f"model non_embedding_params={n_params:,}")
```

Then in `train.py`:

```python
if hasattr(model, "n_params"):
    print(f"model non_embedding_params={model.n_params:,}")
```

This keeps `__init__` pure and testable. The training loop is responsible for all observability output.

### Rule: Replace type-ignore comments with assertions

Old code used `# type: ignore[union-attr]` to suppress static analysis errors. New code uses explicit assertions instead:

```python
# Old (suppress check)
token_ids = self._tokenizer.encode(text)  # type: ignore[union-attr]

# New (assert precondition)
assert self._tokenizer is not None
token_ids = self._tokenizer.encode(text)
```

Assertions fail fast and give runtime visibility into precondition violations.

---

## Testing Rules (Non-Negotiable)

All 28 rules below must be followed for every component. The component build order above determines *when* each test is written.

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
22. A test must assert that the model in inference mode produces identical outputs on two identical inputs — catches dropout left active during inference. Set the model to inference mode with `.eval()` before the two forward passes.
23. A test must assert that the training loop raises explicitly (not silently continues) when loss is `nan`.
24. A test must assert that `GradientLogger.log_layers()` emits per-layer lines for every named parameter, at the correct step cadence, using a fixed synthetic model. Tests use `caplog` — not `capsys` — because output routes through the `logging` module.
25. A test must assert that a `WARNING` line is emitted (not an exception) when any layer's gradient norm exceeds `grad_norm_warn_threshold`.
26. Plot tests must assert that each plot function produces a valid, non-empty `.png` file at the expected path — use synthetic data passed directly to the plot function, do not run a training loop.
27. `TrainConfig` must have a dedicated test file (`test_config.py`) with comprehensive coverage of all `__post_init__` validation rules. Each invalid config must be constructed **inside** `pytest.raises(ValueError)`, never before it.
28. Model initialization must store the non-embedding parameter count as `model.n_params` (an attribute). Tests must assert `model.n_params` exists and is equal to the manually computed count. Do not print parameter counts in `__init__` — let `train.py` surface this value.
