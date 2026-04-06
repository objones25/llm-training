# CONTRIBUTING.md

This document defines the workflow every contributor must follow. The rules here are not
suggestions — they exist to keep the project debuggable at every stage of development.

---

## The Non-Negotiable Order

Every component must be built in this exact sequence:

```
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

```
1.  config.py         — TrainConfig dataclass, no logic
2.  tokenizer.py      — encode, decode, round-trip fidelity
3.  model.py          — Transformer forward pass, shape/dtype contracts
4.  dataset.py        — streaming loader, HuggingFace mock in tests
5.  dataloader.py     — batching, padding, vocab bounds checks
6.  scheduler.py      — cosine LR with warmup, schedule length contract
7.  optimizer.py      — AdamW with correct weight decay grouping
8.  checkpoint.py     — save/load model + optimizer, logit identity check
9.  logger.py         — GradientLogger, per-step and per-layer output contracts
10. plots.py          — all matplotlib plots, file output contracts
11. train.py          — training loop, logging contract, nan detection
12. smoke test        — full mini loop: 10 steps, 2-layer, synthetic data
```

The rationale: each layer depends on the one above it. A bug in `model.py` caught before
`train.py` exists takes minutes to fix. The same bug found during a training run takes hours.
`logger.py` and `plots.py` are built before `train.py` so the training loop is observable
from step one — never add observability after the fact.

---

## Before You Write Any Code

Read the relevant section of `deep_dive.md` first. For each component, the minimum required
reading is:

| Component       | Minimum reading                                    |
| --------------- | -------------------------------------------------- |
| `model.py`      | Sections 2, 3 (architecture, FLOPs accounting)     |
| `optimizer.py`  | Section 4 (AdamW finding)                          |
| `scheduler.py`  | Section 4 (cosine schedule constraint)             |
| `dataloader.py` | Section 8 (critical batch size)                    |
| `logger.py`     | Logging Contract in `CLAUDE.md`                    |
| `plots.py`      | Plots Contract in `CLAUDE.md`                      |
| `train.py`      | Sections 6, 7 (compute-optimal, inference scaling) |
| `config.py`     | Pre-Training Checklist in `CLAUDE.md`              |

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

### `model.py` — init sanity log

The model's `__init__` must log the non-embedding parameter count on construction:

```python
n_params = sum(
    p.numel() for name, p in self.named_parameters()
    if "embedding" not in name
)
print(f"model non_embedding_params={n_params:,}")
```

### `scheduler.py` — schedule length guard

```python
if cosine_steps != max_steps:
    raise ValueError(
        f"Cosine cycle length ({cosine_steps}) must equal max_steps ({max_steps}). "
        "Mismatches >25% measurably degrade final loss."
    )
```

### `optimizer.py` — weight decay grouping

Weight decay must never be applied to biases or LayerNorm parameters. Implemented as
explicit parameter groups, not by setting `weight_decay=0` globally:

```python
decay_params = [
    p for n, p in model.named_parameters()
    if p.requires_grad and not any(nd in n for nd in ["bias", "norm"])
]
no_decay_params = [
    p for n, p in model.named_parameters()
    if p.requires_grad and any(nd in n for nd in ["bias", "norm"])
]
```

### `plots.py` — Agg backend

Must be set at the top of the file, before any other matplotlib import:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Never use the default backend. Plots must work on a headless SSH session.

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

1. `pytest -x --tb=short` passes with zero failures and zero warnings.
2. `pytest --cov=src --cov-report=term-missing` shows ≥90% line coverage for the changed module.
3. All 26 testing rules in `CLAUDE.md` are satisfied for the new component.
4. The Pre-Training Checklist in `CLAUDE.md` is updated if any hyperparameter defaults changed.
5. Any new dependency is pinned in `pyproject.toml` and added to the Library Policy table in `CLAUDE.md`.

---

## Common Mistakes to Avoid

| Mistake                                  | Consequence                                   | Fix                                           |
| ---------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| Using `Adam` instead of `AdamW`          | Worse final loss, especially at scale         | Always use `AdamW`                            |
| Setting cosine length > `max_steps`      | Measurable loss degradation                   | Enforce the guard in `scheduler.py`           |
| Comparing `N` including embeddings       | Misleading scale comparisons                  | Always report non-embedding N                 |
| Not checking `nan` gradients             | Silent corruption of all subsequent steps     | Rule 10: check for `nan` and `inf` explicitly |
| Hardcoding hyperparameters in `train.py` | Untestable, non-reproducible runs             | Everything goes in `TrainConfig`              |
| Training to full convergence             | Wastes compute                                | Stop ~10% above converged loss                |
| Writing tests after implementation       | Tests rationalize code rather than specify it | Interface → tests → implementation, always    |
| Putting plot logic in `train.py`         | Untestable, bloated training loop             | All plots live in `plots.py`                  |
| Putting log formatting in `train.py`     | Same problem                                  | All logging lives in `logger.py`              |
| Using a display backend for matplotlib   | Crashes on headless SSH                       | Always set `matplotlib.use("Agg")` first      |
