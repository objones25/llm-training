# Training Operations Guide

Operational reference for running training jobs. For coding guidance and implementation contracts, see `CLAUDE.md`. For contributor workflow, see `CONTRIBUTING.md`.

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

### Dataset Scale (measured)

`train.bin` = 9.90B tokens (19.80 GB), `val.bin` = 100.1M tokens. Total ~10.0B tokens. (Tokenized with vocab_size=32768; prior measurement of 11.78B used vocab_size=8192 — larger vocab produces fewer, longer tokens.)

Chinchilla-optimal for this dataset: `N_optimal = 10.0B / 20 ≈ 500M non-embedding params`. The current run uses `n_layers=12, d_model=2048, n_heads=32, d_ff=8192` (~604M params) — slightly over-parameterized but close. Expected wall time: ~12 GPU-hours on an H100.

---

## Cloud Pod Data Pipeline

On a cloud pod the local disk (`/root`, typically 20–50 GB) will fill up before sample-10BT finishes downloading. Always redirect the HF shard cache and output to the persistent volume:

```bash
# 1. Train the tokenizer (saved to ./tokenizer.json by default)
uv run python scripts/train_tokenizer.py --vocab-size 32768 --max-docs 500000

# 2. Move it to the persistent volume
cp ~/llm-training/tokenizer.json /workspace/tokenizer.json

# 3. Pre-tokenize — HF cache and output go to /workspace
uv run python scripts/pretokenize.py --tokenizer /workspace/tokenizer.json --hf-cache-dir /workspace/hf_cache --output-dir /workspace/data

# 4. Train
uv run python scripts/run_training.py --train-bin /workspace/data/train.bin --val-bin /workspace/data/val.bin --device cuda --use-muon --use-amp --use-compile --checkpoint-dir /workspace/checkpoints --plot-dir /workspace/plots
```

`--hf-cache-dir` sets `HF_HOME` before any HuggingFace import fires, so shards never touch the small local disk. See `README.md` for full cloud setup steps (SSH, RunPod provisioning, Lambda Labs alternative).

### `torch.compile` / triton dependency notes

- `triton` and `setuptools` are required for `--use-compile` on CUDA. Both are in `pyproject.toml` with Linux/x86_64 platform markers so they don't break `uv sync` on macOS.
- `triton` has no macOS wheels — never add it without a platform marker.
- After `uv add triton`, always run `uv sync` on the pod to actually install it.
- If triton fails to import with `ModuleNotFoundError: No module named 'setuptools'`, run `uv add setuptools`.

---

## Checkpoint Evaluation

After training completes, evaluate the final model on validation data:

```bash
# Evaluate latest checkpoint in checkpoint_dir
uv run python scripts/evaluate.py

# Evaluate a specific checkpoint
uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt

# Override checkpoint or validation data paths
uv run python scripts/evaluate.py --checkpoint-dir /workspace/checkpoints --val-bin /workspace/data/val.bin

# Skip text generation and report perplexity only
uv run python scripts/evaluate.py --no-sample
```

---

## Resuming from a Checkpoint

```bash
uv run python scripts/run_training.py --train-bin /workspace/data/train.bin --val-bin /workspace/data/val.bin --device cuda --use-muon --resume checkpoints/best.pt
```

`best.pt` is overwritten in place on each validation improvement — safe to download at any point during training.
