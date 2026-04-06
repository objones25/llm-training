# llm-training

A small GPT-style language model trained on [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT), built incrementally for learning purposes.

## Setup

```bash
uv sync
```

---

## Scripts

### 1. Train the tokenizer

Streams documents from fineweb-edu and trains a ByteLevel BPE tokenizer. Never loads the full dataset into memory.

```bash
uv run python scripts/train_tokenizer.py
```

**Options:**

| Flag              | Default                     | Description                                |
| ----------------- | --------------------------- | ------------------------------------------ |
| `--vocab-size`    | `8192`                      | Target vocabulary size                     |
| `--max-docs`      | `200000`                    | Number of documents to stream for training |
| `--output`        | `tokenizer.json`            | Output path for the saved tokenizer        |
| `--dataset-name`  | `HuggingFaceFW/fineweb-edu` | HuggingFace dataset name                   |
| `--dataset-split` | `sample-10BT`               | Dataset split                              |

**Example — smaller vocab for a quick experiment:**

```bash
uv run python scripts/train_tokenizer.py --vocab-size 4096 --max-docs 50000 --output tokenizer_small.json
```

Training 200k documents takes a few minutes. The output is a single JSON file that can be loaded with `BPETokenizer.load("tokenizer.json")`.

---

### 2. Pre-tokenize the dataset

Streams the full dataset, encodes every document, and writes two binary files (`train.bin` / `val.bin`) to disk as raw `uint16` arrays. This is a one-time preprocessing step — the training loop reads from these files directly via `numpy.memmap`.

Requires a tokenizer trained in step 1.

```bash
uv run python scripts/pretokenize.py --tokenizer tokenizer.json
```

**Options:**

| Flag              | Default                     | Description                                         |
| ----------------- | --------------------------- | --------------------------------------------------- |
| `--tokenizer`     | _(required)_                | Path to a tokenizer JSON from step 1                |
| `--seq-len`       | `512`                       | Fixed sequence length for each chunk                |
| `--val-every`     | `100`                       | Route every Nth document to the val split (~1% val) |
| `--output-dir`    | `data`                      | Directory for `train.bin` and `val.bin`             |
| `--dataset-name`  | `HuggingFaceFW/fineweb-edu` | HuggingFace dataset name                            |
| `--dataset-split` | `sample-10BT`               | Dataset split                                       |

**Example — custom output directory and sequence length:**

```bash
uv run python scripts/pretokenize.py \
    --tokenizer tokenizer.json \
    --seq-len 1024 \
    --output-dir data/processed
```

**Output files:**

```text
data/
  train.bin   # ~99% of documents, raw uint16 tokens
  val.bin     #  ~1% of documents, raw uint16 tokens
```

Each file is a flat sequence of `uint16` values. At `seq_len=512` and `vocab_size=8192`, the full sample-10BT run produces roughly 20 GB across both files. The training dataloader will read them with `numpy.memmap` — no file needs to fit in RAM.

**Memory usage:** peak RAM is bounded by one document's tokens plus one `seq_len`-sized buffer at any point during the run. Progress is printed every 10,000 documents.

---

### 3. Run the training loop

Reads from the pre-tokenized binary file and runs the full pretraining loop. Checkpoints are saved to `checkpoints/` and plots to `plots/` by default.

Requires a binary dataset from step 2.

```bash
uv run python scripts/run_training.py --train-bin data/train.bin
```

**Options:**

| Flag                         | Default          | Description                                               |
| ---------------------------- | ---------------- | --------------------------------------------------------- |
| `--train-bin`                | `data/train.bin` | Path to the uint16 training token file                    |
| `--val-bin`                  | `data/val.bin`   | Path to the uint16 validation token file (skipped if absent) |
| `--max-steps`                | `10000`          | Total training steps                                      |
| `--batch-size`               | `32`             | Sequences per batch                                       |
| `--learning-rate`            | `3e-4`           | Peak learning rate (after warmup)                         |
| `--checkpoint-dir`           | `checkpoints`    | Directory for checkpoint `.pt` files                      |
| `--plot-dir`                 | `plots`          | Directory for saved plot images                           |
| `--device`                   | `cpu`            | Training device: `cpu`, `mps`, `cuda`                     |
| `--early-stopping-patience`  | `0`              | Val evals without improvement before stopping; 0 disables |

**Example — Apple Silicon with validation and early stopping:**

```bash
uv run python scripts/run_training.py \
    --train-bin data/train.bin \
    --val-bin data/val.bin \
    --device mps \
    --early-stopping-patience 5
```

**Example — shorter run with faster learning rate:**

```bash
uv run python scripts/run_training.py \
    --train-bin data/train.bin \
    --max-steps 5000 \
    --learning-rate 1e-3 \
    --checkpoint-dir runs/exp1/ckpts \
    --plot-dir runs/exp1/plots
```

**Output:**

- A summary line is printed before training starts (token count, model size, estimated compute in EFLOPs).
- Per-step log lines are written to stdout: `step=N loss=X.XXXX lr=X.Xe-XX grad_norm=X.XXXX ...`
- When `--val-bin` is provided, validation loss is evaluated every `val_every` steps (default: 250) and logged as `val step=N val_loss=X.XXXX`.
- When `--early-stopping-patience` is set, training stops automatically if val loss has not improved for that many consecutive evaluations.
- Checkpoints are saved as `checkpoint_NNNNNNN.pt` (7-digit zero-padded step number) every `checkpoint_every` steps (default: 1,000).
- Six plot files are updated every `plot_every` steps (default: 500): `loss.png`, `lr.png`, `grad_norm.png`, `grad_heatmap.png`, `weight_norm.png`, `grad_hist.png`.

**Architecture and other hyperparameters** (`n_layers`, `d_model`, `n_heads`, `d_ff`, `warmup_steps`, `weight_decay`, etc.) require editing `src/config.py` directly — they are not exposed as CLI flags.

**Advanced options in `src/config.py`:**

| Field         | Default      | Purpose                                        |
| ------------- | ------------ | ---------------------------------------------- |
| `adamw_betas` | (0.9, 0.999) | AdamW momentum and variance decay rates        |
| `adamw_eps`   | 1e-8         | AdamW numerical stability constant             |
| `use_compile` | False        | Enable `torch.compile()` (adds 30-60s startup) |
| `use_amp`     | False        | Automatic mixed precision (CUDA only)          |

**Device selection:**

| Device | Flag            | Notes                                                                  |
| ------ | --------------- | ---------------------------------------------------------------------- |
| CPU    | `--device cpu`  | Default. Works everywhere, slowest.                                    |
| MPS    | `--device mps`  | Apple Silicon (M1/M2/M3/M4). Requires macOS ≥13 and PyTorch ≥2.0.      |
| CUDA   | `--device cuda` | NVIDIA GPU. Use on cloud instances (Lambda Labs, RunPod, Colab, etc.). |

```bash
# Apple Silicon (M1–M4)
uv run python scripts/run_training.py --train-bin data/train.bin --device mps

# NVIDIA GPU (cloud)
uv run python scripts/run_training.py --train-bin data/train.bin --device cuda
```

---

### 4. Evaluate the trained model

Loads the latest checkpoint (or a specific one), computes perplexity on the validation set, and optionally generates a short text sample as a qualitative sanity check.

Requires a checkpoint from step 3 and `val.bin` from step 2.

```bash
uv run python scripts/evaluate.py
```

**Options:**

| Flag                  | Default          | Description                                                    |
| --------------------- | ---------------- | -------------------------------------------------------------- |
| `--checkpoint`        | _(auto-detect)_  | Path to a specific `.pt` file; overrides `--checkpoint-dir`    |
| `--checkpoint-dir`    | `checkpoints`    | Directory to search for the latest checkpoint                  |
| `--val-bin`           | `data/val.bin`   | Path to the uint16 validation token file                       |
| `--device`            | _(from cfg)_     | Evaluation device: `cpu`, `mps`, `cuda`                        |
| `--val-batches`       | _(all)_          | Max number of val batches to evaluate                          |
| `--no-sample`         | off              | Skip text generation (useful in CI / headless environments)    |
| `--tokenizer`         | _(auto-detect)_  | Path to tokenizer JSON; auto-discovered from cwd if omitted    |
| `--top-k`             | `50`             | Top-k value for sampling; set to `1` for greedy               |
| `--max-new-tokens`    | `100`            | Number of tokens to generate in the sample                     |

**Example — evaluate the latest checkpoint on MPS, no sample output:**

```bash
uv run python scripts/evaluate.py --device mps --no-sample
```

**Example — evaluate a specific checkpoint with a custom val file:**

```bash
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/checkpoint_0005000.pt \
    --val-bin data/val.bin
```

**Output:**

```text
checkpoint : checkpoints/checkpoint_0005000.pt
step       : 5,000
config     : 6-layer  d_model=512  n_heads=8  vocab=8192
val data   : 40 batches  (655,360 tokens)

val loss   : 3.2841
perplexity : 26.72

── sample (decoded) ──────────────────────────────────
The study found that students who...
```

If no tokenizer is found, the sample is printed as raw token IDs instead.

---

## Running tests

```bash
# All tests
uv run python -m pytest -x --tb=short

# Single file
uv run python -m pytest tests/test_tokenizer.py -x --tb=short

# With coverage
uv run python -m pytest --cov=src --cov-report=term-missing
```

All tests run on CPU with synthetic data — no GPU or network access required.

---

## Checkpoints and Resuming

Checkpoints are saved automatically every `checkpoint_every` steps (default: 1,000) as `checkpoint_NNNNNNN.pt` files in `checkpoint_dir`. Each checkpoint contains:

- Model weights
- Optimizer state (momentum and variance buffers for AdamW)
- Scheduler state (current LR schedule position)
- Training step counter
- Configuration (`TrainConfig`)

To resume training from a checkpoint, load it and pass the same config plus restored optimizer and scheduler:

```python
from src.checkpoint import load_checkpoint

step = load_checkpoint("checkpoints/checkpoint_0005000.pt", model, optimizer, scheduler=scheduler)
cfg.max_steps += 5000  # Extend training by another 5k steps
train(cfg, model=model, token_stream=resumed_stream)
```

**Important:** Always pass `scheduler=scheduler` to both `save_checkpoint()` and `load_checkpoint()`. Without it, the LR schedule state is lost and training diverges on resume — the scheduler resets to initial state instead of resuming from where it left off.

Checkpoints created without scheduler state (pre-v2 format) will emit a warning on load and the scheduler will start fresh. New checkpoints always include scheduler state.
