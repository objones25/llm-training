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

```
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

| Flag               | Default           | Description                            |
| ------------------ | ----------------- | -------------------------------------- |
| `--train-bin`      | `data/train.bin`  | Path to the uint16 training token file |
| `--max-steps`      | `10000`           | Total training steps                   |
| `--batch-size`     | `32`              | Sequences per batch                    |
| `--learning-rate`  | `3e-4`            | Peak learning rate (after warmup)      |
| `--checkpoint-dir` | `checkpoints`     | Directory for checkpoint `.pt` files   |
| `--plot-dir`       | `plots`           | Directory for saved plot images        |

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
- Checkpoints are saved as `checkpoint_NNNNNNN.pt` (7-digit zero-padded step number) every `checkpoint_every` steps (default: 1,000).
- Six plot files are updated every `plot_every` steps (default: 500): `loss.png`, `lr.png`, `grad_norm.png`, `grad_heatmap.png`, `weight_norm.png`, `grad_hist.png`.

**Architecture and other hyperparameters** (`n_layers`, `d_model`, `n_heads`, `d_ff`, `warmup_steps`, `weight_decay`, etc.) require editing `src/config.py` directly — they are not exposed as CLI flags.

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
