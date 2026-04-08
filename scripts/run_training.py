"""Run the pretraining loop on pre-tokenized binary data.

Reads a flat uint16 binary file produced by ``scripts/pretokenize.py``
and streams tokens lazily through the training loop.  The token stream
is memory-mapped — only the pages accessed by the current batch are
loaded into RAM regardless of file size.

Usage::

    uv run python scripts/run_training.py --train-bin data/train.bin
    uv run python scripts/run_training.py \\
        --train-bin data/train.bin \\
        --max-steps 10000 \\
        --batch-size 32 \\
        --learning-rate 3e-4 \\
        --checkpoint-dir checkpoints \\
        --plot-dir plots
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Generator
from pathlib import Path


def _token_stream(path: Path) -> Generator[int, None, None]:
    """Lazily yield token IDs from a flat uint16 binary file.

    Uses ``numpy.memmap`` so only the pages currently accessed are loaded
    into RAM.  The generator is single-pass — suitable for passing directly
    to ``train()``.
    """
    import numpy as np

    tokens = np.memmap(path, dtype="<u2", mode="r")
    for tok in tokens:
        yield int(tok)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pretraining on a pre-tokenized binary dataset"
    )
    parser.add_argument(
        "--train-bin",
        default="data/train.bin",
        help="Path to the uint16 training token file (default: data/train.bin)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override TrainConfig.max_steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override TrainConfig.batch_size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override TrainConfig.learning_rate",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Override TrainConfig.checkpoint_dir",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Override TrainConfig.plot_dir",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Training device: "cpu", "mps" (Apple Silicon), or "cuda" (NVIDIA GPU)',
    )
    parser.add_argument(
        "--val-bin",
        default="data/val.bin",
        help="Path to the uint16 validation token file (default: data/val.bin). "
             "Skipped if the file does not exist.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Override TrainConfig.early_stopping_patience (0 = disabled)",
    )
    parser.add_argument(
        "--use-muon",
        action="store_true",
        default=False,
        help="Use Muon optimizer for weight matrix params; AdamW for ln+embed",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=False,
        help="Enable automatic mixed precision (CUDA only)",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        default=False,
        help="Enable torch.compile with reduce-overhead mode (adds 30-60s cold start)",
    )
    args = parser.parse_args()

    import numpy as np

    from src.config import TrainConfig

    train_bin = Path(args.train_bin)

    # ── Build config with any CLI overrides ───────────────────────────────────
    overrides: dict = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.plot_dir is not None:
        overrides["plot_dir"] = args.plot_dir
    if args.device is not None:
        overrides["device"] = args.device
    if args.early_stopping_patience is not None:
        overrides["early_stopping_patience"] = args.early_stopping_patience
    if args.use_muon:
        overrides["use_muon"] = True
    if args.use_amp:
        overrides["use_amp"] = True
    if args.use_compile:
        overrides["use_compile"] = True
    cfg = TrainConfig(**overrides)

    # ── Pre-training summary ───────────────────────────────────────────────────
    tokens_mmap = np.memmap(train_bin, dtype="<u2", mode="r")
    n_tokens = len(tokens_mmap)
    del tokens_mmap  # release mmap; _token_stream() opens a fresh one

    tokens_per_step = cfg.batch_size * cfg.seq_len
    n_params_approx = (
        cfg.n_layers
        * (
            4 * cfg.d_model * cfg.d_model   # attention projections (Q,K,V,O)
            + 2 * cfg.d_model * cfg.d_ff    # FF up + down
        )
    )
    compute_flops = 6 * n_params_approx * tokens_per_step * cfg.max_steps

    val_bin = Path(args.val_bin)
    val_stream = _token_stream(val_bin) if val_bin.exists() else None

    print(f"train.bin : {n_tokens:,} tokens  ({train_bin.stat().st_size / 1e9:.2f} GB)")
    if val_stream is not None:
        val_n_tokens = len(np.memmap(val_bin, dtype="<u2", mode="r"))
        print(f"val.bin   : {val_n_tokens:,} tokens  ({val_bin.stat().st_size / 1e6:.1f} MB)  "
              f"val_every={cfg.val_every}  val_batches={cfg.val_batches}")
    else:
        print("val.bin   : not found — validation loss disabled")
    print(f"config    : {cfg.n_layers}-layer  d_model={cfg.d_model}  n_heads={cfg.n_heads}  "
          f"d_ff={cfg.d_ff}  vocab={cfg.vocab_size}")
    print(f"training  : max_steps={cfg.max_steps:,}  batch_size={cfg.batch_size}  "
          f"seq_len={cfg.seq_len}  lr={cfg.learning_rate}  device={cfg.device}")
    print(f"compute   : ~{compute_flops / 1e18:.2f} EFLOPs  "
          f"(N≈{n_params_approx / 1e6:.0f}M non-emb params)")
    print(f"outputs   : checkpoints → {cfg.checkpoint_dir}  plots → {cfg.plot_dir}")
    print()
    print("Starting training...")
    print()

    from src.train import train

    train(cfg, token_stream=_token_stream(train_bin), val_token_stream=val_stream)

    print()
    print("Training complete.")


if __name__ == "__main__":
    main()
