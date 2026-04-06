"""Evaluate a pretrained GPT model loaded from a checkpoint.

Discovers the latest checkpoint in ``cfg.checkpoint_dir``, loads the model,
evaluates perplexity on ``data/val.bin``, and optionally generates a short
text sample as a qualitative sanity check.

Usage::

    # Evaluate using defaults (latest checkpoint, data/val.bin)
    uv run python scripts/evaluate.py

    # Override checkpoint dir or val file
    uv run python scripts/evaluate.py --checkpoint-dir checkpoints --val-bin data/val.bin

    # Skip text generation
    uv run python scripts/evaluate.py --no-sample

    # Evaluate on a specific checkpoint
    uv run python scripts/evaluate.py --checkpoint checkpoints/checkpoint_0005000.pt
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

from src.config import TrainConfig
from src.dataloader import make_batches
from src.model import GPT


# ── Checkpoint discovery ──────────────────────────────────────────────────────


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the highest-numbered ``checkpoint_*.pt`` file, or None if absent.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory to scan.  Returns None if the directory does not exist or
        contains no matching files.
    """
    if not checkpoint_dir.is_dir():
        return None
    candidates = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
    return candidates[-1] if candidates else None


# ── Checkpoint loading (inference-only — no optimizer needed) ─────────────────


def load_checkpoint_for_eval(path: Path) -> tuple[GPT, TrainConfig, int]:
    """Load model weights and config from a checkpoint file.

    Unlike ``src.checkpoint.load_checkpoint``, this function requires no
    optimizer argument — it is intended for inference only.

    Parameters
    ----------
    path : Path
        Path to a ``.pt`` file written by :func:`src.checkpoint.save_checkpoint`.

    Returns
    -------
    tuple[GPT, TrainConfig, int]
        ``(model, cfg, step)`` where *model* is on CPU with weights loaded,
        *cfg* is the :class:`TrainConfig` stored in the checkpoint, and *step*
        is the training step at which the checkpoint was saved.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If the checkpoint is missing expected keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    torch.serialization.add_safe_globals([TrainConfig])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    cfg: TrainConfig = ckpt["cfg"]
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    step: int = ckpt["step"]
    return model, cfg, step


# ── Perplexity evaluation ─────────────────────────────────────────────────────


def compute_perplexity(
    model: GPT,
    val_batches: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[float, float]:
    """Compute average cross-entropy loss and perplexity over *val_batches*.

    The model is set to eval mode and gradients are disabled.

    Parameters
    ----------
    model : GPT
    val_batches : list of (inputs, targets) tensor pairs
    cfg : TrainConfig
        Provides ``vocab_size`` for the loss reshape.
    device : torch.device
        Device to move batches to.

    Returns
    -------
    tuple[float, float]
        ``(avg_loss, perplexity)`` where ``perplexity = exp(avg_loss)``.

    Raises
    ------
    ValueError
        If *val_batches* is empty or computed loss is not finite.
    """
    if not val_batches:
        raise ValueError("val_batches is empty — nothing to evaluate.")

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_batches:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                targets.reshape(-1),
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(val_batches)
    if not math.isfinite(avg_loss):
        raise ValueError(
            f"Loss is {avg_loss} — model may have diverged. "
            "Check the checkpoint and validation data."
        )
    return avg_loss, math.exp(avg_loss)


# ── Text sampling ─────────────────────────────────────────────────────────────


def sample_text(
    model: GPT,
    cfg: TrainConfig,
    device: torch.device,
    seed_tokens: list[int],
    max_new_tokens: int = 100,
    top_k: int = 50,
) -> list[int]:
    """Autoregressively sample *max_new_tokens* tokens from the model.

    Parameters
    ----------
    model : GPT
    cfg : TrainConfig
    device : torch.device
    seed_tokens : list[int]
        Prompt token IDs (must be non-empty).
    max_new_tokens : int
        Number of tokens to generate beyond the seed.
    top_k : int
        Sample from the top-k logits at each step.  Set to 1 for greedy.

    Returns
    -------
    list[int]
        The generated token IDs (seed + newly generated).
    """
    model.eval()
    generated = list(seed_tokens)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[-cfg.seq_len:]
            idx = torch.tensor([context], dtype=torch.long, device=device)
            logits = model(idx)
            next_logits = logits[0, -1, :]  # (vocab_size,)
            if top_k < cfg.vocab_size:
                top_vals, _ = torch.topk(next_logits, top_k)
                threshold = top_vals[-1]
                next_logits = next_logits.masked_fill(
                    next_logits < threshold, float("-inf")
                )
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(int(next_token))
    return generated


def _token_stream_from_bin(path: Path) -> Iterator[int]:
    """Lazily yield token IDs from a flat uint16 binary file."""
    import numpy as np
    tokens = np.memmap(path, dtype="<u2", mode="r")
    for tok in tokens:
        yield int(tok)


def _find_tokenizer(explicit: str | None) -> Path | None:
    """Return first existing tokenizer file, or None."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for name in ("tokenizer.json", "tokenizer.model"):
        p = Path(name)
        if p.exists():
            return p
    return None


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained GPT checkpoint on a validation set"
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a specific .pt checkpoint file.  "
             "When omitted, the latest checkpoint in --checkpoint-dir is used.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory to search for checkpoints (default: TrainConfig.checkpoint_dir).",
    )
    parser.add_argument(
        "--val-bin",
        default="data/val.bin",
        help="Path to the uint16 validation token file (default: data/val.bin).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Evaluation device: "cpu", "mps", or "cuda" (default: same as checkpoint cfg).',
    )
    parser.add_argument(
        "--val-batches",
        type=int,
        default=None,
        help="Max number of val batches to evaluate (default: all).",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Skip text generation (useful in headless/CI environments).",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to tokenizer JSON file for decoding generated text.  "
             "Auto-discovered from cwd if omitted.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k value for sampling (default: 50).  Set to 1 for greedy.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate in the sample (default: 100).",
    )
    args = parser.parse_args()

    # ── Resolve checkpoint path ───────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise SystemExit(f"Error: checkpoint not found: {ckpt_path}")
    else:
        search_dir = (
            Path(args.checkpoint_dir)
            if args.checkpoint_dir
            else Path(TrainConfig().checkpoint_dir)
        )
        ckpt_path = find_latest_checkpoint(search_dir)
        if ckpt_path is None:
            raise SystemExit(
                f"Error: no checkpoint found in '{search_dir}'."
                "\nRun training first with:  uv run python scripts/run_training.py"
            )

    print(f"checkpoint : {ckpt_path}")

    # ── Load model from checkpoint ────────────────────────────────────────────
    model, cfg, step = load_checkpoint_for_eval(ckpt_path)
    print(f"step       : {step:,}")
    print(f"config     : {cfg.n_layers}-layer  d_model={cfg.d_model}  "
          f"n_heads={cfg.n_heads}  vocab={cfg.vocab_size}")

    device_str = args.device or cfg.device
    device = torch.device(device_str)
    model = model.to(device)

    # ── Load validation data ──────────────────────────────────────────────────
    val_bin = Path(args.val_bin)
    if not val_bin.exists():
        raise SystemExit(
            f"Error: validation data not found: {val_bin}"
            "\nGenerate it with:  uv run python scripts/pretokenize.py"
        )

    val_batches = list(make_batches(_token_stream_from_bin(val_bin), cfg))
    if args.val_batches is not None:
        val_batches = val_batches[: args.val_batches]

    n_tokens = len(val_batches) * cfg.batch_size * cfg.seq_len
    print(f"val data   : {len(val_batches)} batches  ({n_tokens:,} tokens)")
    print()

    # ── Compute perplexity ────────────────────────────────────────────────────
    avg_loss, perplexity = compute_perplexity(model, val_batches, cfg, device)
    print(f"val loss   : {avg_loss:.4f}")
    print(f"perplexity : {perplexity:.2f}")
    print()

    # ── Optional text sampling ────────────────────────────────────────────────
    if not args.no_sample:
        seed_inputs, _ = val_batches[0]
        seed_tokens = seed_inputs[0, :5].tolist()

        generated = sample_text(
            model, cfg, device, seed_tokens,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
        )

        tokenizer_path = _find_tokenizer(args.tokenizer)
        if tokenizer_path:
            try:
                from src.tokenizer import BPETokenizer
                tok = BPETokenizer.load(str(tokenizer_path))
                decoded = tok.decode(generated)
                print("── sample (decoded) ──────────────────────────────────")
                print(decoded)
            except Exception as exc:
                print(f"(tokenizer decode failed: {exc} — printing token IDs)")
                print("── sample (token IDs) ────────────────────────────────")
                print(generated)
        else:
            print("── sample (token IDs — no tokenizer found) ───────────")
            print(generated)
        print()


if __name__ == "__main__":
    main()
