"""Train a BPE tokenizer on a streaming subset of fineweb-edu.

Never loads more than one document into RAM at a time.

Usage::

    uv run python scripts/train_tokenizer.py
    uv run python scripts/train_tokenizer.py --max-docs 500000 --vocab-size 8192
"""

from __future__ import annotations

import argparse
from collections.abc import Generator
from pathlib import Path


def _text_stream(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    max_docs: int,
    token: str | None = None,
) -> Generator[str, None, None]:
    """Lazily yield text strings from a streaming HuggingFace dataset.

    ``dataset_config`` is the subset/config name (e.g. "sample-10BT"),
    passed as the second positional argument to ``load_dataset``.
    ``dataset_split`` is the split within that config (e.g. "train").

    Imports datasets inside the function so the module is importable
    without the library being installed (e.g. in test environments).
    """
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        streaming=True,
        trust_remote_code=False,
        token=token,
    )
    for i, sample in enumerate(dataset):
        if i >= max_docs:
            break
        text = sample.get("text", "")
        if text:
            yield text


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on fineweb-edu")
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=500_000,
        help="Number of documents to stream for training (default: 200k)",
    )
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--output", default="tokenizer.json")
    args = parser.parse_args()

    from scripts._auth import load_hf_token
    from src.tokenizer import BPETokenizer

    token = load_hf_token()
    print(f"HF_TOKEN: {'set' if token else 'not set (unauthenticated)'}")
    print(
        f"Training BPE tokenizer (vocab_size={args.vocab_size}) "
        f"on up to {args.max_docs:,} documents from {args.dataset_name}..."
    )

    tok = BPETokenizer()
    tok.train(
        _text_stream(
            args.dataset_name,
            args.dataset_config,
            args.dataset_split,
            args.max_docs,
            token=token,
        ),
        vocab_size=args.vocab_size,
    )

    output_path = Path(args.output)
    tok.save(str(output_path))
    print(f"Saved tokenizer ({tok.vocab_size:,} tokens) → {output_path}")


if __name__ == "__main__":
    main()
