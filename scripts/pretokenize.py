"""Pre-tokenize fineweb-edu to binary uint16 files for training.

All three generator stages are lazy — peak RAM usage is bounded by a
single document plus one chunk buffer (``seq_len`` tokens):

    Dataset stream → text_stream → token_stream → chunk_stream → write_bin

The train/val split is made per document (every ``val_every``-th document
goes to val), so both files contain complete sequences with no partial
documents that span the split boundary.

Output
------
    data/train.bin   — uint16 numpy array, shape (N_train,)
    data/val.bin     — uint16 numpy array, shape (N_val,)

Usage::

    uv run python scripts/pretokenize.py --tokenizer tokenizer.json
    uv run python scripts/pretokenize.py --tokenizer tokenizer.json \\
        --seq-len 512 --val-every 100 --output-dir data
"""
from __future__ import annotations

import argparse
import struct
from collections.abc import Generator
from pathlib import Path


# ── Generator pipeline ────────────────────────────────────────────────────────


def _text_stream(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    token: str | None = None,
) -> Generator[str, None, None]:
    """Lazily yield non-empty text strings from a streaming dataset.

    ``dataset_config`` is the subset/config name (e.g. "sample-10BT"),
    passed as the second positional argument to ``load_dataset``.
    ``dataset_split`` is the split within that config (e.g. "train").
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        streaming=True,
        trust_remote_code=False,
        token=token,
    )
    for sample in dataset:
        text = sample.get("text", "")
        if text:
            yield text


def _token_stream(
    texts: Generator[str, None, None],
    encode,
    bos_id: int,
    eos_id: int,
) -> Generator[int, None, None]:
    """Lazily yield individual token IDs, wrapping each document with BOS/EOS.

    Memory at any point: the tokens for a single document.
    """
    for text in texts:
        yield bos_id
        yield from encode(text)
        yield eos_id


def _chunk_stream(
    tokens: Generator[int, None, None],
    chunk_size: int,
) -> Generator[list[int], None, None]:
    """Lazily yield non-overlapping fixed-size chunks of token IDs.

    The final partial chunk (if any) is discarded to keep all sequences
    the same length.

    Memory at any point: one chunk (``chunk_size`` integers).
    """
    buf: list[int] = []
    for tok in tokens:
        buf.append(tok)
        if len(buf) == chunk_size:
            yield buf
            buf = []


# ── Routing: per-document train/val split ─────────────────────────────────────


def _split_text_streams(
    all_texts: Generator[str, None, None],
    val_every: int,
) -> tuple[Generator[str, None, None], Generator[str, None, None]]:
    """Route documents to train or val by position.

    Every ``val_every``-th document (0-indexed) goes to val; the rest go to
    train. Both outputs are lazy generators backed by a shared in-process
    queue so the caller can consume them independently.

    A simpler single-pass approach is used here: iterate once, write tokens
    immediately to the correct file. The two generators are therefore fused
    into the writing loop in ``main()``.
    """
    # This function is a placeholder for documentation clarity.
    # The actual routing happens in main() with a modulo check per document.
    raise NotImplementedError("Use _route_and_write() directly.")


# ── Writer ────────────────────────────────────────────────────────────────────


def _write_chunks(
    path: Path,
    chunks: Generator[list[int], None, None],
) -> int:
    """Append token chunks to a binary uint16 file.

    Each token is written as a 2-byte little-endian unsigned integer.
    Returns the total number of tokens written.

    Memory at any point: one chunk.
    """
    total = 0
    with path.open("wb") as f:
        for chunk in chunks:
            # struct.pack is faster than np.array(...).tofile for small chunks
            f.write(struct.pack(f"<{len(chunk)}H", *chunk))
            total += len(chunk)
    return total


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize fineweb-edu to uint16 binary files"
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to a saved tokenizer JSON (from train_tokenizer.py)",
    )
    parser.add_argument("--dataset-name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", default="sample-10BT")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Fixed sequence length for each chunk (default: 512)",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=100,
        help="Route every Nth document to val split (default: 100 → ~1%% val)",
    )
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    from src.tokenizer import BPETokenizer

    from scripts._auth import load_hf_token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    token = load_hf_token()
    print(f"HF_TOKEN: {'set' if token else 'not set (unauthenticated)'}")

    tok = BPETokenizer.load(args.tokenizer)
    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")

    print(f"Tokenizer loaded: {tok.vocab_size:,} tokens")
    print(f"BOS={bos_id}  EOS={eos_id}  seq_len={args.seq_len}")
    print(f"Val routing: every {args.val_every}th document → {val_path}")
    print(f"Writing to {train_path} and {val_path} ...")

    train_tokens = 0
    val_tokens = 0

    # Single streaming pass: route each document to train or val by position.
    # Both files are written incrementally — never more than one chunk in RAM.
    with train_path.open("wb") as train_f, val_path.open("wb") as val_f:
        train_buf: list[int] = []
        val_buf: list[int] = []

        def _flush(buf: list[int], f) -> int:
            """Write all complete seq_len chunks from buf; return tokens written."""
            written = 0
            while len(buf) >= args.seq_len:
                chunk = buf[: args.seq_len]
                del buf[: args.seq_len]
                f.write(struct.pack(f"<{len(chunk)}H", *chunk))
                written += len(chunk)
            return written

        for doc_idx, text in enumerate(_text_stream(args.dataset_name, args.dataset_config, args.dataset_split, token=token)):
            tokens = [bos_id] + tok.encode(text) + [eos_id]

            if doc_idx % args.val_every == 0:
                val_buf.extend(tokens)
                val_tokens += _flush(val_buf, val_f)
            else:
                train_buf.extend(tokens)
                train_tokens += _flush(train_buf, train_f)

            if doc_idx % 10_000 == 0 and doc_idx > 0:
                print(
                    f"  doc={doc_idx:,}  "
                    f"train={train_tokens:,} tokens  "
                    f"val={val_tokens:,} tokens"
                )

    print(
        f"\nDone.\n"
        f"  train.bin : {train_tokens:,} tokens  "
        f"({train_path.stat().st_size / 1e9:.2f} GB)\n"
        f"  val.bin   : {val_tokens:,} tokens  "
        f"({val_path.stat().st_size / 1e9:.2f} GB)"
    )


if __name__ == "__main__":
    main()
