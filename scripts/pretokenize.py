"""Pre-tokenize fineweb-edu to binary uint16 files.

Parallelizes BPE tokenization across CPU cores using ProcessPoolExecutor.
The dataset is streamed from HuggingFace (single-threaded) while tokenization
runs in parallel worker processes that each hold a loaded tokenizer in memory.

    Download stream → batch docs → parallel tokenize → route train/val → write

The train/val split is made per document (every ``val_every``-th document
goes to val), so both files contain complete sequences with no partial
documents that span the split boundary.

Output
------
    <output-dir>/train.bin   — uint16 numpy array, shape (N_train,)
    <output-dir>/val.bin     — uint16 numpy array, shape (N_val,)

Cloud / pod usage
-----------------
HuggingFace caches downloaded dataset shards to ``~/.cache/huggingface``
by default.  On a cloud pod the local disk is small (~20-50 GB) and will
fill up long before sample-10BT (~100 GB raw) finishes downloading.

Pass ``--hf-cache-dir`` and ``--output-dir`` to redirect everything to the
mounted persistent volume::

    uv run python scripts/pretokenize.py \\
        --tokenizer /workspace/tokenizer.json \\
        --hf-cache-dir /workspace/hf_cache \\
        --output-dir /workspace/data

Usage::

    uv run python scripts/pretokenize.py --tokenizer tokenizer.json
    uv run python scripts/pretokenize.py --tokenizer tokenizer.json \\
        --seq-len 1024 --val-every 100 --num-proc 8 --output-dir data
"""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# SIGTERM (e.g. `kill <pid>` or `pkill`) does not trigger atexit by default.
# Converting it to SystemExit lets atexit handlers run and cleans up workers.
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


# Must be set before importing datasets/huggingface_hub so the cache
# directory is redirected before any initialisation happens.
def _set_hf_cache(cache_dir: str | None) -> None:
    """Redirect the HuggingFace cache to ``cache_dir`` if given."""
    if cache_dir:
        p = Path(cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(p)


# ── Worker state (one instance per worker process) ────────────────────────────

_worker_tok = None
_worker_bos: int = -1
_worker_eos: int = -1


def _init_worker(tokenizer_path: str) -> None:
    """Load tokenizer once into this worker process at startup."""
    global _worker_tok, _worker_bos, _worker_eos
    from src.tokenizer import BPETokenizer

    _worker_tok = BPETokenizer.load(tokenizer_path)
    _worker_bos = _worker_tok.token_to_id("[BOS]")
    _worker_eos = _worker_tok.token_to_id("[EOS]")


def _tokenize_doc(text: str) -> list[int]:
    """Tokenize one document; return [BOS] + token_ids + [EOS]."""
    assert _worker_tok is not None
    return [_worker_bos] + _worker_tok.encode(text) + [_worker_eos]


# ── Writer helper ─────────────────────────────────────────────────────────────


def _flush_buf(buf: list[int], f, seq_len: int) -> int:
    """Write all complete seq_len chunks from buf; return tokens written."""
    written = 0
    while len(buf) >= seq_len:
        chunk = buf[:seq_len]
        del buf[:seq_len]
        f.write(struct.pack(f"<{seq_len}H", *chunk))
        written += seq_len
    return written


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
        default=1024,
        help="Fixed sequence length for each chunk (default: 1024)",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=100,
        help="Route every Nth document to val split (default: 100 → ~1%% val)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=os.cpu_count(),
        help="Number of tokenizer worker processes (default: os.cpu_count())",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Documents per parallel batch submitted to the pool (default: 512)",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help=(
            "Override the HuggingFace cache root (sets HF_HOME). "
            "On cloud pods point this at the mounted volume to avoid filling "
            "the local disk with dataset shards, e.g. /workspace/hf_cache"
        ),
    )
    args = parser.parse_args()

    # Redirect HF cache BEFORE importing datasets so the env var takes effect.
    _set_hf_cache(args.hf_cache_dir)

    from datasets import load_dataset

    from scripts._auth import load_hf_token

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    hf_token = load_hf_token()
    print(f"HF_TOKEN   : {'set' if hf_token else 'not set (unauthenticated)'}")
    print(f"HF_HOME    : {os.environ.get('HF_HOME', '~/.cache/huggingface (default)')}")
    print(
        f"Workers={args.num_proc}  batch={args.batch_size}  "
        f"seq_len={args.seq_len}  val_every={args.val_every}"
    )
    print(f"Writing to {train_path} and {val_path} ...")

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.dataset_split,
        streaming=True,
        trust_remote_code=False,
        token=hf_token,
    )

    train_tokens = 0
    val_tokens = 0
    doc_idx = 0

    with (
        train_path.open("wb") as train_f,
        val_path.open("wb") as val_f,
        ProcessPoolExecutor(
            max_workers=args.num_proc,
            initializer=_init_worker,
            initargs=(args.tokenizer,),
        ) as executor,
    ):
        atexit.register(executor.shutdown, wait=False, cancel_futures=True)
        train_buf: list[int] = []
        val_buf: list[int] = []
        batch: list[str] = []

        def _submit_batch(texts: list[str], base_idx: int) -> None:
            nonlocal train_tokens, val_tokens
            for rel_i, token_ids in enumerate(executor.map(_tokenize_doc, texts)):
                idx = base_idx + rel_i
                if idx % args.val_every == 0:
                    val_buf.extend(token_ids)
                    val_tokens += _flush_buf(val_buf, val_f, args.seq_len)
                else:
                    train_buf.extend(token_ids)
                    train_tokens += _flush_buf(train_buf, train_f, args.seq_len)

        for sample in dataset:
            text = sample.get("text", "")
            if not text:
                continue
            batch.append(text)
            if len(batch) == args.batch_size:
                _submit_batch(batch, doc_idx)
                doc_idx += len(batch)
                batch = []
                if doc_idx % 50_000 == 0:
                    print(
                        f"  doc={doc_idx:,}  "
                        f"train={train_tokens:,} tokens  "
                        f"val={val_tokens:,} tokens"
                    )

        if batch:
            _submit_batch(batch, doc_idx)
            doc_idx += len(batch)

    print(
        f"\nDone. {doc_idx:,} documents processed.\n"
        f"  train.bin : {train_tokens:,} tokens  "
        f"({train_path.stat().st_size / 1e9:.2f} GB)\n"
        f"  val.bin   : {val_tokens:,} tokens  "
        f"({val_path.stat().st_size / 1e9:.2f} GB)"
    )


if __name__ == "__main__":
    main()
