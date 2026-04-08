"""Batching, packing, and token ID validation for LM pretraining.

Consumes a flat stream of integer token IDs and yields fixed-size
(inputs, targets) tensor pairs suitable for a GPT-style language model.

Public API
----------
    make_batches(
        token_stream: Iterable[int],
        cfg: TrainConfig,
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]

Design notes
------------
- Packed sequences: tokens are concatenated end-to-end with no padding.
  The last incomplete batch is silently dropped.
- targets = inputs shifted left by one position (standard LM objective).
- Token IDs are validated against cfg.vocab_size on entry; out-of-range
  IDs raise ValueError immediately rather than corrupting a batch silently.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable

import torch

from src.config import TrainConfig


def make_batches(
    token_stream: Iterable[int],
    cfg: TrainConfig,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """Pack token IDs into (inputs, targets) batches for language model training.

    Parameters
    ----------
    token_stream : Iterable[int]
        Flat sequence of integer token IDs, typically produced by applying a
        tokenizer to the output of ``stream_documents``.
    cfg : TrainConfig
        Supplies ``batch_size``, ``seq_len``, and ``vocab_size``.

    Yields
    ------
    inputs  : LongTensor[batch_size, seq_len]
    targets : LongTensor[batch_size, seq_len]
        ``targets[b, t] == inputs[b, t+1]`` for all ``t < seq_len - 1``.
        The final target token in each sequence is the first token of the
        next sequence in the packed stream.

    Raises
    ------
    ValueError
        If any token ID is outside ``[0, cfg.vocab_size)``.
    """
    tokens_needed = cfg.batch_size * (cfg.seq_len + 1)
    buf: list[int] = []
    for tok in token_stream:
        if not (0 <= tok < cfg.vocab_size):
            raise ValueError(
                f"Token ID {tok} is out of range [0, {cfg.vocab_size}). "
                "Check tokenizer vocab_size matches TrainConfig.vocab_size."
            )
        buf.append(tok)
        if len(buf) >= tokens_needed:
            chunk = torch.tensor(buf[:tokens_needed], dtype=torch.long)
            chunk = chunk.view(cfg.batch_size, cfg.seq_len + 1)
            yield chunk[:, :-1], chunk[:, 1:]
            buf = buf[tokens_needed:]
