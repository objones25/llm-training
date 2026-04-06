"""Tests for src/dataloader.py.

All tests use synthetic token streams — no network, no tokenizer, no GPU.
"""
from __future__ import annotations

import types

import pytest
import torch

from src.config import TrainConfig
from src.dataloader import make_batches


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> TrainConfig:
    # Small values so tests are fast; vocab_size large enough for test tokens.
    return TrainConfig(
        vocab_size=256,
        batch_size=2,
        seq_len=8,
        n_layers=2,
        d_model=64,
        n_heads=2,
        d_ff=128,
    )


def _token_stream(cfg: TrainConfig, n_batches: int) -> list[int]:
    """Return exactly enough tokens to produce *n_batches* complete batches."""
    count = cfg.batch_size * (cfg.seq_len + 1) * n_batches
    return [i % cfg.vocab_size for i in range(count)]


# ── Generator contract ────────────────────────────────────────────────────────


def test_is_generator(cfg: TrainConfig) -> None:
    """make_batches must return a generator object."""
    stream = _token_stream(cfg, 1)
    result = make_batches(iter(stream), cfg)
    assert isinstance(result, types.GeneratorType)


# ── Shape / dtype contracts ───────────────────────────────────────────────────


def test_output_shapes(cfg: TrainConfig) -> None:
    """inputs and targets must both be shape (batch_size, seq_len)."""
    stream = _token_stream(cfg, 3)
    for inputs, targets in make_batches(iter(stream), cfg):
        assert inputs.shape == (cfg.batch_size, cfg.seq_len)
        assert targets.shape == (cfg.batch_size, cfg.seq_len)


def test_output_dtype(cfg: TrainConfig) -> None:
    """Both tensors must be torch.long (int64)."""
    stream = _token_stream(cfg, 1)
    inputs, targets = next(make_batches(iter(stream), cfg))
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


# ── Value contracts ───────────────────────────────────────────────────────────


def test_targets_are_inputs_shifted(cfg: TrainConfig) -> None:
    """targets[b, t] must equal the token that follows inputs[b, t] in the stream.

    Each row is an independent (seq_len+1)-token chunk split into
    inputs[:seq_len] and targets[1:seq_len+1], so the overlap is:
        targets[b, :-1] == inputs[b, 1:]
    """
    n_tokens = cfg.batch_size * (cfg.seq_len + 1)
    stream = [i % cfg.vocab_size for i in range(n_tokens)]
    inputs, targets = next(make_batches(iter(stream), cfg))

    for b in range(cfg.batch_size):
        # All interior positions: target[t] == input[t+1]
        assert torch.equal(targets[b, :-1], inputs[b, 1:]), (
            "targets[b, t] must equal inputs[b, t+1] for t < seq_len-1"
        )


def test_targets_match_next_token_across_rows(cfg: TrainConfig) -> None:
    """The last target in row b is one position before the first input of row b+1.

    With packed rows of width (seq_len+1), row b covers flat tokens
    [b*(seq_len+1) .. (b+1)*(seq_len+1) - 1].  targets[b, -1] is token
    (b+1)*(seq_len+1)-1, and inputs[b+1, 0] is token (b+1)*(seq_len+1).
    For a monotone stream they differ by exactly 1.
    """
    n_tokens = cfg.batch_size * (cfg.seq_len + 1)
    stream = [i % cfg.vocab_size for i in range(n_tokens)]
    inputs, targets = next(make_batches(iter(stream), cfg))

    for b in range(cfg.batch_size - 1):
        # targets[b, -1] and inputs[b+1, 0] are adjacent in the flat stream.
        last_target = targets[b, -1].item()
        next_row_first = inputs[b + 1, 0].item()
        assert (last_target + 1) % cfg.vocab_size == next_row_first % cfg.vocab_size, (
            "last target in row b must be immediately before first input of row b+1"
        )


def test_token_ids_within_vocab_bounds(cfg: TrainConfig) -> None:
    """All yielded token IDs must be in [0, cfg.vocab_size)."""
    stream = _token_stream(cfg, 3)
    for inputs, targets in make_batches(iter(stream), cfg):
        assert inputs.min().item() >= 0
        assert inputs.max().item() < cfg.vocab_size
        assert targets.min().item() >= 0
        assert targets.max().item() < cfg.vocab_size


def test_no_padding_only_batches(cfg: TrainConfig) -> None:
    """No batch may consist entirely of a single repeated padding value.

    In a packed (no-padding) dataloader this should never happen; the test
    acts as a regression guard.
    """
    stream = _token_stream(cfg, 5)
    for inputs, _ in make_batches(iter(stream), cfg):
        # A padding-only batch would have all identical values.
        unique_values = inputs.unique()
        assert len(unique_values) > 1, "Batch contains only a single token value"


# ── Error / edge cases ────────────────────────────────────────────────────────


def test_raises_on_out_of_bounds_token(cfg: TrainConfig) -> None:
    """ValueError must be raised for any token ID >= vocab_size."""
    bad_stream = [0, 1, cfg.vocab_size]  # last token is out of range
    with pytest.raises(ValueError, match=str(cfg.vocab_size)):
        list(make_batches(iter(bad_stream), cfg))


def test_raises_on_negative_token(cfg: TrainConfig) -> None:
    """ValueError must be raised for any negative token ID."""
    bad_stream = [0, 1, -1]
    with pytest.raises(ValueError):
        list(make_batches(iter(bad_stream), cfg))


def test_incomplete_final_batch_dropped(cfg: TrainConfig) -> None:
    """Trailing tokens that cannot fill a complete batch must be silently dropped."""
    full = cfg.batch_size * (cfg.seq_len + 1)  # tokens for exactly 1 batch
    # Add a few extra tokens that do not form a second full batch
    stream = [i % cfg.vocab_size for i in range(full + 3)]
    batches = list(make_batches(iter(stream), cfg))
    assert len(batches) == 1


def test_exact_multiple_yields_correct_count(cfg: TrainConfig) -> None:
    """A stream with exactly N * tokens_per_batch tokens yields N batches."""
    n_batches = 4
    stream = _token_stream(cfg, n_batches)
    batches = list(make_batches(iter(stream), cfg))
    assert len(batches) == n_batches


def test_empty_stream_yields_nothing(cfg: TrainConfig) -> None:
    """An empty token stream must produce zero batches without error."""
    batches = list(make_batches(iter([]), cfg))
    assert batches == []
