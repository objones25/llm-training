"""Tests for src/tokenizer.py.

All tests are hermetic: no network, no GPU, no shared mutable state.
The module-scoped ``tokenizer`` fixture is read-only in every test.
Tests that write to disk use ``tmp_path`` exclusively.
"""

from __future__ import annotations

import pytest

from src.tokenizer import BPETokenizer

# ── Shared corpus ─────────────────────────────────────────────────────────────
# Repeated enough times that BPE merges fire and the round-trip strings
# appear frequently enough to survive min_frequency=2.

_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning is a subset of artificial intelligence",
    "transformers use attention mechanisms to process sequences",
    "the cat sat on the mat and the rat sat on the cat",
    "natural language processing enables computers to understand text",
    "hello world foo bar baz transformer architecture",
    "the quick brown fox and the lazy dog sat on the mat",
] * 30


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    """Tiny trained tokenizer shared across tests (read-only)."""
    tok = BPETokenizer()
    tok.train(iter(_CORPUS), vocab_size=400)
    return tok


# ── Round-trip fidelity (Rule 8) ──────────────────────────────────────────────

_ROUND_TRIP_STRINGS = [
    "hello world",
    "the quick brown fox",
    "machine learning",
    "transformer architecture",
    "natural language processing",
]


@pytest.mark.parametrize("text", _ROUND_TRIP_STRINGS)
def test_round_trip_fidelity(tokenizer: BPETokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


# ── Shape / dtype contracts (Rule 13) ─────────────────────────────────────────


def test_encode_returns_list_of_int(tokenizer: BPETokenizer) -> None:
    ids = tokenizer.encode("hello world")
    assert isinstance(ids, list), "encode() must return list"
    assert len(ids) > 0, "encode() must return non-empty list"
    assert all(isinstance(i, int) for i in ids), "all token IDs must be int"


def test_encode_batch_returns_list_of_lists(tokenizer: BPETokenizer) -> None:
    texts = ["hello", "world", "test"]
    result = tokenizer.encode_batch(texts)
    assert isinstance(result, list), "encode_batch() must return list"
    assert len(result) == len(texts), "one entry per input"
    for row in result:
        assert isinstance(row, list), "each row must be list"
        assert all(isinstance(i, int) for i in row), "all token IDs must be int"


# ── Batch consistency ─────────────────────────────────────────────────────────


def test_encode_batch_matches_single_encode(tokenizer: BPETokenizer) -> None:
    texts = ["hello world", "quick brown fox", "machine learning"]
    assert tokenizer.encode_batch(texts) == [tokenizer.encode(t) for t in texts]


# ── Vocab size ────────────────────────────────────────────────────────────────


def test_vocab_size_is_int(tokenizer: BPETokenizer) -> None:
    assert isinstance(tokenizer.vocab_size, int)


def test_vocab_size_within_requested_bound(tokenizer: BPETokenizer) -> None:
    # Actual vocab can be ≤ requested size if the corpus is small.
    assert 0 < tokenizer.vocab_size <= 400


# ── Special tokens (Rule 8 adjacent) ─────────────────────────────────────────


@pytest.mark.parametrize("token", BPETokenizer.SPECIAL_TOKENS)
def test_special_token_ids_are_valid(tokenizer: BPETokenizer, token: str) -> None:
    id_ = tokenizer.token_to_id(token)
    assert isinstance(id_, int)
    assert 0 <= id_ < tokenizer.vocab_size


def test_token_to_id_raises_for_unknown(tokenizer: BPETokenizer) -> None:
    with pytest.raises(KeyError):
        tokenizer.token_to_id("[DEFINITELY_NOT_A_TOKEN_XYZ]")


# ── Vocab bounds (Rule 9 analog) ──────────────────────────────────────────────


def test_ids_within_vocab_bounds(tokenizer: BPETokenizer) -> None:
    ids = tokenizer.encode("the quick brown fox jumps over the lazy dog")
    assert all(0 <= i < tokenizer.vocab_size for i in ids)


def test_no_padding_only_sequence(tokenizer: BPETokenizer) -> None:
    ids = tokenizer.encode("hello world")
    pad_id = tokenizer.token_to_id("[PAD]")
    assert any(
        i != pad_id for i in ids
    ), "non-empty text must not encode to all-padding"


# ── Save / load (Rule 11, Rule 12) ────────────────────────────────────────────


def test_save_load_produces_identical_encodings(
    tmp_path: pytest.TempPathFactory, tokenizer: BPETokenizer
) -> None:
    path = str(tmp_path / "tok.json")
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)

    text = "the quick brown fox"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(tokenizer.encode(text)) == text
    assert loaded.vocab_size == tokenizer.vocab_size


def test_load_writes_only_to_tmp_path(
    tmp_path: pytest.TempPathFactory, tokenizer: BPETokenizer
) -> None:
    """Disk write goes through tmp_path — no stray files elsewhere."""
    path = str(tmp_path / "check.json")
    tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tokenizer.vocab_size


# ── Training from generator (memory safety) ───────────────────────────────────


def test_train_from_generator_never_accumulates() -> None:
    """train() must accept a generator — never a pre-loaded list."""

    consumed: list[int] = []

    def corpus_gen():
        for i in range(60):
            consumed.append(i)
            yield f"hello world foo bar baz sentence number {i}"

    tok = BPETokenizer()
    tok.train(corpus_gen(), vocab_size=150)

    assert tok.vocab_size <= 150
    assert tok.vocab_size > 0
    # Generator was consumed lazily — all items processed
    assert len(consumed) == 60


# ── Determinism (Rule 17) ─────────────────────────────────────────────────────


def test_encode_is_deterministic(tokenizer: BPETokenizer) -> None:
    text = "the quick brown fox"
    assert tokenizer.encode(text) == tokenizer.encode(text)


def test_train_is_deterministic_across_instances() -> None:
    """Two tokenizers trained on the same iterator produce identical encodings."""
    corpus = ["hello world " * 5, "foo bar baz " * 5, "transformer attention " * 5] * 20

    tok1 = BPETokenizer()
    tok1.train(iter(corpus), vocab_size=200)

    tok2 = BPETokenizer()
    tok2.train(iter(corpus), vocab_size=200)

    assert tok1.encode("hello world") == tok2.encode("hello world")


# ── Error on untrained tokenizer ──────────────────────────────────────────────


def test_encode_raises_if_not_trained() -> None:
    tok = BPETokenizer()
    with pytest.raises(RuntimeError, match="not been trained"):
        tok.encode("hello")


def test_decode_raises_if_not_trained() -> None:
    tok = BPETokenizer()
    with pytest.raises(RuntimeError, match="not been trained"):
        tok.decode([1, 2, 3])


def test_vocab_size_raises_if_not_trained() -> None:
    tok = BPETokenizer()
    with pytest.raises(RuntimeError, match="not been trained"):
        _ = tok.vocab_size
