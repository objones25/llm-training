"""BPE tokenizer wrapping the HuggingFace tokenizers library.

Pipeline: ByteLevel pre-tokenizer → BPE model → ByteLevel decoder.
No normalizer: ByteLevel handles the full byte range without lossy normalization.
No post-processor: BOS/EOS injection is the dataloader's responsibility.
"""

from __future__ import annotations

from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    """ByteLevel BPE tokenizer.

    Usage::

        tok = BPETokenizer()
        tok.train(iter(corpus), vocab_size=8192)
        ids = tok.encode("hello world")
        text = tok.decode(ids)
        tok.save("tokenizer.json")

        tok2 = BPETokenizer.load("tokenizer.json")
    """

    SPECIAL_TOKENS: list[str] = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

    def __init__(self) -> None:
        self._tokenizer: Tokenizer | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, iterator: Iterable[str], vocab_size: int) -> None:
        """Train BPE from a lazy string iterator.

        Never materialises the full corpus in RAM: the tokenizers library
        consumes the iterator one document at a time.

        add_prefix_space=False ensures decode(encode(text)) == text for all
        inputs, including those that start without a space.
        """
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=self.SPECIAL_TOKENS,
            show_progress=False,
        )
        tokenizer.train_from_iterator(iterator, trainer=trainer)
        self._tokenizer = tokenizer

    # ── Encoding / decoding ───────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of integer token IDs."""
        self._require_trained()
        assert self._tokenizer is not None
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to a string."""
        self._require_trained()
        assert self._tokenizer is not None
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode a batch of strings. Equivalent to ``[encode(t) for t in texts]``."""
        self._require_trained()
        assert self._tokenizer is not None
        return [enc.ids for enc in self._tokenizer.encode_batch(texts)]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise the tokenizer to a JSON file at *path*."""
        self._require_trained()
        assert self._tokenizer is not None
        self._tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> BPETokenizer:
        """Deserialise a tokenizer from a JSON file at *path*."""
        instance = cls()
        instance._tokenizer = Tokenizer.from_file(path)
        return instance

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary (including special tokens)."""
        self._require_trained()
        assert self._tokenizer is not None
        return self._tokenizer.get_vocab_size()

    def token_to_id(self, token: str) -> int:
        """Return the integer ID for *token*. Raises ``KeyError`` if absent."""
        self._require_trained()
        assert self._tokenizer is not None
        id_ = self._tokenizer.token_to_id(token)
        if id_ is None:
            raise KeyError(f"Token {token!r} not in vocabulary.")
        return id_

    # ── Internal ─────────────────────────────────────────────────────────────

    def _require_trained(self) -> None:
        if self._tokenizer is None:
            raise RuntimeError(
                "Tokenizer has not been trained. Call .train() or .load() first."
            )
