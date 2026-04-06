"""Tests for src/dataset.py.

All tests mock datasets.load_dataset — no network access, no HuggingFace calls.
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from src.config import TrainConfig
from src.dataset import stream_documents


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> TrainConfig:
    return TrainConfig(
        dataset_name="HuggingFaceFW/fineweb-edu",
        dataset_config="sample-10BT",
        dataset_split="train",
    )


def _make_mock_dataset(docs: list[str]) -> MagicMock:
    """Return a mock IterableDataset yielding dicts with a 'text' field."""
    mock_ds = MagicMock()
    mock_ds.__iter__ = MagicMock(return_value=iter({"text": d} for d in docs))
    return mock_ds


# ── Type / generator contract ─────────────────────────────────────────────────


def test_returns_generator(cfg: TrainConfig) -> None:
    """stream_documents must return a generator, not a list or other iterable."""
    mock_ds = _make_mock_dataset(["hello world"])
    with patch("src.dataset.load_dataset", return_value=mock_ds):
        result = stream_documents(cfg)
    assert isinstance(result, types.GeneratorType)


def test_yields_strings(cfg: TrainConfig) -> None:
    """Each item yielded must be a plain str."""
    docs = ["first document", "second document", "third document"]
    mock_ds = _make_mock_dataset(docs)
    with patch("src.dataset.load_dataset", return_value=mock_ds):
        yielded = list(stream_documents(cfg))
    assert all(isinstance(item, str) for item in yielded)


def test_yields_correct_text(cfg: TrainConfig) -> None:
    """Yielded strings must match the 'text' field of each row exactly."""
    docs = ["alpha", "beta", "gamma"]
    mock_ds = _make_mock_dataset(docs)
    with patch("src.dataset.load_dataset", return_value=mock_ds):
        yielded = list(stream_documents(cfg))
    assert yielded == docs


# ── load_dataset call contract ────────────────────────────────────────────────


def test_uses_correct_dataset_name(cfg: TrainConfig) -> None:
    """load_dataset must be called with cfg.dataset_name as the first argument."""
    mock_ds = _make_mock_dataset([])
    with patch("src.dataset.load_dataset", return_value=mock_ds) as mock_load:
        list(stream_documents(cfg))
    args, _ = mock_load.call_args
    assert args[0] == cfg.dataset_name


def test_uses_correct_dataset_config(cfg: TrainConfig) -> None:
    """load_dataset must be called with cfg.dataset_config as the second argument."""
    mock_ds = _make_mock_dataset([])
    with patch("src.dataset.load_dataset", return_value=mock_ds) as mock_load:
        list(stream_documents(cfg))
    args, _ = mock_load.call_args
    assert args[1] == cfg.dataset_config


def test_uses_correct_split(cfg: TrainConfig) -> None:
    """load_dataset must be called with split=cfg.dataset_split."""
    mock_ds = _make_mock_dataset([])
    with patch("src.dataset.load_dataset", return_value=mock_ds) as mock_load:
        list(stream_documents(cfg))
    _, kwargs = mock_load.call_args
    assert kwargs["split"] == cfg.dataset_split


def test_uses_streaming_mode(cfg: TrainConfig) -> None:
    """load_dataset must be called with streaming=True."""
    mock_ds = _make_mock_dataset([])
    with patch("src.dataset.load_dataset", return_value=mock_ds) as mock_load:
        list(stream_documents(cfg))
    _, kwargs = mock_load.call_args
    assert kwargs.get("streaming") is True


# ── Authentication ────────────────────────────────────────────────────────────


def test_passes_hf_token_from_env(cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """HF_TOKEN from the environment must be forwarded to load_dataset as token=."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_abc123")
    mock_ds = _make_mock_dataset([])
    with patch("src.dataset.load_dataset", return_value=mock_ds) as mock_load:
        list(stream_documents(cfg))
    _, kwargs = mock_load.call_args
    assert kwargs.get("token") == "hf_test_token_abc123"


def test_passes_none_token_when_env_absent(
    cfg: TrainConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When HF_TOKEN is not set, token passed to load_dataset must be None."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    mock_ds = _make_mock_dataset([])
    with patch("src.dataset.load_dataset", return_value=mock_ds) as mock_load:
        list(stream_documents(cfg))
    _, kwargs = mock_load.call_args
    assert kwargs.get("token") is None


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_empty_dataset_yields_nothing(cfg: TrainConfig) -> None:
    """An empty dataset must produce an empty generator without error."""
    mock_ds = _make_mock_dataset([])
    with patch("src.dataset.load_dataset", return_value=mock_ds):
        yielded = list(stream_documents(cfg))
    assert yielded == []


def test_lazy_evaluation(cfg: TrainConfig) -> None:
    """stream_documents must not consume the dataset until iterated.

    Calling stream_documents() alone must not call load_dataset — the
    generator body only runs when the caller begins iteration.
    """
    mock_ds = _make_mock_dataset(["doc"])
    with patch("src.dataset.load_dataset", return_value=mock_ds) as mock_load:
        gen = stream_documents(cfg)
        # load_dataset should NOT have been called yet
        mock_load.assert_not_called()
        # Only after we pull the first item does the body execute
        next(gen)
        mock_load.assert_called_once()
