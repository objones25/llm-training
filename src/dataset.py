"""HuggingFace dataset streaming for fineweb-edu.

Provides a single generator function that yields raw text documents one at a
time from the configured dataset without materialising the full corpus in RAM.

Public API
----------
    stream_documents(cfg: TrainConfig) -> Generator[str, None, None]
"""

from __future__ import annotations

import os
from collections.abc import Generator

from datasets import load_dataset

from src.config import TrainConfig


def stream_documents(cfg: TrainConfig) -> Generator[str, None, None]:
    """Yield raw text strings from the configured HuggingFace dataset.

    Uses HuggingFace ``datasets`` streaming mode so the corpus is never
    loaded into RAM.  Authentication is handled via ``HF_TOKEN`` from the
    environment; if the variable is absent the request is made unauthenticated
    (works for public datasets).

    Parameters
    ----------
    cfg : TrainConfig
        Supplies ``dataset_name``, ``dataset_config``, and ``dataset_split``.

    Yields
    ------
    str
        One raw text document per iteration.
    """
    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(
        cfg.dataset_name,
        cfg.dataset_config,
        split=cfg.dataset_split,
        streaming=True,
        token=token,
    )
    for row in ds:
        yield row["text"]
