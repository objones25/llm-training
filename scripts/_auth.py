"""Load HuggingFace authentication token from .env."""
from __future__ import annotations

import os
from pathlib import Path


def load_hf_token() -> str | None:
    """Return HF_TOKEN from .env or environment, whichever is set.

    Searches for .env starting from the project root (parent of scripts/).
    Returns None if the variable is not set — callers should decide whether
    to fail hard or fall back to unauthenticated access.
    """
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    return os.getenv("HF_TOKEN") or None
