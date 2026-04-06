"""Shared pytest fixtures for the llm-training test suite."""
from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def reset_llm_training_logger() -> None:
    """Restore the 'llm_training' logger to its pre-test state after each test.

    configure_logging() adds handlers and sets propagate=False on a module-level
    logger that persists for the lifetime of the process.  Without this fixture,
    handlers accumulate across tests and caplog-based tests in test_logger.py
    break if they run after any test that calls train() (which calls
    configure_logging()).
    """
    log = logging.getLogger("llm_training")
    original_handlers = log.handlers[:]
    original_propagate = log.propagate
    original_level = log.level
    yield
    # Close any handlers the test added (prevents resource leaks).
    for h in log.handlers:
        if h not in original_handlers:
            h.close()
    log.handlers = original_handlers
    log.propagate = original_propagate
    log.setLevel(original_level)
