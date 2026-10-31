"""Tests for logging configuration."""

import logging

from ecommerce_purchase_propensity.logging_config import setup_logging


def test_setup_logging_uses_environment_level(monkeypatch) -> None:
    """Logging should respect the LOG_LEVEL environment variable."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    setup_logging()

    assert logging.getLogger().level == logging.DEBUG
