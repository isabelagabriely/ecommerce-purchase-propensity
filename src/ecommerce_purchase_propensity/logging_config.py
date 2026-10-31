"""Logging configuration utilities."""

from __future__ import annotations

import logging.config
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "logging.yaml"


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load logging configuration from a YAML file."""
    with config_path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def _apply_log_level(config: dict[str, Any], level: str) -> None:
    """Apply the configured log level to root logger and handlers."""
    config["root"]["level"] = level

    for handler in config.get("handlers", {}).values():
        handler["level"] = level


def setup_logging(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Configure application logging."""
    load_dotenv()

    config = _load_config(config_path)
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    _apply_log_level(config, log_level)
    logging.config.dictConfig(config)
