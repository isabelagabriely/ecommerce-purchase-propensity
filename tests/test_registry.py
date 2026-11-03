"""Tests for MLflow registry utilities."""

import json
from pathlib import Path

import pytest

from ecommerce_purchase_propensity.modeling.registry import load_test_report


def test_load_test_report_reads_champion_metadata(
    tmp_path: Path,
) -> None:
    """Registry should load champion information from the test report."""
    report_path = tmp_path / "test_metrics.json"

    report = {
        "champion_model": "random_forest",
        "selection_metric": "average_precision",
        "test": {
            "average_precision": 0.72,
            "roc_auc": 0.91,
        },
    }

    report_path.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    loaded_report = load_test_report(report_path)

    assert loaded_report["champion_model"] == "random_forest"
    assert loaded_report["selection_metric"] == "average_precision"
    assert loaded_report["test"]["average_precision"] == 0.72


def test_load_test_report_requires_existing_file(
    tmp_path: Path,
) -> None:
    """Registry should reject a missing evaluation report."""
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_test_report(missing_path)
