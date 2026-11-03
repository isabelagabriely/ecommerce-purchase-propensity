"""Register the champion model in the MLflow Model Registry."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion

from ecommerce_purchase_propensity.config import PROJECT_ROOT
from ecommerce_purchase_propensity.logging_config import setup_logging
from ecommerce_purchase_propensity.modeling.tracking import configure_mlflow

LOGGER = logging.getLogger(__name__)

TEST_METRICS_PATH = PROJECT_ROOT / "reports" / "test_metrics.json"


def load_test_report(
    path: Path = TEST_METRICS_PATH,
) -> dict[str, Any]:
    """Load the final champion evaluation report."""
    if not path.exists():
        raise FileNotFoundError(
            f"Test metrics report not found at {path}. Run the training stage first."
        )

    with path.open(encoding="utf-8") as file:
        report = json.load(file)

    if "champion_model" not in report:
        raise ValueError("Test metrics report has no champion model.")

    if "test" not in report:
        raise ValueError("Test metrics report has no test metrics.")

    return report


def find_champion_run(
    client: MlflowClient,
    *,
    experiment_name: str,
    champion_model: str,
) -> Run:
    """Find the most recent MLflow run marked as champion."""
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.model_role = 'champion' AND params.model_name = '{champion_model}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No champion run found for model: {champion_model}")

    return runs[0]


def find_existing_version(
    client: MlflowClient,
    *,
    registered_model_name: str,
    run_id: str,
) -> ModelVersion | None:
    """Find a registry version already created from the same run."""
    versions = client.search_model_versions(
        filter_string=f"name = '{registered_model_name}'"
    )

    for version in versions:
        source_run_id = version.tags.get("source_run_id")

        if version.run_id == run_id or source_run_id == run_id:
            return version

    return None


def register_model_version(
    client: MlflowClient,
    *,
    registered_model_name: str,
    champion_run: Run,
    champion_model: str,
    selection_metric: str,
) -> ModelVersion:
    """Register the champion logged model if not already registered."""
    existing_version = find_existing_version(
        client,
        registered_model_name=registered_model_name,
        run_id=champion_run.info.run_id,
    )

    if existing_version is not None:
        LOGGER.info(
            "Model version already registered: %s version %s",
            registered_model_name,
            existing_version.version,
        )
        return existing_version

    model_uri = champion_run.data.tags.get("logged_model_uri")

    if not model_uri:
        raise ValueError("Champion run does not contain a logged_model_uri tag.")

    LOGGER.info(
        "Registering champion model from %s",
        model_uri,
    )

    return mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
        tags={
            "source_model": champion_model,
            "source_run_id": champion_run.info.run_id,
            "selection_metric": selection_metric,
        },
    )


def update_registry_metadata(
    client: MlflowClient,
    *,
    registered_model_name: str,
    model_version: ModelVersion,
    report: dict[str, Any],
) -> None:
    """Add descriptions and evaluation metadata to the registry."""
    client.update_registered_model(
        name=registered_model_name,
        description=(
            "Purchase propensity classifier trained with Scikit-Learn. "
            "Candidate models are compared using stratified "
            "cross-validation and Average Precision."
        ),
    )

    champion_model = str(report["champion_model"])

    client.update_model_version(
        name=registered_model_name,
        version=model_version.version,
        description=(
            f"Champion model: {champion_model}. "
            "Selected using cross-validation and evaluated on the "
            "held-out test dataset."
        ),
    )

    for metric_name, metric_value in report["test"].items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_version.version,
            key=f"test_{metric_name}",
            value=str(metric_value),
        )

    client.set_model_version_tag(
        name=registered_model_name,
        version=model_version.version,
        key="champion_model",
        value=champion_model,
    )

    client.set_model_version_tag(
        name=registered_model_name,
        version=model_version.version,
        key="selection_metric",
        value=str(report["selection_metric"]),
    )


def assign_champion_alias(
    client: MlflowClient,
    *,
    registered_model_name: str,
    model_version: ModelVersion,
    alias: str,
) -> None:
    """Assign the champion alias to a registered model version."""
    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=model_version.version,
    )

    LOGGER.info(
        "Alias '%s' now points to %s version %s",
        alias,
        registered_model_name,
        model_version.version,
    )


def run_registration() -> None:
    """Register and promote the current champion model."""
    configure_mlflow()

    report = load_test_report()

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "ecommerce-purchase-propensity",
    )
    registered_model_name = os.getenv(
        "MLFLOW_REGISTERED_MODEL_NAME",
        "purchase-propensity-classifier",
    )
    alias = os.getenv(
        "MLFLOW_MODEL_ALIAS",
        "champion",
    )

    champion_model = str(report["champion_model"])
    selection_metric = str(report["selection_metric"])

    client = MlflowClient()

    champion_run = find_champion_run(
        client,
        experiment_name=experiment_name,
        champion_model=champion_model,
    )

    LOGGER.info(
        "Champion MLflow run found: %s",
        champion_run.info.run_id,
    )

    model_version = register_model_version(
        client,
        registered_model_name=registered_model_name,
        champion_run=champion_run,
        champion_model=champion_model,
        selection_metric=selection_metric,
    )

    update_registry_metadata(
        client,
        registered_model_name=registered_model_name,
        model_version=model_version,
        report=report,
    )

    assign_champion_alias(
        client,
        registered_model_name=registered_model_name,
        model_version=model_version,
        alias=alias,
    )

    resolved_version = client.get_model_version_by_alias(
        registered_model_name,
        alias,
    )

    LOGGER.info(
        "Registry validation successful: %s@%s -> version %s",
        registered_model_name,
        alias,
        resolved_version.version,
    )


def main() -> None:
    """Run champion model registration."""
    setup_logging()
    run_registration()


if __name__ == "__main__":
    main()
