"""MLflow experiment tracking utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from ecommerce_purchase_propensity.config import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)

PARAMS_PATH = PROJECT_ROOT / "params.yaml"
DVC_LOCK_PATH = PROJECT_ROOT / "dvc.lock"


def configure_mlflow() -> None:
    """Configure the MLflow tracking server and experiment."""
    load_dotenv(PROJECT_ROOT / ".env")

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000",
    )
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "ecommerce-purchase-propensity",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    LOGGER.info("MLflow tracking URI: %s", tracking_uri)
    LOGGER.info("MLflow experiment: %s", experiment_name)


def _normalize_param(value: Any) -> Any:
    """Convert values unsupported as MLflow params."""
    if value is None:
        return "None"

    return value


def _build_run_params(
    params: dict[str, Any],
    model_name: str,
    *,
    random_state: int,
    cv_folds: int,
    selection_metric: str,
    train_rows: int,
    feature_count: int,
) -> dict[str, Any]:
    """Build parameters to log for a candidate run."""
    model_params = {
        f"model_{key}": _normalize_param(value)
        for key, value in params["models"][model_name].items()
    }

    run_params = {
        "model_name": model_name,
        "random_state": random_state,
        "cv_folds": cv_folds,
        "selection_metric": selection_metric,
        "train_rows": train_rows,
        "feature_count": feature_count,
    }

    return run_params | model_params


def _log_reproducibility_artifacts() -> None:
    """Log configuration files required to reproduce the experiment."""
    for path in (PARAMS_PATH, DVC_LOCK_PATH):
        if path.exists():
            mlflow.log_artifact(
                str(path),
                artifact_path="reproducibility",
            )


def log_candidate_run(
    model_name: str,
    model: Pipeline,
    features: pd.DataFrame,
    cv_result: dict[str, float | str],
    params: dict[str, Any],
    *,
    random_state: int,
    cv_folds: int,
    selection_metric: str,
) -> str:
    """Log a fitted candidate model and its validation results."""
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tags(
            {
                "model_name": model_name,
                "model_role": "candidate",
                "pipeline_stage": "model_comparison",
            }
        )

        run_params = _build_run_params(
            params,
            model_name,
            random_state=random_state,
            cv_folds=cv_folds,
            selection_metric=selection_metric,
            train_rows=len(features),
            feature_count=features.shape[1],
        )
        mlflow.log_params(run_params)

        cv_metrics = {
            key: float(value) for key, value in cv_result.items() if key != "model"
        }
        mlflow.log_metrics(cv_metrics)

        input_example = features.head(5)
        predictions = model.predict(input_example)
        signature = infer_signature(input_example, predictions)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        if model_info.model_id:
            mlflow.set_tag("logged_model_id", model_info.model_id)

        mlflow.set_tag("logged_model_uri", model_info.model_uri)

        _log_reproducibility_artifacts()

        LOGGER.info(
            "MLflow run created for %s: %s",
            model_name,
            run.info.run_id,
        )

        return run.info.run_id


def log_champion_evaluation(
    run_id: str,
    test_metrics: dict[str, float],
    comparison_path: Path,
    test_metrics_path: Path,
) -> None:
    """Add final holdout metrics and reports to the champion run."""
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("model_role", "champion")

        mlflow.log_metrics(
            {
                f"test_{metric_name}": float(value)
                for metric_name, value in test_metrics.items()
            }
        )

        mlflow.log_artifact(
            str(comparison_path),
            artifact_path="reports",
        )
        mlflow.log_artifact(
            str(test_metrics_path),
            artifact_path="reports",
        )

    LOGGER.info("Champion evaluation logged to MLflow run %s", run_id)
