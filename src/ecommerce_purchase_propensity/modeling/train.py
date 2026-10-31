"""Train and compare purchase propensity classifiers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ecommerce_purchase_propensity.config import PROJECT_ROOT, load_params
from ecommerce_purchase_propensity.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from ecommerce_purchase_propensity.logging_config import setup_logging
from ecommerce_purchase_propensity.modeling.evaluate import (
    build_cross_validator,
    cross_validate_model,
    evaluate_holdout,
    select_champion,
)
from ecommerce_purchase_propensity.modeling.models import build_models

LOGGER = logging.getLogger(__name__)

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
TEST_METRICS_PATH = REPORTS_DIR / "test_metrics.json"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a processed dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}.")

    return pd.read_csv(path)


def split_features_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataset into features and target."""
    features = dataframe[FEATURE_COLUMNS].copy()
    target = dataframe[TARGET_COLUMN].copy()

    return features, target


def save_comparison(results: list[dict[str, float | str]]) -> None:
    """Save cross-validation results."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame(results)
    comparison.to_csv(COMPARISON_PATH, index=False)

    LOGGER.info("Model comparison saved to %s", COMPARISON_PATH)


def save_test_metrics(
    champion_name: str,
    selection_metric: str,
    metrics: dict[str, float],
) -> None:
    """Save final held-out test metrics."""
    payload = {
        "champion_model": champion_name,
        "selection_metric": selection_metric,
        "test": metrics,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with TEST_METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    LOGGER.info("Test metrics saved to %s", TEST_METRICS_PATH)


def compare_models(
    models: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
    *,
    cv_folds: int,
    random_state: int,
) -> list[dict[str, float | str]]:
    """Compare candidate models using stratified cross-validation."""
    cross_validator = build_cross_validator(
        n_splits=cv_folds,
        random_state=random_state,
    )

    results = []

    for model_name, model in models.items():
        LOGGER.info("Cross-validating model: %s", model_name)

        result = cross_validate_model(
            model_name,
            model,
            features,
            target,
            cross_validator,
        )

        results.append(result)

    return results


def run_training(params: dict[str, Any]) -> None:
    """Train, compare, select, and evaluate candidate models."""
    data_params = params["data"]
    training_params = params["training"]

    random_state = int(data_params["random_state"])
    cv_folds = int(training_params["cv_folds"])
    selection_metric = str(training_params["selection_metric"])

    train_data = load_dataset(TRAIN_DATA_PATH)
    test_data = load_dataset(TEST_DATA_PATH)

    train_features, train_target = split_features_target(train_data)
    test_features, test_target = split_features_target(test_data)

    models = build_models(
        params,
        random_state=random_state,
    )

    results = compare_models(
        models,
        train_features,
        train_target,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    save_comparison(results)

    champion_name = select_champion(
        results,
        selection_metric=selection_metric,
    )

    LOGGER.info("Selected champion model: %s", champion_name)

    champion_model = models[champion_name]
    champion_model.fit(train_features, train_target)

    test_metrics = evaluate_holdout(
        champion_model,
        test_features,
        test_target,
    )

    save_test_metrics(
        champion_name,
        selection_metric,
        test_metrics,
    )

    LOGGER.info(
        "Champion test average precision: %.4f",
        test_metrics["average_precision"],
    )


def main() -> None:
    """Run model training and evaluation."""
    setup_logging()
    params = load_params()
    run_training(params)


if __name__ == "__main__":
    main()
