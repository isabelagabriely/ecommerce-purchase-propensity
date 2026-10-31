"""Prepare and split the Online Shoppers dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ecommerce_purchase_propensity.config import PROJECT_ROOT, load_params
from ecommerce_purchase_propensity.data.schema import (
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
)
from ecommerce_purchase_propensity.logging_config import setup_logging

LOGGER = logging.getLogger(__name__)

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "online_shoppers_intention.csv"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset."""
    LOGGER.info("Loading raw dataset from %s", path)

    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run the download stage first."
        )

    return pd.read_csv(path)


def validate_schema(dataframe: pd.DataFrame) -> None:
    """Validate expected columns and target availability."""
    missing_columns = set(EXPECTED_COLUMNS) - set(dataframe.columns)
    unexpected_columns = set(dataframe.columns) - set(EXPECTED_COLUMNS)

    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    if unexpected_columns:
        raise ValueError(
            f"Dataset contains unexpected columns: {sorted(unexpected_columns)}"
        )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")


def normalize_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert the binary target to integer values."""
    dataframe = dataframe.copy()

    target_values = set(dataframe[TARGET_COLUMN].dropna().unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(f"Unexpected target values: {sorted(target_values)}")

    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int).astype("int8")

    return dataframe


def prepare_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare the raw dataframe."""
    validate_schema(dataframe)

    dataframe = dataframe.loc[:, EXPECTED_COLUMNS].copy()
    dataframe = normalize_target(dataframe)

    LOGGER.info(
        "Dataset prepared with %s rows and %s columns",
        len(dataframe),
        len(dataframe.columns),
    )

    return dataframe


def split_data(
    dataframe: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible stratified train and test datasets."""
    train_data, test_data = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        stratify=dataframe[TARGET_COLUMN],
    )

    LOGGER.info(
        "Dataset split into %s training rows and %s test rows",
        len(train_data),
        len(test_data),
    )

    return train_data, test_data


def save_dataset(dataframe: pd.DataFrame, path: Path) -> None:
    """Save a dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)

    LOGGER.info("Dataset saved to %s", path)


def log_target_distribution(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Log the positive target rate for a dataset."""
    positive_rate = dataframe[TARGET_COLUMN].mean()

    LOGGER.info(
        "%s positive target rate: %.4f",
        dataset_name,
        positive_rate,
    )


def run_preprocessing(params: dict[str, Any]) -> None:
    """Run the preprocessing pipeline."""
    data_params = params["data"]

    dataframe = load_raw_data()
    dataframe = prepare_data(dataframe)

    train_data, test_data = split_data(
        dataframe,
        test_size=float(data_params["test_size"]),
        random_state=int(data_params["random_state"]),
    )

    log_target_distribution(train_data, "Training")
    log_target_distribution(test_data, "Test")

    save_dataset(train_data, TRAIN_DATA_PATH)
    save_dataset(test_data, TEST_DATA_PATH)


def main() -> None:
    """Run dataset preprocessing."""
    setup_logging()

    params = load_params()
    run_preprocessing(params)


if __name__ == "__main__":
    main()
