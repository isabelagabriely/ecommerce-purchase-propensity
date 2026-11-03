"""Tests for dataset preprocessing."""

import pandas as pd

from ecommerce_purchase_propensity.data.preprocess import (
    prepare_data,
    split_data,
)
from ecommerce_purchase_propensity.data.schema import (
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
)


def test_prepare_data_normalizes_target(
    sample_dataset: pd.DataFrame,
) -> None:
    """Preprocessing should normalize the binary target."""
    prepared = prepare_data(sample_dataset)

    assert list(prepared.columns) == EXPECTED_COLUMNS
    assert set(prepared[TARGET_COLUMN].unique()) == {0, 1}
    assert str(prepared[TARGET_COLUMN].dtype) == "int8"


def test_split_data_is_reproducible_and_stratified(
    sample_dataset: pd.DataFrame,
) -> None:
    """Train/test split should be reproducible and stratified."""
    prepared = prepare_data(sample_dataset)

    train_first, test_first = split_data(
        prepared,
        test_size=0.20,
        random_state=42,
    )

    train_second, test_second = split_data(
        prepared,
        test_size=0.20,
        random_state=42,
    )

    pd.testing.assert_frame_equal(train_first, train_second)
    pd.testing.assert_frame_equal(test_first, test_second)

    assert len(train_first) == 16
    assert len(test_first) == 4

    assert train_first[TARGET_COLUMN].mean() == 0.5
    assert test_first[TARGET_COLUMN].mean() == 0.5
