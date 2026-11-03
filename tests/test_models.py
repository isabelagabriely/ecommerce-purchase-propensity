"""Tests for classification pipelines."""

import pandas as pd

from ecommerce_purchase_propensity.data.preprocess import prepare_data
from ecommerce_purchase_propensity.data.schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from ecommerce_purchase_propensity.modeling.models import build_models

TEST_PARAMS = {
    "models": {
        "logistic_regression": {
            "max_iter": 100,
            "solver": "lbfgs",
        },
        "random_forest": {
            "n_estimators": 5,
            "max_depth": 3,
            "min_samples_split": 2,
            "n_jobs": 1,
        },
    }
}


def test_candidate_models_fit_and_predict(
    sample_dataset: pd.DataFrame,
) -> None:
    """Both candidate pipelines should fit and generate predictions."""
    prepared = prepare_data(sample_dataset)

    features = prepared[FEATURE_COLUMNS]
    target = prepared[TARGET_COLUMN]

    models = build_models(
        TEST_PARAMS,
        random_state=42,
    )

    assert set(models) == {
        "logistic_regression",
        "random_forest",
    }

    for model in models.values():
        model.fit(features, target)

        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        assert len(predictions) == len(features)
        assert probabilities.shape == (len(features), 2)
        assert set(predictions).issubset({0, 1})
