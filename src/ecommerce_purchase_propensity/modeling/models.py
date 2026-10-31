"""Model and preprocessing pipeline definitions."""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ecommerce_purchase_propensity.data.schema import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


def build_preprocessor() -> ColumnTransformer:
    """Build the feature preprocessing transformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_pipeline(classifier: BaseEstimator) -> Pipeline:
    """Combine preprocessing and a classifier in one pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def build_models(
    params: dict[str, Any],
    *,
    random_state: int,
) -> dict[str, Pipeline]:
    """Build candidate classification pipelines."""
    logistic_params = params["models"]["logistic_regression"]
    forest_params = params["models"]["random_forest"]

    logistic_regression = LogisticRegression(
        max_iter=int(logistic_params["max_iter"]),
        solver=str(logistic_params["solver"]),
    )

    random_forest = RandomForestClassifier(
        n_estimators=int(forest_params["n_estimators"]),
        max_depth=forest_params["max_depth"],
        min_samples_split=int(forest_params["min_samples_split"]),
        random_state=random_state,
        n_jobs=int(forest_params["n_jobs"]),
    )

    return {
        "logistic_regression": build_model_pipeline(logistic_regression),
        "random_forest": build_model_pipeline(random_forest),
    }
