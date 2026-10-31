"""Model evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

CV_SCORING: dict[str, Any] = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
    "f1": make_scorer(f1_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
}


def build_cross_validator(
    *,
    n_splits: int,
    random_state: int,
) -> StratifiedKFold:
    """Build a reproducible stratified cross-validator."""
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def cross_validate_model(
    model_name: str,
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    cross_validator: StratifiedKFold,
) -> dict[str, float | str]:
    """Evaluate one candidate using cross-validation."""
    scores = cross_validate(
        model,
        features,
        target,
        cv=cross_validator,
        scoring=CV_SCORING,
        return_train_score=False,
    )

    result: dict[str, float | str] = {"model": model_name}

    for metric_name in CV_SCORING:
        values = scores[f"test_{metric_name}"]
        result[f"cv_{metric_name}_mean"] = float(np.mean(values))
        result[f"cv_{metric_name}_std"] = float(np.std(values))

    return result


def select_champion(
    results: list[dict[str, float | str]],
    *,
    selection_metric: str,
) -> str:
    """Select the model with the highest validation metric."""
    score_key = f"cv_{selection_metric}_mean"

    champion = max(
        results,
        key=lambda result: float(result[score_key]),
    )

    return str(champion["model"])


def evaluate_holdout(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Evaluate a fitted classifier on held-out test data."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return {
        "average_precision": float(average_precision_score(target, probabilities)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "accuracy": float(accuracy_score(target, predictions)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
    }
