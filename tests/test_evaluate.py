"""Tests for model evaluation utilities."""

from ecommerce_purchase_propensity.modeling.evaluate import select_champion


def test_select_champion_uses_configured_metric() -> None:
    """Champion selection should use the configured validation metric."""
    results = [
        {
            "model": "logistic_regression",
            "cv_average_precision_mean": 0.65,
        },
        {
            "model": "random_forest",
            "cv_average_precision_mean": 0.74,
        },
    ]

    champion = select_champion(
        results,
        selection_metric="average_precision",
    )

    assert champion == "random_forest"
