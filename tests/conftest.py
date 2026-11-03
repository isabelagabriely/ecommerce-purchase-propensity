"""Shared pytest fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Create a small dataset matching the production schema."""
    rows = 20

    return pd.DataFrame(
        {
            "Administrative": [index % 5 for index in range(rows)],
            "Administrative_Duration": [float(index * 2) for index in range(rows)],
            "Informational": [index % 3 for index in range(rows)],
            "Informational_Duration": [float(index) for index in range(rows)],
            "ProductRelated": [index + 1 for index in range(rows)],
            "ProductRelated_Duration": [float(index * 10) for index in range(rows)],
            "BounceRates": [index / 100 for index in range(rows)],
            "ExitRates": [(index + 1) / 100 for index in range(rows)],
            "PageValues": [float(index * 3) for index in range(rows)],
            "SpecialDay": [0.0] * rows,
            "Month": ["Feb", "Mar", "May", "Nov", "Dec"] * 4,
            "OperatingSystems": [1, 2] * 10,
            "Browser": [1, 2, 3, 4] * 5,
            "Region": [1, 2, 3, 4, 5] * 4,
            "TrafficType": [1, 2, 3, 4, 5] * 4,
            "VisitorType": [
                "Returning_Visitor",
                "New_Visitor",
            ]
            * 10,
            "Weekend": [index % 2 == 0 for index in range(rows)],
            "Revenue": [index % 2 == 0 for index in range(rows)],
        }
    ).assign(
        Administrative_Duration=lambda dataframe: dataframe[
            "Administrative_Duration"
        ].mask(dataframe.index == 0, np.nan),
        VisitorType=lambda dataframe: dataframe["VisitorType"].mask(
            dataframe.index == 1,
            None,
        ),
    )
