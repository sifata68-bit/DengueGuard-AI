
import pandas as pd
import pytest

from src.risk import classify_risk, add_risk_categories


@pytest.mark.parametrize(
    "cases, expected",
    [
        (0, "Low"),
        (2.99, "Low"),
        (3, "Medium"),
        (5.99, "Medium"),
        (6, "High"),
        (9.99, "High"),
        (10, "Critical"),
        (15, "Critical"),
    ],
)
def test_classify_risk_boundaries(cases, expected):
    assert classify_risk(cases) == expected


@pytest.mark.parametrize("cases", [-1, float("nan"), float("inf"), float("-inf")])
def test_classify_risk_rejects_invalid_values(cases):
    with pytest.raises(ValueError):
        classify_risk(cases)


def test_add_risk_categories():
    predictions = pd.DataFrame({
        "predicted_cases": [1, 4, 8, 12]
    })

    result = add_risk_categories(predictions)

    assert result["risk_category"].tolist() == [
        "Low", "Medium", "High", "Critical"
    ]

    # Original DataFrame must remain unchanged.
    assert "risk_category" not in predictions.columns


def test_add_risk_categories_requires_prediction_column():
    predictions = pd.DataFrame({"other_column": [1, 2, 3]})

    with pytest.raises(ValueError):
        add_risk_categories(predictions)
