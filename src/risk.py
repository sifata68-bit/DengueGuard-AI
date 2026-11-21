
"""Risk classification for DengueGuard AI.

Thresholds are based on non-zero training-period
(June-October, 2016-2021) dengue case quantiles.

These are provisional operational thresholds, not
official public-health alert standards.
"""

import numpy as np
import pandas as pd


RISK_THRESHOLDS = {
    "low": 3.0,
    "medium": 6.0,
    "high": 10.0,
}


def classify_risk(predicted_cases):
    """Convert a predicted case count into a risk category."""

    if pd.isna(predicted_cases):
        raise ValueError("Predicted cases cannot be missing.")

    if not np.isfinite(predicted_cases):
        raise ValueError("Predicted cases must be finite.")

    if predicted_cases < 0:
        raise ValueError("Predicted cases cannot be negative.")

    if predicted_cases < RISK_THRESHOLDS["low"]:
        return "Low"

    if predicted_cases < RISK_THRESHOLDS["medium"]:
        return "Medium"

    if predicted_cases < RISK_THRESHOLDS["high"]:
        return "High"

    return "Critical"


def add_risk_categories(predictions):
    """Add a risk category to every forecast row.

    Expects a DataFrame containing a predicted_cases column.
    Returns a copy; the original DataFrame is unchanged.
    """

    if "predicted_cases" not in predictions.columns:
        raise ValueError(
            "Predictions must contain a 'predicted_cases' column."
        )

    result = predictions.copy()

    result["risk_category"] = result["predicted_cases"].apply(
        classify_risk
    )

    return result
