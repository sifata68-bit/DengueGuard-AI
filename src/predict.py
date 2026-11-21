
"""Production prediction engine for DengueGuard AI.

Loads trained models and generates ward-level dengue forecasts
for 2, 3, and 4 weeks ahead.

Uses the production feature pipeline. Current-week dengue cases
are never used as prediction features.
"""

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CITY_COL,
    DATE_COL,
    HORIZONS,
    MODELS_DIR,
    WARD_COL,
)
from src.preprocessing import get_clean_data
from src.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
    build_production_features,
)
from src.risk import add_risk_categories


MODEL_DIR = MODELS_DIR / "production"


def load_production_model(horizon):
    """Load a trained production model for the requested horizon."""

    if horizon not in HORIZONS:
        raise ValueError(
            f"Unsupported forecast horizon: {horizon}. "
            f"Available horizons: {HORIZONS}"
        )

    model_path = MODEL_DIR / f"dengue_h{horizon}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    artifact = joblib.load(model_path)

    if artifact["horizon_weeks"] != horizon:
        raise ValueError(
            f"Model horizon mismatch in {model_path}"
        )

    if artifact["feature_columns"] != PRODUCTION_FEATURE_COLUMNS:
        raise ValueError(
            f"Feature columns do not match the current "
            f"production pipeline for horizon {horizon}."
        )

    return artifact["model"]


def predict_ward(clean_data, ward_id, as_of_date=None):
    """Generate 2-, 3-, and 4-week forecasts for one ward.

    Args:
        clean_data: Cleaned historical dataset.
        ward_id: Ward identifier.
        as_of_date: Optional exact forecast issue date.
                    Defaults to the latest available date for the ward.

    Returns:
        DataFrame containing predictions and risk categories.
    """

    if ward_id not in clean_data[WARD_COL].unique():
        raise ValueError(f"Ward not found: {ward_id}")

    # Build features using the same pipeline used during training.
    features = build_production_features(clean_data)

    ward_features = features[
        features[WARD_COL] == ward_id
    ].copy()

    if ward_features.empty:
        raise ValueError(
            f"No feature records found for ward {ward_id}."
        )

    ward_features[DATE_COL] = pd.to_datetime(
        ward_features[DATE_COL]
    )

    if as_of_date is None:
        selected_date = ward_features[DATE_COL].max()
    else:
        selected_date = pd.to_datetime(as_of_date)

    # Require an exact observation date.
    selected_rows = ward_features[
        ward_features[DATE_COL] == selected_date
    ]

    if selected_rows.empty:
        raise ValueError(
            f"No data found for ward {ward_id} "
            f"on {selected_date.date()}."
        )

    row = selected_rows.iloc[[0]]

    # Check that every required production feature is available.
    missing_features = row[
        PRODUCTION_FEATURE_COLUMNS
    ].isna().any()

    unavailable = missing_features[
        missing_features
    ].index.tolist()

    if unavailable:
        raise ValueError(
            "Insufficient historical data to generate a forecast. "
            f"Missing features: {unavailable}"
        )

    predictions = []

    for horizon in HORIZONS:
        model = load_production_model(horizon)

        X = row[PRODUCTION_FEATURE_COLUMNS].astype(float)

        predicted_cases = float(
            np.maximum(model.predict(X)[0], 0.0)
        )

        result = {
            WARD_COL: ward_id,
            DATE_COL: selected_date,
            "horizon_weeks": horizon,
            "predicted_cases": round(predicted_cases, 2),
        }

        if CITY_COL in clean_data.columns:
            city_values = clean_data.loc[
                clean_data[WARD_COL] == ward_id,
                CITY_COL,
            ].dropna()

            if not city_values.empty:
                result[CITY_COL] = city_values.iloc[0]

        predictions.append(result)

    # Attach Low, Medium, High, or Critical risk categories.
    forecast_df = pd.DataFrame(predictions)

    return add_risk_categories(forecast_df)


def predict_all_wards(clean_data, as_of_date=None):
    """Generate 2-, 3-, and 4-week forecasts for all wards.

    Builds production features once and predicts all wards in batches.

    Args:
        clean_data: Cleaned historical dataset.
        as_of_date: Optional exact forecast issue date.
                    Defaults to each ward's latest available date.

    Returns:
        DataFrame containing predictions and risk categories.
    """

    # Build the production features only once.
    features = build_production_features(clean_data)

    features[DATE_COL] = pd.to_datetime(features[DATE_COL])

    # Select the latest available observation for each ward.
    if as_of_date is None:
        selected_rows = (
            features.sort_values(DATE_COL)
            .drop_duplicates(subset=[WARD_COL], keep="last")
            .copy()
        )
    else:
        selected_date = pd.to_datetime(as_of_date)

        selected_rows = features[
            features[DATE_COL] == selected_date
        ].copy()

    if selected_rows.empty:
        raise ValueError("No ward records available for prediction.")

    # Check that all required features are available.
    missing_rows = selected_rows[
        PRODUCTION_FEATURE_COLUMNS
    ].isna().any(axis=1)

    if missing_rows.any():
        missing_wards = selected_rows.loc[
            missing_rows, WARD_COL
        ].tolist()

        raise ValueError(
            "Some wards have insufficient historical features: "
            f"{missing_wards}"
        )

    X = selected_rows[
        PRODUCTION_FEATURE_COLUMNS
    ].astype(float)

    predictions = []

    # Load each model once and predict all wards together.
    for horizon in HORIZONS:
        model = load_production_model(horizon)

        predicted_cases = np.maximum(
            model.predict(X),
            0.0,
        )

        horizon_results = selected_rows[
            [WARD_COL, DATE_COL]
        ].copy()

        if CITY_COL in clean_data.columns:
            city_lookup = (
                clean_data[[WARD_COL, CITY_COL]]
                .dropna(subset=[WARD_COL])
                .drop_duplicates(subset=[WARD_COL])
            )

            horizon_results = horizon_results.merge(
                city_lookup,
                on=WARD_COL,
                how="left",
                validate="one_to_one",
            )

        horizon_results["horizon_weeks"] = horizon
        horizon_results["predicted_cases"] = np.round(
            predicted_cases, 2
        )

        predictions.append(horizon_results)

    forecast_df = pd.concat(
        predictions,
        ignore_index=True,
    )

    forecast_df = add_risk_categories(forecast_df)

    return forecast_df.sort_values(
        [CITY_COL, WARD_COL, "horizon_weeks"]
        if CITY_COL in forecast_df.columns
        else [WARD_COL, "horizon_weeks"]
    ).reset_index(drop=True)

def main():
    """Run a basic prediction using the latest available ward record."""

    print("Loading cleaned data...")
    clean = get_clean_data()

    first_ward = clean[WARD_COL].iloc[0]

    print(f"\nGenerating forecasts for ward {first_ward}...")

    forecasts = predict_ward(
        clean_data=clean,
        ward_id=first_ward,
    )

    print("\nDENGUE FORECAST")
    print("=" * 65)

    print(
        forecasts.to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )


if __name__ == "__main__":
    main()
