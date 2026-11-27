
import math

import numpy as np
import pandas as pd


DEFAULT_BEDS_PER_CASE = 0.253

REQUIRED_FORECAST_COLUMNS = {
    "ward_id",
    "city",
    "horizon_weeks",
    "predicted_cases",
}

REQUIRED_CAPACITY_COLUMNS = {
    "ward_id",
    "city",
    "capacity_beds",
}

OVERFLOW_COLUMNS = [
    "horizon_weeks",
    "city",
    "from_ward",
    "to_ward",
    "beds_to_transfer",
    "unmet_beds",
    "status",
]


def _validate_inputs(forecasts, hospital_capacity, beds_per_case):
    if not isinstance(forecasts, pd.DataFrame):
        raise TypeError("forecasts must be a pandas DataFrame.")

    if not isinstance(hospital_capacity, pd.DataFrame):
        raise TypeError(
            "hospital_capacity must be a pandas DataFrame."
        )

    missing_forecasts = REQUIRED_FORECAST_COLUMNS - set(
        forecasts.columns
    )
    missing_capacity = REQUIRED_CAPACITY_COLUMNS - set(
        hospital_capacity.columns
    )

    if missing_forecasts:
        raise ValueError(
            f"Missing forecast columns: {sorted(missing_forecasts)}"
        )

    if missing_capacity:
        raise ValueError(
            f"Missing capacity columns: {sorted(missing_capacity)}"
        )

    if forecasts.empty or hospital_capacity.empty:
        raise ValueError("Forecast and capacity data cannot be empty.")

    if (
        isinstance(beds_per_case, (bool, np.bool_))
        or not isinstance(beds_per_case, (int, float, np.integer, np.floating))
        or not math.isfinite(beds_per_case)
        or beds_per_case <= 0
    ):
        raise ValueError(
            "beds_per_case must be a positive finite number."
        )

    forecasts = forecasts.copy()
    hospital_capacity = hospital_capacity.copy()

    if forecasts[list(REQUIRED_FORECAST_COLUMNS)].isna().any().any():
        raise ValueError("Forecast data contains missing required values.")

    if hospital_capacity[
        list(REQUIRED_CAPACITY_COLUMNS)
    ].isna().any().any():
        raise ValueError("Hospital capacity contains missing required values.")

    if forecasts.duplicated(
        subset=["ward_id", "horizon_weeks"]
    ).any():
        raise ValueError(
            "Forecasts must contain one row per ward and horizon."
        )

    if hospital_capacity["ward_id"].duplicated().any():
        raise ValueError(
            "Hospital capacity must contain one row per ward."
        )

    numeric_cases = pd.to_numeric(
        forecasts["predicted_cases"],
        errors="coerce",
    )
    numeric_capacity = pd.to_numeric(
        hospital_capacity["capacity_beds"],
        errors="coerce",
    )

    if numeric_cases.isna().any() or not np.isfinite(numeric_cases).all():
        raise ValueError("Predicted cases must be finite numeric values.")

    if (numeric_cases < 0).any():
        raise ValueError("Predicted cases cannot be negative.")

    if numeric_capacity.isna().any() or not np.isfinite(
        numeric_capacity
    ).all():
        raise ValueError("Hospital capacity must be finite numeric values.")

    if (numeric_capacity < 0).any():
        raise ValueError("Hospital capacity cannot be negative.")

    if (numeric_capacity % 1 != 0).any():
        raise ValueError(
            "Hospital capacity must contain whole numbers of beds."
        )

    forecasts["predicted_cases"] = numeric_cases.astype(float)
    hospital_capacity["capacity_beds"] = numeric_capacity.astype(int)

    if not forecasts["horizon_weeks"].isin([2, 3, 4]).all():
        raise ValueError("Forecast horizons must be 2, 3, or 4 weeks.")

    if not forecasts.groupby("ward_id")["city"].nunique().eq(1).all():
        raise ValueError("Each ward must belong to exactly one city.")

    if not hospital_capacity.groupby("ward_id")["city"].nunique().eq(1).all():
        raise ValueError(
            "Each capacity record must have one city per ward."
        )

    capacity_lookup = hospital_capacity.set_index("ward_id")["city"]

    missing_wards = (
        set(forecasts["ward_id"])
        - set(hospital_capacity["ward_id"])
    )

    if missing_wards:
        raise ValueError(
            f"Missing hospital capacity for wards: {sorted(missing_wards)}"
        )

    for ward_id in forecasts["ward_id"].unique():
        forecast_city = forecasts.loc[
            forecasts["ward_id"] == ward_id, "city"
        ].iloc[0]

        capacity_city = capacity_lookup.loc[ward_id]

        if forecast_city != capacity_city:
            raise ValueError(f"City mismatch for ward {ward_id}.")

    return forecasts, hospital_capacity


def assess_hospital_surge(
    forecasts,
    hospital_capacity,
    beds_per_case=DEFAULT_BEDS_PER_CASE,
    near_capacity_threshold=0.80,
):
    forecasts, hospital_capacity = _validate_inputs(
        forecasts,
        hospital_capacity,
        beds_per_case,
    )

    if (
        isinstance(near_capacity_threshold, (bool, np.bool_))
        or not isinstance(
            near_capacity_threshold,
            (int, float, np.integer, np.floating),
        )
        or not math.isfinite(near_capacity_threshold)
        or not 0 < near_capacity_threshold <= 1
    ):
        raise ValueError(
            "near_capacity_threshold must be between 0 and 1."
        )

    result = forecasts.merge(
        hospital_capacity,
        on=["ward_id", "city"],
        how="left",
        validate="many_to_one",
    )

    result["estimated_bed_demand"] = (
        result["predicted_cases"] * beds_per_case
    ).apply(math.ceil).astype(int)

    result["spare_beds"] = (
        result["capacity_beds"]
        - result["estimated_bed_demand"]
    ).clip(lower=0)

    result["excess_beds"] = (
        result["estimated_bed_demand"]
        - result["capacity_beds"]
    ).clip(lower=0)

    result["capacity_utilization"] = (
        result["estimated_bed_demand"]
        / result["capacity_beds"].replace(0, np.nan)
    )

    result.loc[
        (result["capacity_beds"] == 0)
        & (result["estimated_bed_demand"] == 0),
        "capacity_utilization",
    ] = 0.0

    result.loc[
        (result["capacity_beds"] == 0)
        & (result["estimated_bed_demand"] > 0),
        "capacity_utilization",
    ] = np.inf

    result["surge_status"] = "Normal"

    result.loc[
        result["capacity_utilization"] >= near_capacity_threshold,
        "surge_status",
    ] = "Near Capacity"

    result.loc[
        result["excess_beds"] > 0,
        "surge_status",
    ] = "Overloaded"

    result["demand_basis"] = "Scenario estimate"
    result["beds_per_case_assumption"] = beds_per_case

    return result.sort_values(
        ["horizon_weeks", "city", "ward_id"]
    ).reset_index(drop=True)


def _validate_assessment(surge_assessment):
    if not isinstance(surge_assessment, pd.DataFrame):
        raise TypeError(
            "surge_assessment must be a pandas DataFrame."
        )

    required = {
        "ward_id",
        "city",
        "horizon_weeks",
        "capacity_beds",
        "estimated_bed_demand",
        "spare_beds",
        "excess_beds",
    }

    missing = required - set(surge_assessment.columns)

    if missing:
        raise ValueError(
            f"Missing surge assessment columns: {sorted(missing)}"
        )

    if surge_assessment.empty:
        raise ValueError("Surge assessment cannot be empty.")

    data = surge_assessment.copy()

    if data[list(required)].isna().any().any():
        raise ValueError("Surge assessment contains missing required values.")

    if data.duplicated(
        subset=["ward_id", "horizon_weeks"]
    ).any():
        raise ValueError(
            "Surge assessment must contain one row per ward and horizon."
        )

    if not data["horizon_weeks"].isin([2, 3, 4]).all():
        raise ValueError("Forecast horizons must be 2, 3, or 4 weeks.")

    numeric_columns = [
        "capacity_beds",
        "estimated_bed_demand",
        "spare_beds",
        "excess_beds",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(data[column], errors="coerce")

        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite numeric values.")

        if (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(
                f"{column} must contain nonnegative whole numbers."
            )

        data[column] = values.astype(int)

    expected_spare = (
        data["capacity_beds"] - data["estimated_bed_demand"]
    ).clip(lower=0)

    expected_excess = (
        data["estimated_bed_demand"] - data["capacity_beds"]
    ).clip(lower=0)

    if not data["spare_beds"].equals(expected_spare):
        raise ValueError("Spare bed values are inconsistent.")

    if not data["excess_beds"].equals(expected_excess):
        raise ValueError("Excess bed values are inconsistent.")

    return data


def generate_overflow_plan(surge_assessment):
    data = _validate_assessment(surge_assessment)

    plan_rows = []

    for (horizon, city), city_data in data.groupby(
        ["horizon_weeks", "city"],
        sort=True,
    ):
        donors = city_data[
            city_data["spare_beds"] > 0
        ].sort_values(
            ["spare_beds", "ward_id"],
            ascending=[False, True],
        )

        receivers = city_data[
            city_data["excess_beds"] > 0
        ].sort_values(
            ["excess_beds", "ward_id"],
            ascending=[False, True],
        )

        donor_remaining = donors.set_index(
            "ward_id"
        )["spare_beds"].to_dict()

        for _, receiver in receivers.iterrows():
            unmet = int(receiver["excess_beds"])

            for donor_id in donors["ward_id"]:
                available = int(donor_remaining[donor_id])

                if available <= 0 or unmet <= 0:
                    continue

                transfer = min(available, unmet)

                plan_rows.append(
                    {
                        "horizon_weeks": horizon,
                        "city": city,
                        "from_ward": receiver["ward_id"],
                        "to_ward": donor_id,
                        "beds_to_transfer": transfer,
                        "unmet_beds": 0,
                        "status": "Transfer recommended",
                    }
                )

                donor_remaining[donor_id] -= transfer
                unmet -= transfer

            if unmet > 0:
                plan_rows.append(
                    {
                        "horizon_weeks": horizon,
                        "city": city,
                        "from_ward": receiver["ward_id"],
                        "to_ward": None,
                        "beds_to_transfer": 0,
                        "unmet_beds": unmet,
                        "status": "Unmet demand",
                    }
                )

    return pd.DataFrame(plan_rows, columns=OVERFLOW_COLUMNS)
