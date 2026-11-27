
import math

import numpy as np
import pandas as pd


RISK_PRIORITY = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
}

HORIZONS = (2, 3, 4)

DEFAULT_SALINE_PER_CASE = 2.377868192238584
DEFAULT_BEDS_PER_CASE = 0.253
DEFAULT_DIAGNOSTIC_KITS_PER_CASE = 1.146037712468246

ALLOWED_INTERVENTIONS = {"larvicide", "fogging"}


def _require_columns(data, required, name):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {sorted(missing)}"
        )


def _validate_nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative number.")

    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric.") from None

    if not math.isfinite(number) or number < 0:
        raise ValueError(
            f"{name} must be finite and nonnegative."
        )

    return number


def _validate_inputs(
    forecasts,
    ward_information,
    selected_wards,
    total_teams,
    intervention_type,
    effectiveness_by_horizon,
):
    _require_columns(
        forecasts,
        [
            "ward_id",
            "horizon_weeks",
            "predicted_cases",
            "risk_category",
        ],
        "forecasts",
    )

    _require_columns(
        ward_information,
        [
            "ward_id",
            "city",
            "population_density_sqkm",
        ],
        "ward_information",
    )

    if forecasts.empty or ward_information.empty:
        raise ValueError("Forecast and ward information cannot be empty.")

    if (
        isinstance(total_teams, (bool, np.bool_))
        or not isinstance(total_teams, (int, np.integer))
        or total_teams < 0
    ):
        raise ValueError("total_teams must be a nonnegative integer.")

    if not isinstance(selected_wards, (list, tuple, set)):
        raise TypeError("selected_wards must be a list, tuple, or set.")

    selected_wards = list(selected_wards)

    try:
        if len(selected_wards) != len(set(selected_wards)):
            raise ValueError("Selected wards cannot contain duplicates.")
    except TypeError:
        raise ValueError("Selected ward IDs must be hashable.") from None

    if not selected_wards:
        raise ValueError("At least one ward must be selected.")

    if intervention_type not in ALLOWED_INTERVENTIONS:
        raise ValueError(
            "intervention_type must be 'larvicide' or 'fogging'."
        )

    if not isinstance(effectiveness_by_horizon, dict):
        raise TypeError(
            "effectiveness_by_horizon must be a dictionary."
        )

    if set(effectiveness_by_horizon) != set(HORIZONS):
        raise ValueError(
            "Effectiveness must be supplied for horizons 2, 3, and 4."
        )

    effectiveness = {}

    for horizon in HORIZONS:
        value = _validate_nonnegative(
            effectiveness_by_horizon[horizon],
            f"effectiveness for horizon {horizon}",
        )

        if value > 1:
            raise ValueError(
                "Effectiveness values must be between 0 and 1."
            )

        effectiveness[horizon] = value

    forecasts = forecasts.copy()
    ward_information = ward_information.copy()

    if forecasts[
        ["ward_id", "horizon_weeks", "predicted_cases", "risk_category"]
    ].isna().any().any():
        raise ValueError("Forecasts contain missing required values.")

    if ward_information[
        ["ward_id", "city", "population_density_sqkm"]
    ].isna().any().any():
        raise ValueError("Ward information contains missing values.")

    if forecasts.duplicated(
        ["ward_id", "horizon_weeks"]
    ).any():
        raise ValueError(
            "Forecasts must contain one row per ward and horizon."
        )

    if ward_information["ward_id"].duplicated().any():
        raise ValueError(
            "Ward information must contain one row per ward."
        )

    if not forecasts["horizon_weeks"].isin(HORIZONS).all():
        raise ValueError("Forecast horizons must be 2, 3, or 4.")

    if not forecasts["risk_category"].isin(RISK_PRIORITY).all():
        raise ValueError("Forecasts contain invalid risk categories.")

    forecasts["predicted_cases"] = pd.to_numeric(
        forecasts["predicted_cases"],
        errors="coerce",
    )

    if (
        forecasts["predicted_cases"].isna().any()
        or not np.isfinite(forecasts["predicted_cases"]).all()
        or (forecasts["predicted_cases"] < 0).any()
    ):
        raise ValueError(
            "Predicted cases must be finite and nonnegative."
        )

    ward_information["population_density_sqkm"] = pd.to_numeric(
        ward_information["population_density_sqkm"],
        errors="coerce",
    )

    if (
        ward_information["population_density_sqkm"].isna().any()
        or not np.isfinite(
            ward_information["population_density_sqkm"]
        ).all()
        or (ward_information["population_density_sqkm"] < 0).any()
    ):
        raise ValueError(
            "Population density must be finite and nonnegative."
        )

    known_wards = set(ward_information["ward_id"])
    forecast_wards = set(forecasts["ward_id"])

    unknown = set(selected_wards) - known_wards

    if unknown:
        raise ValueError(
            f"Unknown selected wards: {sorted(unknown)}"
        )

    missing_forecasts = set(selected_wards) - forecast_wards

    if missing_forecasts:
        raise ValueError(
            f"Missing forecasts for selected wards: "
            f"{sorted(missing_forecasts)}"
        )

    selected_forecasts = forecasts[
        forecasts["ward_id"].isin(selected_wards)
    ].copy()

    for ward_id in selected_wards:
        ward_horizons = set(
            selected_forecasts.loc[
                selected_forecasts["ward_id"] == ward_id,
                "horizon_weeks",
            ]
        )

        if ward_horizons != set(HORIZONS):
            raise ValueError(
                f"Ward {ward_id} must have forecasts for horizons 2, 3, and 4."
            )

    ward_information = ward_information[
        ward_information["ward_id"].isin(selected_wards)
    ].copy()

    merged = selected_forecasts.merge(
        ward_information,
        on="ward_id",
        how="left",
        validate="many_to_one",
        suffixes=("_forecast", "_ward"),
    )

    if "city_forecast" in merged.columns:
        if (
            merged["city_forecast"].notna()
            & (merged["city_forecast"] != merged["city_ward"])
        ).any():
            raise ValueError(
                "Forecast city does not match ward information."
            )

        merged = merged.drop(columns=["city_forecast"])
        merged = merged.rename(columns={"city_ward": "city"})

    return merged, effectiveness


def simulate_vector_control(
    forecasts,
    ward_information,
    selected_wards,
    total_teams,
    intervention_type,
    effectiveness_by_horizon,
    saline_per_case=DEFAULT_SALINE_PER_CASE,
    beds_per_case=DEFAULT_BEDS_PER_CASE,
    diagnostic_kits_per_case=DEFAULT_DIAGNOSTIC_KITS_PER_CASE,
):
    data, effectiveness = _validate_inputs(
        forecasts,
        ward_information,
        selected_wards,
        total_teams,
        intervention_type,
        effectiveness_by_horizon,
    )

    saline_ratio = _validate_nonnegative(
        saline_per_case,
        "saline_per_case",
    )

    bed_ratio = _validate_nonnegative(
        beds_per_case,
        "beds_per_case",
    )

    diagnostic_ratio = _validate_nonnegative(
        diagnostic_kits_per_case,
        "diagnostic_kits_per_case",
    )

    priorities = (
        data.groupby("ward_id", as_index=False)
        .agg(
            risk_priority=(
                "risk_category",
                lambda values: min(
                    RISK_PRIORITY[value] for value in values
                ),
            ),
            population_density_sqkm=(
                "population_density_sqkm",
                "first",
            ),
        )
    )

    priority_to_risk = {
        priority: risk
        for risk, priority in RISK_PRIORITY.items()
    }

    priorities["risk_category"] = priorities[
        "risk_priority"
    ].map(priority_to_risk)

    priorities = priorities.sort_values(
        [
            "risk_priority",
            "population_density_sqkm",
            "ward_id",
        ],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    priorities["team_assigned"] = False

    assigned_count = min(
        int(total_teams),
        len(priorities),
    )

    priorities.loc[
        priorities.index[:assigned_count],
        "team_assigned",
    ] = True

    assignments = priorities[
        [
            "ward_id",
            "risk_category",
            "population_density_sqkm",
            "team_assigned",
        ]
    ].copy()

    assignments["intervention_type"] = intervention_type

    data = data.merge(
        assignments[["ward_id", "team_assigned"]],
        on="ward_id",
        how="left",
        validate="many_to_one",
    )

    data["intervention_effectiveness_assumption"] = (
        data["horizon_weeks"].map(effectiveness)
    )

    data["baseline_cases"] = data["predicted_cases"]

    data["scenario_reduction_rate"] = np.where(
        data["team_assigned"],
        data["intervention_effectiveness_assumption"],
        0.0,
    )

    data["scenario_cases"] = (
        data["baseline_cases"]
        * (1 - data["scenario_reduction_rate"])
    ).clip(lower=0)

    data["cases_prevented"] = (
        data["baseline_cases"] - data["scenario_cases"]
    ).clip(lower=0)

    data["baseline_beds"] = np.ceil(
        data["baseline_cases"] * bed_ratio
    ).astype(int)

    data["scenario_beds"] = np.ceil(
        data["scenario_cases"] * bed_ratio
    ).astype(int)

    data["beds_avoided"] = (
        data["baseline_beds"] - data["scenario_beds"]
    ).clip(lower=0)

    data["baseline_saline"] = np.ceil(
        data["baseline_cases"] * saline_ratio
    ).astype(int)

    data["scenario_saline"] = np.ceil(
        data["scenario_cases"] * saline_ratio
    ).astype(int)

    data["saline_saved"] = (
        data["baseline_saline"] - data["scenario_saline"]
    ).clip(lower=0)

    data["baseline_diagnostic_kits"] = np.ceil(
        data["baseline_cases"] * diagnostic_ratio
    ).astype(int)

    data["scenario_diagnostic_kits"] = np.ceil(
        data["scenario_cases"] * diagnostic_ratio
    ).astype(int)

    data["diagnostic_kits_saved"] = (
        data["baseline_diagnostic_kits"]
        - data["scenario_diagnostic_kits"]
    ).clip(lower=0)

    data["roi_cases_prevented_per_crew"] = np.where(
        data["team_assigned"],
        data["cases_prevented"],
        0.0,
    )

    ward_results = data.sort_values(
        ["horizon_weeks", "city", "ward_id"]
    ).reset_index(drop=True)

    roi_ranking = ward_results[
        [
            "horizon_weeks",
            "ward_id",
            "city",
            "risk_category",
            "team_assigned",
            "cases_prevented",
        ]
    ].copy()

    roi_ranking["roi_cases_prevented_per_crew"] = np.where(
        roi_ranking["team_assigned"],
        roi_ranking["cases_prevented"],
        0.0,
    )

    roi_ranking = roi_ranking.sort_values(
        [
            "horizon_weeks",
            "roi_cases_prevented_per_crew",
            "ward_id",
        ],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    summary = (
        ward_results.groupby(
            "horizon_weeks",
            as_index=False,
        )
        .agg(
            baseline_cases=("baseline_cases", "sum"),
            scenario_cases=("scenario_cases", "sum"),
            cases_prevented=("cases_prevented", "sum"),
            beds_avoided=("beds_avoided", "sum"),
            saline_saved=("saline_saved", "sum"),
            diagnostic_kits_saved=("diagnostic_kits_saved", "sum"),
        )
    )

    return {
        "ward_results": ward_results,
        "team_assignments": assignments.reset_index(drop=True),
        "roi_ranking": roi_ranking,
        "summary": summary,
        "assumptions": {
            "intervention_type": intervention_type,
            "total_teams_available": int(total_teams),
            "teams_deployed": int(assigned_count),
            "effectiveness_by_horizon": effectiveness,
            "effectiveness_basis": "User-supplied scenario assumption",
            "saline_per_case": saline_ratio,
            "beds_per_case": bed_ratio,
            "diagnostic_kits_per_case": diagnostic_ratio,
            "resource_ratio_basis": (
                "Historical training-period pooled ratios, "
                "except beds_per_case, which uses the existing "
                "hospital surge assumption."
            ),
        },
    }
