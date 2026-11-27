import math

import numpy as np
import pandas as pd

from src.hospital_surge import DEFAULT_BEDS_PER_CASE


RISK_PRIORITY = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
}

RESOURCE_COLUMNS = {
    "saline_bags": "saline_bags",
    "diagnostic_kits": "diagnostic_kits",
}

ALLOWED_PARAMETER_STATUSES = {
    "verified",
    "user_supplied",
    "scenario_assumption",
}


def _require_columns(dataframe, required, name):
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")

    missing = sorted(set(required) - set(dataframe.columns))

    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_nonnegative_numeric(series, name):
    values = pd.to_numeric(series, errors="coerce")

    if values.isna().any():
        raise ValueError(f"{name} must contain valid numeric values.")

    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must contain finite values.")

    if (values < 0).any():
        raise ValueError(f"{name} cannot contain negative values.")

    return values


def _validate_inventory(inventory):
    required = ["city", "saline_bags", "diagnostic_kits"]
    _require_columns(inventory, required, "regional_inventory")

    inventory = inventory.copy()

    if inventory["city"].isna().any():
        raise ValueError("Inventory city values cannot be missing.")

    if inventory["city"].duplicated().any():
        raise ValueError("Inventory must contain one row per city.")

    for resource in RESOURCE_COLUMNS.values():
        inventory[resource] = _validate_nonnegative_numeric(
            inventory[resource],
            resource,
        )

        if (inventory[resource] % 1 != 0).any():
            raise ValueError(
                f"{resource} inventory must contain whole numbers."
            )

        inventory[resource] = inventory[resource].astype(int)

    return inventory


def _validate_demand_parameters(demand_parameters):
    if not isinstance(demand_parameters, dict):
        raise TypeError("demand_parameters must be a dictionary.")

    for resource in RESOURCE_COLUMNS:
        if resource not in demand_parameters:
            raise ValueError(
                f"Missing demand parameters for {resource}."
            )

        parameters = demand_parameters[resource]

        if not isinstance(parameters, dict):
            raise TypeError(
                f"Parameters for {resource} must be a dictionary."
            )

        required = {"per_case", "source", "status"}
        missing = required - set(parameters)

        if missing:
            raise ValueError(
                f"Missing {resource} parameter fields: {sorted(missing)}"
            )

        try:
            per_case = float(parameters["per_case"])
        except (TypeError, ValueError):
            raise ValueError(
                f"{resource} per_case must be numeric."
            ) from None

        if not math.isfinite(per_case) or per_case < 0:
            raise ValueError(
                f"{resource} per_case must be finite and nonnegative."
            )

        source = parameters["source"]

        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"{resource} must include a nonempty source."
            )

        if parameters["status"] not in ALLOWED_PARAMETER_STATUSES:
            raise ValueError(
                f"{resource} has an invalid provenance status."
            )

    return {
        resource: {
            "per_case": float(demand_parameters[resource]["per_case"]),
            "source": demand_parameters[resource]["source"].strip(),
            "status": demand_parameters[resource]["status"],
        }
        for resource in RESOURCE_COLUMNS
    }


def _validate_crew_limits(spraying_crews_by_city, cities):
    if spraying_crews_by_city is None:
        spraying_crews_by_city = {city: 0 for city in cities}

    if not isinstance(spraying_crews_by_city, dict):
        raise TypeError("spraying_crews_by_city must be a dictionary.")

    if set(spraying_crews_by_city) != set(cities):
        raise ValueError(
            "Crew limits must be supplied for exactly the forecast cities."
        )

    limits = {}

    for city, crews in spraying_crews_by_city.items():
        if isinstance(crews, (bool, np.bool_)):
            raise ValueError("Crew limits must be nonnegative whole numbers.")

        try:
            numeric_crews = float(crews)
        except (TypeError, ValueError):
            raise ValueError(
                "Crew limits must be nonnegative whole numbers."
            ) from None

        if (
            not math.isfinite(numeric_crews)
            or numeric_crews < 0
            or numeric_crews % 1 != 0
        ):
            raise ValueError(
                "Crew limits must be nonnegative whole numbers."
            )

        limits[city] = int(numeric_crews)

    return limits


def _prepare_inputs(
    forecasts,
    ward_information,
    regional_inventory,
    horizon_weeks,
):
    if (
        isinstance(horizon_weeks, (bool, np.bool_))
        or horizon_weeks not in (2, 3, 4)
    ):
        raise ValueError("horizon_weeks must be 2, 3, or 4.")

    _require_columns(
        forecasts,
        [
            "ward_id",
            "date",
            "horizon_weeks",
            "predicted_cases",
            "risk_category",
        ],
        "forecasts",
    )

    _require_columns(
        ward_information,
        ["ward_id", "city", "population_density_sqkm"],
        "ward_information",
    )

    selected = forecasts.loc[
        forecasts["horizon_weeks"] == horizon_weeks
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No forecasts found for horizon {horizon_weeks}."
        )

    if selected["ward_id"].isna().any():
        raise ValueError("Forecast ward IDs cannot be missing.")

    if selected["ward_id"].duplicated().any():
        raise ValueError(
            "Forecasts must contain one row per ward for the selected horizon."
        )

    if selected["date"].isna().any():
        raise ValueError("Forecast dates cannot be missing.")

    if not selected["risk_category"].isin(RISK_PRIORITY).all():
        raise ValueError("Forecasts contain invalid risk categories.")

    selected["predicted_cases"] = _validate_nonnegative_numeric(
        selected["predicted_cases"],
        "predicted_cases",
    )

    ward_info = ward_information[
        ["ward_id", "city", "population_density_sqkm"]
    ].copy()

    if ward_info["ward_id"].isna().any():
        raise ValueError("Ward information contains missing ward IDs.")

    if ward_info["ward_id"].duplicated().any():
        raise ValueError(
            "Ward information must contain one row per ward."
        )

    if ward_info["city"].isna().any():
        raise ValueError("Ward city values cannot be missing.")

    ward_info["population_density_sqkm"] = (
        _validate_nonnegative_numeric(
            ward_info["population_density_sqkm"],
            "population_density_sqkm",
        )
    )

    if "city" in selected.columns:
        checked = selected[
            ["ward_id", "city"]
        ].merge(
            ward_info[["ward_id", "city"]],
            on="ward_id",
            how="left",
            suffixes=("_forecast", "_ward"),
            validate="one_to_one",
        )

        comparable = (
            checked["city_forecast"].notna()
            & checked["city_ward"].notna()
        )

        if (
            checked.loc[comparable, "city_forecast"]
            != checked.loc[comparable, "city_ward"]
        ).any():
            raise ValueError(
                "Forecast city does not match ward information."
            )

    prepared = selected.merge(
        ward_info,
        on="ward_id",
        how="left",
        validate="one_to_one",
        suffixes=("_forecast", "_ward"),
    )

    if "city_forecast" in prepared.columns:
        prepared["city"] = prepared["city_forecast"].fillna(
            prepared["city_ward"]
        )
        prepared = prepared.drop(
            columns=["city_forecast", "city_ward"]
        )

    if prepared["city"].isna().any():
        raise ValueError(
            "Some forecast wards have no matching city information."
        )

    if prepared["population_density_sqkm"].isna().any():
        raise ValueError(
            "Some forecast wards have no population density."
        )

    inventory = _validate_inventory(regional_inventory)

    missing_cities = set(prepared["city"]) - set(inventory["city"])

    if missing_cities:
        raise ValueError(
            f"Regional inventory is missing cities: {sorted(missing_cities)}"
        )

    return prepared, inventory


def allocate_resources(
    forecasts,
    ward_information,
    regional_inventory,
    stock_percentage,
    demand_parameters,
    horizon_weeks,
    spraying_crews_by_city=None,
):
    if isinstance(stock_percentage, (bool, np.bool_)):
        raise ValueError("stock_percentage must be between 30 and 100.")

    try:
        stock_percentage = float(stock_percentage)
    except (TypeError, ValueError):
        raise ValueError(
            "stock_percentage must be numeric."
        ) from None

    if (
        not math.isfinite(stock_percentage)
        or not 30 <= stock_percentage <= 100
    ):
        raise ValueError(
            "stock_percentage must be between 30 and 100."
        )

    parameters = _validate_demand_parameters(demand_parameters)

    prepared, inventory = _prepare_inputs(
        forecasts,
        ward_information,
        regional_inventory,
        horizon_weeks,
    )

    crew_limits = _validate_crew_limits(
        spraying_crews_by_city,
        prepared["city"].unique().tolist(),
    )

    allocation_records = []
    regional_records = []
    crew_records = []

    for city, city_wards in prepared.groupby("city", sort=True):
        city_stock = inventory.loc[
            inventory["city"] == city
        ].iloc[0]

        available = {
            resource: math.floor(
                city_stock[resource] * stock_percentage / 100
            )
            for resource in RESOURCE_COLUMNS
        }

        remaining = available.copy()
        city_wards = city_wards.copy()

        city_wards["risk_priority"] = (
            city_wards["risk_category"].map(RISK_PRIORITY)
        )

        city_wards = city_wards.sort_values(
            by=[
                "risk_priority",
                "population_density_sqkm",
                "ward_id",
            ],
            ascending=[True, False, True],
            kind="stable",
        )

        city_demand = {
            resource: 0 for resource in RESOURCE_COLUMNS
        }
        city_allocated = {
            resource: 0 for resource in RESOURCE_COLUMNS
        }

        crew_limit = crew_limits[city]
        assigned_crews = 0

        for city_rank, (_, ward) in enumerate(
            city_wards.iterrows(),
            start=1,
        ):
            crew_assigned = assigned_crews < crew_limit

            if crew_assigned:
                assigned_crews += 1

            predicted_cases = float(ward["predicted_cases"])
            beds_required = math.ceil(
                predicted_cases * DEFAULT_BEDS_PER_CASE
            )

            record = {
                "city": city,
                "ward_id": ward["ward_id"],
                "date": ward["date"],
                "horizon_weeks": horizon_weeks,
                "predicted_cases": predicted_cases,
                "risk_category": ward["risk_category"],
                "population_density_sqkm": float(
                    ward["population_density_sqkm"]
                ),
                "priority_rank": city_rank,
                "hospital_beds_required": beds_required,
                "vector_control_crews_assigned": int(crew_assigned),
                "crew_assignment_status": (
                    "Assigned" if crew_assigned else "Unassigned"
                ),
            }

            for resource in RESOURCE_COLUMNS:
                demand = math.ceil(
                    predicted_cases * parameters[resource]["per_case"]
                )

                allocated = min(demand, remaining[resource])
                unmet = demand - allocated

                remaining[resource] -= allocated
                city_demand[resource] += demand
                city_allocated[resource] += allocated

                record[f"{resource}_demand"] = demand
                record[f"{resource}_needed"] = demand
                record[f"{resource}_allocated"] = allocated
                record[f"{resource}_unmet"] = unmet

            allocation_records.append(record)

            crew_records.append(
                {
                    "city": city,
                    "ward_id": ward["ward_id"],
                    "risk_category": ward["risk_category"],
                    "population_density_sqkm": float(
                        ward["population_density_sqkm"]
                    ),
                    "priority_rank": city_rank,
                    "crew_assigned": int(crew_assigned),
                    "assignment_status": (
                        "Assigned" if crew_assigned else "Unassigned"
                    ),
                }
            )

        regional_record = {
            "city": city,
            "horizon_weeks": horizon_weeks,
            "stock_percentage": stock_percentage,
            "spraying_crews_available": crew_limit,
            "spraying_crews_assigned": assigned_crews,
            "spraying_crews_unassigned": (
                crew_limit - assigned_crews
            ),
            "spraying_crews_unused": crew_limit - assigned_crews,
            "wards_without_crews": len(city_wards) - assigned_crews,
            "hospital_beds_required": sum(
                math.ceil(
                    float(cases) * DEFAULT_BEDS_PER_CASE
                )
                for cases in city_wards["predicted_cases"]
            ),
        }

        for resource in RESOURCE_COLUMNS:
            regional_record[f"{resource}_inventory"] = int(
                city_stock[resource]
            )
            regional_record[f"{resource}_available"] = available[resource]
            regional_record[f"{resource}_demand"] = city_demand[resource]
            regional_record[f"{resource}_allocated"] = (
                city_allocated[resource]
            )
            regional_record[f"{resource}_unmet"] = (
                city_demand[resource] - city_allocated[resource]
            )
            regional_record[f"{resource}_remaining"] = remaining[resource]

        regional_records.append(regional_record)

    ward_allocations = pd.DataFrame(allocation_records)
    regional_summary = pd.DataFrame(regional_records)
    crew_assignments = pd.DataFrame(crew_records)

    resource_records = []

    for resource in RESOURCE_COLUMNS:
        resource_records.append(
            {
                "resource": resource,
                "total_inventory": int(
                    regional_summary[f"{resource}_inventory"].sum()
                ),
                "total_available": int(
                    regional_summary[f"{resource}_available"].sum()
                ),
                "total_demand": int(
                    regional_summary[f"{resource}_demand"].sum()
                ),
                "total_allocated": int(
                    regional_summary[f"{resource}_allocated"].sum()
                ),
                "total_unmet": int(
                    regional_summary[f"{resource}_unmet"].sum()
                ),
                "total_remaining": int(
                    regional_summary[f"{resource}_remaining"].sum()
                ),
            }
        )

    resource_summary = pd.DataFrame(resource_records)

    return {
        "ward_allocations": ward_allocations,
        "regional_summary": regional_summary,
        "resource_summary": resource_summary,
        "crew_assignments": crew_assignments,
        "crew_summary": regional_summary[
            [
                "city",
                "spraying_crews_available",
                "spraying_crews_assigned",
                "spraying_crews_unassigned",
                "spraying_crews_unused",
                "wards_without_crews",
            ]
        ].copy(),
        "horizon_weeks": horizon_weeks,
        "stock_percentage": stock_percentage,
        "demand_parameters": parameters,
        "assumptions": {
            "beds_per_case": DEFAULT_BEDS_PER_CASE,
            "bed_demand_source": (
                "Historical operational assumption used by Hospital Surge"
            ),
            "bed_capacity_note": (
                "Estimated demand only; this is not actual hospital capacity."
            ),
        },
    }