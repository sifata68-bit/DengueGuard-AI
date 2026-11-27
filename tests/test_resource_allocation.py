
import copy

import pandas as pd
import pytest

from src.resource_allocation import allocate_resources


def make_inputs():
    forecasts = pd.DataFrame(
        {
            "ward_id": ["W1", "W2", "W3"],
            "city": ["Dhaka", "Dhaka", "Dhaka"],
            "date": pd.to_datetime(
                ["2025-12-29", "2025-12-29", "2025-12-29"]
            ),
            "horizon_weeks": [2, 2, 2],
            "predicted_cases": [10, 8, 5],
            "risk_category": ["Critical", "High", "Medium"],
        }
    )

    ward_information = pd.DataFrame(
        {
            "ward_id": ["W1", "W2", "W3"],
            "city": ["Dhaka", "Dhaka", "Dhaka"],
            "population_density_sqkm": [1000, 3000, 2000],
        }
    )

    regional_inventory = pd.DataFrame(
        {
            "city": ["Dhaka"],
            "saline_bags": [30],
            "diagnostic_kits": [20],
        }
    )

    demand_parameters = {
        "saline_bags": {
            "per_case": 2,
            "source": "Test-only supplied parameter",
            "status": "user_supplied",
        },
        "diagnostic_kits": {
            "per_case": 1,
            "source": "Test-only supplied parameter",
            "status": "user_supplied",
        },
    }

    return (
        forecasts,
        ward_information,
        regional_inventory,
        demand_parameters,
    )


def run_allocation(inputs=None, **overrides):
    if inputs is None:
        inputs = make_inputs()

    forecasts, ward_information, inventory, parameters = inputs

    arguments = {
        "forecasts": forecasts,
        "ward_information": ward_information,
        "regional_inventory": inventory,
        "stock_percentage": 100,
        "demand_parameters": parameters,
        "horizon_weeks": 2,
    }

    arguments.update(overrides)

    return allocate_resources(**arguments)


def test_allocation_never_exceeds_available_inventory():
    result = run_allocation()

    allocations = result["ward_allocations"]
    summary = result["regional_summary"]

    assert allocations["saline_bags_allocated"].sum() == 30
    assert allocations["diagnostic_kits_allocated"].sum() == 20

    assert (
        summary["saline_bags_allocated"]
        <= summary["saline_bags_available"]
    ).all()

    assert (
        summary["diagnostic_kits_allocated"]
        <= summary["diagnostic_kits_available"]
    ).all()


def test_critical_risk_ward_receives_highest_priority():
    result = run_allocation()

    allocations = result["ward_allocations"]

    assert allocations.iloc[0]["ward_id"] == "W1"
    assert allocations.iloc[0]["risk_category"] == "Critical"


def test_population_density_breaks_risk_ties():
    inputs = list(make_inputs())

    forecasts = inputs[0].copy()
    forecasts["risk_category"] = "Critical"

    inputs[0] = forecasts

    result = run_allocation(tuple(inputs))

    allocations = result["ward_allocations"]

    assert allocations["ward_id"].tolist() == [
        "W2",
        "W3",
        "W1",
    ]


def test_ward_id_breaks_remaining_priority_ties():
    inputs = list(make_inputs())

    forecasts = inputs[0].copy()
    forecasts["risk_category"] = "Critical"

    ward_information = inputs[1].copy()
    ward_information["population_density_sqkm"] = 1000

    inputs[0] = forecasts
    inputs[1] = ward_information

    result = run_allocation(tuple(inputs))

    allocations = result["ward_allocations"]

    assert allocations["ward_id"].tolist() == [
        "W1",
        "W2",
        "W3",
    ]


def test_unmet_demand_is_reported():
    result = run_allocation()

    allocations = result["ward_allocations"]

    assert allocations["saline_bags_unmet"].sum() == 16
    assert allocations["diagnostic_kits_unmet"].sum() == 3


def test_stock_percentage_reduces_available_inventory():
    result = run_allocation(stock_percentage=50)

    summary = result["regional_summary"].iloc[0]

    assert summary["saline_bags_available"] == 15
    assert summary["diagnostic_kits_available"] == 10


@pytest.mark.parametrize(
    "percentage",
    [29, 101, 0, -10, float("nan"), float("inf")],
)
def test_invalid_stock_percentage_is_rejected(percentage):
    with pytest.raises(ValueError):
        run_allocation(stock_percentage=percentage)


def test_missing_demand_parameter_is_rejected():
    inputs = list(make_inputs())
    inputs[3] = {
        "saline_bags": inputs[3]["saline_bags"]
    }

    with pytest.raises(ValueError):
        run_allocation(tuple(inputs))


def test_missing_parameter_source_is_rejected():
    inputs = list(make_inputs())
    parameters = copy.deepcopy(inputs[3])
    del parameters["saline_bags"]["source"]
    inputs[3] = parameters

    with pytest.raises(ValueError):
        run_allocation(tuple(inputs))


def test_invalid_parameter_status_is_rejected():
    inputs = list(make_inputs())
    parameters = copy.deepcopy(inputs[3])
    parameters["saline_bags"]["status"] = "unknown"
    inputs[3] = parameters

    with pytest.raises(ValueError):
        run_allocation(tuple(inputs))


def test_forecast_city_mismatch_is_rejected():
    inputs = list(make_inputs())

    forecasts = inputs[0].copy()
    forecasts.loc[0, "city"] = "Chattogram"
    inputs[0] = forecasts

    with pytest.raises(ValueError):
        run_allocation(tuple(inputs))


def test_missing_city_inventory_is_rejected():
    inputs = list(make_inputs())

    forecasts = inputs[0].copy()
    forecasts.loc[0, "city"] = "Chattogram"

    ward_information = inputs[1].copy()
    ward_information.loc[0, "city"] = "Chattogram"

    inputs[0] = forecasts
    inputs[1] = ward_information

    with pytest.raises(ValueError):
        run_allocation(tuple(inputs))


def test_only_selected_horizon_is_allocated():
    inputs = list(make_inputs())

    additional = inputs[0].copy()
    additional["horizon_weeks"] = 3

    inputs[0] = pd.concat(
        [inputs[0], additional],
        ignore_index=True,
    )

    result = run_allocation(tuple(inputs))

    assert result["ward_allocations"]["horizon_weeks"].eq(2).all()
    assert len(result["ward_allocations"]) == 3


def test_original_inputs_are_not_modified():
    inputs = make_inputs()

    original = [
        dataframe.copy(deep=True)
        for dataframe in inputs[:3]
    ]

    run_allocation(inputs)

    for before, after in zip(original, inputs[:3]):
        pd.testing.assert_frame_equal(before, after)


def test_inventory_is_allocated_separately_by_city():
    inputs = list(make_inputs())

    forecasts = inputs[0].copy()
    ward_information = inputs[1].copy()

    forecasts.loc[2, "city"] = "Chattogram"
    ward_information.loc[2, "city"] = "Chattogram"

    inventory = pd.DataFrame(
        {
            "city": ["Dhaka", "Chattogram"],
            "saline_bags": [30, 10],
            "diagnostic_kits": [20, 5],
        }
    )

    inputs[0] = forecasts
    inputs[1] = ward_information
    inputs[2] = inventory

    result = run_allocation(tuple(inputs))

    summary = result["regional_summary"].set_index("city")

    assert summary.loc["Dhaka", "saline_bags_allocated"] <= 30
    assert summary.loc["Chattogram", "saline_bags_allocated"] <= 10

    assert summary.loc["Chattogram", "saline_bags_allocated"] == 10

def test_crew_assignments_respect_city_limits():
    result = run_allocation(spraying_crews_by_city={"Dhaka": 1})
    crews = result["crew_assignments"]
    assert crews["crew_assigned"].sum() == 1
    assert crews.loc[crews["crew_assigned"] == 1, "ward_id"].tolist() == ["W1"]


def test_critical_ward_gets_crew_before_lower_risk_wards():
    result = run_allocation(spraying_crews_by_city={"Dhaka": 1})
    crews = result["crew_assignments"]
    assert crews.iloc[0]["risk_category"] == "Critical"
    assert crews.iloc[0]["assignment_status"] == "Assigned"
    assert crews.iloc[1]["assignment_status"] == "Unassigned"


def test_population_density_breaks_crew_priority_ties():
    inputs = list(make_inputs())
    inputs[0]["risk_category"] = "Critical"
    result = run_allocation(tuple(inputs), spraying_crews_by_city={"Dhaka": 1})
    crews = result["crew_assignments"]
    assert crews.iloc[0]["ward_id"] == "W2"
    assert crews.iloc[0]["crew_assigned"] == 1


def test_hospital_bed_demand_uses_shared_ratio():
    result = run_allocation()
    allocations = result["ward_allocations"]
    assert allocations["hospital_beds_required"].tolist() == [3, 3, 2]
    assert result["assumptions"]["beds_per_case"] == 0.253


def test_crew_limits_are_independent_by_city():
    inputs = list(make_inputs())
    inputs[0].loc[2, "city"] = "Chattogram"
    inputs[1].loc[2, "city"] = "Chattogram"
    inputs[2] = pd.DataFrame({
        "city": ["Dhaka", "Chattogram"],
        "saline_bags": [30, 10],
        "diagnostic_kits": [20, 5],
    })
    result = run_allocation(
        tuple(inputs),
        spraying_crews_by_city={"Dhaka": 1, "Chattogram": 1},
    )
    summary = result["crew_summary"].set_index("city")
    assert summary.loc["Dhaka", "spraying_crews_assigned"] == 1
    assert summary.loc["Chattogram", "spraying_crews_assigned"] == 1


@pytest.mark.parametrize(
    "crew_limits",
    [
        {"Dhaka": -1},
        {"Dhaka": 1.5},
        {"Dhaka": float("inf")},
        {"Dhaka": float("nan")},
        {"Dhaka": True},
        {},
        {"Dhaka": 1, "Chattogram": 2},
    ],
)
def test_invalid_crew_limits_are_rejected(crew_limits):
    with pytest.raises((ValueError, TypeError)):
        run_allocation(spraying_crews_by_city=crew_limits)


def test_crew_assignments_report_unassigned_wards():
    result = run_allocation(spraying_crews_by_city={"Dhaka": 1})
    crews = result["crew_assignments"]
    assert (crews["assignment_status"] == "Unassigned").sum() == 2
    assert result["crew_summary"].iloc[0]["wards_without_crews"] == 2

def test_exact_challenge_demand_fields_match_calculated_demand():
    result = run_allocation()
    allocations = result["ward_allocations"]

    assert allocations["saline_bags_needed"].tolist() == [20, 16, 10]
    assert allocations["diagnostic_kits_needed"].tolist() == [10, 8, 5]


def test_unused_crew_summary_is_reported():
    result = run_allocation(spraying_crews_by_city={"Dhaka": 5})
    summary = result["crew_summary"].iloc[0]

    assert summary["spraying_crews_unused"] == 2
    assert summary["spraying_crews_unassigned"] == 2
