
import pandas as pd
import pytest

from src.hospital_surge import (
    DEFAULT_BEDS_PER_CASE,
    assess_hospital_surge,
    generate_overflow_plan,
)


@pytest.fixture
def forecasts():
    return pd.DataFrame(
        {
            "ward_id": ["W1", "W2", "W3"],
            "city": ["Dhaka", "Dhaka", "Dhaka"],
            "horizon_weeks": [2, 2, 2],
            "predicted_cases": [40, 20, 10],
        }
    )


@pytest.fixture
def capacity():
    return pd.DataFrame(
        {
            "ward_id": ["W1", "W2", "W3"],
            "city": ["Dhaka", "Dhaka", "Dhaka"],
            "capacity_beds": [8, 10, 3],
        }
    )


def test_default_bed_conversion_ratio():
    assert DEFAULT_BEDS_PER_CASE == 0.253


def test_bed_demand_uses_ceiling(forecasts, capacity):
    result = assess_hospital_surge(forecasts, capacity)

    assert result["estimated_bed_demand"].tolist() == [11, 6, 3]


def test_overloaded_ward_is_identified(forecasts, capacity):
    result = assess_hospital_surge(forecasts, capacity)

    ward = result.set_index("ward_id").loc["W1"]

    assert ward["surge_status"] == "Overloaded"
    assert ward["excess_beds"] == 3
    assert ward["spare_beds"] == 0


def test_near_capacity_ward_is_identified(forecasts, capacity):
    result = assess_hospital_surge(forecasts, capacity)

    ward = result.set_index("ward_id").loc["W3"]

    assert ward["surge_status"] == "Near Capacity"
    assert ward["capacity_utilization"] == 1.0


def test_normal_ward_is_identified(forecasts, capacity):
    result = assess_hospital_surge(forecasts, capacity)

    ward = result.set_index("ward_id").loc["W2"]

    assert ward["surge_status"] == "Normal"
    assert ward["spare_beds"] == 4


def test_overflow_direction_is_correct(forecasts, capacity):
    assessment = assess_hospital_surge(forecasts, capacity)
    plan = generate_overflow_plan(assessment)

    transfer = plan.iloc[0]

    assert transfer["from_ward"] == "W1"
    assert transfer["to_ward"] == "W2"
    assert transfer["beds_to_transfer"] == 3
    assert transfer["status"] == "Transfer recommended"


def test_overflow_plan_does_not_exceed_donor_capacity(
    forecasts, capacity
):
    assessment = assess_hospital_surge(forecasts, capacity)
    plan = generate_overflow_plan(assessment)

    donor_spare = assessment.set_index("ward_id")["spare_beds"]

    transfers = plan[plan["status"] == "Transfer recommended"]

    transferred = transfers.groupby("to_ward")[
        "beds_to_transfer"
    ].sum()

    for ward_id, amount in transferred.items():
        assert amount <= donor_spare[ward_id]


def test_overflow_plan_does_not_exceed_receiver_demand(
    forecasts, capacity
):
    assessment = assess_hospital_surge(forecasts, capacity)
    plan = generate_overflow_plan(assessment)

    receiver_demand = assessment.set_index("ward_id")["excess_beds"]

    transfers = plan[plan["status"] == "Transfer recommended"]

    transferred = transfers.groupby("from_ward")[
        "beds_to_transfer"
    ].sum()

    for ward_id, amount in transferred.items():
        assert amount <= receiver_demand[ward_id]


def test_unmet_demand_is_reported():
    forecast = pd.DataFrame(
        {
            "ward_id": ["W1", "W2"],
            "city": ["Dhaka", "Dhaka"],
            "horizon_weeks": [2, 2],
            "predicted_cases": [40, 10],
        }
    )

    capacities = pd.DataFrame(
        {
            "ward_id": ["W1", "W2"],
            "city": ["Dhaka", "Dhaka"],
            "capacity_beds": [5, 3],
        }
    )

    assessment = assess_hospital_surge(forecast, capacities)
    plan = generate_overflow_plan(assessment)

    unmet = plan[plan["status"] == "Unmet demand"]

    assert len(unmet) == 1
    assert unmet.iloc[0]["from_ward"] == "W1"
    assert unmet.iloc[0]["unmet_beds"] == 6


def test_no_overflow_plan_when_all_wards_have_capacity(
    capacity,
):
    forecast = pd.DataFrame(
        {
            "ward_id": ["W1", "W2", "W3"],
            "city": ["Dhaka"] * 3,
            "horizon_weeks": [2] * 3,
            "predicted_cases": [5, 5, 5],
        }
    )

    assessment = assess_hospital_surge(forecast, capacity)
    plan = generate_overflow_plan(assessment)

    assert plan.empty



def test_transfers_stay_within_same_city():
    forecast = pd.DataFrame(
        {
            "ward_id": ["W1", "W2"],
            "city": ["Dhaka", "Chattogram"],
            "horizon_weeks": [2, 2],
            "predicted_cases": [40, 20],
        }
    )

    capacities = pd.DataFrame(
        {
            "ward_id": ["W1", "W2"],
            "city": ["Dhaka", "Chattogram"],
            "capacity_beds": [5, 20],
        }
    )

    assessment = assess_hospital_surge(forecast, capacities)
    plan = generate_overflow_plan(assessment)

    transfers = plan[
        plan["status"] == "Transfer recommended"
    ]

    assert transfers.empty
    assert plan["status"].eq("Unmet demand").all()


def test_multiple_horizons_are_allocated_independently():
    forecast = pd.DataFrame(
        {
            "ward_id": ["W1", "W2", "W1", "W2"],
            "city": ["Dhaka"] * 4,
            "horizon_weeks": [2, 2, 3, 3],
            "predicted_cases": [40, 10, 10, 40],
        }
    )

    capacities = pd.DataFrame(
        {
            "ward_id": ["W1", "W2"],
            "city": ["Dhaka", "Dhaka"],
            "capacity_beds": [5, 5],
        }
    )

    assessment = assess_hospital_surge(forecast, capacities)
    plan = generate_overflow_plan(assessment)

    transfers = plan[plan["status"] == "Transfer recommended"]

    h2 = transfers[transfers["horizon_weeks"] == 2]
    h3 = transfers[transfers["horizon_weeks"] == 3]

    assert h2.iloc[0]["from_ward"] == "W1"
    assert h2.iloc[0]["to_ward"] == "W2"

    assert h3.iloc[0]["from_ward"] == "W2"
    assert h3.iloc[0]["to_ward"] == "W1"


def test_missing_capacity_is_rejected(forecasts):
    incomplete = pd.DataFrame(
        {
            "ward_id": ["W1"],
            "city": ["Dhaka"],
            "capacity_beds": [10],
        }
    )

    with pytest.raises(ValueError, match="Missing hospital capacity"):
        assess_hospital_surge(forecasts, incomplete)


def test_negative_predicted_cases_are_rejected(forecasts, capacity):
    forecasts.loc[0, "predicted_cases"] = -1

    with pytest.raises(ValueError, match="negative"):
        assess_hospital_surge(forecasts, capacity)


def test_negative_capacity_is_rejected(forecasts, capacity):
    capacity.loc[0, "capacity_beds"] = -1

    with pytest.raises(ValueError, match="negative"):
        assess_hospital_surge(forecasts, capacity)



def test_fractional_capacity_is_rejected(forecasts, capacity):
    capacity["capacity_beds"] = capacity[
        "capacity_beds"
    ].astype(float)

    capacity.loc[0, "capacity_beds"] = 2.5

    with pytest.raises(ValueError, match="whole numbers"):
        assess_hospital_surge(forecasts, capacity)



def test_duplicate_forecast_rows_are_rejected(forecasts, capacity):
    duplicate = pd.concat(
        [forecasts, forecasts.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="one row per ward"):
        assess_hospital_surge(duplicate, capacity)


def test_invalid_bed_conversion_ratio_is_rejected(forecasts, capacity):
    with pytest.raises(ValueError):
        assess_hospital_surge(
            forecasts,
            capacity,
            beds_per_case=0,
        )


def test_invalid_threshold_is_rejected(forecasts, capacity):
    with pytest.raises(ValueError):
        assess_hospital_surge(
            forecasts,
            capacity,
            near_capacity_threshold=1.5,
        )


def test_zero_capacity_with_positive_demand_is_overloaded():
    forecast = pd.DataFrame(
        {
            "ward_id": ["W1"],
            "city": ["Dhaka"],
            "horizon_weeks": [2],
            "predicted_cases": [10],
        }
    )

    capacities = pd.DataFrame(
        {
            "ward_id": ["W1"],
            "city": ["Dhaka"],
            "capacity_beds": [0],
        }
    )

    result = assess_hospital_surge(forecast, capacities)

    assert result.iloc[0]["surge_status"] == "Overloaded"
    assert result.iloc[0]["capacity_utilization"] == float("inf")


def test_inconsistent_overflow_assessment_is_rejected():
    invalid = pd.DataFrame(
        {
            "ward_id": ["W1"],
            "city": ["Dhaka"],
            "horizon_weeks": [2],
            "capacity_beds": [10],
            "estimated_bed_demand": [5],
            "spare_beds": [2],
            "excess_beds": [0],
        }
    )

    with pytest.raises(ValueError, match="Spare bed"):
        generate_overflow_plan(invalid)


def test_input_data_is_not_modified(forecasts, capacity):
    original_forecasts = forecasts.copy(deep=True)
    original_capacity = capacity.copy(deep=True)

    assess_hospital_surge(forecasts, capacity)

    pd.testing.assert_frame_equal(forecasts, original_forecasts)
    pd.testing.assert_frame_equal(capacity, original_capacity)
