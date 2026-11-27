
import numpy as np
import pandas as pd
import pytest

from src.vector_control import simulate_vector_control


def make_forecasts():
    rows = []

    ward_data = {
        "W1": (10, "Medium"),
        "W2": (20, "Critical"),
        "W3": (30, "High"),
    }

    for ward_id, (cases, risk) in ward_data.items():
        for horizon in (2, 3, 4):
            rows.append(
                {
                    "ward_id": ward_id,
                    "horizon_weeks": horizon,
                    "predicted_cases": cases,
                    "risk_category": risk,
                }
            )

    return pd.DataFrame(rows)


def make_ward_information():
    return pd.DataFrame(
        [
            {
                "ward_id": "W1",
                "city": "Dhaka",
                "population_density_sqkm": 100,
            },
            {
                "ward_id": "W2",
                "city": "Dhaka",
                "population_density_sqkm": 50,
            },
            {
                "ward_id": "W3",
                "city": "Dhaka",
                "population_density_sqkm": 200,
            },
        ]
    )


def run_simulation(
    forecasts=None,
    ward_information=None,
    selected_wards=None,
    total_teams=1,
    effectiveness=None,
    **kwargs,
):
    return simulate_vector_control(
        forecasts=(
            make_forecasts() if forecasts is None else forecasts
        ),
        ward_information=(
            make_ward_information()
            if ward_information is None
            else ward_information
        ),
        selected_wards=(
            ["W1", "W2", "W3"]
            if selected_wards is None
            else selected_wards
        ),
        total_teams=total_teams,
        intervention_type="larvicide",
        effectiveness_by_horizon=(
            {2: 0.2, 3: 0.2, 4: 0.2}
            if effectiveness is None
            else effectiveness
        ),
        **kwargs,
    )


def test_critical_ward_gets_priority():
    result = run_simulation(total_teams=1)

    assigned = result["team_assignments"].loc[
        result["team_assignments"]["team_assigned"],
        "ward_id",
    ].tolist()

    assert assigned == ["W2"]


def test_population_density_breaks_equal_risk_priority():
    result = run_simulation(total_teams=2)

    assigned = result["team_assignments"].loc[
        result["team_assignments"]["team_assigned"],
        "ward_id",
    ].tolist()

    assert assigned == ["W2", "W3"]


def test_zero_teams_assigns_no_wards():
    result = run_simulation(total_teams=0)

    assert result["team_assignments"]["team_assigned"].sum() == 0
    assert result["assumptions"]["teams_deployed"] == 0
    assert result["ward_results"]["cases_prevented"].sum() == 0


def test_team_allocation_uses_highest_risk_across_horizons():
    forecasts = make_forecasts()

    forecasts.loc[
        (forecasts["ward_id"] == "W1")
        & (forecasts["horizon_weeks"] == 4),
        "risk_category",
    ] = "Critical"

    result = run_simulation(
        forecasts=forecasts,
        total_teams=1,
    )

    assigned = result["team_assignments"].loc[
        result["team_assignments"]["team_assigned"],
        "ward_id",
    ].tolist()

    assert assigned == ["W1"]


def test_only_assigned_wards_receive_intervention_effect():
    result = run_simulation(total_teams=1)

    ward_results = result["ward_results"]

    assigned_results = ward_results[
        ward_results["ward_id"] == "W2"
    ]

    unassigned_results = ward_results[
        ward_results["ward_id"] != "W2"
    ]

    assert (assigned_results["cases_prevented"] > 0).all()
    assert (unassigned_results["cases_prevented"] == 0).all()


def test_scenario_cases_use_effectiveness_assumptions():
    result = run_simulation(total_teams=1)

    assigned = result["ward_results"].query(
        "ward_id == 'W2'"
    )

    np.testing.assert_allclose(
        assigned["scenario_cases"],
        [16, 16, 16],
    )

    np.testing.assert_allclose(
        assigned["cases_prevented"],
        [4, 4, 4],
    )


def test_resource_savings_are_calculated():
    forecasts = pd.DataFrame(
        [
            {
                "ward_id": "W1",
                "horizon_weeks": horizon,
                "predicted_cases": 20,
                "risk_category": "Critical",
            }
            for horizon in (2, 3, 4)
        ]
    )

    result = run_simulation(
        forecasts=forecasts,
        selected_wards=["W1"],
        total_teams=1,
        effectiveness={2: 0.5, 3: 0.5, 4: 0.5},
        saline_per_case=2,
        beds_per_case=0.253,
        diagnostic_kits_per_case=1,
    )

    rows = result["ward_results"]

    assert (rows["cases_prevented"] == 10).all()
    assert (rows["saline_saved"] == 20).all()
    assert (rows["diagnostic_kits_saved"] == 10).all()
    assert (rows["beds_avoided"] == 3).all()


def test_roi_is_reported_separately_for_each_horizon():
    result = run_simulation(total_teams=1)

    roi = result["roi_ranking"]

    assert set(roi["horizon_weeks"]) == {2, 3, 4}
    assert len(roi) == 9

    for horizon in (2, 3, 4):
        horizon_roi = roi[
            roi["horizon_weeks"] == horizon
        ]

        assert horizon_roi.iloc[0]["ward_id"] == "W2"
        assert horizon_roi.iloc[0][
            "roi_cases_prevented_per_crew"
        ] == pytest.approx(4)


def test_summary_contains_all_forecast_horizons():
    result = run_simulation()

    summary = result["summary"]

    assert summary["horizon_weeks"].tolist() == [2, 3, 4]
    assert len(summary) == 3


def test_unknown_selected_ward_is_rejected():
    with pytest.raises(ValueError, match="Unknown selected wards"):
        run_simulation(selected_wards=["UNKNOWN"])


def test_missing_forecast_horizon_is_rejected():
    forecasts = make_forecasts()

    forecasts = forecasts[
        ~(
            (forecasts["ward_id"] == "W1")
            & (forecasts["horizon_weeks"] == 4)
        )
    ]

    with pytest.raises(ValueError, match="must have forecasts"):
        run_simulation(forecasts=forecasts)


def test_duplicate_forecast_is_rejected():
    forecasts = make_forecasts()

    forecasts = pd.concat(
        [forecasts, forecasts.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="one row per ward"):
        run_simulation(forecasts=forecasts)


@pytest.mark.parametrize(
    "effectiveness",
    [
        {2: 0.2, 3: 0.2},
        {2: 0.2, 3: 0.2, 4: 1.5},
        {2: 0.2, 3: 0.2, 4: -0.1},
    ],
)
def test_invalid_effectiveness_is_rejected(effectiveness):
    with pytest.raises((ValueError, TypeError)):
        run_simulation(effectiveness=effectiveness)


@pytest.mark.parametrize("teams", [-1, 1.5, True])
def test_invalid_team_count_is_rejected(teams):
    with pytest.raises(ValueError):
        run_simulation(total_teams=teams)


def test_negative_predicted_cases_are_rejected():
    forecasts = make_forecasts()
    forecasts.loc[0, "predicted_cases"] = -5

    with pytest.raises(ValueError, match="nonnegative"):
        run_simulation(forecasts=forecasts)


def test_invalid_intervention_type_is_rejected():
    with pytest.raises(ValueError, match="intervention_type"):
        simulate_vector_control(
            forecasts=make_forecasts(),
            ward_information=make_ward_information(),
            selected_wards=["W1", "W2", "W3"],
            total_teams=1,
            intervention_type="invalid",
            effectiveness_by_horizon={
                2: 0.2,
                3: 0.2,
                4: 0.2,
            },
        )


def test_input_dataframes_are_not_modified():
    forecasts = make_forecasts()
    ward_information = make_ward_information()

    original_forecasts = forecasts.copy(deep=True)
    original_wards = ward_information.copy(deep=True)

    run_simulation(
        forecasts=forecasts,
        ward_information=ward_information,
    )

    pd.testing.assert_frame_equal(forecasts, original_forecasts)
    pd.testing.assert_frame_equal(
        ward_information,
        original_wards,
    )
