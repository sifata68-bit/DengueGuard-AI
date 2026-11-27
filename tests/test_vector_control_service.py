
import pandas as pd
import pytest

from src import vector_control_service


def make_clean_data():
    return pd.DataFrame(
        [
            {
                "ward_id": "W1",
                "city": "Dhaka",
                "population_density_sqkm": 100,
            },
            {
                "ward_id": "W1",
                "city": "Dhaka",
                "population_density_sqkm": 100,
            },
            {
                "ward_id": "W2",
                "city": "Chattogram",
                "population_density_sqkm": 200,
            },
        ]
    )


def test_build_ward_information_returns_one_row_per_ward():
    result = vector_control_service.build_ward_information(
        make_clean_data()
    )

    assert len(result) == 2
    assert result["ward_id"].is_unique
    assert list(result.columns) == [
        "ward_id",
        "city",
        "population_density_sqkm",
    ]


def test_build_ward_information_rejects_missing_columns():
    data = pd.DataFrame(
        {
            "ward_id": ["W1"],
            "city": ["Dhaka"],
        }
    )

    with pytest.raises(ValueError, match="missing required ward columns"):
        vector_control_service.build_ward_information(data)


def test_build_ward_information_rejects_inconsistent_wards():
    data = pd.DataFrame(
        [
            {
                "ward_id": "W1",
                "city": "Dhaka",
                "population_density_sqkm": 100,
            },
            {
                "ward_id": "W1",
                "city": "Dhaka",
                "population_density_sqkm": 200,
            },
        ]
    )

    with pytest.raises(ValueError, match="consistent values"):
        vector_control_service.build_ward_information(data)


def test_build_ward_information_does_not_modify_input():
    data = make_clean_data()
    original = data.copy(deep=True)

    vector_control_service.build_ward_information(data)

    pd.testing.assert_frame_equal(data, original)


def test_run_simulation_uses_selected_reference_date(monkeypatch):
    expected_forecasts = pd.DataFrame(
        {
            "ward_id": ["W1"],
            "horizon_weeks": [2],
        }
    )

    expected_simulation = {
        "ward_results": pd.DataFrame(),
        "team_assignments": pd.DataFrame(),
        "roi_ranking": pd.DataFrame(),
        "summary": pd.DataFrame(),
        "assumptions": {},
    }

    captured = {}

    def fake_predict(data, as_of_date=None):
        captured["as_of_date"] = as_of_date
        return expected_forecasts

    def fake_simulate(**kwargs):
        captured["simulation"] = kwargs
        return expected_simulation

    monkeypatch.setattr(
        vector_control_service,
        "predict_all_wards",
        fake_predict,
    )
    monkeypatch.setattr(
        vector_control_service,
        "simulate_vector_control",
        fake_simulate,
    )

    result = vector_control_service.run_vector_control_simulation(
        clean_data=make_clean_data(),
        as_of_date="2025-12-29",
        selected_wards=["W1"],
        total_teams=1,
        intervention_type="larvicide",
        effectiveness_by_horizon={
            2: 0.10,
            3: 0.15,
            4: 0.20,
        },
    )

    assert captured["as_of_date"] == "2025-12-29"
    assert result["forecasts"] is expected_forecasts
    assert result["ward_results"] is expected_simulation["ward_results"]
    assert captured["simulation"]["selected_wards"] == ["W1"]


def test_run_simulation_requires_reference_date():
    with pytest.raises(ValueError, match="reference date"):
        vector_control_service.run_vector_control_simulation(
            clean_data=make_clean_data(),
            as_of_date=None,
            selected_wards=["W1"],
            total_teams=1,
            intervention_type="larvicide",
            effectiveness_by_horizon={
                2: 0.10,
                3: 0.15,
                4: 0.20,
            },
        )
