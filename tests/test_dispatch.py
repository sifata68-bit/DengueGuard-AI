import pandas as pd
import pytest

from src.dispatch import (
    build_dispatch_notices,
    render_dispatch_html,
    save_dispatch_html,
)
from src.resource_allocation import allocate_resources


def make_allocation_result():
    forecasts = pd.DataFrame(
        {
            "ward_id": ["W1", "W2"],
            "city": ["Dhaka", "Dhaka"],
            "date": pd.to_datetime(["2025-12-29", "2025-12-29"]),
            "horizon_weeks": [2, 2],
            "predicted_cases": [10, 4],
            "risk_category": ["Critical", "Medium"],
        }
    )

    ward_information = pd.DataFrame(
        {
            "ward_id": ["W1", "W2"],
            "city": ["Dhaka", "Dhaka"],
            "population_density_sqkm": [2000, 1000],
        }
    )

    inventory = pd.DataFrame(
        {
            "city": ["Dhaka"],
            "saline_bags": [15],
            "diagnostic_kits": [8],
        }
    )

    parameters = {
        "saline_bags": {
            "per_case": 2,
            "source": "Scenario input for tests",
            "status": "scenario_assumption",
        },
        "diagnostic_kits": {
            "per_case": 1,
            "source": "Scenario input for tests",
            "status": "scenario_assumption",
        },
    }

    return allocate_resources(
        forecasts=forecasts,
        ward_information=ward_information,
        regional_inventory=inventory,
        stock_percentage=100,
        demand_parameters=parameters,
        horizon_weeks=2,
    )


def test_builds_one_notice_per_ward():
    result = build_dispatch_notices(make_allocation_result())

    assert result["ward_id"].tolist() == ["W1", "W2"]
    assert len(result) == 2


def test_message_contains_forecast_risk_and_supply_shortages():
    result = build_dispatch_notices(make_allocation_result())
    message = result.loc[result["ward_id"] == "W1", "message"].iloc[0]

    assert "10.00 cases" in message
    assert "Critical risk" in message
    assert "Saline bags:" in message
    assert "Diagnostic kits:" in message
    assert "unmet" in message


def test_missing_team_assignment_is_explicit():
    result = build_dispatch_notices(make_allocation_result())

    assert result["assigned_teams"].eq("No team assigned").all()
    assert result["message"].str.contains("No team assigned").all()


def test_supplied_team_assignments_are_included():
    assignments = pd.DataFrame(
        {
            "ward_id": ["W1", "W2", "W1"],
            "team_name": ["North Team", "South Team", "Backup Team"],
        }
    )

    result = build_dispatch_notices(
        make_allocation_result(),
        team_assignments=assignments,
    )

    ward_one = result.loc[result["ward_id"] == "W1"].iloc[0]

    assert ward_one["assigned_teams"] == "North Team, Backup Team"
    assert "North Team, Backup Team" in ward_one["message"]


def test_unknown_team_assignment_ward_is_rejected():
    assignments = pd.DataFrame(
        {"ward_id": ["W99"], "team_name": ["Unknown Team"]}
    )

    with pytest.raises(ValueError, match="unknown wards"):
        build_dispatch_notices(
            make_allocation_result(),
            team_assignments=assignments,
        )


def test_invalid_team_assignment_columns_are_rejected():
    assignments = pd.DataFrame({"ward_id": ["W1"], "team": ["Team A"]})

    with pytest.raises(ValueError, match="team_name"):
        build_dispatch_notices(
            make_allocation_result(),
            team_assignments=assignments,
        )


def test_duplicate_ward_allocations_are_rejected():
    allocation = make_allocation_result()
    allocation["ward_allocations"] = pd.concat(
        [allocation["ward_allocations"], allocation["ward_allocations"].iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="one row per ward"):
        build_dispatch_notices(allocation)


def test_invalid_horizon_is_rejected():
    allocation = make_allocation_result()
    allocation["horizon_weeks"] = 5

    with pytest.raises(ValueError, match="2, 3, or 4"):
        build_dispatch_notices(allocation)


def test_negative_allocation_value_is_rejected():
    allocation = make_allocation_result()
    allocation["ward_allocations"].loc[0, "saline_bags_unmet"] = -1

    with pytest.raises(ValueError, match="cannot contain negative values"):
        build_dispatch_notices(allocation)

def test_html_report_contains_notice_and_parameter_provenance():
    notices = build_dispatch_notices(make_allocation_result())
    document = render_dispatch_html(notices)

    assert "DengueGuard AI Ward Dispatch Report" in document
    assert "W1" in document
    assert "scenario_assumption" in document
    assert "Scenario input for tests" in document


def test_html_report_escapes_untrusted_text():
    allocation = make_allocation_result()
    allocation["ward_allocations"].loc[0, "city"] = "<script>alert(1)</script>"

    notices = build_dispatch_notices(allocation)
    document = render_dispatch_html(notices)

    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;" in document


def test_html_export_writes_file_without_network_access(tmp_path):
    notices = build_dispatch_notices(make_allocation_result())
    output = tmp_path / "reports" / "dispatch.html"

    saved_path = save_dispatch_html(notices, output)

    assert saved_path == output
    assert output.exists()
    assert "Ward dispatch notices" in output.read_text(encoding="utf-8")


def test_input_allocation_dataframe_is_not_modified():
    allocation = make_allocation_result()
    original = allocation["ward_allocations"].copy(deep=True)

    build_dispatch_notices(allocation)

    pd.testing.assert_frame_equal(
        allocation["ward_allocations"],
        original,
    )
