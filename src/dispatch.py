from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


RISK_CATEGORIES = {"Low", "Medium", "High", "Critical"}
RESOURCE_FIELDS = {
    "saline_bags": ("saline_bags_demand", "saline_bags_allocated", "saline_bags_unmet"),
    "diagnostic_kits": (
        "diagnostic_kits_demand",
        "diagnostic_kits_allocated",
        "diagnostic_kits_unmet",
    ),
}
ALLOWED_PARAMETER_STATUSES = {
    "verified",
    "user_supplied",
    "scenario_assumption",
}


def _validate_allocation_result(allocation_result):
    if not isinstance(allocation_result, dict):
        raise TypeError("allocation_result must be a dictionary.")

    required_keys = {"ward_allocations", "horizon_weeks", "demand_parameters"}
    missing_keys = required_keys - set(allocation_result)

    if missing_keys:
        raise ValueError(
            f"allocation_result is missing keys: {sorted(missing_keys)}"
        )

    horizon = allocation_result["horizon_weeks"]

    if isinstance(horizon, bool) or horizon not in (2, 3, 4):
        raise ValueError("Allocation horizon must be 2, 3, or 4.")

    allocations = allocation_result["ward_allocations"]

    if not isinstance(allocations, pd.DataFrame):
        raise TypeError("ward_allocations must be a pandas DataFrame.")

    required_columns = {
        "city",
        "ward_id",
        "date",
        "horizon_weeks",
        "predicted_cases",
        "risk_category",
    }

    for fields in RESOURCE_FIELDS.values():
        required_columns.update(fields)

    missing_columns = required_columns - set(allocations.columns)

    if missing_columns:
        raise ValueError(
            f"ward_allocations is missing columns: {sorted(missing_columns)}"
        )

    if allocations.empty:
        raise ValueError("ward_allocations cannot be empty.")

    if allocations["ward_id"].isna().any():
        raise ValueError("Ward IDs cannot be missing.")

    if allocations["ward_id"].duplicated().any():
        raise ValueError("Dispatch input must contain one row per ward.")

    if allocations["city"].isna().any():
        raise ValueError("City values cannot be missing.")

    if allocations["date"].isna().any():
        raise ValueError("Forecast dates cannot be missing.")

    if not allocations["horizon_weeks"].eq(horizon).all():
        raise ValueError(
            "All allocation rows must match the allocation horizon."
        )

    if not allocations["risk_category"].isin(RISK_CATEGORIES).all():
        raise ValueError("Allocation contains an invalid risk category.")

    numeric_columns = ["predicted_cases"]

    for fields in RESOURCE_FIELDS.values():
        numeric_columns.extend(fields)

    prepared = allocations.copy()

    for column in numeric_columns:
        values = pd.to_numeric(prepared[column], errors="coerce")

        if values.isna().any():
            raise ValueError(f"{column} must contain numeric values.")

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain finite values.")

        if (values < 0).any():
            raise ValueError(f"{column} cannot contain negative values.")

        if column != "predicted_cases" and (values % 1 != 0).any():
            raise ValueError(f"{column} must contain whole numbers.")

        prepared[column] = values

    prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")

    if prepared["date"].isna().any():
        raise ValueError("Forecast dates must be valid dates.")

    parameters = allocation_result["demand_parameters"]

    if not isinstance(parameters, dict):
        raise TypeError("demand_parameters must be a dictionary.")

    for resource in RESOURCE_FIELDS:
        if resource not in parameters:
            raise ValueError(f"Missing demand parameters for {resource}.")

        item = parameters[resource]

        if not isinstance(item, dict):
            raise TypeError(f"Parameters for {resource} must be a dictionary.")

        if not {"per_case", "source", "status"}.issubset(item):
            raise ValueError(
                f"Parameters for {resource} require per_case, source, and status."
            )

        try:
            per_case = float(item["per_case"])
        except (TypeError, ValueError):
            raise ValueError(f"{resource} per_case must be numeric.") from None

        if not np.isfinite(per_case) or per_case < 0:
            raise ValueError(
                f"{resource} per_case must be finite and nonnegative."
            )

        if not isinstance(item["source"], str) or not item["source"].strip():
            raise ValueError(f"{resource} source must be a nonempty string.")

        if item["status"] not in ALLOWED_PARAMETER_STATUSES:
            raise ValueError(f"{resource} has an invalid provenance status.")

    return prepared, parameters, horizon


def _prepare_team_assignments(team_assignments, ward_ids):
    if team_assignments is None:
        return {}

    if not isinstance(team_assignments, pd.DataFrame):
        raise TypeError("team_assignments must be a pandas DataFrame.")

    required = {"ward_id", "team_name"}
    missing = required - set(team_assignments.columns)

    if missing:
        raise ValueError(
            f"team_assignments is missing columns: {sorted(missing)}"
        )

    assignments = team_assignments[["ward_id", "team_name"]].copy()

    if assignments["ward_id"].isna().any():
        raise ValueError("Team assignment ward IDs cannot be missing.")

    if assignments["team_name"].isna().any():
        raise ValueError("Team names cannot be missing.")

    assignments["team_name"] = assignments["team_name"].astype(str).str.strip()

    if assignments["team_name"].eq("").any():
        raise ValueError("Team names cannot be empty.")

    unknown_wards = set(assignments["ward_id"]) - set(ward_ids)

    if unknown_wards:
        raise ValueError(
            f"Team assignments contain unknown wards: {sorted(unknown_wards)}"
        )

    grouped = assignments.groupby("ward_id", sort=False)["team_name"].apply(
        lambda names: list(dict.fromkeys(names))
    )

    return grouped.to_dict()


def _format_message(row):
    return (
        f"DENGUEGUARD ALERT | {row['city']} {row['ward_id']} | "
        f"{row['horizon_weeks']}-week forecast: {row['predicted_cases']:.2f} cases "
        f"({row['risk_category']} risk). "
        f"Saline bags: {int(row['saline_bags_allocated'])}/"
        f"{int(row['saline_bags_demand'])} allocated; "
        f"{int(row['saline_bags_unmet'])} unmet. "
        f"Diagnostic kits: {int(row['diagnostic_kits_allocated'])}/"
        f"{int(row['diagnostic_kits_demand'])} allocated; "
        f"{int(row['diagnostic_kits_unmet'])} unmet. "
        f"Assigned team(s): {row['assigned_teams']}."
    )


def build_dispatch_notices(allocation_result, team_assignments=None):
    allocations, parameters, horizon = _validate_allocation_result(
        allocation_result
    )

    team_lookup = _prepare_team_assignments(
        team_assignments,
        allocations["ward_id"],
    )

    allocations["assigned_teams"] = allocations["ward_id"].map(
        lambda ward_id: ", ".join(team_lookup.get(ward_id, []))
        or "No team assigned"
    )

    allocations["saline_bags_parameter_source"] = parameters[
        "saline_bags"
    ]["source"]
    allocations["saline_bags_parameter_status"] = parameters[
        "saline_bags"
    ]["status"]
    allocations["diagnostic_kits_parameter_source"] = parameters[
        "diagnostic_kits"
    ]["source"]
    allocations["diagnostic_kits_parameter_status"] = parameters[
        "diagnostic_kits"
    ]["status"]

    allocations["message"] = allocations.apply(_format_message, axis=1)

    return allocations.sort_values(
        ["city", "ward_id"],
        kind="stable",
    ).reset_index(drop=True)


def render_dispatch_html(notices, title="DengueGuard AI Ward Dispatch Report"):
    if not isinstance(notices, pd.DataFrame):
        raise TypeError("notices must be a pandas DataFrame.")

    required = {
        "city",
        "ward_id",
        "date",
        "horizon_weeks",
        "predicted_cases",
        "risk_category",
        "assigned_teams",
        "message",
    }

    for fields in RESOURCE_FIELDS.values():
        required.update(fields)

    missing = required - set(notices.columns)

    if missing:
        raise ValueError(f"notices is missing columns: {sorted(missing)}")

    safe_title = escape(str(title))
    rows = []

    for _, row in notices.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['city']))}</td>"
            f"<td>{escape(str(row['ward_id']))}</td>"
            f"<td>{escape(pd.Timestamp(row['date']).strftime('%Y-%m-%d'))}</td>"
            f"<td>{int(row['horizon_weeks'])} weeks</td>"
            f"<td>{float(row['predicted_cases']):.2f}</td>"
            f"<td>{escape(str(row['risk_category']))}</td>"
            f"<td>{int(row['saline_bags_allocated'])} / "
            f"{int(row['saline_bags_demand'])} "
            f"(unmet {int(row['saline_bags_unmet'])})</td>"
            f"<td>{int(row['diagnostic_kits_allocated'])} / "
            f"{int(row['diagnostic_kits_demand'])} "
            f"(unmet {int(row['diagnostic_kits_unmet'])})</td>"
            f"<td>{escape(str(row['assigned_teams']))}</td>"
            f"<td>{escape(str(row['message']))}</td>"
            "</tr>"
        )

    provenance_rows = []

    for resource, label in (
        ("saline_bags", "Saline bags"),
        ("diagnostic_kits", "Diagnostic kits"),
    ):
        source_column = f"{resource}_parameter_source"
        status_column = f"{resource}_parameter_status"

        if source_column in notices.columns and status_column in notices.columns:
            source_values = notices[source_column].dropna().astype(str).unique()
            status_values = notices[status_column].dropna().astype(str).unique()
            source = ", ".join(source_values)
            status = ", ".join(status_values)
        else:
            source = "Not supplied"
            status = "Not supplied"

        provenance_rows.append(
            f"<li><strong>{escape(label)} demand parameter:</strong> "
            f"{escape(status)} — {escape(source)}</li>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{safe_title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #172033; }}
h1, h2 {{ color: #12345a; }}
.notice {{ background: #f2f6fb; padding: 14px; border-radius: 8px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; min-width: 1100px; }}
th, td {{ border: 1px solid #d7deea; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #eaf0f8; }}
li {{ margin: 6px 0; }}
@media print {{ body {{ margin: 10mm; }} }}
</style>
</head>
<body>
<h1>{safe_title}</h1>
<div class="notice">
<p>Generated from the supplied DengueGuard AI resource-allocation result.</p>
<p>Forecast horizon: {int(notices["horizon_weeks"].iloc[0]) if not notices.empty else "Not available"} weeks</p>
<p>Notices generated: {len(notices)}</p>
</div>
<h2>Demand parameter provenance</h2>
<ul>{"".join(provenance_rows)}</ul>
<h2>Ward dispatch notices</h2>
<div class="table-wrap">
<table>
<thead>
<tr>
<th>City</th><th>Ward</th><th>Forecast date</th><th>Horizon</th>
<th>Predicted cases</th><th>Risk</th><th>Saline bags allocated / demand</th>
<th>Diagnostic kits allocated / demand</th><th>Assigned teams</th><th>Message</th>
</tr>
</thead>
<tbody>{"".join(rows)}</tbody>
</table>
</div>
</body>
</html>"""


def save_dispatch_html(notices, output_path, title="DengueGuard AI Ward Dispatch Report"):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    html_document = render_dispatch_html(notices, title=title)
    path.write_text(html_document, encoding="utf-8")
    return path
