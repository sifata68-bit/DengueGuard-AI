
import pandas as pd

from src.config import CITY_COL, WARD_COL
from src.predict import predict_all_wards
from src.vector_control import simulate_vector_control


WARD_INFORMATION_COLUMNS = [
    WARD_COL,
    CITY_COL,
    "population_density_sqkm",
]


def build_ward_information(clean_data):
    if not isinstance(clean_data, pd.DataFrame):
        raise TypeError("clean_data must be a pandas DataFrame.")

    missing = sorted(
        set(WARD_INFORMATION_COLUMNS) - set(clean_data.columns)
    )

    if missing:
        raise ValueError(
            f"Clean data is missing required ward columns: {missing}"
        )

    ward_information = (
        clean_data[WARD_INFORMATION_COLUMNS]
        .drop_duplicates()
        .copy()
    )

    if ward_information[WARD_COL].duplicated().any():
        raise ValueError(
            "Ward information must contain consistent values "
            "for each ward."
        )

    if ward_information.isna().any().any():
        raise ValueError(
            "Ward information contains missing required values."
        )

    return ward_information.sort_values(
        WARD_COL,
        kind="stable",
    ).reset_index(drop=True)


def run_vector_control_simulation(
    clean_data,
    as_of_date,
    selected_wards,
    total_teams,
    intervention_type,
    effectiveness_by_horizon,
):
    if not as_of_date:
        raise ValueError("A forecast reference date is required.")

    ward_information = build_ward_information(clean_data)

    forecasts = predict_all_wards(
        clean_data,
        as_of_date=as_of_date,
    )

    simulation = simulate_vector_control(
        forecasts=forecasts,
        ward_information=ward_information,
        selected_wards=selected_wards,
        total_teams=total_teams,
        intervention_type=intervention_type,
        effectiveness_by_horizon=effectiveness_by_horizon,
    )

    return {
        "forecasts": forecasts,
        **simulation,
    }
