
import numpy as np
import pandas as pd

from src.config import (
    DATE_COL,
    IN_SEASON_MONTHS,
    STATIC_COLS,
    TARGET_COL,
    WARD_COL,
    WEATHER_COLS,
)
from src.feature_engineering import (
    KEY_COLUMNS,
    LAG_FEATURE_COLUMNS,
    build_lag_features,
)

CASE_LAGS = (1, 2, 3)
CASE_LAG_COLUMNS = [f"cases_lag_{k}" for k in CASE_LAGS]

CASE_ROLL_COLUMN = "cases_prev4w_mean"

CALENDAR_COLUMNS = [
    "month_sin",
    "month_cos",
    "is_dengue_season",
]

PRODUCTION_FEATURE_COLUMNS = (
    WEATHER_COLS
    + LAG_FEATURE_COLUMNS
    + STATIC_COLS
    + CALENDAR_COLUMNS
    + CASE_LAG_COLUMNS
    + [CASE_ROLL_COLUMN]
)

PRODUCTION_TABLE_COLUMNS = KEY_COLUMNS + PRODUCTION_FEATURE_COLUMNS


def build_production_features(df: pd.DataFrame) -> pd.DataFrame:
    required = set(KEY_COLUMNS + STATIC_COLS + [TARGET_COL])
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required input columns: {sorted(missing)}"
        )

    weather = build_lag_features(df)

    out = df.sort_values(
        [WARD_COL, DATE_COL]
    ).reset_index(drop=True).copy()

    if len(weather) != len(out):
        raise ValueError("Weather feature row count does not match input.")

    static_features = out[KEY_COLUMNS + STATIC_COLS].copy()

    month = out[DATE_COL].dt.month

    calendar = pd.DataFrame(
        {
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "is_dengue_season": month.isin(
                IN_SEASON_MONTHS
            ).astype(int),
        }
    )

    cases_by_ward = out.groupby(
        WARD_COL, sort=False
    )[TARGET_COL]

    case_features = {}

    for lag in CASE_LAGS:
        case_features[f"cases_lag_{lag}"] = (
            cases_by_ward.shift(lag)
        )

    previous_cases = cases_by_ward.shift(1)

    case_features[CASE_ROLL_COLUMN] = (
        previous_cases.groupby(out[WARD_COL], sort=False)
        .transform(
            lambda s: s.rolling(4, min_periods=4).mean()
        )
    )

    case_history = pd.DataFrame(case_features)

    result = pd.concat(
        [
            weather.reset_index(drop=True),
            static_features[STATIC_COLS].reset_index(drop=True),
            calendar.reset_index(drop=True),
            case_history.reset_index(drop=True),
        ],
        axis=1,
    )

    return result[PRODUCTION_TABLE_COLUMNS]
