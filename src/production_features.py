
"""Production feature pipeline for DengueGuard AI.

One row represents one ward at an as-of date t.

All features use information available at or before t, except dengue
case-history features, which deliberately stop at t-1 because current-week
case counts may not yet be available when a forecast is issued.

The target and split columns are never included in the feature table.
"""

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

# Previous dengue case counts only. No lag 0.
CASE_LAGS = (1, 2, 3)
CASE_LAG_COLUMNS = [f"cases_lag_{k}" for k in CASE_LAGS]

# Mean of the four weeks BEFORE the as-of week.
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
    """Build the complete leakage-safe production feature table.

    Includes:
      - Current weather and historical rainfall features.
      - Static ward vulnerability features.
      - Calendar seasonality features.
      - Dengue case history from previous weeks only.

    Excludes:
      - Current-week dengue cases.
      - Future dengue cases.
      - Target and split columns.

    Early rows without sufficient history retain NaN values.
    """
    required = set(
        KEY_COLUMNS + STATIC_COLS + [TARGET_COL]
    )
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"Missing required input columns: {sorted(missing)}"
        )

    # Existing weather and rainfall pipeline performs the weekly-grid check.
    weather = build_lag_features(df)

    # Sort and reset indices to ensure safe row alignment.
    out = df.sort_values(
        [WARD_COL, DATE_COL]
    ).reset_index(drop=True).copy()

    if len(weather) != len(out):
        raise ValueError("Weather feature row count does not match input.")

    # Static ward vulnerability.
    static_features = out[KEY_COLUMNS + STATIC_COLS].copy()

    # Calendar features based on the as-of date.
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

    # Dengue case history: all shifts are strictly before week t.
    cases_by_ward = out.groupby(
        WARD_COL, sort=False
    )[TARGET_COL]

    case_features = {}

    for lag in CASE_LAGS:
        case_features[f"cases_lag_{lag}"] = (
            cases_by_ward.shift(lag)
        )

    # Shift by one week BEFORE calculating the rolling mean.
    # Therefore, the rolling window contains t-1, t-2, t-3 and t-4.
    previous_cases = cases_by_ward.shift(1)

    case_features[CASE_ROLL_COLUMN] = (
        previous_cases.groupby(out[WARD_COL], sort=False)
        .transform(
            lambda s: s.rolling(4, min_periods=4).mean()
        )
    )

    case_history = pd.DataFrame(case_features)

    # Combine all feature groups.
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
