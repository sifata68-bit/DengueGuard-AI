"""Phase 5: multi-week rainfall lag pipeline (grouped per ward_id, leakage-safe).

AS-OF CONVENTION (an assumption, disclosed in ARCHITECTURE.md)
    The dataset does not say whether a Monday date marks the start or the end of the week it
    describes. We assume the row for week t is complete and available when the forecast for
    horizon h is issued, and that the target is the row h weeks later (t + h weeks).
    Every feature built here for as-of week t therefore uses ONLY rows dated t or earlier.

How leakage is prevented
    * Only non-negative shifts (shift(k) with k >= 0) and trailing windows ending at t.
    * All lags are computed inside each ward (groupby ward_id), never across wards.
    * Row shifts equal "k weeks ago" only on a strict weekly grid, which Phase 4 guarantees
      and build_lag_features() re-checks, refusing to run otherwise.
    * Early weeks without full history stay NaN (no partial sums); training code drops them.
    * The target and split columns are never read.

Phase 6 will extend this module with case-history, vulnerability and seasonal features.
"""
import pandas as pd

from src.config import (
    CITY_COL,
    DATE_COL,
    RAIN_LAGS,
    RAIN_ROLL_WINDOWS,
    WARD_COL,
    WEATHER_COLS,
)

ASOF_CONVENTION = (
    "Row for week t is assumed complete and available when the forecast is issued; "
    "features use only rows dated t or earlier; the target is the row h weeks after t."
)

RAIN_COL = "rainfall_mm"
KEY_COLUMNS = [WARD_COL, CITY_COL, DATE_COL]
RAIN_LAG_COLUMNS = [f"rain_lag_{k}" for k in RAIN_LAGS]
RAIN_ROLL_COLUMNS = [f"rain_sum_{w}w" for w in RAIN_ROLL_WINDOWS]
LAG_FEATURE_COLUMNS = RAIN_LAG_COLUMNS + RAIN_ROLL_COLUMNS
LAG_TABLE_COLUMNS = KEY_COLUMNS + WEATHER_COLS + LAG_FEATURE_COLUMNS


def assert_weekly_grid(df: pd.DataFrame) -> None:
    """Raise if any ward has duplicate dates or a step other than exactly 7 days."""
    if df.duplicated([WARD_COL, DATE_COL]).any():
        raise ValueError("Duplicate (ward_id, date) rows found; run src.preprocessing.clean() first.")
    ordered = df.sort_values([WARD_COL, DATE_COL])
    steps = ordered.groupby(WARD_COL)[DATE_COL].diff().dropna()
    if (steps != pd.Timedelta(days=7)).any():
        raise ValueError(
            "Weekly grid is not regular (missing or irregular weeks); lags would be wrong. "
            "Run src.preprocessing.clean() first."
        )


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (ward_id, date): keys, current-week weather and rainfall lag features.

    Columns (see LAG_TABLE_COLUMNS):
      rain_lag_k   rainfall k weeks before the as-of week (k = 0 is the as-of week itself)
      rain_sum_Ww  rainfall summed over the W weeks ending at the as-of week (inclusive)
    Rows lacking enough history for a feature get NaN for it.
    """
    assert_weekly_grid(df)
    out = df.sort_values([WARD_COL, DATE_COL]).reset_index(drop=True)
    rain_by_ward = out.groupby(WARD_COL, sort=False)[RAIN_COL]

    features = {}
    for k in RAIN_LAGS:
        features[f"rain_lag_{k}"] = rain_by_ward.shift(k)
    for window in RAIN_ROLL_WINDOWS:
        features[f"rain_sum_{window}w"] = rain_by_ward.transform(
            lambda s, w=window: s.rolling(w, min_periods=w).sum()
        )
    return pd.concat([out[KEY_COLUMNS + WEATHER_COLS], pd.DataFrame(features)], axis=1)[LAG_TABLE_COLUMNS]