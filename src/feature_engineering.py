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
)

RAIN_COL = "rainfall_mm"
KEY_COLUMNS = [WARD_COL, CITY_COL, DATE_COL]
RAIN_LAG_COLUMNS = [f"rain_lag_{k}" for k in RAIN_LAGS]
RAIN_ROLL_COLUMNS = [f"rain_sum_{w}w" for w in RAIN_ROLL_WINDOWS]
LAG_FEATURE_COLUMNS = RAIN_LAG_COLUMNS + RAIN_ROLL_COLUMNS
LAG_TABLE_COLUMNS = KEY_COLUMNS + WEATHER_COLS + LAG_FEATURE_COLUMNS


def assert_weekly_grid(df: pd.DataFrame) -> None:
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
