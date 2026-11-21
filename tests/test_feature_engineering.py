"""Leakage and correctness tests for the Phase 5 rainfall lag pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.config import CITY_COL, DATE_COL, RAIN_LAGS, RAIN_ROLL_WINDOWS, TARGET_COL, WARD_COL, WEATHER_COLS
from src.feature_engineering import (
    LAG_FEATURE_COLUMNS,
    LAG_TABLE_COLUMNS,
    RAIN_COL,
    build_lag_features,
)
from src.preprocessing import get_clean_data

N_WEEKS = 60
N_WARDS = 3
ENCODE = 100_000  # rainfall = ward_number * ENCODE + week_index  (ward and week are readable)


def make_panel() -> pd.DataFrame:
    dates = pd.date_range("2018-01-01", periods=N_WEEKS, freq="7D")
    frames = []
    for w in range(1, N_WARDS + 1):
        frames.append(
            pd.DataFrame(
                {
                    DATE_COL: dates,
                    CITY_COL: f"C{w}",
                    WARD_COL: f"W{w}",
                    "avg_temp_c": 25.0 + np.arange(N_WEEKS) * 0.01,
                    "max_temp_c": 30.0 + np.arange(N_WEEKS) * 0.01,
                    "min_temp_c": 20.0 + np.arange(N_WEEKS) * 0.01,
                    RAIN_COL: w * ENCODE + np.arange(N_WEEKS, dtype=float),
                    "relative_humidity_pct": 70.0 + np.arange(N_WEEKS) * 0.1,
                    TARGET_COL: np.arange(N_WEEKS) % 7,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_panel()


@pytest.fixture(scope="module")
def features(panel) -> pd.DataFrame:
    return build_lag_features(panel)


@pytest.mark.parametrize("k", RAIN_LAGS)
def test_lag_k_is_the_same_wards_rainfall_k_weeks_earlier(features, k):
    ok = features[f"rain_lag_{k}"].notna()
    lagged = features.loc[ok, f"rain_lag_{k}"]
    current = features.loc[ok, RAIN_COL]
    assert (lagged == current - k).all()  # exactly k weeks earlier
    assert (lagged // ENCODE == current // ENCODE).all()  # same ward, no cross-ward values


def test_lag_0_equals_the_current_weeks_rainfall(features):
    assert (features["rain_lag_0"] == features[RAIN_COL]).all()


@pytest.mark.parametrize("k", RAIN_LAGS)
def test_only_the_first_k_weeks_of_each_ward_are_missing(features, k):
    for _, group in features.groupby(WARD_COL):
        group = group.sort_values(DATE_COL)
        nan_flags = group[f"rain_lag_{k}"].isna().to_numpy()
        assert nan_flags.sum() == k
        assert nan_flags[:k].all()


@pytest.mark.parametrize("window", RAIN_ROLL_WINDOWS)
def test_rolling_sum_is_a_trailing_window_including_the_as_of_week(features, window):
    ok = features[f"rain_sum_{window}w"].notna()
    expected = window * features.loc[ok, RAIN_COL] - window * (window - 1) / 2
    np.testing.assert_allclose(features.loc[ok, f"rain_sum_{window}w"], expected)
    for _, group in features.groupby(WARD_COL):
        assert group[f"rain_sum_{window}w"].isna().sum() == window - 1


def test_future_values_never_change_past_features(panel):
    cutoff = pd.Timestamp("2018-06-04")
    before = build_lag_features(panel)
    altered = panel.copy()
    future = altered[DATE_COL] > cutoff
    altered.loc[future, [TARGET_COL, *WEATHER_COLS]] = 999_999.0  # WEATHER_COLS includes rainfall_mm
    after = build_lag_features(altered)
    past = before[DATE_COL] <= cutoff
    pd.testing.assert_frame_equal(before[past].reset_index(drop=True), after[past].reset_index(drop=True))


def test_building_on_truncated_data_matches_building_on_full_data(panel, features):
    for cutoff in ("2018-03-05", "2018-07-02", "2018-10-01"):
        cut = pd.Timestamp(cutoff)
        truncated = build_lag_features(panel[panel[DATE_COL] <= cut])
        full_then_cut = features[features[DATE_COL] <= cut]
        pd.testing.assert_frame_equal(truncated.reset_index(drop=True), full_then_cut.reset_index(drop=True))


def test_missing_week_is_rejected_instead_of_silently_mis_shifting(panel):
    gapped = panel.drop(index=panel.index[10])
    with pytest.raises(ValueError):
        build_lag_features(gapped)


def test_duplicate_rows_are_rejected(panel):
    doubled = pd.concat([panel, panel.iloc[[5]]], ignore_index=True)
    with pytest.raises(ValueError):
        build_lag_features(doubled)


def test_feature_columns_contain_no_targets_or_labels():
    banned = ("target", "split", TARGET_COL)
    assert not any(any(b in col for b in banned) for col in LAG_FEATURE_COLUMNS)


def test_output_columns_and_order_are_the_documented_ones(features):
    assert list(features.columns) == LAG_TABLE_COLUMNS
    assert len(LAG_FEATURE_COLUMNS) == len(RAIN_LAGS) + len(RAIN_ROLL_WINDOWS)


def test_real_dataset_only_the_first_six_weeks_of_each_ward_lack_history():
    clean = get_clean_data()
    table = build_lag_features(clean)
    assert len(table) == len(clean) == 31_320
    assert table[WEATHER_COLS + ["rain_lag_0"]].notna().all().all()
    has_nan = table[LAG_FEATURE_COLUMNS].isna().any(axis=1)
    assert has_nan.sum() == 60 * 6  # 60 wards x first 6 weeks
    first_six_dates = sorted(clean[DATE_COL].unique())[:6]
    assert set(table.loc[has_nan, DATE_COL]) == set(first_six_dates)