import numpy as np
import pandas as pd
import pytest

from src.config import DATE_COL, SPLITS, WARD_COL
from src.feature_engineering import LAG_FEATURE_COLUMNS, RAIN_COL, build_lag_features
from src.lag_checks import (
    check_dimensions,
    check_feature_table,
    check_future_invariance,
    check_no_banned_columns,
    check_truncation_invariance,
    check_weekly_grid,
    reference_lag_values,
    run_all_checks,
)
from src.preprocessing import get_clean_data
from tests.test_feature_engineering import make_panel


def leaky_builder(df: pd.DataFrame) -> pd.DataFrame:
    table = build_lag_features(df).sort_values([WARD_COL, DATE_COL]).reset_index(drop=True)
    table["rain_lag_1"] = table.groupby(WARD_COL)[RAIN_COL].shift(-1)
    return table


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return get_clean_data()


def test_all_checks_pass_on_the_real_dataset(clean):
    results = run_all_checks(clean)
    assert len(results) == 8
    assert all(r.passed for r in results), [r.detail for r in results if not r.passed]


def test_dimension_check_fails_on_truncated_data(clean):
    assert not check_dimensions(clean.iloc[:-60]).passed


def test_weekly_grid_check_fails_when_a_week_is_missing(clean):
    assert check_weekly_grid(clean).passed
    assert not check_weekly_grid(clean.drop(index=clean.index[100])).passed


def test_independent_recomputation_equals_the_pipeline_on_a_synthetic_panel():
    panel = make_panel()
    reference = reference_lag_values(panel)
    table = build_lag_features(panel)
    for col in LAG_FEATURE_COLUMNS:
        np.testing.assert_allclose(
            reference[col].to_numpy(dtype=float), table[col].to_numpy(dtype=float), equal_nan=True
        )


def test_feature_table_check_detects_a_corrupted_lag_column(clean):
    table = build_lag_features(clean)
    assert check_feature_table(clean, table).passed
    table["rain_lag_2"] = table["rain_lag_2"] + 1.0
    assert not check_feature_table(clean, table).passed


def test_feature_table_check_detects_a_wrongly_shifted_column(clean):
    table = build_lag_features(clean)
    table["rain_lag_3"] = table.groupby(WARD_COL)["rain_lag_3"].shift(-1)
    assert not check_feature_table(clean, table).passed


def test_banned_column_check_flags_target_and_diagnostic_columns(clean):
    table = build_lag_features(clean)
    assert check_no_banned_columns(table).passed
    assert not check_no_banned_columns(table.assign(target_cases_h2=0)).passed
    assert not check_no_banned_columns(table.assign(diag_cases_lag_0=0)).passed


def test_future_invariance_check_passes_for_the_pipeline_and_fails_for_a_leaky_builder(clean):
    cutoff = SPLITS["train"][1]
    assert check_future_invariance(clean, cutoff).passed
    assert not check_future_invariance(clean, cutoff, leaky_builder).passed


def test_truncation_check_passes_for_the_pipeline_and_fails_for_a_leaky_builder(clean):
    cutoff = SPLITS["val"][1]
    assert check_truncation_invariance(clean, cutoff).passed
    assert not check_truncation_invariance(clean, cutoff, leaky_builder).passed


def test_run_all_checks_reports_failures_for_a_leaky_builder(clean):
    results = run_all_checks(clean, builder=leaky_builder)
    failed = [r.name for r in results if not r.passed]
    assert len(failed) >= 3
