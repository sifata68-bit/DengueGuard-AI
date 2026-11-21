from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.diag_models as diag
from src.config import (
    DATE_COL,
    HORIZONS,
    TARGET_COL,
    WARD_COL,
)
from src.feature_engineering import LAG_FEATURE_COLUMNS
from src.preprocessing import get_clean_data


@pytest.fixture(scope="module")
def clean():
    return get_clean_data()


@pytest.fixture(scope="module")
def diagnostic_frame(clean):
    return diag.build_diagnostic_frame(clean)


@pytest.fixture(scope="module")
def diagnostic_results(clean):
    return diag.run_diagnostic(clean)


def test_diagnostic_case_features_are_created_separately(diagnostic_frame):
    for column in diag.DIAG_CASE_COLUMNS:
        assert column in diagnostic_frame.columns

    assert len(diag.DIAG_CASE_COLUMNS) == 5


def test_diagnostic_case_lags_are_correct(diagnostic_frame, clean):
    ward = clean[WARD_COL].iloc[0]

    original = (
        clean.loc[clean[WARD_COL] == ward]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    actual = (
        diagnostic_frame.loc[diagnostic_frame[WARD_COL] == ward]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    assert actual.loc[4, "diag_cases_lag_0"] == original.loc[4, TARGET_COL]

    assert actual.loc[4, "diag_cases_lag_1"] == original.loc[3, TARGET_COL]

    assert actual.loc[4, "diag_cases_lag_3"] == original.loc[1, TARGET_COL]

    expected = original.loc[0:3, TARGET_COL].mean()
    assert actual.loc[3, "diag_cases_roll4_mean"] == pytest.approx(expected)


def test_production_feature_pipeline_has_no_diagnostic_case_features():
    source_path = Path("src/feature_engineering.py")
    source = source_path.read_text(encoding="utf-8")

    assert "diag_cases" not in source
    assert "dengue_cases" not in source


def test_feature_group_sizes():
    assert len(diag.A0_FEATURES) == 5
    assert len(diag.A_FEATURES) == 5 + len(LAG_FEATURE_COLUMNS)
    assert len(diag.B0_FEATURES) == 5
    assert len(diag.B_FEATURES) == len(diag.A_FEATURES) + 5

    assert len(diag.A_FEATURES) == 15
    assert len(diag.B_FEATURES) == 20


def test_common_split_rows_rejects_test_split(diagnostic_frame):
    with pytest.raises(ValueError, match="only permits train and val"):
        diag._common_split_rows(diagnostic_frame, 2, "test")


def test_diagnostic_uses_only_train_and_validation(
    clean,
    monkeypatch,
):
    accessed_splits = []
    original_get_split_frame = diag.get_split_frame

    def tracked_get_split_frame(frame, horizon, split, **kwargs):
        accessed_splits.append(split)
        if split == "test":
            raise AssertionError("The locked test split was accessed.")
        return original_get_split_frame(
            frame,
            horizon,
            split,
            **kwargs,
        )

    monkeypatch.setattr(
        diag,
        "get_split_frame",
        tracked_get_split_frame,
    )

    results = diag.run_diagnostic(clean)

    assert results["horizon_weeks"].nunique() == len(HORIZONS)
    assert set(accessed_splits) == {"train", "val"}
    assert "test" not in accessed_splits


def test_diagnostic_result_schema_and_common_rows(diagnostic_results):
    expected_groups = {"A0", "A", "B0", "B", "C"}

    assert set(diagnostic_results["group"]) == expected_groups
    assert len(diagnostic_results) == len(HORIZONS) * 8

    for horizon in HORIZONS:
        subset = diagnostic_results[
            diagnostic_results["horizon_weeks"] == horizon
        ]

        assert len(subset) == 8
        assert subset["n"].nunique() == 1
        assert subset["n_train"].nunique() == 1

    required_columns = {
        "horizon_weeks",
        "group",
        "model",
        "n_train",
        "n",
        "mae",
        "rmse",
        "mae_in_season",
        "rmse_in_season",
    }
    assert required_columns.issubset(diagnostic_results.columns)


def test_phase2_ward_month_climatology_is_reproduced(diagnostic_results):
    climatology = diagnostic_results[
        (diagnostic_results["group"] == "C")
        & (diagnostic_results["model"] == "ward_month_climatology")
    ]

    assert len(climatology) == len(HORIZONS)

    for _, row in climatology.iterrows():
        assert row["mae_in_season"] == pytest.approx(3.747, abs=0.001)
        assert row["rmse_in_season"] == pytest.approx(5.219, abs=0.001)


def test_diagnostic_results_contain_no_negative_predictions_metrics(
    diagnostic_results,
):
    metric_columns = ["mae", "rmse", "mae_in_season", "rmse_in_season"]

    assert np.isfinite(
        diagnostic_results[metric_columns].to_numpy(dtype=float)
    ).all()
