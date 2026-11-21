
import numpy as np
import pandas as pd
import pytest

from src.config import (
    DATE_COL,
    TARGET_COL,
    WARD_COL,
    STATIC_COLS,
)
from src.preprocessing import get_clean_data
from src.production_features import (
    CASE_LAG_COLUMNS,
    CASE_ROLL_COLUMN,
    PRODUCTION_FEATURE_COLUMNS,
    build_production_features,
)


@pytest.fixture(scope="module")
def clean_data():
    return get_clean_data()


@pytest.fixture(scope="module")
def production_table(clean_data):
    return build_production_features(clean_data)


def test_production_feature_schema(production_table):
    from src.feature_engineering import KEY_COLUMNS

    assert len(PRODUCTION_FEATURE_COLUMNS) == 25
    assert list(production_table.columns) == (
        KEY_COLUMNS + PRODUCTION_FEATURE_COLUMNS
    )


def test_target_and_split_columns_are_excluded(production_table):
    forbidden = {
        TARGET_COL,
        "split",
        "target_cases_h2",
        "target_cases_h3",
        "target_cases_h4",
    }

    assert forbidden.isdisjoint(production_table.columns)


def test_case_history_uses_only_previous_weeks(
    clean_data,
    production_table,
):
    ward = clean_data[WARD_COL].iloc[0]

    original = (
        clean_data[clean_data[WARD_COL] == ward]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    features = (
        production_table[production_table[WARD_COL] == ward]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    row = 10

    assert features.loc[row, CASE_LAG_COLUMNS[0]] == original.loc[
        row - 1, TARGET_COL
    ]
    assert features.loc[row, CASE_LAG_COLUMNS[1]] == original.loc[
        row - 2, TARGET_COL
    ]
    assert features.loc[row, CASE_LAG_COLUMNS[2]] == original.loc[
        row - 3, TARGET_COL
    ]

    expected_mean = original.loc[
        row - 4:row - 1, TARGET_COL
    ].mean()

    assert features.loc[row, CASE_ROLL_COLUMN] == pytest.approx(
        expected_mean
    )


def test_current_and_future_cases_do_not_change_current_features(
    clean_data,
):
    ward = clean_data[WARD_COL].iloc[0]

    original = clean_data.copy()
    original = original.sort_values(
        [WARD_COL, DATE_COL]
    ).reset_index(drop=True)

    ward_indices = original.index[
        original[WARD_COL] == ward
    ].tolist()

    row_index = ward_indices[10]

    modified = original.copy()

    modified.loc[
        row_index:ward_indices[-1], TARGET_COL
    ] += 1000

    before = build_production_features(original)
    after = build_production_features(modified)

    pd.testing.assert_series_equal(
        before.loc[row_index, PRODUCTION_FEATURE_COLUMNS],
        after.loc[row_index, PRODUCTION_FEATURE_COLUMNS],
        check_names=False,
    )


def test_static_features_are_preserved(
    clean_data,
    production_table,
):
    for column in STATIC_COLS:
        expected = clean_data[column].reset_index(drop=True)
        actual = production_table[column].reset_index(drop=True)

        pd.testing.assert_series_equal(
            actual,
            expected,
            check_names=False,
        )


def test_early_case_history_is_missing(production_table):
    ward = production_table[WARD_COL].iloc[0]

    first_rows = production_table[
        production_table[WARD_COL] == ward
    ].head(4)

    assert first_rows.iloc[0][CASE_LAG_COLUMNS].isna().all()

    assert first_rows.iloc[0][CASE_ROLL_COLUMN] != (
        first_rows.iloc[0][CASE_ROLL_COLUMN]
    )

    assert first_rows.iloc[1]["cases_lag_1"] == 0
    assert first_rows.iloc[2]["cases_lag_1"] == 0
    assert first_rows.iloc[3]["cases_lag_1"] == 0

    assert first_rows[CASE_ROLL_COLUMN].isna().all()
