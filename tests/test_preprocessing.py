import json

import numpy as np
import pandas as pd
import pytest

from src.config import (
    CITY_COL,
    DATE_COL,
    OPERATIONAL_COLS,
    REQUIRED_COLS,
    SHEET_NAME,
    TARGET_COL,
    WARD_COL,
)
from src.preprocessing import DataQualityError, clean, clean_with_report, get_clean_data

N_WEEKS = 20


def make_valid() -> pd.DataFrame:
    dates = pd.date_range("2020-01-06", periods=N_WEEKS, freq="7D")
    frames = []
    for i, (ward, city) in enumerate([("W1", "A"), ("W2", "B")]):
        frames.append(
            pd.DataFrame(
                {
                    DATE_COL: dates,
                    CITY_COL: city,
                    WARD_COL: ward,
                    "population_density_sqkm": 1000 * (i + 1),
                    "water_logging_risk_index": 0.3 + 0.1 * i,
                    "sanitation_score": 0.5 + 0.1 * i,
                    "avg_temp_c": 25.0 + np.arange(N_WEEKS) * 0.1,
                    "max_temp_c": 30.0 + np.arange(N_WEEKS) * 0.1,
                    "min_temp_c": 20.0 + np.arange(N_WEEKS) * 0.1,
                    "rainfall_mm": 10.0 + np.arange(N_WEEKS),
                    "relative_humidity_pct": 70.0 + np.arange(N_WEEKS) * 0.5,
                    TARGET_COL: np.arange(N_WEEKS) + 5 * i,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[REQUIRED_COLS]


def test_valid_panel_is_unchanged_and_report_counters_are_zero():
    df = make_valid()
    out, report = clean_with_report(df)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df, check_dtype=False)
    counters = [k for k in report if k not in ("rows_in", "rows_out", "wards", "date_min", "date_max")]
    assert all(report[k] == 0 for k in counters)


def test_duplicate_ward_date_rows_are_dropped():
    df = make_valid()
    doubled = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    out, report = clean_with_report(doubled)
    assert report["duplicate_ward_date_rows_dropped"] == 1
    assert len(out) == len(df)


def test_humidity_clipped_and_negative_rain_is_repaired_from_the_past_only():
    df = make_valid()
    df.loc[5, "relative_humidity_pct"] = 130.0
    df.loc[7, "rainfall_mm"] = -4.0
    out, report = clean_with_report(df)
    assert report["humidity_above_100_clipped"] == 1
    assert report["negative_rainfall_set_missing"] == 1
    assert out.loc[5, "relative_humidity_pct"] == 100.0
    assert out.loc[7, "rainfall_mm"] == df.loc[6, "rainfall_mm"]  # previous week, never next


def test_weather_gap_fill_is_forward_only_and_limited():
    df = make_valid()
    df.loc[[10, 11], "avg_temp_c"] = np.nan  # 2 weeks: fillable
    out = clean(df)
    assert (out.loc[[10, 11], "avg_temp_c"] == df.loc[9, "avg_temp_c"]).all()
    df.loc[[10, 11, 12], "avg_temp_c"] = np.nan  # 3 weeks: beyond the limit
    with pytest.raises(DataQualityError):
        clean(df)


def test_missing_week_is_inserted_then_fails_because_cases_are_never_imputed():
    df = make_valid().drop(index=8)
    with pytest.raises(DataQualityError):
        clean(df)


def test_ward_in_two_cities_raises():
    df = make_valid()
    df.loc[3, CITY_COL] = "Z"
    with pytest.raises(DataQualityError):
        clean(df)


def test_static_attribute_varying_within_a_ward_raises():
    df = make_valid()
    df.loc[3, "sanitation_score"] = 0.99
    with pytest.raises(DataQualityError):
        clean(df)


def test_off_lattice_date_raises():
    df = make_valid()
    df.loc[4, DATE_COL] = df.loc[4, DATE_COL] + pd.Timedelta(days=1)
    with pytest.raises(DataQualityError):
        clean(df)


def test_missing_required_column_raises():
    with pytest.raises(DataQualityError):
        clean(make_valid().drop(columns=["rainfall_mm"]))


def _write_workbook(path, df):
    df.to_excel(path, sheet_name=SHEET_NAME, index=False)


def test_cache_is_reused_and_rebuilt_only_when_the_source_changes(tmp_path):
    raw, csv, meta, rep = (tmp_path / n for n in ("raw.xlsx", "c.csv", "m.json", "r.json"))
    df = make_valid()
    _write_workbook(raw, df)
    first = get_clean_data(raw_path=raw, clean_path=csv, meta_path=meta, report_path=rep)
    assert csv.exists() and meta.exists() and rep.exists()

    stamp = csv.stat().st_mtime_ns
    second = get_clean_data(raw_path=raw, clean_path=csv, meta_path=meta, report_path=rep)
    assert csv.stat().st_mtime_ns == stamp  # cache hit: file not rewritten
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True), check_dtype=False)

    changed = df.copy()
    changed.loc[0, TARGET_COL] = 999
    _write_workbook(raw, changed)
    third = get_clean_data(raw_path=raw, clean_path=csv, meta_path=meta, report_path=rep)
    assert third.sort_values([WARD_COL, DATE_COL]).iloc[0][TARGET_COL] == 999  # source edit -> rebuild
    assert json.loads(meta.read_text())["source_sha256"]



def make_valid_operational() -> pd.DataFrame:
    df = make_valid()
    df[OPERATIONAL_COLS[0]] = 10
    df[OPERATIONAL_COLS[1]] = 20
    df[OPERATIONAL_COLS[2]] = 5
    return df


def test_valid_operational_columns_are_preserved_as_integers():
    df = make_valid_operational()
    out = clean(df)

    for col in OPERATIONAL_COLS:
        assert col in out.columns
        assert out[col].eq(df[col]).all()
        assert out[col].dtype == "int64"


@pytest.mark.parametrize("invalid_value", [np.nan, "invalid"])
def test_missing_or_invalid_operational_values_raise(invalid_value):
    df = make_valid_operational()
    df[OPERATIONAL_COLS[0]] = df[OPERATIONAL_COLS[0]].astype(object)
    df.loc[0, OPERATIONAL_COLS[0]] = invalid_value

    with pytest.raises(DataQualityError):
        clean(df)


@pytest.mark.parametrize("invalid_value", [-1, -10])
def test_negative_operational_values_raise(invalid_value):
    df = make_valid_operational()
    df.loc[0, OPERATIONAL_COLS[0]] = invalid_value

    with pytest.raises(DataQualityError):
        clean(df)


@pytest.mark.parametrize("invalid_value", [1.5, 2.25])
def test_non_integer_operational_values_raise(invalid_value):
    df = make_valid_operational()
    df[OPERATIONAL_COLS[0]] = df[OPERATIONAL_COLS[0]].astype(object)
    df.loc[0, OPERATIONAL_COLS[0]] = invalid_value

    with pytest.raises(DataQualityError):
        clean(df)


@pytest.mark.parametrize("missing_col", OPERATIONAL_COLS)
def test_partial_operational_columns_raise(missing_col):
    df = make_valid_operational().drop(columns=[missing_col])

    with pytest.raises(DataQualityError, match="Incomplete operational columns"):
        clean(df)


def test_phase1_dataset_without_operational_columns_remains_compatible():
    df = make_valid()
    out = clean(df)

    assert len(out) == len(df)
    assert not any(col in out.columns for col in OPERATIONAL_COLS)
