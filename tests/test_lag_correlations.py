"""Tests for src/lag_correlations.py (descriptive correlations; train split only)."""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

import src.lag_correlations as lag_correlations
from src.config import (
    CITY_COL,
    DATE_COL,
    HORIZONS,
    RAIN_LAGS,
    RAIN_ROLL_WINDOWS,
    SPLITS,
    TARGET_COL,
    WARD_COL,
    WEATHER_COLS,
)
from src.lag_correlations import (
    HINTS,
    build_analysis_frame,
    compare_with_hints,
    correlation_table,
    save_outputs,
)
from src.preprocessing import get_clean_data
from src.targets import get_split_frame

KNOWN_DELAY = 3  # in the synthetic data, cases(t) = 2 * rainfall(t - 3)


def make_known_delay_panel(weeks: int = 200, n_wards: int = 4, seed: int = 7) -> pd.DataFrame:
    """Random rainfall; cases are exactly 2 x the rainfall observed KNOWN_DELAY weeks earlier."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2016-01-04", periods=weeks, freq="7D")  # all inside the train period
    frames = []
    for w in range(n_wards):
        rain = rng.gamma(2.0, 20.0, size=weeks)
        cases = 2.0 * pd.Series(rain).shift(KNOWN_DELAY).fillna(0.0).to_numpy()
        frames.append(
            pd.DataFrame(
                {
                    DATE_COL: dates,
                    CITY_COL: "C",
                    WARD_COL: f"W{w}",
                    "avg_temp_c": 27.0,
                    "max_temp_c": 31.0,
                    "min_temp_c": 23.0,
                    "rainfall_mm": rain,
                    "relative_humidity_pct": 80.0,
                    TARGET_COL: cases,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def synthetic_table() -> pd.DataFrame:
    return correlation_table(build_analysis_frame(make_known_delay_panel()))


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return get_clean_data()


@pytest.fixture(scope="module")
def real_frame(clean) -> pd.DataFrame:
    return build_analysis_frame(clean)


@pytest.fixture(scope="module")
def real_table(real_frame) -> pd.DataFrame:
    return correlation_table(real_frame)


# ---------- synthetic data with a KNOWN answer ----------

def test_known_rain_to_case_delay_is_recovered_at_every_usable_horizon(synthetic_table):
    lags = synthetic_table[synthetic_table["kind"] == "lag"]
    for h in (2, 3):  # delay 3 minus horizon = lag 1 and lag 0, both usable
        part = lags[lags["horizon"] == h]
        best = part.loc[part["r_all"].idxmax()]
        assert int(best["lag_weeks"]) == KNOWN_DELAY - h
        assert best["r_all"] > 0.99
    # Horizon 4 would need lag -1 (future rain): no usable lag may look strongly related.
    part4 = lags[lags["horizon"] == 4]
    assert part4["r_all"].abs().max() < 0.3


def test_hint_comparison_flags_matches_and_mismatches(synthetic_table):
    r2 = round(float(synthetic_table.query("horizon == 2 and feature == 'rain_lag_1'")["r_all"].iloc[0]), 3)
    r3 = round(float(synthetic_table.query("horizon == 3 and feature == 'rain_lag_0'")["r_all"].iloc[0]), 3)
    good = compare_with_hints(synthetic_table, hints={2: (1, r2), 3: (0, r3)})
    assert good["lag_matches_hint"].all() and good["r_matches_hint_3dp"].all()
    bad = compare_with_hints(synthetic_table, hints={2: (4, 0.123), 3: (5, 0.456)})
    assert not bad["lag_matches_hint"].any() and not bad["r_matches_hint_3dp"].any()


def test_ideal_three_week_delay_is_reported_unavailable_at_horizon_4(synthetic_table):
    summary = compare_with_hints(synthetic_table, hints={2: (1, 0.0), 3: (0, 0.0), 4: (0, 0.0)})
    flags = dict(zip(summary["horizon"], summary["ideal_3w_delay_available_at_asof"]))
    assert flags == {2: True, 3: True, 4: False}
    assert np.isnan(summary.loc[summary["horizon"] == 4, "r_at_ideal_3w_delay"].iloc[0])


def test_table_has_one_row_per_horizon_and_rainfall_feature(synthetic_table):
    assert len(synthetic_table) == len(HORIZONS) * (len(RAIN_LAGS) + len(RAIN_ROLL_WINDOWS))
    assert {"horizon", "feature", "kind", "r_all", "r_in_season", "n_all", "n_in_season"} <= set(
        synthetic_table.columns
    )


def test_total_delay_is_lag_plus_horizon_and_only_defined_for_lags(synthetic_table):
    lags = synthetic_table[synthetic_table["kind"] == "lag"]
    assert (lags["total_delay_weeks"] == lags["lag_weeks"] + lags["horizon"]).all()
    rolling = synthetic_table[synthetic_table["kind"] == "rolling_sum"]
    assert rolling["lag_weeks"].isna().all() and rolling["total_delay_weeks"].isna().all()


# ---------- real data ----------

def test_sample_size_loses_one_week_per_ward_for_each_extra_lag(real_frame, real_table):
    n_train = len(get_split_frame(real_frame, 2, "train"))
    lags = real_table[(real_table["horizon"] == 2) & (real_table["kind"] == "lag")]
    assert int(lags.loc[lags["lag_weeks"] == 0, "n_all"].iloc[0]) == n_train
    for _, row in lags.iterrows():
        assert row["n_all"] == n_train - 60 * int(row["lag_weeks"])


def test_in_season_row_counts_are_identical_across_features_within_a_horizon(real_table):
    for _, part in real_table.groupby("horizon"):
        assert part["n_in_season"].nunique() == 1  # early-2016 gaps fall outside the season


def test_real_data_reproduces_the_phase_2_hints(real_table):
    summary = compare_with_hints(real_table, hints=HINTS)
    assert summary["lag_matches_hint"].all()
    assert summary["r_matches_hint_3dp"].all()
    assert summary.loc[summary["horizon"] == 4, "ideal_3w_delay_available_at_asof"].iloc[0] == False  # noqa: E712


def test_in_season_correlation_is_lower_than_the_all_row_correlation(real_table):
    summary = compare_with_hints(real_table, hints=HINTS)
    assert (summary["best_r_in_season"] < summary["observed_best_r"]).all()


def test_results_ignore_all_data_after_the_train_period(clean, real_table):
    altered = clean.copy()
    later = altered[DATE_COL] > SPLITS["train"][1]
    columns = [*WEATHER_COLS, TARGET_COL]
    altered.loc[later, columns] = altered.loc[later, columns] * 3.0 + 12_345.0
    recomputed = correlation_table(build_analysis_frame(altered))
    pd.testing.assert_frame_equal(real_table, recomputed)


# ---------- guards and outputs ----------

def test_module_never_requests_the_locked_test_split():
    source = inspect.getsource(lag_correlations)
    assert "final_test_evaluation" not in source


def test_outputs_are_written_and_can_be_read_back(real_table, tmp_path):
    summary = compare_with_hints(real_table)
    csv_path, json_path = tmp_path / "corr.csv", tmp_path / "summary.json"
    save_outputs(real_table, summary, csv_path=csv_path, json_path=json_path)
    assert len(pd.read_csv(csv_path)) == len(real_table)
    records = json.loads(json_path.read_text())
    assert [r["horizon"] for r in records] == [2, 3, 4]
    assert all("observed_best_lag" in r for r in records)