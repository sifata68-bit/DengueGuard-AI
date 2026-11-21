"""Phase 5, Step 2A: ingestion checks and leakage self-checks for the rainfall-lag table.

Every check returns a CheckResult instead of raising, so a report can show all results at once.
Run on its own with:   python -m src.lag_checks

Two ideas make these checks trustworthy:
  * The lag table is verified against an INDEPENDENT recomputation that uses date arithmetic
    (merge on "date minus k weeks"), not the shift() calls used by the pipeline itself.
  * The leakage checks accept a `builder` function, so the tests can feed them a deliberately
    leaky builder and prove the checks actually fail when leakage exists.
"""
import sys
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.config import (
    DATE_COL,
    RAIN_LAGS,
    RAIN_ROLL_WINDOWS,
    SPLITS,
    TARGET_COL,
    WARD_COL,
    WEATHER_COLS,
)
from src.feature_engineering import (
    LAG_FEATURE_COLUMNS,
    LAG_TABLE_COLUMNS,
    RAIN_COL,
    build_lag_features,
)
from src.preprocessing import get_clean_data

# Facts established in Phases 2 and 4 about the provided dataset.
EXPECTED = {
    "rows": 31_320,
    "wards": 60,
    "weeks": 522,
    "date_min": "2016-01-04",
    "date_max": "2025-12-29",
}
# Weeks of history needed before every lag / rolling feature is defined.
MAX_HISTORY_WEEKS = max(max(RAIN_LAGS), max(RAIN_ROLL_WINDOWS) - 1)
BANNED_IN_FEATURES = ("target", "split", TARGET_COL, "diag_")

Builder = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values([WARD_COL, DATE_COL]).reset_index(drop=True)


def check_dimensions(clean: pd.DataFrame, expected: dict = EXPECTED) -> CheckResult:
    actual = {
        "rows": int(len(clean)),
        "wards": int(clean[WARD_COL].nunique()),
        "weeks": int(clean[DATE_COL].nunique()),
        "date_min": str(clean[DATE_COL].min().date()),
        "date_max": str(clean[DATE_COL].max().date()),
    }
    return CheckResult("dataset_dimensions", actual == expected, f"actual={actual} expected={expected}")


def check_weekly_grid(clean: pd.DataFrame) -> CheckResult:
    ordered = _sorted(clean)
    steps = ordered.groupby(WARD_COL)[DATE_COL].diff().dropna()
    all_weekly = bool((steps == pd.Timedelta(days=7)).all())
    rows_per_ward = ordered.groupby(WARD_COL).size()
    same_length = rows_per_ward.nunique() == 1
    same_dates = ordered.groupby(WARD_COL)[DATE_COL].apply(tuple).nunique() == 1
    detail = (
        f"all steps exactly 7 days={all_weekly}; same row count in every ward={same_length}; "
        f"every ward has the same dates={same_dates}"
    )
    return CheckResult("weekly_grid", all_weekly and same_length and same_dates, detail)


def reference_lag_values(clean: pd.DataFrame) -> pd.DataFrame:
    """Recompute the rainfall features with DATE ARITHMETIC, independent of shift()/rolling().

    The value k weeks before (ward, date) is looked up by merging on (ward, date - k weeks).
    Rolling sums are the sum of the needed lags and are NaN if any of them is missing.
    """
    base = _sorted(clean)[[WARD_COL, DATE_COL, RAIN_COL]]
    needed = sorted(set(RAIN_LAGS) | set(range(max(RAIN_ROLL_WINDOWS))))
    out = base[[WARD_COL, DATE_COL]].copy()
    for k in needed:
        past = base.copy()
        past[DATE_COL] = past[DATE_COL] + pd.Timedelta(weeks=k)  # value observed at d appears at d + k weeks
        past = past.rename(columns={RAIN_COL: f"_ref_lag_{k}"})
        out = out.merge(past, on=[WARD_COL, DATE_COL], how="left")
    for k in RAIN_LAGS:
        out[f"rain_lag_{k}"] = out[f"_ref_lag_{k}"]
    for window in RAIN_ROLL_WINDOWS:
        cols = [f"_ref_lag_{k}" for k in range(window)]
        out[f"rain_sum_{window}w"] = out[cols].sum(axis=1, skipna=False)
    return out[[WARD_COL, DATE_COL, *LAG_FEATURE_COLUMNS]]


def check_feature_table(clean: pd.DataFrame, table: pd.DataFrame) -> CheckResult:
    problems = []
    if list(table.columns) != LAG_TABLE_COLUMNS:
        problems.append("column list/order differs from LAG_TABLE_COLUMNS")
    ref = reference_lag_values(clean)
    tab = _sorted(table)
    if len(ref) != len(tab):
        problems.append(f"row count {len(tab)} != {len(ref)}")
    else:
        same_keys = (ref[WARD_COL].to_numpy() == tab[WARD_COL].to_numpy()).all() and (
            ref[DATE_COL].to_numpy() == tab[DATE_COL].to_numpy()
        ).all()
        if not same_keys:
            problems.append("(ward_id, date) keys are not aligned with the cleaned data")
        else:
            for col in LAG_FEATURE_COLUMNS:
                same = np.allclose(
                    ref[col].to_numpy(dtype=float), tab[col].to_numpy(dtype=float), equal_nan=True
                )
                if not same:
                    problems.append(f"{col} differs from the independent date-based recomputation")
    n_nan_rows = int(tab[LAG_FEATURE_COLUMNS].isna().any(axis=1).sum()) if len(tab) else 0
    expected_nan_rows = int(clean[WARD_COL].nunique()) * MAX_HISTORY_WEEKS
    if n_nan_rows != expected_nan_rows:
        problems.append(f"rows with NaN features: {n_nan_rows}, expected {expected_nan_rows}")
    detail = (
        f"{len(LAG_FEATURE_COLUMNS)} lag features match the independent recomputation; "
        f"rows lacking history = {n_nan_rows} (expected {expected_nan_rows})"
        if not problems
        else "; ".join(problems)
    )
    return CheckResult("lag_table_matches_independent_recomputation", not problems, detail)


def check_no_banned_columns(table: pd.DataFrame) -> CheckResult:
    features = [c for c in table.columns if c in LAG_FEATURE_COLUMNS or c in WEATHER_COLS]
    offenders = [c for c in table.columns if any(b in c for b in BANNED_IN_FEATURES)]
    return CheckResult(
        "no_target_split_or_diagnostic_columns_in_feature_table",
        not offenders,
        f"{len(features)} feature columns checked; offending columns: {offenders or 'none'}",
    )


def check_future_invariance(
    clean: pd.DataFrame, cutoff: pd.Timestamp, builder: Builder = build_lag_features
) -> CheckResult:
    """Altering every observation AFTER `cutoff` must not change any feature dated <= cutoff."""
    base = _sorted(clean)
    altered = base.copy()
    future = altered[DATE_COL] > cutoff
    columns = [*WEATHER_COLS, TARGET_COL]
    altered.loc[future, columns] = altered.loc[future, columns] * 3.0 + 12_345.0
    before = _sorted(builder(base))
    after = _sorted(builder(altered))
    past = before[DATE_COL] <= cutoff
    try:
        pd.testing.assert_frame_equal(before[past].reset_index(drop=True), after[past].reset_index(drop=True))
        passed, note = True, "features dated on or before the cutoff are unchanged"
    except AssertionError:
        passed, note = False, "FUTURE DATA CHANGED PAST FEATURES"
    return CheckResult(f"future_invariance_{cutoff.date()}", passed, f"{int(future.sum())} future rows altered; {note}")


def check_truncation_invariance(
    clean: pd.DataFrame, cutoff: pd.Timestamp, builder: Builder = build_lag_features
) -> CheckResult:
    """Features built on data cut at `cutoff` must equal features built on all data, then cut."""
    base = _sorted(clean)
    full = _sorted(builder(base))
    truncated = _sorted(builder(base[base[DATE_COL] <= cutoff]))
    expected = full[full[DATE_COL] <= cutoff].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(truncated.reset_index(drop=True), expected)
        passed, note = True, "truncated build equals the full build cut at the same date"
    except AssertionError:
        passed, note = False, "TRUNCATED BUILD DIFFERS FROM FULL BUILD"
    return CheckResult(f"truncation_invariance_{cutoff.date()}", passed, f"{len(truncated)} rows; {note}")


def run_all_checks(clean: pd.DataFrame, builder: Builder = build_lag_features) -> list[CheckResult]:
    """All Step 2A checks on the real cleaned data."""
    table = builder(clean)
    cutoffs = [SPLITS["train"][1], SPLITS["val"][1]]  # the two chronological boundaries
    results = [
        check_dimensions(clean),
        check_weekly_grid(clean),
        check_feature_table(clean, table),
        check_no_banned_columns(table),
    ]
    results += [check_future_invariance(clean, c, builder) for c in cutoffs]
    results += [check_truncation_invariance(clean, c, builder) for c in cutoffs]
    return results


def main() -> int:
    """python -m src.lag_checks : print every check; exit code 1 if any failed."""
    results = run_all_checks(get_clean_data())
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.name}\n       {r.detail}")
    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())