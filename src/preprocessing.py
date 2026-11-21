"""Phase 4: validation, cleaning and weekly-grid regularisation, plus a cached loader.

Design rules
  * Fix only what is physically unambiguous (duplicates, humidity above 100 %, negative
    rainfall). Everything else that looks wrong is REPORTED, never silently altered.
  * Lag features later use row shifts inside each ward, which are only correct if every ward
    has one row per week. clean() therefore inserts any missing weeks as explicit rows.
  * Gaps in weather are filled FORWARD ONLY (last observed value, limited number of weeks),
    so no future information can leak backwards. Missing case counts are never imputed:
    the pipeline fails loudly instead of inventing outcomes.
  * The provided dataset needs none of these repairs (all counters are 0); the code exists
    so the same pipeline stays safe on new data in later stages.
"""
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from src.config import (
    CITY_COL,
    CLEAN_CSV_PATH,
    CLEAN_META_PATH,
    CLEAN_VERSION,
    DATE_COL,
    MAX_WEATHER_FFILL_WEEKS,
    QUALITY_REPORT_PATH,
    RAW_DATA_PATH,
    REQUIRED_COLS,
    STATIC_COLS,
    TARGET_COL,
    WARD_COL,
    WEATHER_COLS,
)
from src.data_loader import load_raw

NUMERIC_COLS = [*STATIC_COLS, *WEATHER_COLS, TARGET_COL]


class DataQualityError(ValueError):
    """Raised when the data cannot be made safe for forecasting without guessing."""


def clean_with_report(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (cleaned frame, report of everything that was checked or changed)."""
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        raise DataQualityError(f"Missing required columns: {missing_cols}")

    report: dict = {"rows_in": int(len(df))}
    out = df[REQUIRED_COLS].copy()

    # 1. Types: dates, identifiers, numerics (non-numeric text becomes NaN and is counted).
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    for col in (WARD_COL, CITY_COL):
        out[col] = out[col].astype("string").str.strip()
    before_nan = out[NUMERIC_COLS].isna().sum()
    for col in NUMERIC_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    report["cells_failed_numeric_conversion"] = int((out[NUMERIC_COLS].isna().sum() - before_nan).sum())

    # 2. Rows without a usable key cannot be placed on the grid.
    bad_key = out[DATE_COL].isna() | out[WARD_COL].isna() | (out[WARD_COL] == "")
    report["rows_dropped_missing_key"] = int(bad_key.sum())
    out = out[~bad_key]

    # 3. Duplicate (ward, date) rows: keep the first occurrence.
    dup = out.duplicated([WARD_COL, DATE_COL], keep="first")
    report["duplicate_ward_date_rows_dropped"] = int(dup.sum())
    out = out[~dup]

    # 4. Ward identity: one city per ward, constant vulnerability attributes per ward.
    multi_city = out.groupby(WARD_COL)[CITY_COL].nunique(dropna=True)
    if (multi_city > 1).any():
        raise DataQualityError(f"Wards assigned to more than one city: {list(multi_city[multi_city > 1].index)}")
    for col in STATIC_COLS:
        varying = out.groupby(WARD_COL)[col].nunique(dropna=True)
        if (varying > 1).any():
            raise DataQualityError(f"Static column '{col}' varies within wards: {list(varying[varying > 1].index)[:5]}")

    # 5. Physical range rules.
    rain_bad = out["rainfall_mm"] < 0
    report["negative_rainfall_set_missing"] = int(rain_bad.sum())
    out.loc[rain_bad, "rainfall_mm"] = float("nan")
    hum = out["relative_humidity_pct"]
    report["humidity_above_100_clipped"] = int((hum > 100).sum())
    report["humidity_below_0_set_missing"] = int((hum < 0).sum())
    out.loc[hum < 0, "relative_humidity_pct"] = float("nan")
    out["relative_humidity_pct"] = out["relative_humidity_pct"].clip(upper=100)
    cases_bad = out[TARGET_COL] < 0
    report["negative_case_counts_set_missing"] = int(cases_bad.sum())
    out.loc[cases_bad, TARGET_COL] = float("nan")
    non_integer = out[TARGET_COL].notna() & (out[TARGET_COL] % 1 != 0)
    report["non_integer_case_counts_rounded"] = int(non_integer.sum())
    out[TARGET_COL] = out[TARGET_COL].round()
    # Flagged only (never altered): temperature ordering must be min <= avg <= max.
    report["temperature_order_violations_flagged"] = int(
        ((out["min_temp_c"] > out["avg_temp_c"]) | (out["avg_temp_c"] > out["max_temp_c"])).sum()
    )

    # 6. Weekly grid: every ward gets one row per 7-day step between its first and last date.
    offset_days = (out[DATE_COL] - out.groupby(WARD_COL)[DATE_COL].transform("min")).dt.days
    if (offset_days % 7 != 0).any():
        raise DataQualityError("Some dates are not on the ward's 7-day weekly lattice.")
    bounds = out.groupby(WARD_COL)[DATE_COL].agg(["min", "max"])
    grid = pd.concat(
        [
            pd.DataFrame({WARD_COL: ward, DATE_COL: pd.date_range(row["min"], row["max"], freq="7D")})
            for ward, row in bounds.iterrows()
        ],
        ignore_index=True,
    )
    grid[WARD_COL] = grid[WARD_COL].astype("string")
    full = grid.merge(out, on=[WARD_COL, DATE_COL], how="left")
    report["missing_week_rows_inserted"] = int(len(full) - len(out))
    for col in (CITY_COL, *STATIC_COLS):
        full[col] = full.groupby(WARD_COL)[col].transform("first")

    # 7. Weather gaps: forward fill only (past information), limited number of weeks.
    full = full.sort_values([WARD_COL, DATE_COL]).reset_index(drop=True)
    nan_before = int(full[WEATHER_COLS].isna().sum().sum())
    full[WEATHER_COLS] = full.groupby(WARD_COL)[WEATHER_COLS].ffill(limit=MAX_WEATHER_FFILL_WEEKS)
    report["weather_cells_forward_filled"] = nan_before - int(full[WEATHER_COLS].isna().sum().sum())

    # 8. Anything still missing cannot be repaired without guessing -> fail loudly.
    remaining = full[REQUIRED_COLS].isna().sum()
    remaining = remaining[remaining > 0]
    if len(remaining):
        raise DataQualityError(
            "Unrepairable missing values remain (case counts are never imputed and weather "
            f"gaps longer than {MAX_WEATHER_FFILL_WEEKS} weeks are not filled): {remaining.to_dict()}"
        )
    full[TARGET_COL] = full[TARGET_COL].astype("int64")
    for col in (WARD_COL, CITY_COL):
        full[col] = full[col].astype(object)

    report["rows_out"] = int(len(full))
    report["wards"] = int(full[WARD_COL].nunique())
    report["date_min"] = str(full[DATE_COL].min().date())
    report["date_max"] = str(full[DATE_COL].max().date())
    return full[REQUIRED_COLS], report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Interface from CONTRACT.md: cleaned frame, sorted by ward_id then date."""
    return clean_with_report(df)[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def get_clean_data(
    force: bool = False,
    raw_path: Path = RAW_DATA_PATH,
    clean_path: Path = CLEAN_CSV_PATH,
    meta_path: Path = CLEAN_META_PATH,
    report_path: Path = QUALITY_REPORT_PATH,
) -> pd.DataFrame:
    """Cleaned data, parsed from Excel only when the cache is missing or stale.

    The cache is keyed on the SHA-256 of the raw file plus CLEAN_VERSION, so editing the
    data or the cleaning logic always triggers a rebuild, and nothing else does.
    """
    raw_path, clean_path, meta_path, report_path = map(Path, (raw_path, clean_path, meta_path, report_path))
    digest = _sha256(raw_path) if raw_path.exists() else None
    if not force and digest and clean_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("source_sha256") == digest and meta.get("clean_version") == CLEAN_VERSION:
            return pd.read_csv(clean_path, parse_dates=[DATE_COL])
    cleaned, report = clean_with_report(load_raw(raw_path))
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(clean_path, index=False)
    meta_path.write_text(json.dumps({"source_sha256": digest, "clean_version": CLEAN_VERSION}, indent=2))
    report_path.write_text(json.dumps(report, indent=2))
    return cleaned


def main() -> None:
    """python -m src.preprocessing : rebuild the cache, show the report and cache timings."""
    start = time.perf_counter()
    df = get_clean_data(force=True)
    cold = time.perf_counter() - start
    start = time.perf_counter()
    get_clean_data()
    warm = time.perf_counter() - start
    print("Data quality report (data/processed/data_quality_report.json):")
    for key, value in json.loads(QUALITY_REPORT_PATH.read_text()).items():
        print(f"  {key:42s} {value}")
    print(f"\nCleaned frame: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Rebuild from Excel: {cold:.2f}s   |   cached load: {warm:.2f}s")


if __name__ == "__main__":
    main()