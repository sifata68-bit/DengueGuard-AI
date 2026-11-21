"""Central configuration. Every other module imports constants from here."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "hackathon_preli_dataset.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

SHEET_NAME = "bangladesh_dengue_large_dataset"

# --- Column names (exactly as they appear in the provided dataset) ---
DATE_COL = "date"
CITY_COL = "city"
WARD_COL = "ward_id"
TARGET_COL = "dengue_cases"
WEATHER_COLS = [
    "avg_temp_c",
    "max_temp_c",
    "min_temp_c",
    "rainfall_mm",
    "relative_humidity_pct",
]
STATIC_COLS = [
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
]
REQUIRED_COLS = [DATE_COL, CITY_COL, WARD_COL, *STATIC_COLS, *WEATHER_COLS, TARGET_COL]

# --- Forecast horizons (weeks ahead of the as-of date) ---
HORIZONS = (2, 3, 4)

# --- Chronological split. Bounds are inclusive and apply to BOTH the as-of date
# and the target date of a row (see src/targets.py). ---
SPLITS = {
    "train": (pd.Timestamp("2016-01-04"), pd.Timestamp("2021-12-31")),
    "val": (pd.Timestamp("2022-01-03"), pd.Timestamp("2023-12-31")),
    "test": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31")),
}

# Dengue season used for the "in-season" metrics. Months are of the TARGET date,
# i.e. the week being forecast.
IN_SEASON_MONTHS = (6, 7, 8, 9, 10)

# --- Phase 4: cleaning and cache ---
CLEAN_VERSION = 1  # bump when cleaning logic changes, so cached files are rebuilt
MAX_WEATHER_FFILL_WEEKS = 2  # causal (past-only) gap filling limit for weather columns
CLEAN_CSV_PATH = PROCESSED_DIR / "clean_weekly.csv"
CLEAN_META_PATH = PROCESSED_DIR / "clean_weekly.meta.json"
QUALITY_REPORT_PATH = PROCESSED_DIR / "data_quality_report.json"

# --- Phase 5: rainfall lag pipeline ---
RAIN_LAGS = tuple(range(0, 7))  # weeks BEFORE the as-of week (0 = the as-of week itself)
RAIN_ROLL_WINDOWS = (2, 4, 6)  # trailing rainfall sums ending at the as-of week (inclusive)