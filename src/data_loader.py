"""Load and validate the raw hackathon workbook. (Cleaning is extended in Phase 4.)"""
from pathlib import Path

import pandas as pd

from src.config import (
    DATE_COL,
    RAW_DATA_PATH,
    REQUIRED_COLS,
    SHEET_NAME,
    WARD_COL,
)


def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the Excel file, check the schema and return a ward/date-sorted frame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Copy hackathon_preli_dataset.xlsx into data/raw/."
        )
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values([WARD_COL, DATE_COL]).reset_index(drop=True)


def validate_panel(df: pd.DataFrame) -> dict:
    """Facts about the panel that later code relies on (regular weekly spacing, balance)."""
    dup = int(df.duplicated([WARD_COL, DATE_COL]).sum())
    gaps = df.groupby(WARD_COL)[DATE_COL].diff().dropna().dt.days.value_counts().to_dict()
    per_ward = df.groupby(WARD_COL).size()
    return {
        "rows": len(df),
        "wards": int(df[WARD_COL].nunique()),
        "duplicate_ward_date_pairs": dup,
        "day_gaps_between_consecutive_rows": {int(k): int(v) for k, v in gaps.items()},
        "rows_per_ward_min": int(per_ward.min()),
        "rows_per_ward_max": int(per_ward.max()),
        "all_mondays": bool((df[DATE_COL].dt.dayofweek == 0).all()),
        "null_cells": int(df.isna().sum().sum()),
    }