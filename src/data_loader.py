
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Check that the Phase 2 dataset exists in data/raw/."
        )
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df.sort_values([WARD_COL, DATE_COL]).reset_index(drop=True)


def validate_panel(df: pd.DataFrame) -> dict:
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
