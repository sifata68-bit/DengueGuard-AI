import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.baselines import add_baseline_inputs, baseline_predictions
from src.config import (
    DATE_COL,
    HORIZONS,
    PROCESSED_DIR,
    SPLITS,
    TARGET_COL,
    WARD_COL,
    CITY_COL,
    WEATHER_COLS,
)
from src.evaluation import score
from src.feature_engineering import (
    KEY_COLUMNS,
    LAG_FEATURE_COLUMNS,
    build_lag_features,
)
from src.preprocessing import get_clean_data
from src.targets import (
    add_targets,
    get_split_frame,
    split_name,
    target_date_name,
    target_name,
)

DIAG_CASE_LAGS = (0, 1, 2, 3)
DIAG_CASE_COLUMNS = [
    *(f"diag_cases_lag_{k}" for k in DIAG_CASE_LAGS),
    "diag_cases_roll4_mean",
]

A0_FEATURES = list(WEATHER_COLS)
A_FEATURES = A0_FEATURES + list(LAG_FEATURE_COLUMNS)
B0_FEATURES = list(DIAG_CASE_COLUMNS)
B_FEATURES = A_FEATURES + B0_FEATURES

MODEL_GROUPS = {
    "A0": A0_FEATURES,
    "A": A_FEATURES,
    "B0": B0_FEATURES,
    "B": B_FEATURES,
}

BASELINE_NAMES = (
    "persistence",
    "rolling_mean_4w",
    "ward_month_climatology",
    "global_train_mean",
)

OUTPUT_PATH = PROCESSED_DIR / "lag_abc_diagnostic_val.csv"


def build_diagnostic_frame(clean: pd.DataFrame) -> pd.DataFrame:

    lag_features = build_lag_features(clean)

    ordered = clean.sort_values([WARD_COL, DATE_COL]).reset_index(drop=True)
    cases = ordered.groupby(WARD_COL, sort=False)[TARGET_COL]

    diagnostic = {}
    for k in DIAG_CASE_LAGS:
        diagnostic[f"diag_cases_lag_{k}"] = cases.shift(k)

    diagnostic["diag_cases_roll4_mean"] = cases.transform(
        lambda s: s.rolling(4, min_periods=4).mean()
    )

    diagnostic_frame = pd.concat(
        [
            ordered[KEY_COLUMNS],
            pd.DataFrame(diagnostic),
        ],
        axis=1,
    )

    targeted = add_targets(clean)
    target_columns = [
        target_name(h) for h in HORIZONS
    ] + [
        target_date_name(h) for h in HORIZONS
    ] + [
        split_name(h) for h in HORIZONS
    ]

    frame = lag_features.merge(
        diagnostic_frame,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    frame = frame.merge(
        targeted[KEY_COLUMNS + target_columns],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    baseline_frame = add_baseline_inputs(clean)
    frame = frame.merge(
    baseline_frame[KEY_COLUMNS + [TARGET_COL, "cases_roll4_mean"]],
    on=KEY_COLUMNS,
    how="inner",
    validate="one_to_one",
)


    return frame


def _common_split_rows(
    frame: pd.DataFrame,
    horizon: int,
    split: str,
) -> pd.DataFrame:
    if split not in ("train", "val"):
        raise ValueError("Diagnostic evaluation only permits train and val.")

    part = get_split_frame(frame, horizon, split)

    required = (
        B_FEATURES
        + [target_name(horizon)]
    )

    part = part.dropna(subset=required).copy()

    if part.empty:
        raise ValueError(
            f"No eligible {split} rows remain for horizon {horizon}."
        )

    return part


def run_diagnostic(
    clean: pd.DataFrame,
    horizons=HORIZONS,
) -> pd.DataFrame:
    frame = build_diagnostic_frame(clean)
    results = []

    fit_end = SPLITS["train"][1]

    for h in horizons:
        train = _common_split_rows(frame, h, "train")
        val = _common_split_rows(frame, h, "val")

        if not train.index.is_unique or not val.index.is_unique:
            raise AssertionError("Duplicate rows found in diagnostic splits.")

        y_train = train[target_name(h)].to_numpy(dtype=float)
        y_val = val[target_name(h)].to_numpy(dtype=float)
        target_dates = val[target_date_name(h)]

        for group, feature_columns in MODEL_GROUPS.items():
            model = HistGradientBoostingRegressor(
                early_stopping=False,
                random_state=0,
            )

            model.fit(train[feature_columns], y_train)

            predictions = np.maximum(
                model.predict(val[feature_columns]),
                0.0,
            )

            results.append(
                {
                    "horizon_weeks": h,
                    "group": group,
                    "model": "HistGradientBoostingRegressor",
                    "n_train": len(train),
                    **score(y_val, predictions, target_dates),
                }
            )

        baseline_preds = baseline_predictions(
            frame,
            val,
            h,
            fit_end,
        )

        for name in BASELINE_NAMES:
            predictions = np.maximum(
                baseline_preds[name].to_numpy(dtype=float),
                0.0,
            )

            results.append(
                {
                    "horizon_weeks": h,
                    "group": "C",
                    "model": name,
                    "n_train": len(train),
                    **score(y_val, predictions, target_dates),
                }
            )

    result = pd.DataFrame(results)

    for h in horizons:
        counts = result.loc[
            result["horizon_weeks"] == h, "n"
        ].unique()

        if len(counts) != 1:
            raise AssertionError(
                f"Different validation row counts for horizon {h}: {counts}"
            )

    return result


def main():
    clean = get_clean_data()
    results = run_diagnostic(clean)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nA/B/C DIAGNOSTIC VALIDATION RESULTS")
    print("=" * 90)

    columns = [
        "horizon_weeks",
        "group",
        "model",
        "n_train",
        "n",
        "mae",
        "rmse",
        "mae_in_season",
        "rmse_in_season",
    ]

    print(
        results[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
