"""Phase 5, Step 2B: rainfall lag correlations versus the Phase 2 findings.

Run on its own with:   python -m src.lag_correlations

What this is and is not
  * It is a DESCRIPTIVE check that the new lag features carry the relationship seen in Phase 2.
  * It is NOT evidence of forecast skill and NOT used to select features or models.
    Skill is judged only by held-out errors against baselines (Step 2C, then Phase 7).
  * Only the TRAIN split is used (rows whose as-of date AND target date are inside the train
    period). The locked test split is never requested by this module.

Method
  * One row = (ward, as-of week t). Target = dengue cases of the same ward h weeks later.
  * Pearson correlation between each rainfall feature and the horizon's target, using every
    train row where both values exist (pairwise-complete, the same rule as Phase 2).
  * Reported twice: over all rows, and over in-season target weeks only, which removes part of
    the seasonal pattern that rainfall and cases share.
  * "total delay" = lag + horizon = weeks between the rain and the cases it is compared with.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATE_COL,
    HORIZONS,
    PROCESSED_DIR,
    RAIN_LAGS,
    RAIN_ROLL_WINDOWS,
    WARD_COL,
)
from src.evaluation import in_season_mask
from src.feature_engineering import build_lag_features
from src.preprocessing import get_clean_data
from src.targets import add_targets, get_split_frame, split_name, target_date_name, target_name

# Phase 2 findings (from the Phase 2 report): strongest rainfall lag and its Pearson r (3 d.p.).
HINTS = {2: (1, 0.738), 3: (0, 0.738), 4: (0, 0.685)}
# Phase 2 also showed the strongest rain-to-cases relationship at a total delay of about 3 weeks.
PHASE2_TOTAL_DELAY_HINT_WEEKS = 3

CORRELATIONS_CSV = PROCESSED_DIR / "lag_correlations.csv"
SUMMARY_JSON = PROCESSED_DIR / "lag_correlations_summary.json"


def build_analysis_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """Lag feature table joined with every horizon's target, target date and split label."""
    table = build_lag_features(clean)
    labelled = add_targets(clean)
    keep = [WARD_COL, DATE_COL]
    for h in HORIZONS:
        keep += [target_name(h), target_date_name(h), split_name(h)]
    return table.merge(labelled[keep], on=[WARD_COL, DATE_COL], how="left", validate="one_to_one")


def correlation_table(frame: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Long table: one row per (horizon, rainfall feature), computed on the train split only."""
    features = [("lag", k, f"rain_lag_{k}") for k in RAIN_LAGS]
    features += [("rolling_sum", w, f"rain_sum_{w}w") for w in RAIN_ROLL_WINDOWS]
    rows = []
    for h in horizons:
        train = get_split_frame(frame, h, "train")
        season = in_season_mask(train[target_date_name(h)])
        y = train[target_name(h)]
        for kind, number, col in features:
            x = train[col]
            rows.append(
                {
                    "horizon": h,
                    "feature": col,
                    "kind": kind,
                    "lag_weeks": number if kind == "lag" else np.nan,
                    "window_weeks": number if kind == "rolling_sum" else np.nan,
                    "total_delay_weeks": number + h if kind == "lag" else np.nan,
                    "r_all": x.corr(y),
                    "r_in_season": x[season].corr(y[season]),
                    "n_all": int((x.notna() & y.notna()).sum()),
                    "n_in_season": int((x[season].notna() & y[season].notna()).sum()),
                }
            )
    return pd.DataFrame(rows)


def compare_with_hints(table: pd.DataFrame, hints: dict = HINTS) -> pd.DataFrame:
    """For each horizon: strongest observed lag versus the Phase 2 hint, plus the 3-week ideal."""
    rows = []
    for h, (hint_lag, hint_r) in hints.items():
        lags = table[(table["horizon"] == h) & (table["kind"] == "lag")]
        best = lags.loc[lags["r_all"].idxmax()]
        best_season = lags.loc[lags["r_in_season"].idxmax()]
        rolling = table[(table["horizon"] == h) & (table["kind"] == "rolling_sum")]
        best_roll = rolling.loc[rolling["r_all"].idxmax()]
        ideal_lag = PHASE2_TOTAL_DELAY_HINT_WEEKS - h
        ideal_available = ideal_lag >= 0
        r_ideal = (
            float(lags.loc[lags["lag_weeks"] == ideal_lag, "r_all"].iloc[0]) if ideal_available else np.nan
        )
        rows.append(
            {
                "horizon": h,
                "hint_lag": hint_lag,
                "hint_r": hint_r,
                "observed_best_lag": int(best["lag_weeks"]),
                "observed_best_r": float(best["r_all"]),
                "lag_matches_hint": int(best["lag_weeks"]) == hint_lag,
                "r_matches_hint_3dp": round(float(best["r_all"]), 3) == hint_r,
                "best_total_delay_weeks": int(best["total_delay_weeks"]),
                "best_lag_in_season": int(best_season["lag_weeks"]),
                "best_r_in_season": float(best_season["r_in_season"]),
                "ideal_3w_delay_lag": ideal_lag,
                "ideal_3w_delay_available_at_asof": ideal_available,
                "r_at_ideal_3w_delay": r_ideal,
                "best_rolling_feature": best_roll["feature"],
                "best_rolling_r_all": float(best_roll["r_all"]),
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    table: pd.DataFrame,
    hints_table: pd.DataFrame,
    csv_path: Path = CORRELATIONS_CSV,
    json_path: Path = SUMMARY_JSON,
) -> None:
    """Write the long correlation table (CSV) and the hint comparison (JSON)."""
    csv_path, json_path = Path(csv_path), Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(hints_table.to_dict(orient="records"), indent=2, default=str))


def _print_horizon_tables(table: pd.DataFrame) -> None:
    for h in sorted(table["horizon"].unique()):
        part = table[table["horizon"] == h]
        lags = part[part["kind"] == "lag"][
            ["lag_weeks", "total_delay_weeks", "r_all", "r_in_season", "n_all", "n_in_season"]
        ].copy()
        lags[["lag_weeks", "total_delay_weeks"]] = lags[["lag_weeks", "total_delay_weeks"]].astype(int)
        print(f"\nHorizon {h} weeks: rainfall lag features (train split)")
        print(lags.round(3).to_string(index=False))
        roll = part[part["kind"] == "rolling_sum"][["feature", "r_all", "r_in_season", "n_all"]]
        print(roll.round(3).to_string(index=False))


def main() -> int:
    clean = get_clean_data()
    frame = build_analysis_frame(clean)
    table = correlation_table(frame)
    hints_table = compare_with_hints(table)
    save_outputs(table, hints_table)

    print("=" * 78 + "\nRAINFALL LAG CORRELATIONS (train split only; test split never touched)\n" + "=" * 78)
    _print_horizon_tables(table)

    print("\n" + "=" * 78 + "\nCOMPARISON WITH PHASE 2 FINDINGS\n" + "=" * 78)
    show = hints_table[
        [
            "horizon",
            "hint_lag",
            "hint_r",
            "observed_best_lag",
            "observed_best_r",
            "lag_matches_hint",
            "r_matches_hint_3dp",
            "best_total_delay_weeks",
            "best_lag_in_season",
            "best_r_in_season",
        ]
    ]
    print(show.round(3).to_string(index=False))
    print("\nThe ideal 3-week rain-to-cases delay, and whether it can be used at the as-of date:")
    ideal = hints_table[
        ["horizon", "ideal_3w_delay_lag", "ideal_3w_delay_available_at_asof", "r_at_ideal_3w_delay"]
    ]
    print(ideal.round(3).to_string(index=False))

    print(
        "\nHow to read this:\n"
        "  * These are correlations, not forecast skill. Nothing was selected from them.\n"
        "  * In-season correlations are lower than all-row ones: part of the all-row association\n"
        "    is the shared seasonal cycle of rainfall and cases.\n"
        "  * At horizon 4 the ideal 3-week delay would need rainfall from the week AFTER the as-of\n"
        "    week (lag -1), which does not exist yet, so the best usable lag is the as-of week.\n"
        f"\nSaved: {CORRELATIONS_CSV}\n       {SUMMARY_JSON}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())