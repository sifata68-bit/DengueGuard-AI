import pandas as pd

from src.baselines import add_baseline_inputs, baseline_predictions
from src.config import (
    DATE_COL,
    HORIZONS,
    IN_SEASON_MONTHS,
    PROCESSED_DIR,
    SPLITS,
    STATIC_COLS,
    TARGET_COL,
    WARD_COL,
)
from src.data_loader import load_raw, validate_panel
from src.evaluation import comparison_table, in_season_mask
from src.targets import (
    add_targets,
    audit_splits,
    get_split_frame,
    target_date_name,
    target_name,
)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> None:
    df = load_raw()

    section("1. PANEL VALIDATION")
    for key, value in validate_panel(df).items():
        print(f"{key:40s} {value}")

    df = add_baseline_inputs(add_targets(df))

    section("2. LEAKAGE AUDIT (asserts inside; counts and dates only)")
    audit = audit_splits(df)
    print(audit.to_string(index=False))
    print("\nAll leakage assertions passed.")

    section("3. TARGET DISTRIBUTION (train and validation only)")
    for h in HORIZONS:
        rows = []
        for split in ("train", "val"):
            y = get_split_frame(df, h, split)[target_name(h)]
            rows.append(
                {
                    "horizon": h,
                    "split": split,
                    "rows": len(y),
                    "mean": y.mean(),
                    "median": y.median(),
                    "p90": y.quantile(0.90),
                    "p95": y.quantile(0.95),
                    "p99": y.quantile(0.99),
                    "max": y.max(),
                    "share_zero": (y == 0).mean(),
                }
            )
        print(pd.DataFrame(rows).round(3).to_string(index=False))

    section("4. SEASONALITY (train + validation weeks, by calendar month)")
    dev = df[df[DATE_COL] <= SPLITS["val"][1]]
    monthly = dev.groupby(dev[DATE_COL].dt.month)[TARGET_COL].agg(["mean", "sum"])
    monthly["share_of_cases_%"] = 100 * monthly["sum"] / monthly["sum"].sum()
    print(monthly.round(3).to_string())
    season_share = monthly.loc[list(IN_SEASON_MONTHS), "share_of_cases_%"].sum()
    print(f"\nIN_SEASON_MONTHS = {IN_SEASON_MONTHS} hold {season_share:.1f}% of train+val cases.")

    section("5. RAINFALL LAG vs FUTURE CASES (train split only)")
    print("Pearson r between rainfall at (t - k) and cases at (t + h). In-season columns use")
    print("only target weeks inside the season, which removes the shared seasonal pattern.")
    lagged = df.copy()
    grouped = lagged.groupby(WARD_COL, sort=False)["rainfall_mm"]
    for k in range(0, 7):
        lagged[f"rain_lag{k}"] = grouped.shift(k)
    table = {}
    for h in HORIZONS:
        train = get_split_frame(lagged, h, "train")
        mask = in_season_mask(train[target_date_name(h)])
        for k in range(0, 7):
            col = f"rain_lag{k}"
            table.setdefault(f"h{h}_all", {})[k] = train[col].corr(train[target_name(h)])
            table.setdefault(f"h{h}_inseason", {})[k] = (
                train.loc[mask, col].corr(train.loc[mask, target_name(h)])
            )
    out = pd.DataFrame(table)
    out.index.name = "rain_lag_weeks_k"
    print(out.round(3).to_string())
    print("\nNOTE: correlation is a feature-design hint, NOT evidence of forecast skill.")
    print("Skill is judged only by trained-model vs baseline errors on held-out data.")

    section("6. WARD VULNERABILITY vs MEAN CASES (train split rows, h=2 labels)")
    train2 = get_split_frame(df, 2, "train")
    ward = train2.groupby(WARD_COL).agg(
        mean_target=(target_name(2), "mean"), **{c: (c, "first") for c in STATIC_COLS}
    )
    print(ward.corr()["mean_target"].drop("mean_target").round(3).to_string())

    section("7. BASELINE ERRORS ON VALIDATION (no ML; fitted on train period only)")
    fit_end = SPLITS["train"][1]
    saved = []
    for h in HORIZONS:
        val = get_split_frame(df, h, "val")
        preds = baseline_predictions(df, val, h, fit_end)
        table = comparison_table(val[target_name(h)], val[target_date_name(h)], preds)
        print(f"\nHorizon {h} weeks  (validation rows: {len(val)})")
        print(table.round(3).to_string(index=False))
        table.insert(0, "horizon_weeks", h)
        saved.append(table)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "baseline_val_metrics.csv"
    pd.concat(saved).to_csv(out_path, index=False)
    print(f"\nSaved baseline validation metrics to {out_path}")
    print("The trained models (Phase 7) must beat the best of these, especially in-season.")


if __name__ == "__main__":
    main()