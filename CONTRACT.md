# Interface contract (v2, solo workflow)

## Shared constants
All column names, horizons, split dates, season months and feature settings live in
`src/config.py`. Never hard-code them elsewhere.

## As-of convention (assumption)
The dataset does not say whether a Monday date is the start or end of the week it describes.
We assume the row for week t is complete and available when the forecast is issued. Features
for as-of week t use only rows dated t or earlier. The target for horizon h is the row h weeks
after t.

## Modules and interfaces
| Module | Public interface | Phase |
|---|---|---|
| `src/data_loader.py` | `load_raw()`, `validate_panel(df)` | 2 |
| `src/preprocessing.py` | `clean(df)`, `clean_with_report(df)`, `get_clean_data(force=False)` (cached; all later stages load data through this) | 4 |
| `src/targets.py` | `add_targets`, `get_split_frame` (test split locked), `audit_splits` | 2 |
| `src/baselines.py`, `src/evaluation.py` | baseline forecasters; MAE/RMSE overall and in-season | 2 |
| `src/feature_engineering.py` | Phase 5: `build_lag_features(df)`, `LAG_FEATURE_COLUMNS`, `LAG_TABLE_COLUMNS`. Phase 6 adds `build_features(df)` and `FEATURE_COLUMNS`. | 5-6 |
| `src/train.py` | `python -m src.train` writes every artifact listed below | 7 |
| `src/risk_classifier.py` | `load_thresholds()`, `classify(predicted_cases)`, `classify_series(s)` | 8 |
| `src/predict.py` | `load_models()`, `predict(asof_date, horizon_weeks, wards=None)`, `explain(ward_id, asof_date, horizon_weeks)` | 9 |
| `app.py` | Streamlit only; reads data through `predict.py`, never trains | 10 |

## Prediction output schema (`predict()`)
`ward_id, city, asof_date, horizon_weeks, target_date, predicted_cases, lower, upper, risk_tier`
- `risk_tier` is one of `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`.
- `lower` / `upper` are NaN when intervals are not available.

## Artifacts in `models/`
`model_h2.joblib`, `model_h3.joblib`, `model_h4.joblib`, `feature_columns.json`,
`risk_thresholds.json`, `selection.json`, `test_metrics.json`, `deployment_meta.json`

## Rules
- Test-split metrics exist only in `test_metrics.json` (written once, Phase 8).
- No mocked or hardcoded predictions anywhere in the shipped app.