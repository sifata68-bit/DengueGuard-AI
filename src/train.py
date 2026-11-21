
"""Production training pipeline for DengueGuard AI.

Trains one model per forecast horizon using leakage-safe production features.

Training data is used for fitting.
Validation data is used for model selection.
The test period (2024-2025) is excluded entirely from this pipeline.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import (
    DATE_COL,
    HORIZONS,
    MODELS_DIR,
    PROCESSED_DIR,
    SPLITS,
)
from src.evaluation import score
from src.feature_engineering import KEY_COLUMNS
from src.preprocessing import get_clean_data
from src.production_features import (
    PRODUCTION_FEATURE_COLUMNS,
    build_production_features,
)
from src.targets import (
    add_targets,
    get_split_frame,
    target_date_name,
    target_name,
    split_name,
)


MODEL_CANDIDATES = {
    "hist_gradient_boosting": {
        "learning_rate": 0.1,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 0,
    },
    "hist_gradient_boosting_small": {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 10,
        "min_samples_leaf": 50,
        "l2_regularization": 2.0,
        "early_stopping": False,
        "random_state": 0,
    },
}

MODEL_OUTPUT_DIR = MODELS_DIR / "production"
METRICS_PATH = PROCESSED_DIR / "production_validation_metrics.csv"
SUMMARY_PATH = MODEL_OUTPUT_DIR / "training_summary.json"


def prepare_split(frame, horizon, split):
    """Return complete feature and target rows for one horizon and split."""
    part = get_split_frame(frame, horizon, split)

    target_column = target_name(horizon)

    part = part.dropna(
        subset=PRODUCTION_FEATURE_COLUMNS + [target_column]
    ).copy()

    if part.empty:
        raise ValueError(
            f"No eligible {split} rows for horizon {horizon}."
        )

    X = part[PRODUCTION_FEATURE_COLUMNS].astype(float)
    y = part[target_column].astype(float)

    return part, X, y


def train_models(clean, horizons=HORIZONS):
    """Train candidates and select one model per horizon using validation."""

    # Exclude the test period entirely from model development.
    development_data = clean[
        clean[DATE_COL] <= SPLITS["val"][1]
    ].copy()

    if development_data.empty:
        raise ValueError("No training or validation data available.")

    # Build features and targets using development data only.
    features = build_production_features(development_data)
    targets = add_targets(development_data)

    # Keep only target and split columns needed for the requested horizons.
    target_columns = (
        [
            target_name(h)
            for h in horizons
        ]
        + [
            target_date_name(h)
            for h in horizons
        ]
        + [
            split_name(h)
            for h in horizons
        ]
    )

    frame = features.merge(
        targets[KEY_COLUMNS + target_columns],
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )

    results = []
    selected_models = {}
    summary = {}

    for horizon in horizons:
        train, X_train, y_train = prepare_split(
            frame, horizon, "train"
        )

        val, X_val, y_val = prepare_split(
            frame, horizon, "val"
        )

        target_dates = val[target_date_name(horizon)]

        candidates = []

        print(f"\nTraining {horizon}-week forecast models")
        print("-" * 55)

        for model_name, parameters in MODEL_CANDIDATES.items():
            model = HistGradientBoostingRegressor(**parameters)

            # Fit using training data only.
            model.fit(X_train, y_train)

            # Evaluate using validation data only.
            predictions = np.maximum(
                model.predict(X_val),
                0.0,
            )

            metrics = score(
                y_val,
                predictions,
                target_dates,
            )

            result = {
                "horizon_weeks": horizon,
                "model": model_name,
                "n_train": len(X_train),
                **metrics,
            }

            results.append(result)
            candidates.append((model_name, model, metrics))

            print(
                f"{model_name}: "
                f"validation RMSE={metrics['rmse']:.3f}, "
                f"in-season RMSE={metrics['rmse_in_season']:.3f}"
            )

        # Select using in-season RMSE, with overall RMSE as a tie-breaker.
        best_name, best_model, best_metrics = min(
            candidates,
            key=lambda item: (
                item[2]["rmse_in_season"],
                item[2]["rmse"],
            ),
        )

        selected_models[horizon] = best_model

        summary[str(horizon)] = {
            "selected_model": best_name,
            "feature_count": len(PRODUCTION_FEATURE_COLUMNS),
            "feature_columns": PRODUCTION_FEATURE_COLUMNS,
            "n_train": len(X_train),
            "n_validation": len(X_val),
            "validation_metrics": best_metrics,
        }

        print(
            f"Selected: {best_name} "
            f"(in-season RMSE={best_metrics['rmse_in_season']:.3f})"
        )

    return selected_models, pd.DataFrame(results), summary


def main():
    print("Loading cleaned data...")
    clean = get_clean_data()

    print("Training production models...")
    models, metrics, summary = train_models(clean)

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save the selected model for each forecast horizon.
    for horizon, model in models.items():
        model_path = (
            MODEL_OUTPUT_DIR / f"dengue_h{horizon}_model.joblib"
        )

        joblib.dump(
            {
                "model": model,
                "horizon_weeks": horizon,
                "feature_columns": PRODUCTION_FEATURE_COLUMNS,
            },
            model_path,
        )

        print(f"Saved: {model_path}")

    # Save validation metrics and training metadata.
    metrics.to_csv(METRICS_PATH, index=False)

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\nValidation metrics saved: {METRICS_PATH}")
    print(f"Training summary saved: {SUMMARY_PATH}")

    print("\nFINAL VALIDATION COMPARISON")
    print("=" * 75)

    print(
        metrics[
            [
                "horizon_weeks",
                "model",
                "n_train",
                "n",
                "mae",
                "rmse",
                "mae_in_season",
                "rmse_in_season",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )


if __name__ == "__main__":
    main()
