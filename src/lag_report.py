
"""Phase 5 Step 2D: assemble the rainfall-lag analysis report.

Combines:
- Step 2A independent ingestion and leakage checks
- Step 2B train-only rainfall correlation analysis
- Step 2C validation-only A/B/C diagnostic comparison

The locked test split is never accessed.
Run with: python -m src.lag_report
"""

import json
import sys

import pandas as pd

from src.config import HORIZONS, PROCESSED_DIR
from src.diag_models import OUTPUT_PATH as DIAGNOSTIC_CSV
from src.diag_models import run_diagnostic
from src.lag_checks import run_all_checks
from src.lag_correlations import (
    build_analysis_frame,
    compare_with_hints,
    correlation_table,
    save_outputs,
)
from src.preprocessing import get_clean_data


SUMMARY_JSON = PROCESSED_DIR / "lag_report_summary.json"
CHECKS_JSON = PROCESSED_DIR / "lag_checks.json"


def build_diagnostic_summary(results: pd.DataFrame) -> list[dict]:
    """Summarise the best in-season validation result for each horizon."""
    summaries = []

    for horizon in HORIZONS:
        part = results[results["horizon_weeks"] == horizon].copy()

        if part.empty:
            raise ValueError(
                f"No diagnostic results found for horizon {horizon}."
            )

        best = part.sort_values("rmse_in_season").iloc[0]

        summaries.append(
            {
                "horizon_weeks": int(horizon),
                "best_group": str(best["group"]),
                "best_model": str(best["model"]),
                "mae_in_season": float(best["mae_in_season"]),
                "rmse_in_season": float(best["rmse_in_season"]),
                "validation_rows": int(best["n"]),
            }
        )

    return summaries


def run_report(clean: pd.DataFrame) -> dict:
    """Run all Phase 5 Step 2 analyses and assemble their outputs."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 2A: independent checks.
    checks = run_all_checks(clean)
    check_records = [check.as_dict() for check in checks]

    CHECKS_JSON.write_text(
        json.dumps(check_records, indent=2, default=str),
        encoding="utf-8",
    )

    # Step 2B: train-only rainfall correlations.
    analysis_frame = build_analysis_frame(clean)
    correlations = correlation_table(analysis_frame)
    hint_comparison = compare_with_hints(correlations)

    save_outputs(
        correlations,
        hint_comparison,
        csv_path=PROCESSED_DIR / "lag_correlations.csv",
        json_path=PROCESSED_DIR / "lag_correlations_summary.json",
    )

    # Step 2C: train-fit / validation-score A/B/C comparison.
    diagnostic_results = run_diagnostic(clean)
    diagnostic_results.to_csv(DIAGNOSTIC_CSV, index=False)

    # Assemble the final summary.
    summary = {
        "phase": "Phase 5 Step 2",
        "test_split_accessed": False,
        "checks": {
            "total": len(check_records),
            "passed": sum(item["passed"] for item in check_records),
            "failed": sum(not item["passed"] for item in check_records),
            "results": check_records,
        },
        "rainfall_correlations": {
            "rows": len(correlations),
            "phase2_hint_comparison": hint_comparison.to_dict(
                orient="records"
            ),
        },
        "abc_diagnostic": {
            "rows": len(diagnostic_results),
            "best_validation_models": build_diagnostic_summary(
                diagnostic_results
            ),
        },
        "outputs": {
            "checks": str(CHECKS_JSON),
            "correlations_csv": str(
                PROCESSED_DIR / "lag_correlations.csv"
            ),
            "correlations_summary_json": str(
                PROCESSED_DIR / "lag_correlations_summary.json"
            ),
            "diagnostic_csv": str(DIAGNOSTIC_CSV),
            "summary_json": str(SUMMARY_JSON),
        },
    }

    SUMMARY_JSON.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    return summary


def main() -> int:
    clean = get_clean_data()
    summary = run_report(clean)

    print("\nPHASE 5 STEP 2 — FINAL REPORT")
    print("=" * 72)

    print(
        f"Independent checks: "
        f"{summary['checks']['passed']}/{summary['checks']['total']} passed"
    )

    print(
        "\nRainfall correlation rows:",
        summary["rainfall_correlations"]["rows"],
    )

    print("\nBest A/B/C validation models (in-season RMSE):")

    for item in summary["abc_diagnostic"]["best_validation_models"]:
        print(
            f"  Horizon {item['horizon_weeks']} weeks: "
            f"{item['best_group']} / {item['best_model']} — "
            f"MAE={item['mae_in_season']:.3f}, "
            f"RMSE={item['rmse_in_season']:.3f}"
        )

    print("\nIndependent check details:")

    for check in summary["checks"]["results"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['name']}: {check['detail']}")

    print("\nSaved report:", SUMMARY_JSON)

    if summary["checks"]["failed"]:
        print("\nERROR: One or more independent checks failed.")
        return 1

    print("\nAll independent checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
