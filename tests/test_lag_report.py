import json

import pandas as pd
import pytest

import src.lag_report as report
from src.config import HORIZONS
from src.preprocessing import get_clean_data


@pytest.fixture(scope="module")
def clean():
    return get_clean_data()


def test_diagnostic_summary_selects_best_in_season_rmse():
    rows = []

    for horizon in HORIZONS:
        rows.extend([
            {
                "horizon_weeks": horizon,
                "group": "A",
                "model": "model_a",
                "mae_in_season": 3.0,
                "rmse_in_season": 5.0,
                "n": 100,
            },
            {
                "horizon_weeks": horizon,
                "group": "B",
                "model": "model_b",
                "mae_in_season": 2.5,
                "rmse_in_season": 4.0,
                "n": 100,
            },
        ])

    result = report.build_diagnostic_summary(pd.DataFrame(rows))

    assert len(result) == len(HORIZONS)

    for item in result:
        assert item["best_group"] == "B"
        assert item["best_model"] == "model_b"
        assert item["rmse_in_season"] == 4.0

def test_diagnostic_summary_covers_all_horizons():
    rows = []

    for horizon in HORIZONS:
        rows.append(
            {
                "horizon_weeks": horizon,
                "group": "A",
                "model": "test_model",
                "mae_in_season": 3.0,
                "rmse_in_season": 4.0,
                "n": 100,
            }
        )

    result = report.build_diagnostic_summary(pd.DataFrame(rows))

    assert [item["horizon_weeks"] for item in result] == list(HORIZONS)


def test_report_saves_all_expected_outputs(tmp_path, clean, monkeypatch):
    checks_path = tmp_path / "lag_checks.json"
    summary_path = tmp_path / "lag_report_summary.json"
    diagnostic_path = tmp_path / "lag_abc_diagnostic_val.csv"

    monkeypatch.setattr(report, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(report, "CHECKS_JSON", checks_path)
    monkeypatch.setattr(report, "SUMMARY_JSON", summary_path)
    monkeypatch.setattr(report, "DIAGNOSTIC_CSV", diagnostic_path)

    import src.lag_correlations as correlations

    monkeypatch.setattr(
        correlations,
        "CORRELATIONS_CSV",
        tmp_path / "lag_correlations.csv",
    )
    monkeypatch.setattr(
        correlations,
        "SUMMARY_JSON",
        tmp_path / "lag_correlations_summary.json",
    )

    result = report.run_report(clean)

    assert checks_path.exists()
    assert summary_path.exists()
    assert diagnostic_path.exists()
    assert (tmp_path / "lag_correlations.csv").exists()
    assert (tmp_path / "lag_correlations_summary.json").exists()

    saved = json.loads(summary_path.read_text(encoding="utf-8"))

    assert saved["test_split_accessed"] is False
    assert saved["checks"]["total"] == 8
    assert saved["checks"]["passed"] == 8
    assert saved["checks"]["failed"] == 0

    assert result["abc_diagnostic"]["rows"] == len(HORIZONS) * 8
    assert len(result["abc_diagnostic"]["best_validation_models"]) == len(HORIZONS)


def test_report_does_not_claim_success_when_checks_fail(
    clean,
    monkeypatch,
    tmp_path,
):
    class FailedCheck:
        name = "deliberately_failed_check"
        passed = False
        detail = "intentional test failure"

        def as_dict(self):
            return {
                "name": self.name,
                "passed": self.passed,
                "detail": self.detail,
            }

    monkeypatch.setattr(
        report,
        "run_all_checks",
        lambda _clean: [FailedCheck()],
    )

    monkeypatch.setattr(report, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(report, "CHECKS_JSON", tmp_path / "checks.json")
    monkeypatch.setattr(report, "SUMMARY_JSON", tmp_path / "summary.json")
    monkeypatch.setattr(
        report,
        "DIAGNOSTIC_CSV",
        tmp_path / "diagnostic.csv",
    )

    import src.lag_correlations as correlations

    monkeypatch.setattr(
        correlations,
        "CORRELATIONS_CSV",
        tmp_path / "correlations.csv",
    )
    monkeypatch.setattr(
        correlations,
        "SUMMARY_JSON",
        tmp_path / "correlations_summary.json",
    )

    result = report.run_report(clean)

    assert result["checks"]["failed"] == 1
    assert result["checks"]["passed"] == 0
