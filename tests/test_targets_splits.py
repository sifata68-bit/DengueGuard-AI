import pandas as pd
import pytest

from src.baselines import add_baseline_inputs
from src.config import DATE_COL, HORIZONS, SPLITS, TARGET_COL, WARD_COL
from src.targets import (
    add_targets,
    audit_splits,
    get_split_frame,
    split_name,
    target_date_name,
    target_name,
)

N_WARDS = 3
ENCODE = 100_000


def make_panel() -> pd.DataFrame:
    dates = pd.date_range("2016-01-04", "2025-12-29", freq="7D")
    frames = []
    for w in range(1, N_WARDS + 1):
        frames.append(
            pd.DataFrame(
                {
                    DATE_COL: dates,
                    WARD_COL: f"W{w}",
                    TARGET_COL: [w * ENCODE + i for i in range(len(dates))],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return add_targets(make_panel())


@pytest.mark.parametrize("h", HORIZONS)
def test_target_is_same_ward_value_exactly_h_weeks_ahead(panel, h):
    rows = panel[panel[target_name(h)].notna()]
    assert (rows[target_name(h)] - rows[TARGET_COL] == h).all()
    assert (rows[target_name(h)] // ENCODE == rows[TARGET_COL] // ENCODE).all()


@pytest.mark.parametrize("h", HORIZONS)
def test_train_and_val_targets_never_cross_their_boundary(panel, h):
    train = get_split_frame(panel, h, "train")
    val = get_split_frame(panel, h, "val")
    assert train[target_date_name(h)].max() <= SPLITS["train"][1]
    assert train[target_date_name(h)].max() < SPLITS["val"][0]
    assert val[DATE_COL].min() >= SPLITS["val"][0]
    assert val[target_date_name(h)].max() <= SPLITS["val"][1]
    assert val[target_date_name(h)].max() < SPLITS["test"][0]


@pytest.mark.parametrize("h", HORIZONS)
def test_exactly_h_weeks_per_ward_are_purged_at_each_boundary(panel, h):
    for split in ("train", "val"):
        start, end = SPLITS[split]
        purged = panel[
            (panel[split_name(h)] == "purged")
            & panel[DATE_COL].between(start, end)
        ]
        assert len(purged) == h * N_WARDS


@pytest.mark.parametrize("h", HORIZONS)
def test_last_h_weeks_of_data_have_no_target(panel, h):
    no_target = panel[panel[split_name(h)] == "no_target"]
    assert len(no_target) == h * N_WARDS
    assert no_target[DATE_COL].min() > panel[DATE_COL].max() - pd.Timedelta(weeks=h + 1)


def test_audit_passes_on_clean_panel(panel):
    table = audit_splits(panel)
    assert set(table["split"]) == {"train", "val", "test"}


def test_missing_week_gives_nan_target_not_a_shifted_one():
    base = make_panel()
    gap_date = pd.Timestamp("2019-06-03")
    broken = base[~((base[WARD_COL] == "W1") & (base[DATE_COL] == gap_date))]
    out = add_targets(broken)
    w1 = out[out[WARD_COL] == "W1"].set_index(DATE_COL)
    for h in HORIZONS:
        crossing = gap_date - pd.Timedelta(weeks=h)
        assert pd.isna(w1.loc[crossing, target_name(h)])


def test_test_split_is_locked_until_explicitly_opened(panel):
    with pytest.raises(PermissionError):
        get_split_frame(panel, 2, "test")
    opened = get_split_frame(panel, 2, "test", final_test_evaluation=True)
    assert len(opened) > 0


def test_rolling_baseline_input_ignores_the_future():
    base = make_panel()
    original = add_baseline_inputs(base)
    changed = base.copy()
    idx = changed.index[
        (changed[WARD_COL] == "W1")
        & (changed[DATE_COL] == "2020-03-09")
    ][0]
    changed.loc[idx, TARGET_COL] = 10**9
    altered = add_baseline_inputs(changed)
    before = original[DATE_COL] < pd.Timestamp("2020-03-09")
    assert (
        original.loc[before, "cases_roll4_mean"]
        == altered.loc[before, "cases_roll4_mean"]
    ).all()
