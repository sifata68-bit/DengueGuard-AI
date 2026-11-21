
"""Direct multi-horizon targets and leakage-safe chronological split assignment.

Row convention: one row = (ward, as-of date t). Features may only use information
observed at or before t. The label for horizon h is dengue_cases of the SAME ward
at t + h weeks; the date of that observation is stored as target_date_h{h}.

Split rule (the core of leakage prevention): a row belongs to split S only if
BOTH its as-of date AND its target date lie inside S's date range. A row whose
as-of date is in S but whose target date falls beyond S's end is labelled
"purged" and is dropped from every split. Hence:
  * no training label comes from the validation period or later,
  * no validation label comes from the test period,
  * no test label comes from beyond the end of the data.
"""

import pandas as pd

from src.config import DATE_COL, HORIZONS, SPLITS, TARGET_COL, WARD_COL


def target_name(h: int) -> str:
    return f"target_cases_h{h}"


def target_date_name(h: int) -> str:
    return f"target_date_h{h}"


def split_name(h: int) -> str:
    return f"split_h{h}"


def assign_split(asof: pd.Series, target_date: pd.Series) -> pd.Series:
    """Label each row train / val / test / purged / no_target."""
    label = pd.Series("unassigned", index=asof.index, dtype="object")
    has_target = target_date.notna()

    for name, (start, end) in SPLITS.items():
        asof_in = asof.between(start, end)
        target_in = target_date.between(start, end)

        label[asof_in & has_target & target_in] = name
        label[asof_in & has_target & ~target_in] = "purged"
        label[asof_in & ~has_target] = "no_target"

    return label


def add_targets(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Add target_cases_h{h}, target_date_h{h} and split_h{h} for every horizon.

    The target is taken with a within-ward shift of -h rows, but is only kept
    when the shifted row's date is EXACTLY date + h weeks. A missing week
    therefore gives a NaN target instead of silently pointing at the wrong week.
    """
    out = df.sort_values([WARD_COL, DATE_COL]).reset_index(drop=True).copy()
    grouped = out.groupby(WARD_COL, sort=False)

    for h in horizons:
        shifted_date = grouped[DATE_COL].shift(-h)
        shifted_cases = grouped[TARGET_COL].shift(-h)

        expected = out[DATE_COL] + pd.Timedelta(weeks=h)
        ok = shifted_date.eq(expected)

        out[target_date_name(h)] = expected.where(ok)
        out[target_name(h)] = shifted_cases.where(ok)
        out[split_name(h)] = assign_split(
            out[DATE_COL],
            out[target_date_name(h)],
        )

    return out


def get_split_frame(
    df: pd.DataFrame,
    h: int,
    split: str,
    *,
    final_test_evaluation: bool = False,
) -> pd.DataFrame:
    """Return labelled rows for one split and horizon.

    The test split is locked: it can only be requested with
    final_test_evaluation=True, which is reserved for the one-shot
    final test evaluation in src/evaluation.py.
    """
    if split not in SPLITS:
        raise ValueError(
            f"Unknown split '{split}'. Choose from {list(SPLITS)}."
        )

    if split == "test" and not final_test_evaluation:
        raise PermissionError(
            "The test split is locked during exploration, model selection "
            "and tuning. It is opened once, via the final test evaluation."
        )

    return df[df[split_name(h)] == split].copy()


def audit_splits(df: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Return counts/date ranges per horizon and split, asserting no leakage.

    Only counts and dates are reported; test target values are never reported.
    """
    rows = []

    for h in horizons:
        tcol = target_name(h)
        dcol = target_date_name(h)
        scol = split_name(h)

        for name, (start, end) in SPLITS.items():
            # Select the rows belonging to this specific split.
            part = df[df[scol] == name].copy()

            assert part[tcol].notna().all(), (
                f"h={h} {name}: row without target"
            )
            assert part[DATE_COL].between(start, end).all(), (
                f"h={h} {name}: as-of outside split"
            )
            assert part[dcol].between(start, end).all(), (
                f"h={h} {name}: target date outside split"
            )
            assert (
                part[dcol] == part[DATE_COL] + pd.Timedelta(weeks=h)
            ).all(), f"h={h} {name}: incorrect target date"

            rows.append(
                {
                    "horizon_weeks": h,
                    "split": name,
                    "rows": len(part),
                    "asof_first": (
                        part[DATE_COL].min().date()
                        if not part.empty else None
                    ),
                    "asof_last": (
                        part[DATE_COL].max().date()
                        if not part.empty else None
                    ),
                    "target_date_last": (
                        part[dcol].max().date()
                        if not part.empty else None
                    ),
                    "purged_rows": int(
                        (
                            (df[scol] == "purged")
                            & df[DATE_COL].between(start, end)
                        ).sum()
                    ),
                }
            )

        # Cross-split ordering: no labels cross into the next split.
        tr = df[df[scol] == "train"]
        va = df[df[scol] == "val"]
        te = df[df[scol] == "test"]

        assert tr[dcol].max() < SPLITS["val"][0], (
            f"h={h}: train target reaches validation period"
        )
        assert va[dcol].max() < SPLITS["test"][0], (
            f"h={h}: validation target reaches test period"
        )
        assert te[dcol].max() <= df[DATE_COL].max(), (
            f"h={h}: test target beyond data"
        )

    return pd.DataFrame(rows)
