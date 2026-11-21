
import pandas as pd

from src.config import DATE_COL, HORIZONS, SPLITS, TARGET_COL, WARD_COL


def target_name(h: int) -> str:
    return f"target_cases_h{h}"


def target_date_name(h: int) -> str:
    return f"target_date_h{h}"


def split_name(h: int) -> str:
    return f"split_h{h}"


def assign_split(asof: pd.Series, target_date: pd.Series) -> pd.Series:
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
    rows = []

    for h in horizons:
        tcol = target_name(h)
        dcol = target_date_name(h)
        scol = split_name(h)

        for name, (start, end) in SPLITS.items():
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
