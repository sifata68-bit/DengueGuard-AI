"""Reference forecasters every trained model must beat (no machine learning, no test data).

All baselines use only information available at the as-of date t.
  persistence            : cases(t+h) = cases(t)
  rolling_mean_4w        : mean of the ward's cases over weeks t-3..t
  ward_month_climatology : mean cases of that ward in the TARGET month, fitted only on
                           data up to `fit_end` (seasonal climatology)
  global_train_mean      : one constant, fitted only on data up to `fit_end`
"""
import pandas as pd

from src.config import DATE_COL, TARGET_COL, WARD_COL
from src.targets import target_date_name


def add_baseline_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Add cases_roll4_mean (trailing window that includes t and never looks forward)."""
    out = df.sort_values([WARD_COL, DATE_COL]).reset_index(drop=True).copy()
    out["cases_roll4_mean"] = out.groupby(WARD_COL, sort=False)[TARGET_COL].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    return out


def fit_climatology(df: pd.DataFrame, fit_end: pd.Timestamp) -> pd.Series:
    hist = df[df[DATE_COL] <= fit_end]
    return hist.groupby([WARD_COL, hist[DATE_COL].dt.month])[TARGET_COL].mean()


def predict_climatology(frame: pd.DataFrame, h: int, clim: pd.Series) -> pd.Series:
    idx = pd.MultiIndex.from_arrays([frame[WARD_COL], frame[target_date_name(h)].dt.month])
    return pd.Series(clim.reindex(idx).to_numpy(), index=frame.index)


def baseline_predictions(
    df: pd.DataFrame, frame: pd.DataFrame, h: int, fit_end: pd.Timestamp
) -> dict:
    """Predictions of all baselines for `frame` (rows of one split).

    `df` must come from add_baseline_inputs(), so `frame` already carries cases_roll4_mean.
    """
    clim = fit_climatology(df, fit_end)
    global_mean = float(df.loc[df[DATE_COL] <= fit_end, TARGET_COL].mean())
    return {
        "persistence": frame[TARGET_COL],
        "rolling_mean_4w": frame["cases_roll4_mean"],
        "ward_month_climatology": predict_climatology(frame, h, clim),
        "global_train_mean": pd.Series(global_mean, index=frame.index),
    }