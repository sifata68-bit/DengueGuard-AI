import numpy as np
import pandas as pd

from src.config import IN_SEASON_MONTHS


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))


def in_season_mask(target_dates) -> np.ndarray:
    return pd.Series(target_dates).dt.month.isin(IN_SEASON_MONTHS).to_numpy()


def score(y_true, y_pred, target_dates) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = in_season_mask(target_dates)
    return {
        "n": int(len(y_true)),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "n_in_season": int(mask.sum()),
        "mae_in_season": mae(y_true[mask], y_pred[mask]),
        "rmse_in_season": rmse(y_true[mask], y_pred[mask]),
    }


def comparison_table(y_true, target_dates, predictions: dict) -> pd.DataFrame:
    rows = [
        {"model": name, **score(y_true, np.asarray(pred), target_dates)}
        for name, pred in predictions.items()
    ]
    return pd.DataFrame(rows).sort_values("rmse_in_season").reset_index(drop=True)
