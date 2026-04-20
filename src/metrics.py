from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n_obs": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "pearson_ic": np.nan,
            "spearman_ic": np.nan,
        }

    pearson_ic = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan

    true_rank = pd.Series(y_true).rank(method="average").to_numpy()
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy()
    spearman_ic = np.corrcoef(true_rank, pred_rank)[0, 1] if len(y_true) > 1 else np.nan

    return {
        "n_obs": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "pearson_ic": float(pearson_ic),
        "spearman_ic": float(spearman_ic),
    }


def evaluate_predictions(
    df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    model_name: str,
    horizon_sec: int,
    split_col: str = "split",
) -> pd.DataFrame:
    rows = []
    for split_name, part in df.groupby(split_col):
        m = regression_metrics(part[target_col], part[pred_col])
        m["model"] = model_name
        m["horizon_sec"] = horizon_sec
        m["split"] = split_name
        rows.append(m)
    return pd.DataFrame(rows)