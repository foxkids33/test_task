from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def zero_baseline(df: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(df))


def microprice_gap_score(df: pd.DataFrame, spread_eps: float = 1e-12) -> np.ndarray:
    denom = np.maximum(np.abs(df["spread"].to_numpy()), spread_eps)
    return (df["microprice"].to_numpy() - df["mid_price"].to_numpy()) / denom


def imbalance_score(df: pd.DataFrame) -> np.ndarray:
    if "imbalance_l1" in df.columns:
        return df["imbalance_l1"].to_numpy()
    if "imbalance_depth_weighted" in df.columns:
        return df["imbalance_depth_weighted"].to_numpy()
    raise ValueError("No imbalance feature found.")


def fit_score_calibrator(score: np.ndarray, y: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(score.reshape(-1, 1), y)
    return model


def apply_score_calibrator(model: LinearRegression, score: np.ndarray) -> np.ndarray:
    return model.predict(score.reshape(-1, 1))