from __future__ import annotations

import numpy as np
import pandas as pd


def add_split_inplace(
    df: pd.DataFrame,
    train_dates: list[str],
    val_dates: list[str],
    test_dates: list[str],
) -> None:
    df["split"] = "unused"
    df.loc[df["file_date"].astype(str).isin(train_dates), "split"] = "train"
    df.loc[df["file_date"].astype(str).isin(val_dates), "split"] = "val"
    df.loc[df["file_date"].astype(str).isin(test_dates), "split"] = "test"


def add_time_split_within_subset_inplace(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    time_col: str = "event_dt",
) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame")

    n = len(df)
    if n == 0:
        raise ValueError("Cannot split empty DataFrame")

    order = np.argsort(df[time_col].to_numpy())

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    split = np.empty(n, dtype=object)
    split[order[:train_end]] = "train"
    split[order[train_end:val_end]] = "val"
    split[order[val_end:]] = "test"

    df["split"] = split


def add_time_split_with_purge_inplace(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    purge_sec: int,
    time_col: str = "event_dt",
) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")

    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame")

    n = len(df)
    if n == 0:
        raise ValueError("Cannot split empty DataFrame")

    time_arr = pd.to_datetime(df[time_col])
    order = np.argsort(time_arr.to_numpy())

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    split = np.empty(n, dtype=object)
    split[order[:train_end]] = "train"
    split[order[train_end:val_end]] = "val"
    split[order[val_end:]] = "test"

    embargo = pd.Timedelta(seconds=purge_sec)

    if 0 < train_end < n:
        boundary_1 = time_arr.iloc[order[train_end]]
        mask_1 = (time_arr >= boundary_1 - embargo) & (time_arr <= boundary_1 + embargo)
        split[mask_1.to_numpy()] = "purged"

    if 0 < val_end < n:
        boundary_2 = time_arr.iloc[order[val_end]]
        mask_2 = (time_arr >= boundary_2 - embargo) & (time_arr <= boundary_2 + embargo)
        split[mask_2.to_numpy()] = "purged"

    df["split"] = split


def split_summary(df: pd.DataFrame, time_col: str = "event_dt") -> pd.DataFrame:
    if "split" not in df.columns:
        raise ValueError("Column 'split' not found in DataFrame")

    return (
        df.groupby("split")[time_col]
        .agg(["min", "max", "count"])
        .reset_index()
        .sort_values("min")
    )