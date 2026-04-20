from __future__ import annotations

import numpy as np
import pandas as pd


def add_event_dt_inplace(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    unit: str = "ns",
) -> None:
    df["event_dt"] = pd.to_datetime(df[timestamp_col], unit=unit)


def add_seconds_of_day_inplace(df: pd.DataFrame) -> None:
    h = df["event_dt"].dt.hour
    m = df["event_dt"].dt.minute
    s = df["event_dt"].dt.second
    df["seconds_of_day"] = h * 3600 + m * 60 + s


def classify_session_inplace(
    df: pd.DataFrame,
    main_start_sec: int,
    main_end_sec: int,
    evening_start_sec: int,
    evening_end_sec: int,
) -> None:
    sec = df["seconds_of_day"]

    main_mask = (sec >= main_start_sec) & (sec <= main_end_sec)
    evening_mask = (sec >= evening_start_sec) & (sec <= evening_end_sec)

    session_type = np.select(
        [main_mask, evening_mask],
        ["main", "evening"],
        default=None,
    )

    df["session_type"] = pd.Series(session_type, index=df.index, dtype="object")
    df["is_in_session"] = df["session_type"].notna()


def add_session_id_inplace(df: pd.DataFrame) -> None:
    df["session_id"] = np.where(
        df["is_in_session"],
        df["file_date"].astype(str) + "_" + df["session_type"].astype(str),
        None,
    )


def add_book_sanity_cols_inplace(df: pd.DataFrame) -> None:
    df["spread"] = df["best_ask_price"] - df["best_bid_price"]
    df["mid_reconstructed"] = 0.5 * (df["best_bid_price"] + df["best_ask_price"])
    df["mid_reconstruction_error"] = df["mid_price"] - df["mid_reconstructed"]


def filter_trading_sessions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[df["is_in_session"]].copy()
    out = out.sort_values(["file_date", "session_type", "event_dt"]).reset_index(drop=True)
    return out