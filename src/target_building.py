from __future__ import annotations

import numpy as np
import pandas as pd


def build_targets_by_session_inplace(
    df: pd.DataFrame,
    horizons_sec: list[int],
    event_col: str = "event_dt",
    mid_col: str = "mid_price",
    session_col: str = "session_id",
) -> None:
    """
    Строит future targets внутри каждой торговой сессии отдельно.

    Важно:
    - df должен иметь RangeIndex 0..N-1
    - target строится по calendar-time через merge_asof
    - переходы между session_id запрещены по конструкции
    """
    if not isinstance(df.index, pd.RangeIndex) or df.index.start != 0 or df.index.step != 1:
        raise ValueError("DataFrame must have a simple RangeIndex. Use df.reset_index(drop=True) first.")

    n = len(df)
    grouped_indices = df.groupby(session_col, sort=False).groups

    for h in horizons_sec:
        future_mid_arr = np.full(n, np.nan, dtype="float64")
        future_dt_arr = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        valid_arr = np.zeros(n, dtype=bool)

        for session_id, idx in grouped_indices.items():
            idx = np.asarray(idx, dtype=int)

            part = df.loc[idx, [event_col, mid_col]].sort_values(event_col)
            pos = part.index.to_numpy()

            right = pd.DataFrame({
                "future_dt": part[event_col].to_numpy(),
                "future_mid": part[mid_col].to_numpy(),
            })

            left = pd.DataFrame({
                "orig_pos": pos,
                "target_dt": part[event_col].to_numpy() + pd.to_timedelta(h, unit="s"),
            })

            merged = pd.merge_asof(
                left.sort_values("target_dt"),
                right.sort_values("future_dt"),
                left_on="target_dt",
                right_on="future_dt",
                direction="forward",
                allow_exact_matches=True,
            )

            orig_pos = merged["orig_pos"].to_numpy()
            future_mid_vals = merged["future_mid"].to_numpy()
            future_dt_vals = merged["future_dt"].to_numpy()
            valid_vals = ~pd.isna(future_mid_vals)

            future_mid_arr[orig_pos] = future_mid_vals
            future_dt_arr[orig_pos] = future_dt_vals
            valid_arr[orig_pos] = valid_vals

        df[f"future_dt_{h}s"] = pd.to_datetime(future_dt_arr)
        df[f"future_mid_{h}s"] = future_mid_arr
        df[f"valid_target_{h}s"] = valid_arr
        df[f"delta_mid_{h}s"] = df[f"future_mid_{h}s"] - df[mid_col]
        df[f"y_logret_{h}s"] = np.log(df[f"future_mid_{h}s"] / df[mid_col])


def add_direction_target_inplace(
    df: pd.DataFrame,
    regression_target: str,
    out_col: str,
    eps: float,
) -> None:
    y = df[regression_target]
    df[out_col] = np.where(y > eps, 1, np.where(y < -eps, -1, 0))
    df.loc[y.isna(), out_col] = np.nan