from __future__ import annotations

import pandas as pd


def make_file_date_report(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["source_file", "file_date", "event_dt"]].copy()
    tmp["event_date_yyyymmdd"] = tmp["event_dt"].dt.strftime("%Y%m%d")

    report = (
        tmp.groupby("source_file")
        .agg(
            n_rows=("source_file", "size"),
            file_date=("file_date", "first"),
            min_event_dt=("event_dt", "min"),
            max_event_dt=("event_dt", "max"),
            n_unique_event_dates=("event_date_yyyymmdd", "nunique"),
            event_dates=("event_date_yyyymmdd", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
    )

    report["file_date_matches_all_rows"] = report["file_date"] == report["event_dates"]
    return report


def make_monotonicity_report(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("source_file")["timestamp"]
        .apply(lambda s: bool(s.is_monotonic_increasing))
        .rename("timestamp_monotonic_increasing")
        .reset_index()
    )
    return out


def make_light_audit_report(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "timestamp_dtype": str(df["timestamp"].dtype),
        "timestamp_min": int(df["timestamp"].min()),
        "timestamp_max": int(df["timestamp"].max()),
        "best_bid_price_non_positive": int((df["best_bid_price"] <= 0).sum()),
        "best_ask_price_non_positive": int((df["best_ask_price"] <= 0).sum()),
        "mid_price_non_positive": int((df["mid_price"] <= 0).sum()),
        "spread_negative_count": int((df["spread"] < 0).sum()),
        "spread_zero_count": int((df["spread"] == 0).sum()),
        "spread_mean": float(df["spread"].mean()),
        "spread_median": float(df["spread"].median()),
        "reversed_book_share": float((df["best_ask_price"] < df["best_bid_price"]).mean()),
    }
    return report


def make_null_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "feature": col,
                "dtype": str(df[col].dtype),
                "null_frac": float(df[col].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("null_frac", ascending=False)


def make_session_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["file_date", "session_type"])
        .size()
        .rename("n_events")
        .reset_index()
        .sort_values(["file_date", "session_type"])
    )


def make_event_gap_summary_by_file(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for source_file, part in df.groupby("source_file", sort=False):
        ts = part["timestamp"]
        dt_ns = ts.diff()

        desc = dt_ns.describe(percentiles=[0.5, 0.9, 0.99, 0.999])

        row = {"source_file": source_file}
        for k, v in desc.to_dict().items():
            row[str(k)] = v
        rows.append(row)

    return pd.DataFrame(rows)