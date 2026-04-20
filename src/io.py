from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


DATE_RE = re.compile(r"(\d{8})")


def extract_file_date(filename: str) -> str:
    m = DATE_RE.search(filename)
    if not m:
        raise ValueError(f"Could not extract YYYYMMDD from filename: {filename}")
    return m.group(1)


def list_raw_files(raw_dir: str, pattern: str) -> list[Path]:
    files = sorted(Path(raw_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {raw_dir} matching {pattern}")
    return files


def load_daily_feature_files(raw_dir: str, pattern: str) -> pd.DataFrame:
    files = list_raw_files(raw_dir, pattern)

    parts = []
    for path in files:
        df = pd.read_parquet(path)
        df["source_file"] = path.name
        df["file_date"] = extract_file_date(path.name)
        parts.append(df)

    out = pd.concat(parts, axis=0, ignore_index=True)
    return out


def save_table(df: pd.DataFrame, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if path_obj.suffix == ".parquet":
        df.to_parquet(path_obj, index=False)
    elif path_obj.suffix == ".csv":
        df.to_csv(path_obj, index=False)
    else:
        raise ValueError(f"Unsupported format: {path_obj.suffix}")