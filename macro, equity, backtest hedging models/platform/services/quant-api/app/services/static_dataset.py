from __future__ import annotations

from pathlib import Path

import pandas as pd


def check_hy_oas_dataset(repo_root: Path) -> dict:
    dataset = repo_root / "data" / "HY_OAS.csv"
    if not dataset.exists():
        return {
            "dataset": str(dataset),
            "exists": False,
            "min_date": None,
            "max_date": None,
            "rows": 0,
        }

    df = pd.read_csv(dataset)
    date_col = "observation_date" if "observation_date" in df.columns else None
    if date_col is None:
        return {
            "dataset": str(dataset),
            "exists": True,
            "min_date": None,
            "max_date": None,
            "rows": int(len(df)),
        }

    d = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if d.empty:
        return {
            "dataset": str(dataset),
            "exists": True,
            "min_date": None,
            "max_date": None,
            "rows": int(len(df)),
        }

    return {
        "dataset": str(dataset),
        "exists": True,
        "min_date": d.min().date().isoformat(),
        "max_date": d.max().date().isoformat(),
        "rows": int(len(df)),
    }
