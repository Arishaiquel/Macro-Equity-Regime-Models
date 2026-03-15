from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="all")


def daily_log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff().dropna(how="any")


def rolling_corr_tensor(returns: pd.DataFrame, window: int, method: str) -> dict[pd.Timestamp, pd.DataFrame]:
    corr_map: dict[pd.Timestamp, pd.DataFrame] = {}
    for end_i in range(window - 1, len(returns)):
        window_slice = returns.iloc[end_i - window + 1 : end_i + 1]
        corr_map[returns.index[end_i]] = window_slice.corr(method=method)
    return corr_map


def mean_offdiag_corr(corr_mat: pd.DataFrame) -> float:
    vals = corr_mat.values
    n = vals.shape[0]
    if n < 2:
        return np.nan
    mask = ~np.eye(n, dtype=bool)
    return float(vals[mask].mean())


def run_sector_correlation(
    tickers: list[str],
    start_date: str,
    window: int = 63,
    method: str = "pearson",
) -> dict:
    method = method.lower().strip()
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of: pearson, spearman, kendall")

    tk = [t.strip().upper() for t in tickers if t.strip()]
    px = download_prices(tk, start_date)
    if px.empty:
        raise ValueError("No data returned from yfinance")

    px = px[tk].dropna(how="any")
    rets = daily_log_returns(px)
    if rets.empty:
        raise ValueError("Not enough data for returns")

    full_corr = rets.corr(method=method)
    corr_map = rolling_corr_tensor(rets, window, method)
    if not corr_map:
        raise ValueError("Not enough data for selected rolling window")

    latest_date = max(corr_map.keys())
    latest_corr = corr_map[latest_date]
    cohesion = pd.Series({dt: mean_offdiag_corr(mat) for dt, mat in corr_map.items()})

    full_matrix = full_corr.reset_index().rename(columns={"index": "ticker"}).to_dict(orient="records")
    latest_matrix = latest_corr.reset_index().rename(columns={"index": "ticker"}).to_dict(orient="records")
    cohesion_tail = [
        {"date": d.date().isoformat(), "avg_corr": float(v)}
        for d, v in cohesion.dropna().tail(200).items()
    ]

    return {
        "method": method,
        "window": window,
        "latest_date": latest_date.date().isoformat(),
        "latest_avg_corr": float(cohesion.iloc[-1]),
        "full_matrix": full_matrix,
        "latest_matrix": latest_matrix,
        "cohesion_tail": cohesion_tail,
    }
