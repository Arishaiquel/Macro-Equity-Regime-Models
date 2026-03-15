from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="any")


def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff().dropna(how="any")


def max_drawdown(px: pd.Series) -> float:
    peak = px.cummax()
    dd = 1 - px / peak
    return float(dd.max())


def rolling_max_drawdown(px: pd.Series, window: int) -> pd.Series:
    out = []
    idx = px.index
    for i in range(len(px)):
        if i < window:
            out.append(np.nan)
            continue
        w = px.iloc[i - window + 1 : i + 1]
        out.append(max_drawdown(w))
    return pd.Series(out, index=idx)


def rolling_beta(ra: pd.Series, rb: pd.Series, window: int) -> pd.Series:
    cov = ra.rolling(window).cov(rb)
    var = rb.rolling(window).var()
    return cov / var


def tracking_error(ra: pd.Series, rb: pd.Series, window: int) -> pd.Series:
    diff = ra - rb
    return diff.rolling(window).std() * np.sqrt(252)


def ann_vol(r: pd.Series, window: int) -> pd.Series:
    return r.rolling(window).std() * np.sqrt(252)


def run_pair_correlation(
    asset_a: str,
    asset_b: str,
    start_date: str,
    win_3m: int = 63,
    win_6m: int = 126,
) -> dict:
    a = asset_a.upper()
    b = asset_b.upper()
    px = download_prices([a, b], start_date)
    if px.empty:
        raise ValueError("No data returned for selected assets")

    r = log_returns(px)
    if r.empty:
        raise ValueError("Not enough data to compute log returns")

    roll_corr_3m = r[a].rolling(win_3m).corr(r[b])
    roll_corr_6m = r[a].rolling(win_6m).corr(r[b])

    beta_3m = rolling_beta(r[a], r[b], win_3m)
    beta_6m = rolling_beta(r[a], r[b], win_6m)

    vol_a_3m = ann_vol(r[a], win_3m)
    vol_b_3m = ann_vol(r[b], win_3m)
    te_3m = tracking_error(r[a], r[b], win_3m)

    vol_a_6m = ann_vol(r[a], win_6m)
    vol_b_6m = ann_vol(r[b], win_6m)
    te_6m = tracking_error(r[a], r[b], win_6m)

    mdd_a_3m = rolling_max_drawdown(px[a], win_3m)
    mdd_b_3m = rolling_max_drawdown(px[b], win_3m)
    mdd_a_6m = rolling_max_drawdown(px[a], win_6m)
    mdd_b_6m = rolling_max_drawdown(px[b], win_6m)

    latest = pd.DataFrame(
        {
            "corr_3m": roll_corr_3m,
            "corr_6m": roll_corr_6m,
            "beta_3m": beta_3m,
            "beta_6m": beta_6m,
            "volA_3m": vol_a_3m,
            "volB_3m": vol_b_3m,
            "te_3m": te_3m,
            "volA_6m": vol_a_6m,
            "volB_6m": vol_b_6m,
            "te_6m": te_6m,
            "mddA_3m": mdd_a_3m,
            "mddB_3m": mdd_b_3m,
            "mddA_6m": mdd_a_6m,
            "mddB_6m": mdd_b_6m,
        }
    ).dropna()

    rebased = (px / px.iloc[0] * 100.0).dropna()

    summary = {
        "start": str(px.index.min().date()),
        "end": str(px.index.max().date()),
        "n_days": int(len(px)),
        "corr_daily_returns_full": float(r[a].corr(r[b])),
        "beta_full_a_on_b": float(r[a].cov(r[b]) / r[b].var()),
        "max_dd_a_full": max_drawdown(px[a]),
        "max_dd_b_full": max_drawdown(px[b]),
    }

    latest_metrics = {} if latest.empty else {k: float(v) for k, v in latest.iloc[-1].to_dict().items()}
    rebased_tail = [
        {
            "date": d.date().isoformat(),
            a: float(row[a]),
            b: float(row[b]),
        }
        for d, row in rebased.tail(200).iterrows()
    ]

    return {
        "asset_a": a,
        "asset_b": b,
        "summary": summary,
        "latest_metrics": latest_metrics,
        "rebased_tail": rebased_tail,
    }
