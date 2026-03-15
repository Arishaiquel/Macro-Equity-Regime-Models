from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


def rolling_corr(a: pd.Series, b: pd.Series, win: int) -> pd.Series:
    return a.pct_change().rolling(win, min_periods=max(20, win // 3)).corr(b.pct_change())


def spike_flag(x: pd.Series, win: int, q: float) -> pd.Series:
    thr = x.rolling(win, min_periods=max(50, win // 3)).quantile(q)
    return x > thr


def top_pct_to_quantile(top_pct: float) -> float:
    pct = min(max(float(top_pct), 0.0), 99.0)
    return 1.0 - (pct / 100.0)


def avg_pairwise_corr(ret: pd.DataFrame, win: int) -> pd.Series:
    out = pd.Series(index=ret.index, dtype=float)
    cols = ret.columns
    n = len(cols)
    for t in range(win, len(ret)):
        w = ret.iloc[t - win : t].dropna()
        if w.shape[0] < max(30, win // 2):
            continue
        c = w.corr().values
        out.iloc[t] = (c.sum() - np.trace(c)) / (n * (n - 1))
    return out


def perf_stats(x: pd.Series) -> dict:
    x = x.dropna()
    if x.empty:
        return {"ann_ret": None, "ann_vol": None, "sharpe": None, "sortino": None}
    ann_ret = (1 + x).prod() ** (252 / len(x)) - 1
    ann_vol = x.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    downside = x[x < 0]
    downside_vol = downside.std() * np.sqrt(252)
    sortino = ann_ret / downside_vol if downside_vol > 0 else np.nan
    return {
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": None if np.isnan(sharpe) else float(sharpe),
        "sortino": None if np.isnan(sortino) else float(sortino),
    }


def run_hedged_backtest(
    repo_root: Path,
    start_date: str = "2016-01-01",
    hedge_h: float = 0.5,
    vix_threshold: float = 20.0,
    ema_long_term: int = 180,
    ema_short_term: int = 40,
    corr_window: int = 63,
    spike_history_window: int = 256,
    cross_asset_spike_top_pct: float = 10.0,
    intra_equity_spike_top_pct: float = 10.0,
    vvix_spike_top_pct: float = 10.0,
    hy_level_bps: float = 350.0,
    hy_widen_bps: float = 75.0,
    hy_lookback_days: int = 20,
    borrow_rate_annual: float = 0.02,
    slippage_bps: float = 5.0,
    include_costs: bool = False,
) -> dict:
    sectors = ["XLE", "XLF", "XLI", "XLK", "XLP", "XLV", "XLY"]
    tickers = sorted(set(["QQQ", "^VIX", "SPY", "TLT", "HYG"] + sectors))

    px = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)["Close"].dropna(how="all")
    rets = px.pct_change().dropna()
    idx = rets.index

    qqq = px["QQQ"].loc[idx]
    vix = px["^VIX"].loc[idx]
    spy = px["SPY"].loc[idx]
    tlt = px["TLT"].loc[idx]
    hyg = px["HYG"].loc[idx]

    sector_prices = px[sectors].loc[idx].dropna(how="any")
    sector_ret = sector_prices.pct_change()

    ema_long = qqq.ewm(span=ema_long_term, adjust=False).mean()
    ema_short = qqq.ewm(span=ema_short_term, adjust=False).mean()

    bear_on = qqq < ema_long
    bear_off = qqq > ema_long * 1.02
    bear_state = pd.Series(0, index=idx, dtype=int)
    for i in range(1, len(idx)):
        prev = bear_state.iat[i - 1]
        if prev == 0 and bear_on.iat[i]:
            bear_state.iat[i] = 1
        elif prev == 1 and bear_off.iat[i]:
            bear_state.iat[i] = 0
        else:
            bear_state.iat[i] = prev

    risk3_q = top_pct_to_quantile(cross_asset_spike_top_pct)
    corr_spy_tlt = rolling_corr(spy, tlt, corr_window).abs()
    corr_spy_hyg = rolling_corr(spy, hyg, corr_window).abs()
    risk3 = spike_flag(corr_spy_tlt, spike_history_window, risk3_q) | spike_flag(corr_spy_hyg, spike_history_window, risk3_q)

    risk4_q = top_pct_to_quantile(intra_equity_spike_top_pct)
    eq_corr = avg_pairwise_corr(sector_ret, corr_window)
    risk4 = spike_flag(eq_corr, spike_history_window, risk4_q)

    risk5_q = top_pct_to_quantile(vvix_spike_top_pct)
    vix_chg = vix.pct_change()
    vix_vov = vix_chg.rolling(20, min_periods=15).std()
    risk5 = spike_flag(vix_vov, 252, risk5_q).reindex(idx).fillna(False)

    hy_path = repo_root / "data" / "HY_OAS.csv"
    if not hy_path.exists():
        raise ValueError("Built-in HY OAS dataset not found at data/HY_OAS.csv")

    hy = pd.read_csv(hy_path, parse_dates=["observation_date"])
    hy = hy.rename(columns={"observation_date": "date", "BAMLH0A0HYM2": "HY_OAS_PCT"})
    hy["date"] = pd.to_datetime(hy["date"], errors="coerce")
    hy = hy.dropna(subset=["date"]).set_index("date")
    hy["HY_OAS_PCT"] = pd.to_numeric(hy["HY_OAS_PCT"], errors="coerce")
    hy = hy.loc[start_date:]

    hy_oas_bps = (hy["HY_OAS_PCT"] * 100).rename("HY_OAS_BPS").reindex(idx).ffill()
    hy_widen = hy_oas_bps.diff(hy_lookback_days)
    risk6 = (hy_oas_bps > hy_level_bps) | (hy_widen > hy_widen_bps)

    overlay_on = (
        (bear_state == 1)
        & (qqq < ema_short)
        & (vix > vix_threshold)
        & (risk3 | risk4 | risk5 | risk6)
    ).astype(int)
    overlay_series = overlay_on.reindex(idx).shift(1).fillna(0).astype(int)

    port = rets["QQQ"].copy()
    qqq_ret = rets["QQQ"].loc[idx]
    short_overlay_return = -hedge_h * qqq_ret * overlay_series

    borrow_drag = (borrow_rate_annual / 252.0) * (hedge_h * overlay_series)
    switch = overlay_series.diff().abs().fillna(0)
    slip_cost = (slippage_bps / 10000.0) * switch

    hedged = port + short_overlay_return
    if include_costs:
        hedged = hedged - borrow_drag - slip_cost

    u = perf_stats(port)
    h = perf_stats(hedged)

    latest = idx[-1]
    tail = []
    for d in idx[-200:]:
        tail.append(
            {
                "date": d.date().isoformat(),
                "overlay_on": int(overlay_series.loc[d]),
                "unhedged": float(port.loc[d]),
                "hedged": float(hedged.loc[d]),
            }
        )

    return {
        "latest": {
            "date": latest.date().isoformat(),
            "overlay_on": int(overlay_series.loc[latest]),
            "vix": float(vix.loc[latest]),
            "hy_oas_bps": None if pd.isna(hy_oas_bps.loc[latest]) else float(hy_oas_bps.loc[latest]),
            "risk_composite_active": bool((risk3 | risk4 | risk5 | risk6).reindex([latest]).fillna(False).iloc[0]),
        },
        "overlay_on_pct": float(overlay_series.mean()),
        "unhedged_stats": u,
        "hedged_stats": h,
        "risk_days_pct": {
            "risk3": float(risk3.reindex(idx).fillna(False).mean()),
            "risk4": float(risk4.reindex(idx).fillna(False).mean()),
            "risk5": float(risk5.reindex(idx).fillna(False).mean()),
            "risk6": float(risk6.reindex(idx).fillna(False).mean()),
        },
        "timeseries_tail": tail,
    }
