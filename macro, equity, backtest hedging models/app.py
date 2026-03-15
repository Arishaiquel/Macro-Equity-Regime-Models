from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


ROOT = Path(__file__).resolve().parent
HY_OAS_PATH = ROOT / "data" / "HY_OAS.csv"


@dataclass(frozen=True)
class FeatureDef:
    key: str
    label: str
    script_name: str
    data_mode: str
    blurb: str


FEATURES: list[FeatureDef] = [
    FeatureDef(
        key="valuation_quality",
        label="Valuation vs Quality",
        script_name="Valuation VS Quality Model.py",
        data_mode="Live",
        blurb="Peer-relative valuation and quality scoring.",
    ),
    FeatureDef(
        key="analyst_valuation",
        label="Analyst Valuation + Revisions",
        script_name="Analyst valuation model.py",
        data_mode="Live",
        blurb="Combines peer valuation with analyst revision proxies.",
    ),
    FeatureDef(
        key="crash_risk",
        label="Crash Risk (SPY/QQQ)",
        script_name="UER Model.py",
        data_mode="Live",
        blurb="Estimates probability of forward drawdown risk using Machine Learning (HistGradientBoostingClassifier).",
    ),
    FeatureDef(
        key="macro_equity",
        label="Macro vs Equity",
        script_name="Macro-Equity model.py",
        data_mode="Live",
        blurb="Builds weighted macro score and compares to equity returns.",
    ),
    FeatureDef(
        key="pair_correlation",
        label="Pair Correlation",
        script_name="correlation.py",
        data_mode="Live",
        blurb="Tracks rolling correlation, beta, TE, and drawdown behavior.",
    ),
    FeatureDef(
        key="sector_correlation",
        label="Sector Correlation",
        script_name="sector_correlation.py",
        data_mode="Live",
        blurb="Sector heatmap and cohesion analysis with rolling correlation.",
    ),
    FeatureDef(
        key="hedged_backtest",
        label="Hedged Backtest Overlay",
        script_name="Backtest.py",
        data_mode="Static",
        blurb="Regime overlay backtest using market stress signals.",
    ),
    FeatureDef(
        key="global_indices",
        label="Global Indices",
        script_name="(app-native)",
        data_mode="Live",
        blurb="Compares major global equity indices with live yfinance data.",
    ),
]


def _status_chip(mode: str) -> str:
    if mode.lower() == "static":
        return ":orange-badge[Data Mode: Static]"
    return ":green-badge[Data Mode: Live]"


def _init_state() -> None:
    if "run_history" not in st.session_state:
        st.session_state.run_history = []


def _push_history(feature: str, status: str, detail: str) -> None:
    event = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature": feature,
        "status": status,
        "detail": detail,
    }
    st.session_state.run_history.insert(0, event)
    st.session_state.run_history = st.session_state.run_history[:10]


def _render_history() -> None:
    st.sidebar.subheader("Run History")
    history = st.session_state.run_history
    if not history:
        st.sidebar.caption("No runs yet.")
        return
    for row in history:
        st.sidebar.caption(
            f"[{row['time']}] {row['feature']} -> {row['status']} ({row['detail']})"
        )


def _global_data_status() -> None:
    st.sidebar.subheader("Data Source Status")
    st.sidebar.caption("Live sources: yfinance / FRED")
    if HY_OAS_PATH.exists():
        st.sidebar.success("Backtest static dataset found: data/HY_OAS.csv")
    else:
        st.sidebar.error("Backtest static dataset missing: data/HY_OAS.csv")


def _common_header(feature: FeatureDef) -> None:
    st.title(feature.label)
    st.markdown(feature.blurb)
    st.markdown(_status_chip(feature.data_mode))
    st.caption(f"Linked script: `{feature.script_name}`")
    st.divider()


def _top_buttons(feature: FeatureDef) -> tuple[bool, bool]:
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.2])
    run_clicked = c1.button("Run", use_container_width=True)
    reset_clicked = c2.button("Reset Inputs", use_container_width=True)
    c3.button("Export Table CSV", use_container_width=True, disabled=True)
    c4.button("Export Chart PNG", use_container_width=True, disabled=True)
    if run_clicked:
        _push_history(feature.label, "queued", "frontend-only")
    if reset_clicked:
        _push_history(feature.label, "reset", "inputs")
    return run_clicked, reset_clicked


def _placeholder_tabs(feature: FeatureDef, inputs: dict[str, Any]) -> None:
    charts, tables, diagnostics = st.tabs(["Charts", "Tables", "Diagnostics"])
    with charts:
        st.info("Chart rendering layer will be connected in the execution pass.")
    with tables:
        st.dataframe(
            pd.DataFrame(
                [{"input": k, "value": str(v)} for k, v in inputs.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )
    with diagnostics:
        st.write("Current input snapshot:")
        st.json(inputs)
        if feature.key == "hedged_backtest":
            _render_backtest_dataset_diagnostics()


def _render_backtest_dataset_diagnostics() -> None:
    if not HY_OAS_PATH.exists():
        st.error(
            "Built-in HY OAS dataset not found at `data/HY_OAS.csv`. "
            "Backtest page requires this local static file."
        )
        return
    try:
        df = pd.read_csv(HY_OAS_PATH, nrows=5)
        st.success("Built-in HY OAS dataset is available.")
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Built-in HY OAS dataset could not be read: {exc}")


@st.cache_data(show_spinner=False, ttl=1800)
def _download_prices(tickers: list[str], start_date: str) -> pd.DataFrame:
    px = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="all")


def _compute_mtd_return(px: pd.Series) -> float:
    if px.empty:
        return np.nan
    idx = px.index
    month_start = pd.Timestamp(idx[-1].year, idx[-1].month, 1)
    current_month = px.loc[px.index >= month_start]
    if current_month.empty:
        return np.nan
    base = current_month.iloc[0]
    last = current_month.iloc[-1]
    if base <= 0:
        return np.nan
    return (last / base) - 1.0


def _compute_ytd_return(px: pd.Series) -> float:
    if px.empty:
        return np.nan
    idx = px.index
    year_start = pd.Timestamp(idx[-1].year, 1, 1)
    ytd = px.loc[px.index >= year_start]
    if ytd.empty:
        return np.nan
    base = ytd.iloc[0]
    last = ytd.iloc[-1]
    if base <= 0:
        return np.nan
    return (last / base) - 1.0


def _page_valuation_quality(feature: FeatureDef) -> None:
    _common_header(feature)
    run_clicked, _ = _top_buttons(feature)
    c1, c2 = st.columns(2)
    tickers = c1.text_area("Tickers (comma-separated)", "UNH,ABT,CI,HUM,JNJ,PFE,MRK,LLY")
    highlight = c2.text_input("Highlight Ticker", "ABT")
    c3, c4 = st.columns(2)
    min_peers = c3.number_input("Min Peers", min_value=3, max_value=50, value=5)
    sleep_delay = c4.number_input("Request Delay (sec)", min_value=0.0, max_value=1.0, value=0.15)
    inputs = {
        "tickers": tickers,
        "highlight": highlight,
        "min_peers": min_peers,
        "sleep_delay": sleep_delay,
    }
    _placeholder_tabs(feature, inputs)
    if run_clicked:
        st.warning("Execution wiring is next. UI and config are now in place.")


def _page_analyst_valuation(feature: FeatureDef) -> None:
    _common_header(feature)
    run_clicked, _ = _top_buttons(feature)
    c1, c2 = st.columns(2)
    tickers = c1.text_area("Tickers (comma-separated)", "AMZN,MSFT,GOOGL,META,AAPL,ORCL,WMT,COST,TGT,SHOP,NFLX")
    highlight = c2.text_input("Highlight Ticker", "NFLX")
    c3, c4 = st.columns(2)
    lookback_days = c3.selectbox("Lookback Days", [90, 180, 365], index=1)
    min_peers = c4.number_input("Min Peers", min_value=3, max_value=50, value=5)
    inputs = {
        "tickers": tickers,
        "highlight": highlight,
        "lookback_days": lookback_days,
        "min_peers": min_peers,
    }
    _placeholder_tabs(feature, inputs)
    if run_clicked:
        st.warning("Execution wiring is next. UI and config are now in place.")


def _page_crash_risk(feature: FeatureDef) -> None:
    _common_header(feature)
    run_clicked, _ = _top_buttons(feature)
    c1, c2, c3 = st.columns(3)
    start_date = c1.date_input("Start Date", value=datetime.strptime("2000-01-01", "%Y-%m-%d"))
    horizon = c2.number_input("Horizon Days", min_value=5, max_value=252, value=63)
    dd_thresh = c3.number_input("Drawdown Threshold", min_value=0.05, max_value=0.5, value=0.15, step=0.01)
    folds = st.number_input("CV Splits", min_value=2, max_value=20, value=5)
    inputs = {
        "start_date": str(start_date),
        "horizon": horizon,
        "drawdown_threshold": dd_thresh,
        "cv_splits": folds,
    }
    _placeholder_tabs(feature, inputs)
    if run_clicked:
        st.warning("Execution wiring is next. UI and config are now in place.")


def _page_macro_equity(feature: FeatureDef) -> None:
    _common_header(feature)
    run_clicked, _ = _top_buttons(feature)
    c1, c2, c3 = st.columns(3)
    start_date = c1.date_input("Start Date", value=datetime.strptime("1995-01-01", "%Y-%m-%d"))
    lookback_months = c2.number_input("Z-score Lookback (months)", min_value=12, max_value=240, value=60)
    min_periods = c3.number_input("Min Periods", min_value=12, max_value=120, value=36)
    min_factors = st.number_input("Min Factors Required", min_value=1, max_value=12, value=8)
    inputs = {
        "start_date": str(start_date),
        "lookback_months": lookback_months,
        "min_periods": min_periods,
        "min_factors": min_factors,
    }
    _placeholder_tabs(feature, inputs)
    if run_clicked:
        st.warning("Execution wiring is next. UI and config are now in place.")


def _page_pair_correlation(feature: FeatureDef) -> None:
    _common_header(feature)
    run_clicked, _ = _top_buttons(feature)
    c1, c2, c3 = st.columns(3)
    asset_a = c1.text_input("Asset A", "SPY")
    asset_b = c2.text_input("Asset B", "XLK")
    start_date = c3.date_input("Start Date", value=datetime.strptime("2020-01-01", "%Y-%m-%d"))
    c4, c5 = st.columns(2)
    win_3m = c4.number_input("3M Window", min_value=21, max_value=252, value=63)
    win_6m = c5.number_input("6M Window", min_value=42, max_value=504, value=126)
    inputs = {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "start_date": str(start_date),
        "win_3m": win_3m,
        "win_6m": win_6m,
    }
    _placeholder_tabs(feature, inputs)
    if run_clicked:
        st.warning("Execution wiring is next. UI and config are now in place.")


def _page_sector_correlation(feature: FeatureDef) -> None:
    _common_header(feature)
    run_clicked, _ = _top_buttons(feature)
    c1, c2, c3 = st.columns(3)
    tickers = c1.text_area("Sector Tickers", "XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY")
    start_date = c2.date_input("Start Date", value=datetime.strptime("2018-01-01", "%Y-%m-%d"))
    method = c3.selectbox("Method", ["pearson", "spearman", "kendall"], index=0)
    window = st.number_input("Rolling Window", min_value=21, max_value=252, value=63)
    inputs = {
        "sector_tickers": tickers,
        "start_date": str(start_date),
        "method": method,
        "window": window,
    }
    _placeholder_tabs(feature, inputs)
    if run_clicked:
        st.warning("Execution wiring is next. UI and config are now in place.")


def _page_hedged_backtest(feature: FeatureDef) -> None:
    _common_header(feature)
    run_clicked, _ = _top_buttons(feature)
    c1, c2, c3 = st.columns(3)
    start_date = c1.date_input("Start Date", value=datetime.strptime("2016-01-01", "%Y-%m-%d"))
    hedge_h = c2.number_input("Hedge Ratio H", min_value=0.0, max_value=2.0, value=0.5, step=0.05)
    vix_thresh = c3.number_input("VIX Threshold", min_value=5.0, max_value=80.0, value=20.0, step=1.0)
    c4, c5, c6 = st.columns(3)
    borrow_rate = c4.number_input("Borrow Rate (annual)", min_value=0.0, max_value=0.5, value=0.02, step=0.005)
    slippage_bps = c5.number_input("Slippage (bps)", min_value=0.0, max_value=200.0, value=5.0, step=0.5)
    ema_long_term = c6.selectbox(
        "Long-Term EMA (QQQ/SPY < EMA)", [10, 30, 40, 50, 100, 180, 200], index=5
    )

    c7, c8, c9 = st.columns(3)
    ema_short_term = c7.selectbox(
        "Short-Term EMA (QQQ/SPY < EMA)", [10, 30, 40, 50, 100, 180, 200], index=2
    )
    corr_spike_pct = c8.slider("SPY/TLT or SPY/HYG spike top %", min_value=1, max_value=30, value=10)
    intra_eq_spike_pct = c9.slider("Intra-equity spike top %", min_value=1, max_value=30, value=10)

    c10, c11, c12 = st.columns(3)
    vvix_spike_pct = c10.slider("VVIX spike top %", min_value=1, max_value=30, value=10)
    hy_level = c11.number_input("HY Spread Level (bps)", min_value=100, max_value=1000, value=350, step=10)
    hy_widen_bps = c12.number_input("HY Widening Threshold (bps)", min_value=10, max_value=500, value=75, step=5)

    c13, c14, c15 = st.columns(3)
    hy_lookback_days = c13.number_input("HY Widening Lookback (days)", min_value=5, max_value=126, value=20)
    spike_history_window = c14.number_input("Spike History Window (days)", min_value=63, max_value=756, value=256)
    corr_window = c15.number_input("Rolling Correlation Window (days)", min_value=21, max_value=252, value=63)

    enable_costs = st.toggle("Enable Cost Model", value=False)
    inputs = {
        "start_date": str(start_date),
        "hedge_ratio_h": hedge_h,
        "vix_threshold": vix_thresh,
        "borrow_rate_annual": borrow_rate,
        "slippage_bps": slippage_bps,
        "ema_long_term": ema_long_term,
        "ema_short_term": ema_short_term,
        "cross_asset_spike_top_pct": corr_spike_pct,
        "intra_equity_spike_top_pct": intra_eq_spike_pct,
        "vvix_spike_top_pct": vvix_spike_pct,
        "corr_window": corr_window,
        "spike_history_window": spike_history_window,
        "hy_spread_threshold_bps": hy_level,
        "hy_widen_threshold_bps": hy_widen_bps,
        "hy_widen_lookback_days": hy_lookback_days,
        "enable_cost_model": enable_costs,
        "hy_oas_path": str(HY_OAS_PATH),
    }
    _placeholder_tabs(feature, inputs)
    if run_clicked:
        if HY_OAS_PATH.exists():
            st.success("Static dataset check passed for Backtest.")
        else:
            st.error("Built-in HY OAS dataset not found at `data/HY_OAS.csv`.")


def _page_global_indices(feature: FeatureDef) -> None:
    _common_header(feature)
    run_clicked, _ = _top_buttons(feature)

    default_map = {
        "^GSPC": "S&P 500",
        "^IXIC": "Nasdaq Composite",
        "^DJI": "Dow Jones",
        "^FTSE": "FTSE 100",
        "^GDAXI": "DAX",
        "^FCHI": "CAC 40",
        "^N225": "Nikkei 225",
        "^HSI": "Hang Seng",
        "^STOXX50E": "Euro Stoxx 50",
        "^AXJO": "ASX 200",
    }
    c1, c2 = st.columns(2)
    tickers_raw = c1.text_area(
        "Index Tickers (comma-separated)",
        ",".join(default_map.keys()),
    )
    start_date = c2.date_input(
        "Start Date",
        value=datetime.strptime("2020-01-01", "%Y-%m-%d"),
    )
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    inputs = {"tickers": tickers, "start_date": str(start_date)}

    charts, tables, diagnostics = st.tabs(["Charts", "Tables", "Diagnostics"])
    if run_clicked:
        if not tickers:
            st.error("Please provide at least one ticker.")
            _push_history(feature.label, "failed", "no tickers")
            return
        try:
            px = _download_prices(tickers, str(start_date))
            if px.empty:
                st.error("No data returned from yfinance for the selected tickers/date.")
                _push_history(feature.label, "failed", "no data")
                return

            valid = [c for c in px.columns if px[c].dropna().shape[0] > 0]
            px = px[valid].dropna(how="all")
            latest = px.ffill().iloc[-1]

            rebased = px.ffill().divide(px.ffill().iloc[0]).multiply(100.0)
            perf_rows = []
            for col in px.columns:
                series = px[col].dropna()
                if series.empty:
                    continue
                perf_rows.append(
                    {
                        "Index": default_map.get(col, col),
                        "Ticker": col,
                        "Last": round(float(latest[col]), 2),
                        "Monthly (MTD)": _compute_mtd_return(series),
                        "YTD": _compute_ytd_return(series),
                    }
                )
            perf = pd.DataFrame(perf_rows).sort_values("YTD", ascending=False)

            with charts:
                st.line_chart(rebased, use_container_width=True)
                st.caption("Rebased index levels (100 = start date)")

            with tables:
                show = perf.copy()
                for c in ["Monthly (MTD)", "YTD"]:
                    show[c] = (show[c] * 100).map(lambda x: f"{x:.2f}%")
                st.dataframe(show, use_container_width=True, hide_index=True)

            with diagnostics:
                st.write("Input snapshot:")
                st.json(inputs)
                missing = [t for t in tickers if t not in valid]
                if missing:
                    st.warning(f"No usable data for: {', '.join(missing)}")
                st.caption(f"Rows downloaded: {len(px)} | Symbols used: {len(valid)}")

            _push_history(feature.label, "ok", f"{len(px.columns)} symbols")
        except Exception as exc:
            st.error(f"Failed to load global indices data: {exc}")
            _push_history(feature.label, "failed", "exception")
    else:
        with charts:
            st.info("Click Run to load chart.")
        with tables:
            st.info("Click Run to calculate Monthly (MTD) and YTD table.")
        with diagnostics:
            st.json(inputs)


RENDERERS = {
    "valuation_quality": _page_valuation_quality,
    "analyst_valuation": _page_analyst_valuation,
    "crash_risk": _page_crash_risk,
    "macro_equity": _page_macro_equity,
    "pair_correlation": _page_pair_correlation,
    "sector_correlation": _page_sector_correlation,
    "hedged_backtest": _page_hedged_backtest,
    "global_indices": _page_global_indices,
}


def main() -> None:
    st.set_page_config(page_title="Research Tools", layout="wide")
    _init_state()
    st.sidebar.title("Research Tools")
    st.sidebar.caption("Functions")
    labels = [f.label for f in FEATURES]
    selected_label = st.sidebar.radio("Select Function", labels, index=0)
    _global_data_status()
    _render_history()
    selected_feature = next(f for f in FEATURES if f.label == selected_label)
    RENDERERS[selected_feature.key](selected_feature)


if __name__ == "__main__":
    main()
