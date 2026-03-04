#!/usr/bin/env python3
"""
Volatility regime detector + long/short backtest on SPY or QQQ.
options data -> regime features -> position map -> backtest vs buy/hold
options data are from marketData

flow 
options data -> expiration dates, strikes, historial quote -> features ( IV, skew, smile, PCR Open interest and volume, IV term-structure)
Signal -> core signal (VIX + realized-vol + trend) -> 



Supports:
1) Core regime (VIX + realized-vol + trend)
2) Options regime (ATM IV, skew, smile, PCR, term-structure)
3) Hybrid regime (OR of core and options signals)

Examples:
  python Volatility_Regime.py --asset QQQ --style long_short --start 2015-01-01
  python Volatility_Regime.py --asset SPY --style long_cash --regime-mode core --no-plot
  python Volatility_Regime.py --asset SPY --style long_short --regime-mode options \
    --options-features data/options/SPX_option_features.csv --no-plot
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf


TRADING_DAYS = 252


@dataclass
class Config:
    asset: str
    start: str
    end: str | None
    style: str
    regime_mode: str
    options_features: str | None
    vix_threshold: float
    vol_lookback: int
    vol_z_lookback: int
    vol_z_threshold: float
    opt_lookback: int
    opt_iv_pctl_off: float
    opt_skew_pctl_off: float
    opt_smile_pctl_off: float
    opt_pcr_pctl_off: float
    opt_term_off: float
    opt_score_off: int
    ema_fast: int
    ema_slow: int
    short_borrow_annual: float
    slippage_bps: float
    no_plot: bool
    out_dir: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Volatility regime backtest for SPY/QQQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--asset", choices=["SPY", "QQQ"], default="QQQ")
    parser.add_argument("--start", default="2010-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument(
        "--style",
        choices=["long_short", "long_cash", "short_cash"],
        default="long_short",
        help=(
            "Position map by regime: "
            "long_short=+1 risk-on / -1 risk-off, "
            "long_cash=+1 risk-on / 0 risk-off, "
            "short_cash=0 risk-on / -1 risk-off"
        ),
    )
    parser.add_argument(
        "--regime-mode",
        choices=["core", "options", "hybrid"],
        default="core",
        help="core=price/vol filters, options=options-only filters, hybrid=OR of both",
    )
    parser.add_argument(
        "--options-features",
        default=None,
        help="CSV from Historical_Data/build_option_regime_features.py",
    )
    parser.add_argument("--vix-threshold", type=float, default=20.0)
    parser.add_argument("--vol-lookback", type=int, default=20, help="Realized vol window")
    parser.add_argument("--vol-z-lookback", type=int, default=252, help="Vol z-score window")
    parser.add_argument("--vol-z-threshold", type=float, default=0.5)
    parser.add_argument("--opt-lookback", type=int, default=252, help="Rolling lookback for options percentiles")
    parser.add_argument("--opt-iv-pctl-off", type=float, default=0.80)
    parser.add_argument("--opt-skew-pctl-off", type=float, default=0.70)
    parser.add_argument("--opt-smile-pctl-off", type=float, default=0.70)
    parser.add_argument("--opt-pcr-pctl-off", type=float, default=0.75)
    parser.add_argument("--opt-term-off", type=float, default=1.05)
    parser.add_argument("--opt-score-off", type=int, default=2, help="Risk-off when options score >= this")
    parser.add_argument("--ema-fast", type=int, default=50)
    parser.add_argument("--ema-slow", type=int, default=200)
    parser.add_argument("--short-borrow-annual", type=float, default=0.02)
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Cost on each position switch")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--out-dir", default="data/vol_regime")
    args = parser.parse_args()

    return Config(
        asset=args.asset,
        start=args.start,
        end=args.end,
        style=args.style,
        regime_mode=args.regime_mode,
        options_features=args.options_features,
        vix_threshold=args.vix_threshold,
        vol_lookback=max(5, args.vol_lookback),
        vol_z_lookback=max(30, args.vol_z_lookback),
        vol_z_threshold=args.vol_z_threshold,
        opt_lookback=max(60, args.opt_lookback),
        opt_iv_pctl_off=min(max(args.opt_iv_pctl_off, 0.0), 1.0),
        opt_skew_pctl_off=min(max(args.opt_skew_pctl_off, 0.0), 1.0),
        opt_smile_pctl_off=min(max(args.opt_smile_pctl_off, 0.0), 1.0),
        opt_pcr_pctl_off=min(max(args.opt_pcr_pctl_off, 0.0), 1.0),
        opt_term_off=max(0.01, args.opt_term_off),
        opt_score_off=max(1, args.opt_score_off),
        ema_fast=max(5, args.ema_fast),
        ema_slow=max(20, args.ema_slow),
        short_borrow_annual=max(0.0, args.short_borrow_annual),
        slippage_bps=max(0.0, args.slippage_bps),
        no_plot=args.no_plot,
        out_dir=args.out_dir,
    )


def download_data(cfg: Config) -> pd.DataFrame:
    symbols = [cfg.asset, "^VIX"]
    raw = yf.download(
        symbols,
        start=cfg.start,
        end=cfg.end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError("No data downloaded from yfinance.")

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.rename(columns={cfg.asset: "asset", "^VIX": "vix"})
    close = close[["asset", "vix"]].dropna()
    if len(close) < 300:
        raise RuntimeError("Not enough history for regime model. Increase date range.")
    return close


def load_options_features(path: str) -> pd.DataFrame:
    opt = pd.read_csv(path)
    if opt.empty:
        raise RuntimeError(f"Options features file is empty: {path}")

    date_col = None
    for c in opt.columns:
        if c.lower() in {"date", "updated", "timestamp"}:
            date_col = c
            break
    if date_col is None:
        raise RuntimeError(f"Options features CSV requires a date column. Got: {list(opt.columns)}")

    opt["date"] = pd.to_datetime(opt[date_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    opt = opt.dropna(subset=["date"]).set_index("date").sort_index()
    return opt


def rolling_percentile_of_last(x: pd.Series, lookback: int) -> pd.Series:
    min_periods = max(60, lookback // 4)

    def _pct(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0 or np.isnan(arr[-1]):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        return float((valid <= arr[-1]).mean())

    return x.rolling(lookback, min_periods=min_periods).apply(_pct, raw=True)


def build_regime_features(px: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = px.copy()
    df["asset_ret"] = df["asset"].pct_change()

# Realized volatility as annualized std of returns over lookback window. Then compute z-score of realized vol against its rolling mean/std.
    df["realized_vol"] = df["asset_ret"].rolling(cfg.vol_lookback).std() * np.sqrt(TRADING_DAYS)
    vol_mean = df["realized_vol"].rolling(cfg.vol_z_lookback).mean() #Rolling mean of realized volatility 
    vol_std = df["realized_vol"].rolling(cfg.vol_z_lookback).std(ddof=0).replace(0, np.nan) #Rolling std of realized volatility
    df["vol_z"] = (df["realized_vol"] - vol_mean) / vol_std #Z-score of realized volatility

    df["ema_fast"] = df["asset"].ewm(span=cfg.ema_fast, adjust=False).mean() # trennd filter: 50-day EMA
    df["ema_slow"] = df["asset"].ewm(span=cfg.ema_slow, adjust=False).mean() # trend filter: 200-day EMA
 
 
 # Core risk-off 
    core_risk_off = (   
        (df["vix"] > cfg.vix_threshold) #VIX >Threshold
        | (df["vol_z"] > cfg.vol_z_threshold) #  vol z-score > Threshold
        | (df["asset"] < df["ema_slow"]) #  asset price < 200-day EMA
    )
    df["core_risk_off"] = core_risk_off.fillna(False) # stores core risk-off signal in dataframe, filling NA values with False

    df["option_risk_off"] = False
    df["option_score"] = 0  # initialize option risk-off and score columns to default values. These will be updated if regime_mode includes options filters.

    if cfg.regime_mode in {"options", "hybrid"}:
        if not cfg.options_features:
            raise RuntimeError(
                "regime_mode options/hybrid requires --options-features. "
                "Build it with Historical_Data/build_option_regime_features.py."
            )
        if not os.path.exists(cfg.options_features):
            raise RuntimeError(f"Options features file not found: {cfg.options_features}")

        opt = load_options_features(cfg.options_features)
        wanted = ["iv_atm_30", "skew_25d", "smile_curvature", "iv_term_30_90", "pcr_oi"] # Options features we want to use for regime detection
        available = [c for c in wanted if c in opt.columns]
        if not available:
            raise RuntimeError(f"Options features CSV has none of required columns: {wanted}")

        df = df.join(opt[available], how="left")
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill(limit=3)

        df["sig_iv"] = False
        df["sig_skew"] = False
        df["sig_smile"] = False
        df["sig_pcr"] = False
        df["sig_term"] = False

        if "iv_atm_30" in df.columns:
            df["iv_atm_30_pctl"] = rolling_percentile_of_last(df["iv_atm_30"], cfg.opt_lookback)
            df["sig_iv"] = (df["iv_atm_30_pctl"] > cfg.opt_iv_pctl_off).fillna(False) # Flags IV stress if > percentile threshold
        if "skew_25d" in df.columns:
            df["skew_25d_pctl"] = rolling_percentile_of_last(df["skew_25d"], cfg.opt_lookback)
            df["sig_skew"] = (df["skew_25d_pctl"] > cfg.opt_skew_pctl_off).fillna(False)
        if "smile_curvature" in df.columns:
            df["smile_curvature_pctl"] = rolling_percentile_of_last(df["smile_curvature"], cfg.opt_lookback)
            df["sig_smile"] = (df["smile_curvature_pctl"] > cfg.opt_smile_pctl_off).fillna(False)
        if "pcr_oi" in df.columns:
            df["pcr_oi_pctl"] = rolling_percentile_of_last(df["pcr_oi"], cfg.opt_lookback)
            df["sig_pcr"] = (df["pcr_oi_pctl"] > cfg.opt_pcr_pctl_off).fillna(False)
        if "iv_term_30_90" in df.columns:
            df["sig_term"] = (df["iv_term_30_90"] > cfg.opt_term_off).fillna(False)

        sig_cols = ["sig_iv", "sig_skew", "sig_smile", "sig_pcr", "sig_term"]
        df["option_score"] = df[sig_cols].astype(int).sum(axis=1)
        df["option_risk_off"] = (df["option_score"] >= cfg.opt_score_off).fillna(False)

    if cfg.regime_mode == "core":
        risk_off = df["core_risk_off"]
    elif cfg.regime_mode == "options":
        risk_off = df["option_risk_off"]
    else:
        risk_off = df["core_risk_off"] | df["option_risk_off"]

    risk_on = (~risk_off) & (df["ema_fast"] > df["ema_slow"])
    df["risk_off"] = risk_off.fillna(False)
    df["risk_on"] = risk_on.fillna(False)
    return df.dropna(subset=["asset_ret"]).copy()

# TRADE POSITION
def map_position(risk_on: pd.Series, style: str) -> pd.Series:
    pos = pd.Series(0.0, index=risk_on.index)
    if style == "long_short": # long for risk on and short for risk off
        pos.loc[risk_on] = 1.0
        pos.loc[~risk_on] = -1.0
    elif style == "long_cash":
        pos.loc[risk_on] = 1.0
        pos.loc[~risk_on] = 0.0
    elif style == "short_cash":
        pos.loc[risk_on] = 0.0
        pos.loc[~risk_on] = -1.0
    else:
        raise ValueError(f"Unknown style: {style}")
    return pos


def perf_stats(ret: pd.Series) -> dict[str, float]:
    ret = ret.dropna()
    if ret.empty:
        return {
            "ann_return": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
        }
    equity = (1.0 + ret).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    ann_return = equity.iloc[-1] ** (TRADING_DAYS / len(ret)) - 1.0
    ann_vol = ret.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": dd.min(),
        "hit_rate": (ret > 0).mean(),
    }


def run_backtest(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    raw_pos = map_position(out["risk_on"], cfg.style)

    # Shift one day so today's signal trades tomorrow.
    out["position"] = raw_pos.shift(1).fillna(0.0)
    out["switch"] = out["position"].diff().abs().fillna(0.0)

    short_borrow_daily = cfg.short_borrow_annual / TRADING_DAYS
    borrow_drag = short_borrow_daily * (out["position"] < 0).astype(float) * out["position"].abs()
    slippage = (cfg.slippage_bps / 10000.0) * out["switch"]

    out["strategy_ret"] = out["position"] * out["asset_ret"] - borrow_drag - slippage
    out["buy_hold_ret"] = out["asset_ret"]

    out["strategy_equity"] = (1.0 + out["strategy_ret"]).cumprod()
    out["buy_hold_equity"] = (1.0 + out["buy_hold_ret"]).cumprod()

    summary = pd.DataFrame(
        {
            "strategy": perf_stats(out["strategy_ret"]),
            "buy_hold": perf_stats(out["buy_hold_ret"]),
        }
    )
    return out, summary


def save_outputs(result: pd.DataFrame, summary: pd.DataFrame, cfg: Config) -> tuple[str, str]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    base = f"{cfg.asset}_{cfg.style}_{cfg.regime_mode}_{cfg.start}_to_{cfg.end or 'latest'}"
    result_path = os.path.join(cfg.out_dir, f"{base}_timeseries.csv")
    summary_path = os.path.join(cfg.out_dir, f"{base}_summary.csv")
    result.to_csv(result_path)
    summary.to_csv(summary_path)
    return result_path, summary_path


def plot_results(result: pd.DataFrame, cfg: Config) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(result.index, result["buy_hold_equity"], label=f"Buy/Hold {cfg.asset}", linewidth=1.8)
    axes[0].plot(result.index, result["strategy_equity"], label=f"Regime {cfg.style}", linewidth=1.8)
    axes[0].set_title(f"{cfg.asset} Volatility Regime Backtest")
    axes[0].set_ylabel("Equity")
    axes[0].legend(loc="upper left")

    risk_off = (~result["risk_on"]).astype(int)
    axes[1].plot(result.index, result["vix"], label="VIX", color="tab:red", alpha=0.8)
    axes[1].axhline(cfg.vix_threshold, linestyle="--", linewidth=1.2, color="black", label="VIX threshold")
    axes[1].fill_between(
        result.index,
        0,
        result["vix"],
        where=risk_off > 0,
        alpha=0.1,
        color="tab:red",
        label="Risk-off windows",
    )
    axes[1].set_ylabel("VIX")
    axes[1].legend(loc="upper left")

    axes[2].step(result.index, result["position"], where="post", color="tab:blue")
    axes[2].set_ylabel("Position")
    axes[2].set_xlabel("Date")
    axes[2].set_yticks([-1, 0, 1])
    axes[2].set_title("Executed Position (1=Long, -1=Short, 0=Cash)")

    plt.tight_layout()
    plt.show()


def print_summary(cfg: Config, summary: pd.DataFrame, result: pd.DataFrame) -> None:
    print("\nBacktest Configuration")
    print(
        f"asset={cfg.asset} | style={cfg.style} | regime_mode={cfg.regime_mode} | "
        f"start={cfg.start} | end={cfg.end or 'latest'}"
    )
    print(
        "core filters: "
        f"VIX>{cfg.vix_threshold}, vol_z>{cfg.vol_z_threshold}, "
        f"EMA{cfg.ema_fast}/EMA{cfg.ema_slow}"
    )
    if cfg.regime_mode in {"options", "hybrid"}:
        print(
            "options filters: "
            f"iv_pctl>{cfg.opt_iv_pctl_off}, skew_pctl>{cfg.opt_skew_pctl_off}, "
            f"smile_pctl>{cfg.opt_smile_pctl_off}, pcr_pctl>{cfg.opt_pcr_pctl_off}, "
            f"term>{cfg.opt_term_off}, score>={cfg.opt_score_off}"
        )
        print(f"options_features={cfg.options_features}")

    print("\nPerformance Summary")
    pretty = pd.DataFrame(index=summary.index, columns=summary.columns, dtype=object)
    pct_rows = ["ann_return", "ann_vol", "max_drawdown", "hit_rate"]
    for k in pct_rows:
        if k in summary.index:
            pretty.loc[k] = (summary.loc[k].astype(float) * 100.0).round(2).astype(str) + "%"
    if "sharpe" in summary.index:
        pretty.loc["sharpe"] = summary.loc["sharpe"].astype(float).round(2).astype(str)
    print(pretty)

    print(f"\nObservations: {len(result)}")
    print(f"Risk-off share: {(~result['risk_on']).mean() * 100:.2f}%")
    print(f"Average gross exposure: {result['position'].abs().mean():.2f}x")
    if cfg.regime_mode in {"options", "hybrid"} and "option_score" in result.columns:
        print(f"Average option score: {result['option_score'].mean():.2f}")


def main() -> None:
    cfg = parse_args()
    px = download_data(cfg)
    features = build_regime_features(px, cfg)
    result, summary = run_backtest(features, cfg)
    result_path, summary_path = save_outputs(result, summary, cfg)
    print_summary(cfg, summary, result)
    print(f"\nSaved timeseries: {result_path}")
    print(f"Saved summary:    {summary_path}")
    if not cfg.no_plot:
        plot_results(result, cfg)


if __name__ == "__main__":
    main()
