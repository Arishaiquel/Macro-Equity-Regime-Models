
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import yfinance as yf

tickers = ["QQQ", "SPY"]  # traded + benchmark
data = yf.download(tickers, start="2010-01-01", auto_adjust=True, progress=False)

# If yfinance returns multi-index columns, take Close (or Adj Close depending on auto_adjust)
if isinstance(data.columns, pd.MultiIndex):
    px = data["Close"].copy()
else:
    px = data[["QQQ", "SPY"]].copy()  # fallback if already flat




# -----------------------------
# Mean Reversion Signals
# -----------------------------
def mean_reversion_signals(px: pd.Series, lookback=20, entry_z=2.0, exit_z=0.5) -> pd.Series:
    ma = px.rolling(lookback).mean()
    sd = px.rolling(lookback).std(ddof=0)
    z = (px - ma) / sd

    pos = pd.Series(0.0, index=px.index)
    current = 0.0

    long_entry  = z <= -entry_z
    short_entry = z >=  entry_z
    exit_to_flat = z.abs() <= exit_z

    for t in px.index:
        if np.isnan(z.loc[t]):
            pos.loc[t] = current
            continue

        if current == 0.0:
            if long_entry.loc[t]:
                current = 1.0
            elif short_entry.loc[t]:
                current = -1.0
        else:
            if exit_to_flat.loc[t]:
                current = 0.0

        pos.loc[t] = current

    return pos

# -----------------------------
# Performance Stats
# -----------------------------
def performance_stats(strat_rets: pd.Series, equity: pd.Series, annualization: int = 252) -> pd.Series:
    strat_rets = strat_rets.dropna()
    equity = equity.reindex(strat_rets.index)

    eps = 1e-12
    n = strat_rets.shape[0]
    years = n / annualization if n > 0 else np.nan

    total_return = equity.iloc[-1] - 1.0
    cagr = (equity.iloc[-1] ** (1.0 / years) - 1.0) if (years and years > 0) else np.nan

    vol = strat_rets.std(ddof=0) * np.sqrt(annualization)
    mean_ann = strat_rets.mean() * annualization
    sharpe = mean_ann / (vol + eps)

    downside = strat_rets.clip(upper=0)
    downside_vol = downside.std(ddof=0) * np.sqrt(annualization)
    sortino = mean_ann / (downside_vol + eps)

    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = dd.min()

    calmar = cagr / (abs(max_dd) + eps) if pd.notna(cagr) else np.nan

    win_rate = (strat_rets > 0).mean()
    profit_factor = strat_rets[strat_rets > 0].sum() / (abs(strat_rets[strat_rets < 0].sum()) + eps)

    return pd.Series({
        "Total Return": total_return,
        "CAGR": cagr,
        "Ann. Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Num Obs": n,
        "Years": years,
    })

# -----------------------------
# Backtest Engine
# -----------------------------
def backtest_positions(px: pd.Series, pos: pd.Series, fee_bps=1.0, slippage_bps=0.0, annualization=252) -> dict:
    px = px.dropna()
    pos = pos.reindex(px.index).fillna(0.0)

    rets = px.pct_change().fillna(0.0)
    pos_lag = pos.shift(1).fillna(0.0)

    turnover = pos.diff().abs().fillna(0.0)
    cost_rate = (fee_bps + slippage_bps) / 10000.0
    costs = turnover * cost_rate

    strat_rets = pos_lag * rets - costs
    equity = (1.0 + strat_rets).cumprod()

    stats = performance_stats(strat_rets, equity, annualization=annualization)

    return {"strat_rets": strat_rets, "equity": equity, "stats": stats}

# -----------------------------
# Benchmark (SPY Buy & Hold)
# -----------------------------
def buy_and_hold_equity(px_bench: pd.Series) -> tuple[pd.Series, pd.Series]:
    bench_rets = px_bench.pct_change().fillna(0.0)
    bench_eq = (1.0 + bench_rets).cumprod()
    return bench_rets, bench_eq

# -----------------------------
# Plot: Strategy vs Benchmark
# -----------------------------
def plot_strategy_vs_benchmark(eq_strat: pd.Series, eq_bench: pd.Series, title="Equity Curve: Strategy vs SPY"):
    # align indices
    idx = eq_strat.index.intersection(eq_bench.index)
    eq_strat = eq_strat.reindex(idx)
    eq_bench = eq_bench.reindex(idx)

    plt.figure(figsize=(11, 5))
    plt.plot(eq_strat.index, eq_strat.values, label="Mean Reversion Strategy")
    plt.plot(eq_bench.index, eq_bench.values, label="SPY Buy & Hold")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (Growth of $1)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# HOW TO RUN (Minimal)
# -----------------------------
# Requirements:
# - px is a DataFrame with DatetimeIndex
# - columns include your traded asset (e.g. "QQQ") and "SPY" for benchmark
#
# Example:
# traded = px["QQQ"]
# bench  = px["SPY"]

def run_mean_reversion_with_spy_benchmark(px: pd.DataFrame, traded_ticker="QQQ",
                                         lookback=20, entry_z=2.0, exit_z=0.5,
                                         fee_bps=1.0, slippage_bps=0.0):
    traded = px[traded_ticker].dropna()
    bench = px["SPY"].dropna()

    # Align by date intersection
    idx = traded.index.intersection(bench.index)
    traded = traded.reindex(idx)
    bench = bench.reindex(idx)

    pos = mean_reversion_signals(traded, lookback=lookback, entry_z=entry_z, exit_z=exit_z)
    res = backtest_positions(traded, pos, fee_bps=fee_bps, slippage_bps=slippage_bps)

    bench_rets, bench_eq = buy_and_hold_equity(bench)
    bench_stats = performance_stats(bench_rets, bench_eq)

    print("\n=== Strategy Stats ===")
    print(res["stats"].to_string())

    print("\n=== SPY Buy & Hold Stats ===")
    print(bench_stats.to_string())

    plot_strategy_vs_benchmark(res["equity"], bench_eq,
                               title=f"{traded_ticker} Mean Reversion vs SPY (lookback={lookback}, entry={entry_z}, exit={exit_z})")

    return res, {"bench_rets": bench_rets, "bench_eq": bench_eq, "bench_stats": bench_stats}



# ---- QUICK RUN BLOCK (paste at the bottom) ----
import matplotlib.pyplot as plt

print("px type:", type(px))
print("px shape:", getattr(px, "shape", None))
print("px columns (first 10):", list(px.columns)[:10] if hasattr(px, "columns") else None)
print("px index sample:", px.index[:3] if hasattr(px, "index") else None)

assert isinstance(px, pd.DataFrame), "px must be a pandas DataFrame with columns like 'QQQ' and 'SPY'."
assert "SPY" in px.columns, "Missing benchmark column 'SPY' in px."
assert "QQQ" in px.columns, "Missing traded column 'QQQ' in px. Change traded_ticker to a column that exists."

res, bench = run_mean_reversion_with_spy_benchmark(
    px,
    traded_ticker="QQQ",
    lookback=20,
    entry_z=2.0,
    exit_z=0.5,
    fee_bps=1.0,
    slippage_bps=0.0
)

print("\nFinished. Equity last value:", float(res["equity"].iloc[-1]))

# Force plot display (helps in some environments)
plt.show()
