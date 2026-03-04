import numpy as np
import pandas as pd
import yfinance as yf

NAV0 = 13346
START = "2020-01-01"

# portfolio weights (ABT combined)
weights = {
    "META": 0.3,
    "NFLX": 0.0677,
    "MSFT": 0.06,
    "ORCL": 0.0759,
    "NVDA": 0.0972,
    "UMMA": 0.3,
    "AMZN": 0.063,
    "PG": 0.06,
    "BABA": 0.0365,
    "GLD": 0.0341,
    "ABT": 0.0619,
}
other_w = max(0.0, 1.0 - sum(weights.values()))

# hedge parameters
NET_DELTA_CORE = 0.15      # magnitude; hedge delta is negative
NET_DELTA_OVERLAY = 0.10   # extra magnitude when overlay is ON
PREMIUM_BUDGET_MO = 0.01   # 1% of NAV per month

# pull data
tickers = list(weights.keys()) + ["QQQ", "^VIX"]
px = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"].dropna()
rets = px.pct_change().dropna()

# portfolio returns (proxy "other" as QQQ)
port = sum(rets[t]*w for t,w in weights.items())
if other_w > 0:
    port = port + other_w*rets["QQQ"]

qqq = px["QQQ"].loc[port.index]
vix = px["^VIX"].loc[port.index]

# regime signals
qqq_200 = qqq.rolling(200).mean()
risk1 = qqq < qqq_200
risk2 = vix > 20
overlay_on = (risk1 & risk2).astype(int)

# hedge P&L approximation: -delta * (QQQ return) * (QQQ notional)
# QQQ notional per 1 contract = QQQ*100, so delta-notional exposure changes with QQQ level.
# For backtest simplicity, use daily P&L proportional to index return:
# hedge_return ≈ + net_delta * (-QQQ_return) * (QQQ_notional / NAV)
# We'll compute hedge as return on NAV.

qqq_notional = qqq * 100.0
core_hedge_return = (NET_DELTA_CORE) * (-rets["QQQ"].loc[port.index]) * (qqq_notional / NAV0)
overlay_hedge_return = (NET_DELTA_OVERLAY) * (-rets["QQQ"].loc[port.index]) * (qqq_notional / NAV0) * overlay_on

# premium drag: subtract 1% NAV per month spread across trading days (~21)
daily_premium_drag = (PREMIUM_BUDGET_MO / 21.0)

hedged = port + core_hedge_return + overlay_hedge_return - daily_premium_drag

def perf_stats(x):
    ann_ret = (1 + x).prod() ** (252/len(x)) - 1
    ann_vol = x.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe

u = perf_stats(port)
h = perf_stats(hedged)

print("Unhedged:  ann_ret %.2f%% | ann_vol %.2f%% | sharpe %.2f" % (u[0]*100, u[1]*100, u[2]))
print("Hedged:    ann_ret %.2f%% | ann_vol %.2f%% | sharpe %.2f" % (h[0]*100, h[1]*100, h[2]))

print("\nOverlay ON days (%):", overlay_on.mean()*100)
