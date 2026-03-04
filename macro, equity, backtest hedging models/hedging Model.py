import numpy as np
import pandas as pd
import yfinance as yf

NAV0 = 10000
START = "2010-01-01"

# portfolio weights (ABT combined)
weights = {
    #"META": 0.30,
    #"NFLX": 0.0677,
   # "MSFT": 0.06,
   # "ORCL": 0.0759,
   # "NVDA": 0.0972,
   # "UMMA": 0.70,
      "QQQ": 1.00,
    #"QQQ" : 0.30,
   # "AMZN": 0.063,
   # "PG": 0.06,
  #  "BABA": 0.0365,
    #"GLD": 0.0341,
   # "ABT": 0.0619,
}
other_w = max(0.0, 1.0 - sum(weights.values()))

# hedge parameters
NET_DELTA_CORE = 0     # magnitude; hedge delta is negative
NET_DELTA_OVERLAY = 0.150   # extra magnitude when overlay is ON
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
qqq_50 = qqq.rolling(50).mean()
risk1 = qqq < qqq_50
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

hedged = port + core_hedge_return + overlay_hedge_return - daily_premium_drag * overlay_on

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


import matplotlib.pyplot as plt

# ---- Build equity curves
equity_unhedged = (1 + port).cumprod()
equity_hedged   = (1 + hedged).cumprod()

# ---- Drawdowns
def drawdown(equity_curve: pd.Series) -> pd.Series:
    peak = equity_curve.cummax()
    return equity_curve / peak - 1

dd_unhedged = drawdown(equity_unhedged)
dd_hedged   = drawdown(equity_hedged)

# ---- Rolling stats
WINDOW = 63  # ~3 months
roll_ret_u = (1 + port).rolling(WINDOW).apply(lambda x: x.prod()**(252/WINDOW)-1, raw=False)
roll_ret_h = (1 + hedged).rolling(WINDOW).apply(lambda x: x.prod()**(252/WINDOW)-1, raw=False)

roll_vol_u = port.rolling(WINDOW).std() * np.sqrt(252)
roll_vol_h = hedged.rolling(WINDOW).std() * np.sqrt(252)

roll_sharpe_u = roll_ret_u / roll_vol_u
roll_sharpe_h = roll_ret_h / roll_vol_h

# ---- Regime shading helper
overlay_series = overlay_on.reindex(port.index).fillna(0).astype(int)

def shade_overlay(ax):
    # Shade regions where overlay is ON
    on = overlay_series == 1
    if on.any():
        # find contiguous ON segments
        idx = overlay_series.index
        # identify start/end points of ON blocks
        starts = idx[(~on.shift(1, fill_value=False)) & on]
        ends   = idx[(on) & (~on.shift(-1, fill_value=False))]
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, alpha=0.15)

# =========================
# Plot 1: Equity curves + overlay shading
# =========================
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(equity_unhedged.index, equity_unhedged.values, label="Unhedged")
ax.plot(equity_hedged.index, equity_hedged.values, label="Hedged")
shade_overlay(ax)
ax.set_title("Equity Curve")
ax.set_xlabel("Date")
ax.set_ylabel("Asset Value")
ax.legend()
plt.show()

# =========================
# Plot 2: Drawdowns + overlay shading
# =========================
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dd_unhedged.index, dd_unhedged.values, label="Unhedged DD")
ax.plot(dd_hedged.index, dd_hedged.values, label="Hedged DD")
shade_overlay(ax)
ax.set_title("Drawdowns (Shaded = Overlay ON)")
ax.set_xlabel("Date")
ax.set_ylabel("Drawdown")
ax.legend()
plt.show()

# =========================
# Plot 3: Rolling Sharpe + overlay shading
# =========================
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(roll_sharpe_u.index, roll_sharpe_u.values, label=f"Unhedged {WINDOW}d Sharpe")
ax.plot(roll_sharpe_h.index, roll_sharpe_h.values, label=f"Hedged {WINDOW}d Sharpe")
shade_overlay(ax)
ax.set_title(f"Rolling Sharpe ({WINDOW}d) (Shaded = Overlay ON)")
ax.set_xlabel("Date")
ax.set_ylabel("Sharpe")
ax.legend()
plt.show()

# =========================
# Plot 4: Scatter vs QQQ returns (who behaves better in down markets)
# =========================
qqq_ret = rets["QQQ"].loc[port.index]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(qqq_ret.values, port.values, s=8, alpha=0.3, label="Unhedged")
ax.scatter(qqq_ret.values, hedged.values, s=8, alpha=0.3, label="Hedged")
ax.set_title("Daily Returns Scatter vs QQQ")
ax.set_xlabel("QQQ daily return")
ax.set_ylabel("Strategy daily return")
ax.legend()
plt.show()

# =========================
# Regime performance table: overlay ON vs OFF
# =========================
def stats(x):
    ann_ret = (1 + x).prod() ** (252/len(x)) - 1
    ann_vol = x.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe

mask_on  = overlay_series == 1
mask_off = overlay_series == 0

u_on  = stats(port[mask_on])
h_on  = stats(hedged[mask_on])
u_off = stats(port[mask_off])
h_off = stats(hedged[mask_off])

table = pd.DataFrame({
    "Unhedged (ON)":  [u_on[0], u_on[1], u_on[2]],
    "Hedged (ON)":    [h_on[0], h_on[1], h_on[2]],
    "Unhedged (OFF)": [u_off[0], u_off[1], u_off[2]],
    "Hedged (OFF)":   [h_off[0], h_off[1], h_off[2]],
}, index=["Ann Return", "Ann Vol", "Sharpe"])

print("\n=== Regime split (Hedging ON vs OFF) ===")

table_fmt = table.copy()
table_fmt.loc["Ann Return"] *= 100
table_fmt.loc["Ann Vol"] *= 100

print(table_fmt.round(2))
print("\n(Ann Return/Vol shown in %, Sharpe unitless)")

# =========================
# Calendar-year returns (Unhedged vs Hedged)
# =========================
unhedged_eq = (1 + port).cumprod()
hedged_eq   = (1 + hedged).cumprod()

# Year-end values
un_y = unhedged_eq.resample("Y").last()
hd_y = hedged_eq.resample("Y").last()

# Calendar-year returns = YoY % change in year-end equity
un_cal = un_y.pct_change().dropna()
hd_cal = hd_y.pct_change().dropna()

annual_table = pd.DataFrame({
    "Unhedged": (un_cal * 100).round(2),
    "Hedged":   (hd_cal * 100).round(2),
})

# Make the index show the year (e.g., 2021, 2022)
annual_table.index = annual_table.index.year

print("\n=== Calendar-year returns (%) ===")
print(annual_table.to_string())


print("\nBest/Worst calendar year:")
print("Unhedged best:", annual_table["Unhedged"].max(), "worst:", annual_table["Unhedged"].min())
print("Hedged   best:", annual_table["Hedged"].max(), "worst:", annual_table["Hedged"].min())


effective_hedge = NET_DELTA_OVERLAY * (qqq_notional / NAV0)
print("Avg effective hedge (ON):", effective_hedge[overlay_on==1].mean())
print("Max effective hedge (ON):", effective_hedge[overlay_on==1].max())
