import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
# Conditions 
# risk 1 - regime: QQQ < EMA180 + VIX > 20
# risk 2 - regime: QQQ < EMA40 + VIX > 20
# risk 3 - regime: SPY/TLT or SPY/HYG correlation spike (top 10% of history)
# risk 4 - regime: intra-equity correlation spike (top 10% of history)
# risk 5 - regime: VVIX spike (top 10% of history)
# risk 6 - regime: HY credit spread > 350 bps (or widened >75 bps in last month)

NAV0  = 10000
START = "2016-01-01"

weights = {
    "QQQ": 1.00,
}

# Hedge settings
HEDGE_H = 0.5                # short 60% of QQQ exposure when overlay is ON
BORROW_RATE_ANNUAL = 0.02      # 2% annual borrow/financing cost on short notional
SLIPPAGE_BPS = 5               # 5 bps cost per regime switch (enter/exit)

# Regime settings
VIX_THRESHOLD = 20             # overlay requires VIX > threshold

# Rolling stats window
WINDOW = 63                    # ~3 months

# EMA settings (user-selectable in frontend)
EMA_LONG_TERM = 180
EMA_SHORT_TERM = 40
EMA_EXIT_MARGIN = 1.02


# Spike settings (top X% of history, user-configurable in frontend)
CORR_WIN = 63
HIST_WIN = 256
CROSS_ASSET_SPIKE_TOP_PCT = 10
INTRA_EQUITY_SPIKE_TOP_PCT = 10
VVIX_SPIKE_TOP_PCT = 10

# Credit spread settings (user-configurable in frontend)
HY_LOOKBACK = 20
HY_WIDEN_BPS = 75
HY_LEVEL_BPS = 350

# Sector proxies for intra-equity correlation
SECTORS = ["XLE", "XLF", "XLI", "XLK", "XLP", "XLV", "XLY"]


# ----------------------------
# functions for risk3/4 correlation spikes
# ----------------------------  
def rolling_corr(a: pd.Series, b: pd.Series, win: int) -> pd.Series:   #(win is window length AKA number of trading days used in the calc)
    return a.pct_change().rolling(win, min_periods=max(20, win//3)).corr(b.pct_change())

def spike_flag(x: pd.Series, win: int = 252, q: float = 0.90) -> pd.Series:
    thr = x.rolling(win, min_periods=max(50, win//3)).quantile(q)
    return (x > thr)


def top_pct_to_quantile(top_pct: float) -> float:
    # top 10% -> 0.90 quantile threshold
    pct = min(max(float(top_pct), 0.0), 99.0)
    return 1.0 - (pct / 100.0)


def avg_pairwise_corr(ret: pd.DataFrame, win: int) -> pd.Series:
    """
    Average off-diagonal correlation across columns in a rolling window.
    Fast enough for ~7 sector ETFs.
    """
    out = pd.Series(index=ret.index, dtype=float)
    cols = ret.columns
    n = len(cols)
    for t in range(win, len(ret)):
        w = ret.iloc[t-win:t].dropna()
        if w.shape[0] < max(30, win//2):
            continue
        C = w.corr().values
        out.iloc[t] = (C.sum() - np.trace(C)) / (n * (n - 1))
    return out


# ----------------------------
# DATA
# ----------------------------
# NEW: add SPY/TLT/HYG + SECTORS so risk3/risk4 can be computed
tickers = sorted(set(list(weights.keys()) + ["QQQ", "^VIX", "SPY", "TLT", "HYG"] + SECTORS))

px = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"].dropna(how="all")
rets = px.pct_change().dropna()

# Align everything
idx = rets.index

# Portfolio return series (if weights don't sum to 1, remainder is assumed cash at 0% return)
w_sum = sum(weights.values())
if w_sum > 1.000001:
    raise ValueError(f"Weights sum to {w_sum:.4f} > 1.0. Fix your weights.")

port = pd.Series(0.0, index=idx)
for t, w in weights.items():
    if t not in rets.columns:
        raise ValueError(f"Ticker {t} not in downloaded data. Available: {list(rets.columns)}")
    port = port.add(rets[t] * w, fill_value=0.0)


# ----------------------------
# REGIME (overlay ON/OFF)
# ----------------------------
qqq = px["QQQ"].loc[idx]
vix = px["^VIX"].loc[idx]

# NEW: series needed for risk3/risk4
spy = px["SPY"].loc[idx]
tlt = px["TLT"].loc[idx]
hyg = px["HYG"].loc[idx]
sector_prices = px[SECTORS].loc[idx].dropna(how="any")

# =========================
# Regime signals (sticky bear + tactical EMA)
# =========================
ema_long_term = qqq.ewm(span=EMA_LONG_TERM, adjust=False).mean()
ema_short_term = qqq.ewm(span=EMA_SHORT_TERM, adjust=False).mean()

# Exit bear only if QQQ > exit_margin * long-term EMA
exit_margin = EMA_EXIT_MARGIN

bear_on = (qqq < ema_long_term)
bear_off = (qqq > ema_long_term * exit_margin)

bear_state = pd.Series(0, index=idx, dtype=int)
for i in range(1, len(idx)):
    prev = bear_state.iat[i-1]
    if prev == 0 and bear_on.iat[i]:
        bear_state.iat[i] = 1
    elif prev == 1 and bear_off.iat[i]:
        bear_state.iat[i] = 0
    else:
        bear_state.iat[i] = prev


# ----------------------------
# NEW: risk3 (cross-asset correlation spikes)
# ----------------------------
corr_spy_tlt = rolling_corr(spy, tlt, CORR_WIN).abs()
corr_spy_hyg = rolling_corr(spy, hyg, CORR_WIN).abs()
risk3_q = top_pct_to_quantile(CROSS_ASSET_SPIKE_TOP_PCT)
risk3 = spike_flag(corr_spy_tlt, win=HIST_WIN, q=risk3_q) | spike_flag(corr_spy_hyg, win=HIST_WIN, q=risk3_q)

# ----------------------------
# NEW: risk4 (intra-equity correlation spikes)
# ----------------------------
sector_ret = sector_prices.pct_change()
eq_corr = avg_pairwise_corr(sector_ret, win=CORR_WIN)
risk4_q = top_pct_to_quantile(INTRA_EQUITY_SPIKE_TOP_PCT)
risk4 = spike_flag(eq_corr, win=HIST_WIN, q=risk4_q)


#Risk5 (VVIX)
vix_chg = vix.pct_change()
vix_vov = vix_chg.rolling(20, min_periods=15).std()
risk5_q = top_pct_to_quantile(VVIX_SPIKE_TOP_PCT)
risk5 = spike_flag(vix_vov, win=252, q=risk5_q).reindex(idx).fillna(False)

#Risk6 (Credit spread)
root = Path(__file__).resolve().parent
hy_path_primary = root / "data" / "HY_OAS.csv"
hy_path_legacy = root / "HY_OAS.csv"
hy_path = hy_path_primary if hy_path_primary.exists() else hy_path_legacy
if not hy_path.exists():
    raise FileNotFoundError(
        "Built-in HY OAS dataset not found. Expected data/HY_OAS.csv."
    )
hy = pd.read_csv(hy_path, parse_dates=["observation_date"])
hy = hy.rename(columns={"observation_date": "date", "BAMLH0A0HYM2": "HY_OAS_PCT"})
hy["date"] = pd.to_datetime(hy["date"], errors="coerce")
hy = hy.dropna(subset=["date"]).set_index("date")
hy["HY_OAS_PCT"] = pd.to_numeric(hy["HY_OAS_PCT"], errors="coerce")
hy = hy.loc[START:]  # START like "2010-01-01"

# Convert % to bps: 3.13 -> 313 bps
hy_oas_bps = (hy["HY_OAS_PCT"] * 100).rename("HY_OAS_BPS")

# Align to your trading dates (idx) and forward-fill missing days
hy_oas_bps = hy_oas_bps.reindex(idx).ffill()


hy_widen = hy_oas_bps.diff(HY_LOOKBACK)              # 1-month change in bps
risk6 = (hy_oas_bps > HY_LEVEL_BPS) | (hy_widen > HY_WIDEN_BPS)


#conditions 
overlay_on = (
    (bear_state == 1)
    & (qqq < ema_short_term)
    & (vix > VIX_THRESHOLD)
    & (risk3 | risk4 | risk5 | risk6)
).astype(int)
# Execute next day (realistic)
overlay_series = overlay_on.reindex(idx).shift(1).fillna(0).astype(int)


# ----------------------------
# SHORT QQQ OVERLAY HEDGE
# ----------------------------
qqq_ret = rets["QQQ"].loc[idx]

# Short overlay return: when QQQ falls, short gains (+), when QQQ rises, short loses (-)
short_overlay_return = -HEDGE_H * qqq_ret * overlay_series

# Borrow/financing cost applies only when short is active (proportional to hedge notional)
daily_borrow_cost = (BORROW_RATE_ANNUAL / 252.0)
borrow_drag = daily_borrow_cost * (HEDGE_H * overlay_series)

# Slippage cost on regime switches (enter/exit)
switch = overlay_series.diff().abs().fillna(0)  # 1 on OFF->ON or ON->OFF
slip_cost = (SLIPPAGE_BPS / 10000.0) * switch

# Hedged return series
#hedged = port + short_overlay_return - borrow_drag - slip_cost
hedged = port + short_overlay_return # NO COSTS (for pure performance comparison)


# ----------------------------
# PERFORMANCE STATS
# ----------------------------
def perf_stats(x: pd.Series):
    x = x.dropna()
    ann_ret = (1 + x).prod() ** (252/len(x)) - 1
    ann_vol = x.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    downside = x[x < 0]
    downside_vol = downside.std() * np.sqrt(252)
    sortino = ann_ret / downside_vol if downside_vol > 0 else np.nan
    return ann_ret, ann_vol, sharpe, sortino

u = perf_stats(port)
h = perf_stats(hedged)


print("")
print("")
print("=== Overall Performance ===")
print("Unhedged:  annual returns: %.2f%% | annual volatility %.2f%% | sharpe %.2f | sortino %.2f" % (u[0]*100, u[1]*100, u[2], u[3]))
print("Hedged:    annual returns: %.2f%% | annual volatility %.2f%% | sharpe %.2f | sortino %.2f" % (h[0]*100, h[1]*100, h[2], h[3]))
print("\nRISK ON days (%):", overlay_series.mean()*100)

today = pd.Timestamp.today().normalize()
overlay_today = int(overlay_series.reindex([today], method="ffill").iloc[0])
latest_signal_date = overlay_series.index[-1]
overlay_latest = int(overlay_series.iloc[-1])
print(f"Overlay ON today ({today.date()}): {'YES' if overlay_today == 1 else 'NO'}")
print(f"Latest signal date ({latest_signal_date.date()}): {'ON' if overlay_latest == 1 else 'OFF'}")

# Quick debug for new risks
risk3_days = risk3.reindex(idx).fillna(False).mean() * 100
risk4_days = risk4.reindex(idx).fillna(False).mean() * 100
risk5_days = risk5.reindex(idx).fillna(False).mean() * 100
risk6_days = risk6.reindex(idx).fillna(False).mean() * 100

print(f"\nRisk3 (cross-asset correlation) spike days (%): {risk3_days:.2f}")
print(f"Risk4 (sector correlation) spike days (%): {risk4_days:.2f}")
print(f"Risk5 (V VIX) spike days (%): {risk5_days:.2f}")
print(f"Risk6 (credit spread) spike days (%): {risk6_days:.2f}")
print(
    "Thresholds | "
    f"EMA_LONG_TERM={EMA_LONG_TERM}, EMA_SHORT_TERM={EMA_SHORT_TERM}, "
    f"Risk3 top%={CROSS_ASSET_SPIKE_TOP_PCT}, "
    f"Risk4 top%={INTRA_EQUITY_SPIKE_TOP_PCT}, "
    f"Risk5 top%={VVIX_SPIKE_TOP_PCT}, "
    f"HY level={HY_LEVEL_BPS}bps, HY widen={HY_WIDEN_BPS}bps/{HY_LOOKBACK}d"
)

print("\nEffective hedge when ON (fraction of QQQ):")
print("  Hedge fraction (HEDGE_H):", HEDGE_H)

# ----------------------------
# REGIME PERFORMANCE TABLE (ON vs OFF)
# ----------------------------
def stats(x):
    x = x.dropna()
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

print("\n=== Regime split (Risk ON vs OFF) ===")
table_fmt = table.copy()
table_fmt.loc["Ann Return"] *= 100
table_fmt.loc["Ann Vol"] *= 100
print(table_fmt.round(2))
print("\n(Ann Return/Vol shown in %, Sharpe unitless)")

# ----------------------------
# CALENDAR-YEAR RETURNS
# ----------------------------
unhedged_eq = (1 + port).cumprod()
hedged_eq   = (1 + hedged).cumprod()

un_y = unhedged_eq.resample("YE").last()
hd_y = hedged_eq.resample("YE").last()

un_cal = un_y.pct_change().dropna()
hd_cal = hd_y.pct_change().dropna()

annual_table = pd.DataFrame({
    "Unhedged": (un_cal * 100).round(2),
    "Hedged":   (hd_cal * 100).round(2),
})
annual_table.index = annual_table.index.year

print("\n=== Calendar-year returns (%) ===")
print(annual_table.to_string())

print("\nBest/Worst calendar year:")
print("Unhedged best:", annual_table["Unhedged"].max(), "worst:", annual_table["Unhedged"].min())
print("Hedged   best:", annual_table["Hedged"].max(), "worst:", annual_table["Hedged"].min())


# ----------------------------
# EQUITY CURVES + DRAWDOWN
# ----------------------------
equity_unhedged = (1 + port).cumprod()
equity_hedged   = (1 + hedged).cumprod()

def drawdown(equity_curve: pd.Series) -> pd.Series:
    peak = equity_curve.cummax()
    return equity_curve / peak - 1

dd_unhedged = drawdown(equity_unhedged)
dd_hedged   = drawdown(equity_hedged)

# Rolling Sharpe
roll_ret_u = (1 + port).rolling(WINDOW).apply(lambda x: x.prod()**(252/WINDOW)-1, raw=False)
roll_ret_h = (1 + hedged).rolling(WINDOW).apply(lambda x: x.prod()**(252/WINDOW)-1, raw=False)
roll_vol_u = port.rolling(WINDOW).std() * np.sqrt(252)
roll_vol_h = hedged.rolling(WINDOW).std() * np.sqrt(252)
roll_sharpe_u = roll_ret_u / roll_vol_u
roll_sharpe_h = roll_ret_h / roll_vol_h

# Regime shading helper
def shade_overlay(ax, label: str | None = None):
    on = overlay_series == 1
    if on.any():
        idx2 = overlay_series.index
        starts = idx2[(~on.shift(1, fill_value=False)) & on]
        ends   = idx2[(on) & (~on.shift(-1, fill_value=False))]
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(
                s,
                e,
                alpha=0.15,
                color="grey",
                label=label if (label and i == 0) else None,
            )

# ----------------------------
# PLOTS
# ----------------------------
# Plot 1: Equity curves + VIX + Risk3/Risk4 flags
risk3_plot = risk3.reindex(idx).fillna(False).astype(int)
risk4_plot = risk4.reindex(idx).fillna(False).astype(int)

fig, (ax_eq, ax_vix, ax_risk) = plt.subplots(
    3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
)

ax_eq.plot(equity_unhedged.index, equity_unhedged.values, label="Unhedged")
ax_eq.plot(equity_hedged.index, equity_hedged.values, label="Hedged (Short QQQ ON)")
shade_overlay(ax_eq, label="Overlay ON")
ax_eq.set_title("Equity Curve (Shaded = Overlay ON)")
ax_eq.set_ylabel("Asset value")
ax_eq.legend(loc="upper left")

vix_plot = vix.reindex(idx)
vix_above = vix_plot > VIX_THRESHOLD
ax_vix.plot(vix_plot.index, vix_plot.values, color="tab:red", label="VIX")
ax_vix.axhline(VIX_THRESHOLD, color="black", linestyle="--", linewidth=1.2, label=f"VIX threshold ({VIX_THRESHOLD})")
ax_vix.fill_between(
    vix_plot.index,
    VIX_THRESHOLD,
    vix_plot.values,
    where=vix_above,
    color="tab:red",
    alpha=0.2,
    interpolate=True,
    label="VIX > threshold",
)
shade_overlay(ax_vix)
ax_vix.set_xlabel("Date")
ax_vix.set_ylabel("VIX")
ax_vix.legend(loc="upper left")

ax_risk.step(risk3_plot.index, risk3_plot.values, where="post", color="tab:orange", label="Risk3 cross-asset correlation spikes")
ax_risk.step(risk4_plot.index, risk4_plot.values, where="post", color="tab:green", label="Risk4 intra-equity correlation spikes")
ax_risk.fill_between(
    risk3_plot.index, 0, 1,
    where=risk3_plot.values.astype(bool),
    color="tab:orange", alpha=0.15
)
ax_risk.fill_between(
    risk4_plot.index, 0, 1,
    where=risk4_plot.values.astype(bool),
    color="tab:green", alpha=0.15
)
shade_overlay(ax_risk)
ax_risk.set_title("Risk Flags (Risk3 / Risk4)")
ax_risk.set_ylabel("Flag")
ax_risk.set_yticks([0, 1])
ax_risk.set_xlabel("Date")
ax_risk.legend(loc="upper left")

# Show date ticks on equity and VIX panels as well (sharex normally hides upper labels)
for ax in (ax_eq, ax_vix, ax_risk):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax_eq.tick_params(axis="x", labelbottom=True)
ax_vix.tick_params(axis="x", labelbottom=True)

plt.tight_layout()
plt.show()

# Plot 2: Risk6 (Credit spread stress) + VIX
risk6_plot = risk6.reindex(idx).fillna(False)
hy_oas_plot = hy_oas_bps.reindex(idx).ffill()
hy_widen_plot = hy_widen.reindex(idx)
vix_plot_2 = vix.reindex(idx)
vix_above_2 = vix_plot_2 > VIX_THRESHOLD

fig, (ax_oas, ax_vix2, ax_widen) = plt.subplots(
    3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
)

ax_oas.plot(hy_oas_plot.index, hy_oas_plot.values, color="tab:blue", label="HY OAS (bps)")
ax_oas.axhline(HY_LEVEL_BPS, color="black", linestyle="--", linewidth=1.2, label=f"Level threshold ({HY_LEVEL_BPS} bps)")
ax_oas.fill_between(
    hy_oas_plot.index,
    0,
    hy_oas_plot.values,
    where=risk6_plot.values.astype(bool),
    color="tab:blue",
    alpha=0.15,
    label="Risk6 ON",
)
ax_oas.set_title("Risk6: High-Yield Credit Spread Stress")
ax_oas.set_ylabel("OAS (bps)")
ax_oas.legend(loc="upper left")

ax_vix2.plot(vix_plot_2.index, vix_plot_2.values, color="tab:red", label="VIX")
ax_vix2.axhline(VIX_THRESHOLD, color="black", linestyle="--", linewidth=1.2, label=f"VIX threshold ({VIX_THRESHOLD})")
ax_vix2.fill_between(
    vix_plot_2.index,
    VIX_THRESHOLD,
    vix_plot_2.values,
    where=vix_above_2,
    color="tab:red",
    alpha=0.2,
    interpolate=True,
    label="VIX > threshold",
)
ax_vix2.set_ylabel("VIX")
ax_vix2.legend(loc="upper left")

ax_widen.plot(hy_widen_plot.index, hy_widen_plot.values, color="tab:purple", label=f"{HY_LOOKBACK}d widening (bps)")
ax_widen.axhline(HY_WIDEN_BPS, color="black", linestyle="--", linewidth=1.2, label=f"Widen threshold ({HY_WIDEN_BPS} bps)")
ax_widen.fill_between(
    hy_widen_plot.index,
    0,
    hy_widen_plot.values,
    where=risk6_plot.values.astype(bool),
    color="tab:purple",
    alpha=0.15,
)
ax_widen.set_xlabel("Date")
ax_widen.set_ylabel("Widen (bps)")
ax_widen.legend(loc="upper left")

plt.tight_layout()
plt.show()

# Plot 3: Drawdowns
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dd_unhedged.index, dd_unhedged.values, label="Unhedged DD")
ax.plot(dd_hedged.index, dd_hedged.values, label="Hedged DD")
shade_overlay(ax)
ax.set_title("Drawdowns (Shaded = Overlay ON)")
ax.set_xlabel("Date")
ax.set_ylabel("Drawdown")
ax.legend()
plt.tight_layout()
plt.show()

# Plot 4: Rolling Sharpe
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(roll_sharpe_u.index, roll_sharpe_u.values, label=f"Unhedged {WINDOW}d Sharpe")
#ax.plot(roll_sharpe_h.index, roll_sharpe_h.values, label=f"Hedged {WINDOW}d Sharpe")
#shade_overlay(ax)
#ax.set_title(f"Rolling Sharpe ({WINDOW}d) (Shaded = Overlay ON)")
#ax.set_xlabel("Date")
#ax.set_ylabel("Sharpe")
#ax.legend()
#plt.tight_layout()
#plt.show()

# Plot 5: Scatter vs QQQ returns
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.scatter(qqq_ret.values, port.values, s=8, alpha=0.3, label="Unhedged")
#ax.scatter(qqq_ret.values, hedged.values, s=8, alpha=0.3, label="Hedged")
#ax.set_title("Daily Returns Scatter vs QQQ")
#ax.set_xlabel("QQQ daily return")
#ax.set_ylabel("Strategy daily return")
#ax.legend()
#plt.tight_layout()
#plt.show()
