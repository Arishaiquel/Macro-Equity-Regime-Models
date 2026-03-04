print("RUNNING macrovsIndex.py ✅", flush=True)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# -----------------------
# 1) Your weights + direction
# -----------------------
WEIGHTS = {
    "Policy Rate": 0.17,
    "10Y Yield": 0.16,
    "CPI YoY": 0.11,
    "Core CPI YoY": 0.11,
    "GDP QoQ": 0.10,
    "GDP YoY": 0.07,
    "Jobless Rate": 0.07,
    "Retail Sales": 0.06,
    "Industrial Production": 0.04,
    "Current Account / GDP": 0.04,
    "Govt Debt / GDP": 0.03,
    "Population": 0.03,
}

# Nasdaq/tech lens: "good" should push score UP, "bad" pushes DOWN
DIRECTION = {
    "Policy Rate": -1,
    "10Y Yield": -1,
    "CPI YoY": -1,
    "Core CPI YoY": -1,
    "GDP QoQ": +1,
    "GDP YoY": +1,
    "Jobless Rate": -1,
    "Retail Sales": +1,
    "Industrial Production": +1,
    "Current Account / GDP": +1,
    "Govt Debt / GDP": -1,
    "Population": +1,
}

START = "1995-01-01"

# -----------------------
# 2) Download from FRED (NO pandas_datareader)
# -----------------------
def fred_csv(series_id: str, start: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["DATE", series_id]   # force correct names
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    s = df[series_id].replace(".", np.nan).astype(float)
    return s.loc[start:]


fred_series = {
    # Use FEDFUNDS (longer history) instead of EFFR
    "Policy Rate": "FEDFUNDS",
    "10Y Yield": "DGS10",
    "CPI": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "GDP": "GDP",
    "Jobless Rate": "UNRATE",
    "Retail Sales": "RSAFS",
    "Industrial Production": "INDPRO",
    "Govt Debt / GDP": "GFDEGDQ188S",
    "Current Account": "IEABCA",
    "Population": "POP",
}

print("Downloading FRED series...")
raw = {}
for name, sid in fred_series.items():
    print(" -", name, sid)
    raw[name] = fred_csv(sid, START)

df = pd.concat(raw, axis=1)

# -----------------------
# 3) Transform into model factors (monthly)
# -----------------------
m = df.resample("M").last()

# Inflation YoY
m["CPI YoY"] = m["CPI"].pct_change(12) * 100
m["Core CPI YoY"] = m["Core CPI"].pct_change(12) * 100

# GDP QoQ and YoY from quarterly GDP, forward-fill to monthly
gdp_q = df["GDP"].dropna()
gdp_qoq = gdp_q.pct_change(1) * 100
gdp_yoy = gdp_q.pct_change(4) * 100
g = pd.concat([gdp_qoq.rename("GDP QoQ"), gdp_yoy.rename("GDP YoY")], axis=1).resample("M").ffill()
m = m.join(g)

# Current Account / GDP (quarterly -> monthly ffill)
ca = df["Current Account"].dropna()
ca_gdp = (ca / gdp_q) * 100
m["Current Account / GDP"] = ca_gdp.resample("M").ffill()

# Extract only the exact factor columns you want
factors = m[list(WEIGHTS.keys())].copy()

# -----------------------
# 4) Rolling Z-score per factor (robust to missing history)
# -----------------------
lookback_months = 60     # 5 years rolling window
min_periods = 36         # need 3 years to compute z
z = pd.DataFrame(index=factors.index, columns=factors.columns, dtype=float)

for col in factors.columns:
    mu = factors[col].rolling(lookback_months, min_periods=min_periods).mean()
    sd = factors[col].rolling(lookback_months, min_periods=min_periods).std()
    z[col] = (factors[col] - mu) / sd
    z[col] = z[col] * DIRECTION[col]  # make "good" positive

# Clamp extremes
z = z.clip(-3, 3)

# Weighted macro score: require at least 8/12 factors present
w = pd.Series(WEIGHTS)
min_factors = 8
macro_score = z.mul(w, axis=1).sum(axis=1, min_count=min_factors).rename("Macro Score").dropna()

print("Macro score available from:", macro_score.index.min().date(), "to", macro_score.index.max().date())
print("Number of points:", len(macro_score))

# -----------------------
# 5) Download S&P 500 + Nasdaq proxy and compute monthly returns
# -----------------------
tickers = {"S&P 500 (SPY)": "SPY", "Nasdaq-100 (QQQ)": "QQQ"}

print("Downloading index prices from yfinance...")
px = yf.download(list(tickers.values()), start=START, auto_adjust=True, progress=False)["Close"]

# Make monthly end prices, aligned to macro_score dates
px_m = px.resample("M").last().reindex(macro_score.index)

# 1-month returns (%)
ret_1m = px_m.pct_change(1) * 100

# Cumulative return index (rebased to 100)
cum = (1 + px_m.pct_change().fillna(0)).cumprod() * 100

# Rename columns to friendly labels
inv_map = {v: k for k, v in tickers.items()}
px_m.columns = [inv_map.get(c, c) for c in px_m.columns]
ret_1m.columns = [inv_map.get(c, c) for c in ret_1m.columns]
cum.columns = [inv_map.get(c, c) for c in cum.columns]

# -----------------------
# 6) PLOT 1: Macro Score vs Cumulative Returns (rebased)
# -----------------------
fig1, ax1 = plt.subplots()

ax1.plot(macro_score.index, macro_score.values, label="Macro Score", color='green')
ax1.axhline(0, linewidth=1)
ax1.set_title("Macro Score vs S&P 500 & Nasdaq-100 (Rebased to 100)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Macro Score")

ax2 = ax1.twinx()
for col in cum.columns:
    ax2.plot(cum.index, cum[col], label=col)
ax2.set_ylabel("Cumulative Return Index (100 = start)")

# Combine legends (because twin axes)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

fig1.tight_layout()
fig1.savefig("macro_vs_spy_qqq.png", dpi=200)
print("Saved chart: macro_vs_spy_qqq.png")

# -----------------------
# 7) PLOT 2: Macro Score vs Forward 1-Month Returns (scatter)
# -----------------------
# forward return = next month's return
fwd_1m = ret_1m.shift(-1)

fig2, ax = plt.subplots()
for col in fwd_1m.columns:
    tmp = pd.concat([macro_score, fwd_1m[col].rename("Fwd 1M %")], axis=1).dropna()
    ax.scatter(tmp["Macro Score"], tmp["Fwd 1M %"], label=col, alpha=0.6)

ax.set_title("Macro Score vs Forward 1-Month Returns")
ax.set_xlabel("Macro Score")
ax.set_ylabel("Forward 1M Return (%)")
ax.legend()

fig2.tight_layout()
fig2.savefig("macro_scatter_forward_returns.png", dpi=200)
print("Saved chart: macro_scatter_forward_returns.png")

plt.show(block=True)


print("Done ✅")    