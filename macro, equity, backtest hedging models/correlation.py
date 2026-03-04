import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG (edit these)
# -----------------------------
A = "SPY"
B = "XLK"
START = "2020-01-01"

WIN_3M = 63    # ~3 months trading days
WIN_6M = 126   # ~6 months trading days


# -----------------------------
# Helpers
# -----------------------------
def download_prices(tickers, start):
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna()

def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff().dropna()

def max_drawdown(px: pd.Series) -> float:
    peak = px.cummax()
    dd = 1 - px / peak
    return float(dd.max())

def rolling_max_drawdown(px: pd.Series, window: int) -> pd.Series:
    # max drawdown computed over a rolling window using prices inside that window
    out = []
    idx = px.index
    for i in range(len(px)):
        if i < window:
            out.append(np.nan)
            continue
        w = px.iloc[i-window+1:i+1]
        out.append(max_drawdown(w))
    return pd.Series(out, index=idx, name=f"roll_mdd_{window}")

def rolling_beta(ra: pd.Series, rb: pd.Series, window: int) -> pd.Series:
    cov = ra.rolling(window).cov(rb)
    var = rb.rolling(window).var()
    return (cov / var).rename(f"beta_{window}")

def tracking_error(ra: pd.Series, rb: pd.Series, window: int) -> pd.Series:
    # annualized tracking error of (A - B)
    diff = (ra - rb)
    te = diff.rolling(window).std() * np.sqrt(252)
    return te.rename(f"te_{window}")

def ann_vol(r: pd.Series, window: int) -> pd.Series:
    return (r.rolling(window).std() * np.sqrt(252)).rename(f"vol_{window}")

def summary_stats(px: pd.DataFrame, r: pd.DataFrame):
    stats = {}
    stats["start"] = str(px.index.min().date())
    stats["end"] = str(px.index.max().date())
    stats["n_days"] = int(len(px))
    stats["corr_daily_returns_full"] = float(r[A].corr(r[B]))
    stats["beta_full_A_on_B"] = float(r[A].cov(r[B]) / r[B].var())

    # full-sample max drawdowns
    stats["max_dd_A_full"] = max_drawdown(px[A])
    stats["max_dd_B_full"] = max_drawdown(px[B])

    return stats


# -----------------------------
# Main
# -----------------------------
px = download_prices([A, B], START)
r = log_returns(px)

# Rolling correlation (3m, 6m)
roll_corr_3m = r[A].rolling(WIN_3M).corr(r[B]).rename("corr_3m")
roll_corr_6m = r[A].rolling(WIN_6M).corr(r[B]).rename("corr_6m")

# Rolling beta (3m, 6m): A vs B
beta_3m = rolling_beta(r[A], r[B], WIN_3M)
beta_6m = rolling_beta(r[A], r[B], WIN_6M)

# Rolling vol + tracking error (3m, 6m)
volA_3m = ann_vol(r[A], WIN_3M)
volB_3m = ann_vol(r[B], WIN_3M)
te_3m = tracking_error(r[A], r[B], WIN_3M)

volA_6m = ann_vol(r[A], WIN_6M)
volB_6m = ann_vol(r[B], WIN_6M)
te_6m = tracking_error(r[A], r[B], WIN_6M)

# Rolling max drawdown (3m, 6m) using prices
mddA_3m = rolling_max_drawdown(px[A], WIN_3M)
mddB_3m = rolling_max_drawdown(px[B], WIN_3M)
mddA_6m = rolling_max_drawdown(px[A], WIN_6M)
mddB_6m = rolling_max_drawdown(px[B], WIN_6M)

# Print summary
stats = summary_stats(px, r)
print("\n=== Summary ===")
for k, v in stats.items():
    print(f"{k}: {v}")

print("\n=== Latest rolling metrics ===")
latest = pd.DataFrame({
    "corr_3m": roll_corr_3m,
    "corr_6m": roll_corr_6m,
    "beta_3m": beta_3m,
    "beta_6m": beta_6m,
    "volA_3m": volA_3m,
    "volB_3m": volB_3m,
    "te_3m": te_3m,
    "volA_6m": volA_6m,
    "volB_6m": volB_6m,
    "te_6m": te_6m,
    "mddA_3m": mddA_3m,
    "mddB_3m": mddB_3m,
    "mddA_6m": mddA_6m,
    "mddB_6m": mddB_6m,
}).dropna()

print(latest.tail(1).round(4).to_string())

# -----------------------------
# Plots
# -----------------------------
# 1) Rebased prices
rebased = px / px.iloc[0] * 100
fig, ax = plt.subplots(figsize=(12, 5))
rebased.plot(ax=ax, lw=2)
ax.set_title(f"Rebased prices: {A} vs {B} (100 = start)")
ax.set_ylabel("Index level")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 2) Rolling correlation
fig, ax = plt.subplots(figsize=(12, 4))
roll_corr_3m.plot(ax=ax, lw=2, label="3M corr")
roll_corr_6m.plot(ax=ax, lw=2, label="6M corr")
ax.axhline(0, linestyle="--", alpha=0.4)
ax.set_title(f"Rolling correlation of daily returns: {A} vs {B}")
ax.set_ylabel("Correlation")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 3) Rolling beta
fig, ax = plt.subplots(figsize=(12, 4))
beta_3m.plot(ax=ax, lw=2, label="3M beta (A on B)")
beta_6m.plot(ax=ax, lw=2, label="6M beta (A on B)")
ax.axhline(1, linestyle="--", alpha=0.35)
ax.set_title(f"Rolling beta: {A} on {B}")
ax.set_ylabel("Beta")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 4) Tracking error
fig, ax = plt.subplots(figsize=(12, 4))
te_3m.plot(ax=ax, lw=2, label="3M tracking error (ann.)")
te_6m.plot(ax=ax, lw=2, label="6M tracking error (ann.)")
ax.set_title(f"Tracking error (A-B), annualized: {A} vs {B}")
ax.set_ylabel("Annualized TE")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 5) Rolling max drawdown
fig, ax = plt.subplots(figsize=(12, 4))
mddA_3m.plot(ax=ax, lw=2, label=f"{A} 3M max DD")
mddB_3m.plot(ax=ax, lw=2, label=f"{B} 3M max DD")
mddA_6m.plot(ax=ax, lw=2, label=f"{A} 6M max DD")
mddB_6m.plot(ax=ax, lw=2, label=f"{B} 6M max DD")
ax.set_title("Rolling max drawdown (window)")
ax.set_ylabel("Max drawdown")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
