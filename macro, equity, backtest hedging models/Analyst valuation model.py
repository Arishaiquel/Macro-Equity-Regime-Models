"""
NFLX PEER VALUATION + ANALYST REVISIONS (NO BLOOMBERG) — One-file script
-----------------------------------------------------------------------
What this script does:
  1) Pulls simple fundamentals via yfinance for NFLX + peers
  2) Builds a peer-relative VALUATION model:
       - multiples (P/E, EV/EBITDA, P/B) + FCF yield
       - outputs valuation_pct, quality_pct, combined_pct
  3) Builds an ANALYST REVISIONS PROXY model (no Bloomberg):
       - uses yfinance upgrades/downgrades actions over a lookback window
       - outputs revision_pct vs peers
  4) Plots:
       - Valuation vs Quality scatter (labels)
       - Analyst revisions percentile bar chart (NFLX highlighted)
       - Valuation vs Revisions scatter (labels)
  5) Saves a CSV: nflx_peer_valuation_revisions.csv

Install:
  pip install yfinance pandas numpy matplotlib

Run:
  python nflx_peer_valuation_revisions.py
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# -----------------------------
# CONFIG
# -----------------------------
TICKERS = [
    "AMZN",  # Amazon
    "MSFT",  # cloud + platform
    "GOOGL", # ads + cloud-ish
    "META",  # ads/platform (market comp)
    "AAPL",  # platform comp
    "ORCL",  # cloud/software comp
    "WMT",   # retail comp
    "COST",  # retail comp
    "TGT",   # retail comp
    "SHOP",  # e-commerce enablement
    "NFLX",  # e-commerce marketplace
]
HIGHLIGHT = "NFLX"

SLEEP = 0.15          # be polite to Yahoo
MIN_PEERS = 5         # min peer count for z-score stability
LOOKBACK_DAYS = 180   # analyst actions lookback (90/180/365 are common)


# -----------------------------
# UTILITIES
# -----------------------------
def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def zscore(x: float, peer: pd.Series) -> float:
    peer = peer.dropna()
    if len(peer) < MIN_PEERS:
        return np.nan
    sd = peer.std(ddof=0)
    if sd <= 1e-12:
        return 0.0
    return (x - peer.mean()) / sd


def compute_fcf_yield(fcf: float, mcap: float) -> float:
    if np.isnan(fcf) or np.isnan(mcap) or mcap <= 0:
        return np.nan
    return fcf / mcap


# -----------------------------
# VALUATION MODEL
# -----------------------------
def fetch_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).get_info() or {}
    return {
        "ticker": ticker,
        "name": info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),

        # valuation
        "trailingPE": safe_float(info.get("trailingPE")),
        "forwardPE": safe_float(info.get("forwardPE")),
        "evToEbitda": safe_float(info.get("enterpriseToEbitda")),
        "priceToBook": safe_float(info.get("priceToBook")),
        "freeCashflow": safe_float(info.get("freeCashflow")),
        "marketCap": safe_float(info.get("marketCap")),

        # quality
        "profitMargin": safe_float(info.get("profitMargins")),
        "roe": safe_float(info.get("returnOnEquity")),
        "debtToEquity": safe_float(info.get("debtToEquity")),
        "revenueGrowth": safe_float(info.get("revenueGrowth")),
    }


def build_valuation_df(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        rows.append(fetch_fundamentals(tk))
        time.sleep(SLEEP)

    df = pd.DataFrame(rows).set_index("ticker")
    df["fcfYield"] = [
        compute_fcf_yield(df.loc[idx, "freeCashflow"], df.loc[idx, "marketCap"])
        for idx in df.index
    ]

    # direction: +1 means higher is better/cheaper; -1 means lower is better/cheaper
    valuation_metrics = {
        "trailingPE": -1,
        "forwardPE": -1,
        "evToEbitda": -1,
        "priceToBook": -1,
        "fcfYield": +1,
    }

    quality_metrics = {
        "profitMargin": +1,
        "roe": +1,
        "revenueGrowth": +1,
        "debtToEquity": -1,
    }

    # valuation z-scores aligned so higher = cheaper
    for m, direction in valuation_metrics.items():
        df[f"z_{m}"] = np.nan
        for tk in df.index:
            df.loc[tk, f"z_{m}"] = direction * zscore(df.loc[tk, m], df[m])
    df["valuation_score"] = df[[f"z_{m}" for m in valuation_metrics]].mean(axis=1, skipna=True)
    df["valuation_pct"] = df["valuation_score"].rank(pct=True)

    # quality z-scores aligned so higher = higher quality
    for m, direction in quality_metrics.items():
        df[f"zq_{m}"] = np.nan
        for tk in df.index:
            df.loc[tk, f"zq_{m}"] = direction * zscore(df.loc[tk, m], df[m])
    df["quality_score"] = df[[f"zq_{m}" for m in quality_metrics]].mean(axis=1, skipna=True)
    df["quality_pct"] = df["quality_score"].rank(pct=True)

    # combined
    df["combined_score"] = 0.7 * df["valuation_score"] + 0.3 * df["quality_score"]
    df["combined_pct"] = df["combined_score"].rank(pct=True)

    df["cheap_vs_peers"] = df["valuation_pct"] >= 0.80
    df["rich_vs_peers"] = df["valuation_pct"] <= 0.20
    df["cheap_and_quality"] = (df["valuation_pct"] >= 0.75) & (df["quality_pct"] >= 0.60)

    return df


# -----------------------------
# ANALYST REVISIONS PROXY MODEL (Upgrades/Downgrades)
# -----------------------------
def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        # best-effort conversion
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    return out


def fetch_upgrades_downgrades(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    ud = getattr(t, "upgrades_downgrades", None)
    if ud is None:
        return pd.DataFrame()
    return _to_datetime_index(ud)


def action_score(from_grade: str, to_grade: str) -> float:
    def norm(x):
        if x is None:
            return ""
        return str(x).lower()

    f = norm(from_grade)
    t = norm(to_grade)

    bullish = ["buy", "outperform", "overweight", "strong buy", "positive"]
    neutral = ["hold", "neutral", "market perform", "equal-weight", "in-line"]
    bearish = ["sell", "underperform", "underweight", "negative"]

    def bucket(s: str) -> int:
        if any(w in s for w in bullish):
            return 1
        if any(w in s for w in bearish):
            return -1
        if any(w in s for w in neutral):
            return 0
        return 0

    return float(bucket(t) - bucket(f))  # + upgrade, - downgrade


def build_revision_proxy(tickers: list[str], lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    rows = []
    for tk in tickers:
        ud = fetch_upgrades_downgrades(tk)

        if ud.empty:
            rows.append({"ticker": tk, "net_score": np.nan, "n_actions": 0, "revision_score": np.nan})
            continue

        end = ud.index.max()
        start = end - pd.Timedelta(days=lookback_days)
        recent = ud.loc[(ud.index >= start) & (ud.index <= end)].copy()

        # find FromGrade / ToGrade columns robustly
        from_col = None
        to_col = None
        for c in recent.columns:
            lc = str(c).lower()
            if "from" in lc and "grade" in lc:
                from_col = c
            if "to" in lc and "grade" in lc:
                to_col = c

        if from_col is None or to_col is None or len(recent) == 0:
            rows.append({"ticker": tk, "net_score": np.nan, "n_actions": int(len(recent)), "revision_score": np.nan})
            continue

        scores = []
        for fr, to in zip(recent[from_col].values, recent[to_col].values):
            scores.append(action_score(fr, to))

        scores = np.array(scores, dtype=float)
        net = float(np.nansum(scores))
        n = int(np.sum(~np.isnan(scores)))

        # Stabilize: penalize tiny samples
        rev = net / np.sqrt(max(n, 1))
        rows.append({"ticker": tk, "net_score": net, "n_actions": n, "revision_score": rev})

    df = pd.DataFrame(rows).set_index("ticker")
    df["revision_pct"] = df["revision_score"].rank(pct=True)
    df["revisions_up"] = df["revision_pct"] >= 0.70
    df["revisions_down"] = df["revision_pct"] <= 0.30
    return df


# -----------------------------
# PLOTTING
# -----------------------------
def plot_valuation_quality(df: pd.DataFrame, highlight: str = HIGHLIGHT):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(df["valuation_pct"], df["quality_pct"], alpha=0.65, s=100)

    # labels
    for tk in df.index:
        ax.annotate(
            tk,
            (df.loc[tk, "valuation_pct"], df.loc[tk, "quality_pct"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            alpha=0.85
        )

    # highlight
    if highlight in df.index:
        ax.scatter(
            df.loc[highlight, "valuation_pct"],
            df.loc[highlight, "quality_pct"],
            color="red",
            s=260,
            edgecolor="black",
            zorder=5
        )

    ax.axvline(0.5, linestyle="--", alpha=0.35)
    ax.axhline(0.5, linestyle="--", alpha=0.35)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Valuation percentile (cheap → expensive)")
    ax.set_ylabel("Quality percentile (low → high)")
    ax.set_title("Valuation vs Quality (peer-relative)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def plot_revision_bars(rev, highlight="NFLX", lookback_days=180):
    """
    Bar chart of revision_pct vs peers, with n_actions printed on each bar.
    Requires columns: revision_pct, n_actions.
    """
    d = rev.sort_values("revision_pct").copy()

    colors = ["red" if tk == highlight else "steelblue" for tk in d.index]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(d.index, d["revision_pct"], color=colors, alpha=0.85)

    ax.axhline(0.5, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Analyst revisions percentile (vs peers)")
    ax.set_title(f"Analyst revisions proxy (lookback={lookback_days} days)")

    # --- NEW: show number of actions used (n_actions) on each bar ---
    for bar, tk in zip(bars, d.index):
        n = int(d.loc[tk, "n_actions"]) if pd.notna(d.loc[tk, "n_actions"]) else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()



def plot_valuation_vs_revisions(val: pd.DataFrame, rev: pd.DataFrame, highlight: str = HIGHLIGHT):
    combo = val.join(rev[["revision_pct", "net_score", "n_actions"]], how="left")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(combo["valuation_pct"], combo["revision_pct"], alpha=0.65, s=100)

    for tk in combo.index:
        ax.annotate(
            tk,
            (combo.loc[tk, "valuation_pct"], combo.loc[tk, "revision_pct"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            alpha=0.85
        )

    if highlight in combo.index:
        ax.scatter(
            combo.loc[highlight, "valuation_pct"],
            combo.loc[highlight, "revision_pct"],
            color="red",
            s=260,
            edgecolor="black",
            zorder=5
        )

    ax.axvline(0.5, linestyle="--", alpha=0.35)
    ax.axhline(0.5, linestyle="--", alpha=0.35)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Valuation percentile (cheap → expensive)")
    ax.set_ylabel("Analyst revisions percentile (bearish → bullish)")
    ax.set_title("Valuation vs Analyst Revisions (peer-relative)")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    return combo


# -----------------------------
# MAIN
# -----------------------------
def main():
    # build models
    val = build_valuation_df(TICKERS)
    rev = build_revision_proxy(TICKERS, lookback_days=LOOKBACK_DAYS)

    # combine
    combo = val.join(rev[["revision_pct", "net_score", "n_actions", "revisions_up", "revisions_down"]], how="left")

    # simple combined flags
    combo["cheap_and_up_revisions"] = (combo["valuation_pct"] >= 0.75) & (combo["revision_pct"] >= 0.70)
    combo["value_trap_risk"] = (combo["valuation_pct"] >= 0.75) & (combo["revision_pct"] <= 0.30)

    # print NFLX summary
    cols = [
        "name", "sector", "industry",
        "trailingPE", "forwardPE", "evToEbitda", "priceToBook", "fcfYield",
        "profitMargin", "roe", "debtToEquity", "revenueGrowth",
        "valuation_pct", "quality_pct", "combined_pct",
        "revision_pct", "net_score", "n_actions",
        "cheap_and_quality", "revisions_up", "cheap_and_up_revisions",
    ]
    print("\n=== NFLX SUMMARY (peer-relative) ===\n")
    if HIGHLIGHT in combo.index:
        print(combo.loc[[HIGHLIGHT], cols].round(4).to_string())
    else:
        print(f"{HIGHLIGHT} not found in universe.")

    # save
    combo.to_csv("nflx_peer_valuation_revisions.csv")
    print("\nSaved: nflx_peer_valuation_revisions.csv")

    # plots
    plot_valuation_quality(val, highlight=HIGHLIGHT)
    plot_revision_bars(rev, highlight=HIGHLIGHT)
    plot_valuation_vs_revisions(val, rev, highlight=HIGHLIGHT)


if __name__ == "__main__":
    main()
