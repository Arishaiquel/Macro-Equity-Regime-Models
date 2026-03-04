"""
STOCK VALUATION MODEL (v1) — Industry-Relative Valuation + Quality + Plots
------------------------------------------------------------------------
What it does:
  - Pulls simple valuation + quality metrics for a peer set (via yfinance)
  - Normalizes each metric within the peer set (z-scores / percentiles)
  - Produces:
      valuation_score, valuation_pct  (cheap ↔ rich vs peers)
      quality_score,   quality_pct    (weak ↔ strong vs peers)
      combined_score,  combined_pct   (cheap + quality)
  - Plots:
      1) Valuation vs Quality scatter (best single view)
      2) Bar chart for valuation / quality / combined percentiles

Notes:
  - yfinance fundamentals can be missing for some tickers; the code handles NaNs.
  - For best results, include a reasonable peer set (8–30 names).

Install:
  pip install yfinance pandas numpy matplotlib

Run:
  python valuation_model_v1.py
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
    "UNH",  # managed care + services
    "ABT",  # managed care
    "CI",   # managed care + services
    "HUM",  # medicare advantage

    "JNJ",  # pharma + medtech
    "PFE",  # large-cap pharma
    "MRK",  # large-cap pharma
    "LLY",  # pharma (obesity/diabetes comp)
    "ABBV", # biopharma
    "NVO",  # pharma (diabetes/obesity)

    "TMO",  # life sciences tools
    "DHR",  # life sciences + diagnostics
    "ISRG", # medtech / surgical robotics
]


HIGHLIGHT = "ABT"

SLEEP = 0.15            # be polite to Yahoo
MIN_PEERS = 5           # minimum peer count needed to compute z-scores


# -----------------------------
# HELPERS
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
    std = peer.std(ddof=0)
    if std <= 1e-12:
        return 0.0
    return (x - peer.mean()) / std


def fetch_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).get_info() or {}

    return {
        "ticker": ticker,
        "name": info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),

        # Valuation multiples (lower often = cheaper)
        "trailingPE": safe_float(info.get("trailingPE")),
        "forwardPE": safe_float(info.get("forwardPE")),
        "evToEbitda": safe_float(info.get("enterpriseToEbitda")),
        "priceToBook": safe_float(info.get("priceToBook")),

        # Cash flow + market cap (for FCF yield)
        "freeCashflow": safe_float(info.get("freeCashflow")),
        "marketCap": safe_float(info.get("marketCap")),

        # Quality / fundamentals
        "profitMargin": safe_float(info.get("profitMargins")),
        "roe": safe_float(info.get("returnOnEquity")),
        "debtToEquity": safe_float(info.get("debtToEquity")),
        "revenueGrowth": safe_float(info.get("revenueGrowth")),
    }


def compute_fcf_yield(fcf: float, mcap: float) -> float:
    if np.isnan(fcf) or np.isnan(mcap) or mcap <= 0:
        return np.nan
    return fcf / mcap


# -----------------------------
# CORE: BUILD + SCORE
# -----------------------------

def build_valuation_dataframe(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        rows.append(fetch_fundamentals(t))
        time.sleep(SLEEP)

    df = pd.DataFrame(rows).set_index("ticker")
    df["fcfYield"] = [
        compute_fcf_yield(df.loc[idx, "freeCashflow"], df.loc[idx, "marketCap"])
        for idx in df.index
    ]

    # --- Metric directions: +1 means higher is better/cheaper, -1 means lower is better/cheaper
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

    # --- Valuation z-scores (aligned so higher = cheaper)
    for m, direction in valuation_metrics.items():
        df[f"z_{m}"] = np.nan
        for tk in df.index:
            df.loc[tk, f"z_{m}"] = direction * zscore(df.loc[tk, m], df[m])

    val_cols = [f"z_{m}" for m in valuation_metrics.keys()]
    df["valuation_score"] = df[val_cols].mean(axis=1, skipna=True)
    df["valuation_pct"] = df["valuation_score"].rank(pct=True)

    # --- Quality z-scores (aligned so higher = higher quality)
    for m, direction in quality_metrics.items():
        df[f"zq_{m}"] = np.nan
        for tk in df.index:
            df.loc[tk, f"zq_{m}"] = direction * zscore(df.loc[tk, m], df[m])

    q_cols = [f"zq_{m}" for m in quality_metrics.keys()]
    df["quality_score"] = df[q_cols].mean(axis=1, skipna=True)
    df["quality_pct"] = df["quality_score"].rank(pct=True)

    # --- Combined score (cheap + quality)
    df["combined_score"] = 0.7 * df["valuation_score"] + 0.3 * df["quality_score"]
    df["combined_pct"] = df["combined_score"].rank(pct=True)

    # --- Simple flags
    df["cheap_vs_peers"] = df["valuation_pct"] >= 0.80
    df["rich_vs_peers"] = df["valuation_pct"] <= 0.20
    df["cheap_and_quality"] = (df["valuation_pct"] >= 0.75) & (df["quality_pct"] >= 0.60)

    return df


# -----------------------------
# PLOTTING
# -----------------------------

def plot_valuation_quality(df, highlight="NFLX", show_names=True):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter all peers
    ax.scatter(
        df["valuation_pct"],
        df["quality_pct"],
        alpha=0.6,
        s=90
    )

    # Annotate each peer (small offset to reduce overlap)
    if show_names:
        for ticker in df.index:
            ax.annotate(
                ticker,
                (df.loc[ticker, "valuation_pct"], df.loc[ticker, "quality_pct"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
                alpha=0.8
            )

    # Highlight NFLX more strongly
    if highlight in df.index:
        ax.scatter(
            df.loc[highlight, "valuation_pct"],
            df.loc[highlight, "quality_pct"],
            color="red",
            s=220,
            edgecolor="black",
            zorder=5
        )

    # Reference lines
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



def plot_percentile_bars(df: pd.DataFrame, metric: str = "valuation_pct", highlight: str = "NFLX"):
    if metric not in df.columns:
        raise ValueError(f"Unknown metric '{metric}'. Choose from: valuation_pct, quality_pct, combined_pct")

    df_sorted = df.sort_values(metric)
    colors = ["red" if tk == highlight else "steelblue" for tk in df_sorted.index]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(df_sorted.index, df_sorted[metric], color=colors, alpha=0.85)

    ax.set_ylim(0, 1)
    ax.axhline(0.5, linestyle="--", alpha=0.35)

    ax.set_ylabel("Percentile vs peers")
    ax.set_title(f"{metric} (red = {highlight})")

    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN
# -----------------------------

def main():
    df = build_valuation_dataframe(TICKERS)

    # Print NFLX row
    cols = [
        "name", "sector", "industry",
        "trailingPE", "forwardPE", "evToEbitda", "priceToBook", "fcfYield",
        "profitMargin", "roe", "debtToEquity", "revenueGrowth",
        "valuation_pct", "quality_pct", "combined_pct",
        "cheap_vs_peers", "rich_vs_peers", "cheap_and_quality",
    ]
    print("\n=== NFLX SUMMARY ===\n")
    if HIGHLIGHT in df.index:
        print(df.loc[[HIGHLIGHT], cols].round(4).to_string())
    else:
        print(f"{HIGHLIGHT} not found in universe.")

    # Save full output
    df.to_csv("valuation_output.csv")
    print("\nSaved: valuation_output.csv")

    # Plots
    plot_valuation_quality(df, highlight=HIGHLIGHT)
    plot_percentile_bars(df, metric="valuation_pct", highlight=HIGHLIGHT)
    plot_percentile_bars(df, metric="quality_pct", highlight=HIGHLIGHT)
    plot_percentile_bars(df, metric="combined_pct", highlight=HIGHLIGHT)


if __name__ == "__main__":
    main()
