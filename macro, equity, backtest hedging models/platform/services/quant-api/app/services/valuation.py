from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def zscore(x: float, peer: pd.Series, min_peers: int) -> float:
    peer = peer.dropna()
    if len(peer) < min_peers:
        return np.nan
    sd = peer.std(ddof=0)
    if sd <= 1e-12:
        return 0.0
    return (x - peer.mean()) / sd


def compute_fcf_yield(fcf: float, mcap: float) -> float:
    if np.isnan(fcf) or np.isnan(mcap) or mcap <= 0:
        return np.nan
    return fcf / mcap


def fetch_fundamentals(ticker: str) -> dict:
    info = yf.Ticker(ticker).get_info() or {}
    return {
        "ticker": ticker,
        "name": info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "trailingPE": safe_float(info.get("trailingPE")),
        "forwardPE": safe_float(info.get("forwardPE")),
        "evToEbitda": safe_float(info.get("enterpriseToEbitda")),
        "priceToBook": safe_float(info.get("priceToBook")),
        "freeCashflow": safe_float(info.get("freeCashflow")),
        "marketCap": safe_float(info.get("marketCap")),
        "profitMargin": safe_float(info.get("profitMargins")),
        "roe": safe_float(info.get("returnOnEquity")),
        "debtToEquity": safe_float(info.get("debtToEquity")),
        "revenueGrowth": safe_float(info.get("revenueGrowth")),
    }


def build_valuation_scores(
    tickers: list[str],
    min_peers: int = 5,
    sleep: float = 0.05,
    highlight: str | None = None,
) -> list[dict]:
    rows = []
    for tk in tickers:
        rows.append(fetch_fundamentals(tk))
        time.sleep(max(0.0, sleep))

    df = pd.DataFrame(rows).set_index("ticker")
    if df.empty:
        return []

    df["fcfYield"] = [
        compute_fcf_yield(df.loc[idx, "freeCashflow"], df.loc[idx, "marketCap"])
        for idx in df.index
    ]

    valuation_metrics = {
        "trailingPE": -1,
        "forwardPE": -1,
        "evToEbitda": -1,
        "priceToBook": -1,
        "fcfYield": 1,
    }
    quality_metrics = {
        "profitMargin": 1,
        "roe": 1,
        "revenueGrowth": 1,
        "debtToEquity": -1,
    }

    for m, direction in valuation_metrics.items():
        df[f"z_{m}"] = np.nan
        for tk in df.index:
            df.loc[tk, f"z_{m}"] = direction * zscore(df.loc[tk, m], df[m], min_peers)
    df["valuation_score"] = df[[f"z_{m}" for m in valuation_metrics]].mean(axis=1, skipna=True)
    df["valuation_pct"] = df["valuation_score"].rank(pct=True)

    for m, direction in quality_metrics.items():
        df[f"zq_{m}"] = np.nan
        for tk in df.index:
            df.loc[tk, f"zq_{m}"] = direction * zscore(df.loc[tk, m], df[m], min_peers)
    df["quality_score"] = df[[f"zq_{m}" for m in quality_metrics]].mean(axis=1, skipna=True)
    df["quality_pct"] = df["quality_score"].rank(pct=True)

    df["combined_score"] = 0.7 * df["valuation_score"] + 0.3 * df["quality_score"]
    df["combined_pct"] = df["combined_score"].rank(pct=True)

    df["cheap_vs_peers"] = df["valuation_pct"] >= 0.80
    df["rich_vs_peers"] = df["valuation_pct"] <= 0.20
    df["cheap_and_quality"] = (df["valuation_pct"] >= 0.75) & (df["quality_pct"] >= 0.60)

    out = []
    for tk, row in df.sort_values("combined_pct", ascending=False).iterrows():
        out.append(
            {
                "ticker": tk,
                "name": str(row.get("name") or tk),
                "sector": None if pd.isna(row.get("sector")) else str(row.get("sector")),
                "industry": None if pd.isna(row.get("industry")) else str(row.get("industry")),
                "valuation_pct": None if pd.isna(row.get("valuation_pct")) else float(row.get("valuation_pct")),
                "quality_pct": None if pd.isna(row.get("quality_pct")) else float(row.get("quality_pct")),
                "combined_pct": None if pd.isna(row.get("combined_pct")) else float(row.get("combined_pct")),
                "cheap_vs_peers": bool(row.get("cheap_vs_peers")),
                "rich_vs_peers": bool(row.get("rich_vs_peers")),
                "cheap_and_quality": bool(row.get("cheap_and_quality")),
                "is_highlight": bool(highlight and tk.upper() == highlight.upper()),
            }
        )
    return out


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    return out


def fetch_upgrades_downgrades(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    ud = getattr(t, "upgrades_downgrades", None)
    if ud is None:
        return pd.DataFrame()
    return _to_datetime_index(ud)


def action_score(from_grade: str | None, to_grade: str | None) -> float:
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
        if any(k in s for k in bullish):
            return 1
        if any(k in s for k in bearish):
            return -1
        if any(k in s for k in neutral):
            return 0
        return 0

    return float(bucket(t) - bucket(f))


def build_analyst_revision_scores(tickers: list[str], lookback_days: int = 180) -> dict[str, float | None]:
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=lookback_days)
    raw_scores: dict[str, float | None] = {}

    for tk in tickers:
        ud = fetch_upgrades_downgrades(tk)
        if ud.empty:
            raw_scores[tk] = None
            continue
        ud = ud.loc[(ud.index >= start) & (ud.index <= end)]
        if ud.empty:
            raw_scores[tk] = 0.0
            continue

        score = 0.0
        for _, row in ud.iterrows():
            score += action_score(row.get("FromGrade"), row.get("ToGrade"))
        raw_scores[tk] = score

    s = pd.Series(raw_scores, dtype=float)
    pct = s.rank(pct=True)
    return {k: (None if pd.isna(v) else float(v)) for k, v in pct.to_dict().items()}
