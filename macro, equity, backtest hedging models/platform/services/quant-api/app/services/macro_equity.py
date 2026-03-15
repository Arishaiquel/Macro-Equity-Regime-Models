from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def fred_csv(series_id: str, start: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["DATE", series_id]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE")
    s = df[series_id].replace(".", np.nan).astype(float)
    return s.loc[start:]


def run_macro_equity(
    start_date: str = "1995-01-01",
    lookback_months: int = 60,
    min_periods: int = 36,
    min_factors: int = 8,
) -> dict:
    weights = {
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

    direction = {
        "Policy Rate": -1,
        "10Y Yield": -1,
        "CPI YoY": -1,
        "Core CPI YoY": -1,
        "GDP QoQ": 1,
        "GDP YoY": 1,
        "Jobless Rate": -1,
        "Retail Sales": 1,
        "Industrial Production": 1,
        "Current Account / GDP": 1,
        "Govt Debt / GDP": -1,
        "Population": 1,
    }

    fred_series = {
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

    raw = {name: fred_csv(sid, start_date) for name, sid in fred_series.items()}
    df = pd.concat(raw, axis=1)
    m = df.resample("ME").last()

    m["CPI YoY"] = m["CPI"].pct_change(12) * 100
    m["Core CPI YoY"] = m["Core CPI"].pct_change(12) * 100

    gdp_q = df["GDP"].dropna()
    gdp_qoq = gdp_q.pct_change(1) * 100
    gdp_yoy = gdp_q.pct_change(4) * 100
    g = pd.concat([gdp_qoq.rename("GDP QoQ"), gdp_yoy.rename("GDP YoY")], axis=1).resample("ME").ffill()
    m = m.join(g)

    ca = df["Current Account"].dropna()
    ca_gdp = (ca / gdp_q) * 100
    m["Current Account / GDP"] = ca_gdp.resample("ME").ffill()

    factors = m[list(weights.keys())].copy()

    z = pd.DataFrame(index=factors.index, columns=factors.columns, dtype=float)
    for col in factors.columns:
        mu = factors[col].rolling(lookback_months, min_periods=min_periods).mean()
        sd = factors[col].rolling(lookback_months, min_periods=min_periods).std()
        z[col] = (factors[col] - mu) / sd
        z[col] = z[col] * direction[col]

    z = z.clip(-3, 3)
    w = pd.Series(weights)
    macro_score = z.mul(w, axis=1).sum(axis=1, min_count=min_factors).rename("Macro Score").dropna()
    if macro_score.empty:
        raise ValueError("Not enough data to compute macro score")

    px = yf.download(["SPY", "QQQ"], start=start_date, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        raise ValueError("Expected SPY and QQQ data")

    px_m = px.resample("ME").last().reindex(macro_score.index)
    ret_1m = px_m.pct_change(1)

    out_tail = []
    for dt in macro_score.index[-200:]:
        out_tail.append(
            {
                "date": dt.date().isoformat(),
                "macro_score": float(macro_score.loc[dt]),
                "spy_1m": None if pd.isna(ret_1m.loc[dt, "SPY"]) else float(ret_1m.loc[dt, "SPY"]),
                "qqq_1m": None if pd.isna(ret_1m.loc[dt, "QQQ"]) else float(ret_1m.loc[dt, "QQQ"]),
            }
        )

    latest = macro_score.index[-1]
    return {
        "latest": {
            "date": latest.date().isoformat(),
            "macro_score": float(macro_score.loc[latest]),
            "spy_1m": None if pd.isna(ret_1m.loc[latest, "SPY"]) else float(ret_1m.loc[latest, "SPY"]),
            "qqq_1m": None if pd.isna(ret_1m.loc[latest, "QQQ"]) else float(ret_1m.loc[latest, "QQQ"]),
        },
        "series_tail": out_tail,
    }
