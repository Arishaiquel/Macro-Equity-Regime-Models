from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

NAME_MAP = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq Composite",
    "^DJI": "Dow Jones",
    "^FTSE": "FTSE 100",
    "^GDAXI": "DAX",
    "^FCHI": "CAC 40",
    "^N225": "Nikkei 225",
    "^HSI": "Hang Seng",
    "^STOXX50E": "Euro Stoxx 50",
    "^AXJO": "ASX 200",
}


def download_index_prices(tickers: list[str], start_date: str) -> pd.DataFrame:
    px = yf.download(tickers, start=start_date, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="all")


def _mtd(px: pd.Series) -> float | None:
    if px.empty:
        return None
    month_start = pd.Timestamp(px.index[-1].year, px.index[-1].month, 1)
    current_month = px.loc[px.index >= month_start]
    if current_month.empty:
        return None
    base = current_month.iloc[0]
    last = current_month.iloc[-1]
    if base <= 0:
        return None
    return float((last / base) - 1.0)


def _ytd(px: pd.Series) -> float | None:
    if px.empty:
        return None
    year_start = pd.Timestamp(px.index[-1].year, 1, 1)
    current_year = px.loc[px.index >= year_start]
    if current_year.empty:
        return None
    base = current_year.iloc[0]
    last = current_year.iloc[-1]
    if base <= 0:
        return None
    return float((last / base) - 1.0)


def build_performance_rows(px: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    for col in px.columns:
        s = px[col].dropna()
        if s.empty:
            continue
        out.append(
            {
                "ticker": col,
                "name": NAME_MAP.get(col, col),
                "last": float(s.iloc[-1]),
                "monthly_return": _mtd(s),
                "ytd_return": _ytd(s),
            }
        )
    out.sort(key=lambda x: x["ytd_return"] if x["ytd_return"] is not None else -np.inf, reverse=True)
    return out


def as_of_date(px: pd.DataFrame) -> str:
    if px.empty:
        return date.today().isoformat()
    return px.index[-1].date().isoformat()
