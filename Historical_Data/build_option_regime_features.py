#!/usr/bin/env python3
"""
Build daily option-regime features from MarketData option quotes CSV.

Expected input:
  CSV from Historical_Data/optionsTrdr.py

Example:
  python Historical_Data/build_option_regime_features.py \
    --quotes-csv data/options/SPX_quotes_2025-01-01_to_2025-01-31.csv \
    --underlying SPX \
    --out-csv data/options/SPX_option_features.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Config:
    quotes_csv: str
    out_csv: str
    underlying: str | None
    dte_min: int
    dte_max: int
    dte_target_short: int
    dte_target_long: int
    dte_band: int
    delta_target: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Aggregate options quotes into daily regime features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--quotes-csv", required=True, help="Input options quotes CSV")
    parser.add_argument("--out-csv", default="data/options/option_regime_features.csv", help="Output features CSV")
    parser.add_argument("--underlying", default=None, help="Optional underlying filter (e.g. SPX)")
    parser.add_argument("--dte-min", type=int, default=20, help="Min DTE included")
    parser.add_argument("--dte-max", type=int, default=120, help="Max DTE included")
    parser.add_argument("--dte-target-short", type=int, default=30, help="Target DTE for ATM IV short term")
    parser.add_argument("--dte-target-long", type=int, default=90, help="Target DTE for ATM IV long term")
    parser.add_argument("--dte-band", type=int, default=20, help="Allowed +/- band around DTE target")
    parser.add_argument("--delta-target", type=float, default=0.25, help="Target absolute delta for skew legs")
    args = parser.parse_args()

    return Config(
        quotes_csv=args.quotes_csv,
        out_csv=args.out_csv,
        underlying=args.underlying.upper() if args.underlying else None,
        dte_min=max(0, args.dte_min),
        dte_max=max(1, args.dte_max),
        dte_target_short=max(1, args.dte_target_short),
        dte_target_long=max(2, args.dte_target_long),
        dte_band=max(1, args.dte_band),
        delta_target=max(0.01, min(0.49, args.delta_target)),
    )


def _find_col(df: pd.DataFrame, candidates: set[str], required: bool = True) -> str | None:
    for c in df.columns:
        if c.lower() in candidates:
            return c
    if required:
        raise RuntimeError(f"Missing required column. Need one of {sorted(candidates)} in {list(df.columns)}")
    return None


def load_quotes(cfg: Config) -> pd.DataFrame:
    q = pd.read_csv(cfg.quotes_csv)
    if q.empty:
        raise RuntimeError("Quotes CSV is empty.")

    updated_col = _find_col(q, {"updated", "date", "timestamp"})
    exp_col = _find_col(q, {"expiration"})
    strike_col = _find_col(q, {"strike"})
    side_col = _find_col(q, {"side"})

    q["date"] = pd.to_datetime(q[updated_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    q["expiration_date"] = pd.to_datetime(q[exp_col], errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    q["strike"] = pd.to_numeric(q[strike_col], errors="coerce")
    q["side"] = q[side_col].astype(str).str.lower()

    dte_col = _find_col(q, {"dte"}, required=False)
    if dte_col is not None:
        dte_series = pd.to_numeric(q[dte_col], errors="coerce")
    else:
        dte_series = pd.Series(np.nan, index=q.index, dtype=float)

    dte_calc = (q["expiration_date"] - q["date"]).dt.days.astype(float)
    q["dte"] = dte_series.where(dte_series.notna(), dte_calc)

    iv_col = _find_col(q, {"iv"}, required=False)
    delta_col = _find_col(q, {"delta"}, required=False)
    oi_col = _find_col(q, {"openinterest", "open_interest"}, required=False)
    vol_col = _find_col(q, {"volume"}, required=False)
    und_col = _find_col(q, {"underlyingprice", "underlying_price"}, required=False)

    q["iv"] = pd.to_numeric(q[iv_col], errors="coerce") if iv_col else np.nan
    q["delta"] = pd.to_numeric(q[delta_col], errors="coerce") if delta_col else np.nan
    q["openInterest"] = pd.to_numeric(q[oi_col], errors="coerce") if oi_col else 0.0
    q["volume"] = pd.to_numeric(q[vol_col], errors="coerce") if vol_col else 0.0
    q["underlyingPrice"] = pd.to_numeric(q[und_col], errors="coerce") if und_col else np.nan

    if cfg.underlying:
        und_name_col = _find_col(q, {"underlying"}, required=False)
        if und_name_col:
            q = q[q[und_name_col].astype(str).str.upper() == cfg.underlying]

    q = q.dropna(subset=["date", "expiration_date", "strike"])
    q = q[q["side"].isin(["call", "put"])]
    q = q[(q["dte"] >= cfg.dte_min) & (q["dte"] <= cfg.dte_max)]
    if q.empty:
        raise RuntimeError("No rows remain after cleaning and DTE filters.")
    return q.reset_index(drop=True)


def _pick_nearest_atm_iv(day: pd.DataFrame, target_dte: int, dte_band: int, spot: float) -> float:
    sub = day[day["iv"].notna()].copy()
    if sub.empty:
        return np.nan

    sub["dte_dist"] = (sub["dte"] - target_dte).abs()
    sub = sub[sub["dte_dist"] <= dte_band] if (sub["dte_dist"] <= dte_band).any() else sub
    sub["atm_dist"] = (sub["strike"] - spot).abs()
    chosen = sub.nsmallest(8, ["dte_dist", "atm_dist"])
    return float(chosen["iv"].mean()) if not chosen.empty else np.nan


def _pick_25d_iv(day: pd.DataFrame, side: str, target_dte: int, dte_band: int, delta_target: float, spot: float) -> float:
    sub = day[(day["side"] == side) & day["iv"].notna()].copy()
    if sub.empty:
        return np.nan

    sub["dte_dist"] = (sub["dte"] - target_dte).abs()
    sub = sub[sub["dte_dist"] <= dte_band] if (sub["dte_dist"] <= dte_band).any() else sub

    # Prefer delta-based selection. Fallback to strike-moneyness proxy.
    d = pd.to_numeric(sub["delta"], errors="coerce")
    if d.notna().any():
        target = delta_target if side == "call" else -delta_target
        sub["delta_dist"] = (d - target).abs()
        chosen = sub.nsmallest(4, ["dte_dist", "delta_dist"])
        if not chosen.empty:
            return float(chosen["iv"].mean())

    target_k = 1.05 * spot if side == "call" else 0.95 * spot
    sub["k_dist"] = (sub["strike"] - target_k).abs()
    chosen = sub.nsmallest(4, ["dte_dist", "k_dist"])
    return float(chosen["iv"].mean()) if not chosen.empty else np.nan


def build_daily_features(quotes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows: list[dict] = []

    for dt, day in quotes.groupby("date", sort=True):
        spot = float(day["underlyingPrice"].median()) if day["underlyingPrice"].notna().any() else float(day["strike"].median())

        iv_atm_30 = _pick_nearest_atm_iv(day, cfg.dte_target_short, cfg.dte_band, spot)
        iv_atm_90 = _pick_nearest_atm_iv(day, cfg.dte_target_long, cfg.dte_band, spot)

        iv_put25 = _pick_25d_iv(day, "put", cfg.dte_target_short, cfg.dte_band, cfg.delta_target, spot)
        iv_call25 = _pick_25d_iv(day, "call", cfg.dte_target_short, cfg.dte_band, cfg.delta_target, spot)

        put_oi = float(day.loc[day["side"] == "put", "openInterest"].sum())
        call_oi = float(day.loc[day["side"] == "call", "openInterest"].sum())
        pcr_oi = (put_oi / call_oi) if call_oi > 0 else np.nan

        put_vol = float(day.loc[day["side"] == "put", "volume"].sum())
        call_vol = float(day.loc[day["side"] == "call", "volume"].sum())
        pcr_volume = (put_vol / call_vol) if call_vol > 0 else np.nan

        skew_25d = iv_put25 - iv_call25 if np.isfinite(iv_put25) and np.isfinite(iv_call25) else np.nan
        smile_curvature = (
            0.5 * (iv_put25 + iv_call25) - iv_atm_30
            if np.isfinite(iv_put25) and np.isfinite(iv_call25) and np.isfinite(iv_atm_30)
            else np.nan
        )
        iv_term_30_90 = (iv_atm_30 / iv_atm_90) if np.isfinite(iv_atm_30) and np.isfinite(iv_atm_90) and iv_atm_90 > 0 else np.nan

        rows.append(
            {
                "date": dt,
                "underlying_price": spot,
                "iv_atm_30": iv_atm_30,
                "iv_atm_90": iv_atm_90,
                "iv_put_25d": iv_put25,
                "iv_call_25d": iv_call25,
                "skew_25d": skew_25d,
                "smile_curvature": smile_curvature,
                "iv_term_30_90": iv_term_30_90,
                "pcr_oi": pcr_oi,
                "pcr_volume": pcr_volume,
                "n_contracts": int(len(day)),
                "n_expirations": int(day["expiration_date"].nunique()),
            }
        )

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return out


def main() -> None:
    cfg = parse_args()
    q = load_quotes(cfg)
    feat = build_daily_features(q, cfg)

    os.makedirs(os.path.dirname(cfg.out_csv) or ".", exist_ok=True)
    feat.to_csv(cfg.out_csv, index=False)

    print(f"Saved features: {cfg.out_csv}")
    print(f"Rows: {len(feat)} | Date range: {feat['date'].min()} -> {feat['date'].max()}")
    cols = ["iv_atm_30", "skew_25d", "smile_curvature", "iv_term_30_90", "pcr_oi"]
    present = [c for c in cols if c in feat.columns]
    print("Feature columns:", present)


if __name__ == "__main__":
    main()
