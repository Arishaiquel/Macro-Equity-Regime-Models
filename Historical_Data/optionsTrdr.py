#!/usr/bin/env python3
"""
Download historical options prices + greeks from MarketData.app.

Primary flow (recommended for SPX):
  expirations -> strikes -> synthesize option symbols -> quotes

Examples:
  python Historical_Data/optionsTrdr.py --underlying SPX --from-date 2025-01-01 --to-date 2025-01-31
  python Historical_Data/optionsTrdr.py --underlying SPX --mode exp_strikes --side put --max-expirations 2 --max-strikes 25
  python Historical_Data/optionsTrdr.py --underlying SPX --exp-from 2025-03-01 --exp-to 2025-04-30 --spot 5200

Auth:
  export MARKETDATA_TOKEN='your_token'
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import requests


BASE_URL = "https://api.marketdata.app/v1"
DEFAULT_SLEEP_SECONDS = 0.20


@dataclass
class RunConfig:
    token: str | None
    underlying: str
    mode: str
    chain_date: str
    from_date: str
    to_date: str
    side: str
    expiration: str | None
    exp_from: str | None
    exp_to: str | None
    min_strike: float | None
    max_strike: float | None
    max_expirations: int
    max_strikes: int
    spot: float | None
    option_root: str
    max_contracts: int
    sleep_seconds: float
    out_dir: str


def parse_args() -> RunConfig:
    today = date.today()
    default_chain_date = today.isoformat()
    default_from = (today - timedelta(days=30)).isoformat()
    default_to = today.isoformat()

    parser = argparse.ArgumentParser(
        description="Download historical option prices + greeks from MarketData.app.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--token", default=os.getenv("MARKETDATA_TOKEN"), help="MarketData token")
    parser.add_argument("--underlying", default="SPX", help="Underlying ticker (e.g. SPX, SPY, QQQ)")
    parser.add_argument(
        "--mode",
        choices=["auto", "exp_strikes", "chain"],
        default="auto",
        help="Data universe mode. auto tries chain first then falls back to expirations+strikes.",
    )
    parser.add_argument("--chain-date", default=default_chain_date, help="Chain snapshot date (YYYY-MM-DD)")
    parser.add_argument("--from-date", default=default_from, help="Quote history start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", default=default_to, help="Quote history end date (YYYY-MM-DD)")
    parser.add_argument("--side", choices=["all", "call", "put"], default="all", help="Option side filter")
    parser.add_argument("--expiration", default=None, help="Exact expiration date filter (YYYY-MM-DD)")
    parser.add_argument("--exp-from", default=None, help="Expiration start date filter (YYYY-MM-DD)")
    parser.add_argument("--exp-to", default=None, help="Expiration end date filter (YYYY-MM-DD)")
    parser.add_argument("--min-strike", type=float, default=None, help="Minimum strike filter")
    parser.add_argument("--max-strike", type=float, default=None, help="Maximum strike filter")
    parser.add_argument("--max-expirations", type=int, default=2, help="Max expiration dates to process")
    parser.add_argument(
        "--max-strikes",
        type=int,
        default=25,
        help="Max strikes per expiration after filtering",
    )
    parser.add_argument(
        "--spot",
        type=float,
        default=None,
        help="Reference spot used to pick strikes nearest ATM when max-strikes is applied",
    )
    parser.add_argument(
        "--option-root",
        default="auto",
        help="Option root symbol. auto chooses SPX/SPXW logic for SPX.",
    )
    parser.add_argument(
        "--max-contracts",
        type=int,
        default=250,
        help="Max contracts to download quotes for (safety limit)",
    )
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS, help="Delay between quote requests")
    parser.add_argument("--out-dir", default="data/options", help="Output directory")
    args = parser.parse_args()

    return RunConfig(
        token=args.token,
        underlying=args.underlying.upper(),
        mode=args.mode,
        chain_date=args.chain_date,
        from_date=args.from_date,
        to_date=args.to_date,
        side=args.side,
        expiration=args.expiration,
        exp_from=args.exp_from,
        exp_to=args.exp_to,
        min_strike=args.min_strike,
        max_strike=args.max_strike,
        max_expirations=max(1, args.max_expirations),
        max_strikes=max(1, args.max_strikes),
        spot=args.spot,
        option_root=args.option_root.upper(),
        max_contracts=max(1, args.max_contracts),
        sleep_seconds=max(0.0, args.sleep_seconds),
        out_dir=args.out_dir,
    )


def _json_to_frame(payload: dict) -> pd.DataFrame:
    if payload.get("s") == "error":
        msg = payload.get("errmsg") or payload.get("error") or str(payload)
        raise RuntimeError(f"API error: {msg}")

    list_cols = {k: v for k, v in payload.items() if isinstance(v, list)}
    if not list_cols:
        return pd.DataFrame()

    row_count = max(len(v) for v in list_cols.values())
    out: dict[str, list] = {}
    for key, values in list_cols.items():
        if len(values) < row_count:
            values = values + [None] * (row_count - len(values))
        out[key] = values
    return pd.DataFrame(out)


def _get_endpoint_frame(session: requests.Session, path: str, params: dict, token: str | None) -> pd.DataFrame:
    url = f"{BASE_URL}{path}"
    retries = 5
    auth_values = [None]
    if token:
        auth_values = [f"Bearer {token}", f"Token {token}", token]

    last_error: Exception | None = None
    last_http_error: str | None = None

    for auth_value in auth_values:
        headers = {"Authorization": auth_value} if auth_value else {}
        for attempt in range(1, retries + 1):
            try:
                resp = session.get(url, params=params, headers=headers, timeout=45)
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(min(2**attempt, 10))
                continue

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait_s = float(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt, 15)
                time.sleep(wait_s)
                continue

            if resp.status_code == 401 and token:
                try:
                    data = resp.json()
                    err = data.get("errmsg") or data.get("error") or data
                except Exception:
                    err = resp.text.strip()
                last_http_error = f"HTTP 401 calling {url}: {err}"
                break

            if resp.status_code >= 400:
                try:
                    data = resp.json()
                    err = data.get("errmsg") or data.get("error") or data
                except Exception:
                    err = resp.text.strip()
                raise RuntimeError(f"HTTP {resp.status_code} calling {url}: {err}")

            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return _json_to_frame(resp.json())
            return pd.read_csv(io.StringIO(resp.text))

    if last_http_error:
        raise RuntimeError(last_http_error)
    if last_error:
        raise RuntimeError(f"Failed request after retries: {last_error}") from last_error
    raise RuntimeError("Failed request after retries due to repeated rate limiting")


def _find_col(df: pd.DataFrame, candidates: set[str]) -> str:
    for col in df.columns:
        if col.lower() in candidates:
            return col
    raise RuntimeError(f"Missing required column. Expected one of {sorted(candidates)}; got {list(df.columns)}")


def _normalize_date_str(value: object) -> str:
    dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(dt):
        raise RuntimeError(f"Could not parse date value: {value}")
    return dt.tz_convert(None).date().isoformat()


def _expiration_to_yyMMdd(expiration: str) -> str:
    dt = pd.to_datetime(expiration, errors="raise")
    return dt.strftime("%y%m%d")


def _is_third_friday(dt: pd.Timestamp) -> bool:
    return dt.weekday() == 4 and 15 <= dt.day <= 21


def choose_option_root(cfg: RunConfig, expiration: str) -> str:
    if cfg.option_root != "AUTO":
        return cfg.option_root

    if cfg.underlying != "SPX":
        return cfg.underlying

    exp_dt = pd.to_datetime(expiration)
    return "SPX" if _is_third_friday(exp_dt) else "SPXW"


def format_option_symbol(root: str, expiration: str, side: str, strike: float) -> str:
    exp = _expiration_to_yyMMdd(expiration)
    cp = "C" if side.lower() == "call" else "P"
    strike_int = int(round(float(strike) * 1000.0))
    return f"{root}{exp}{cp}{strike_int:08d}"


def fetch_chain_symbols(session: requests.Session, cfg: RunConfig) -> tuple[pd.DataFrame, list[str]]:
    params: dict[str, str] = {
        "date": cfg.chain_date,
        "expiration": "all",
        "dateformat": "timestamp",
    }
    if cfg.side != "all":
        params["side"] = cfg.side
    if cfg.expiration:
        params["expiration"] = cfg.expiration
    if cfg.exp_from:
        params["from"] = cfg.exp_from
    if cfg.exp_to:
        params["to"] = cfg.exp_to

    chain_df = _get_endpoint_frame(session, f"/options/chain/{cfg.underlying}/", params, cfg.token)
    if chain_df.empty:
        raise RuntimeError("Chain endpoint returned no rows.")

    strike_col = next((c for c in chain_df.columns if c.lower() == "strike"), None)
    if strike_col and cfg.min_strike is not None:
        chain_df = chain_df[chain_df[strike_col] >= cfg.min_strike]
    if strike_col and cfg.max_strike is not None:
        chain_df = chain_df[chain_df[strike_col] <= cfg.max_strike]

    symbol_col = _find_col(chain_df, {"optionsymbol", "option_symbol", "symbol"})
    symbols = chain_df[symbol_col].dropna().astype(str).tolist()[: cfg.max_contracts]
    if not symbols:
        raise RuntimeError("No contract symbols available from chain after filters.")
    return chain_df.reset_index(drop=True), symbols


def fetch_expirations(session: requests.Session, cfg: RunConfig) -> list[str]:
    params: dict[str, str] = {"dateformat": "timestamp"}
    exp_df = _get_endpoint_frame(session, f"/options/expirations/{cfg.underlying}/", params, cfg.token)
    if exp_df.empty:
        raise RuntimeError("Expirations endpoint returned no rows.")

    exp_col = _find_col(exp_df, {"expiration", "expirations", "date"})
    expirations = sorted({_normalize_date_str(x) for x in exp_df[exp_col].dropna().tolist()})

    if cfg.expiration:
        expirations = [e for e in expirations if e == cfg.expiration]
    if cfg.exp_from:
        expirations = [e for e in expirations if e >= cfg.exp_from]
    if cfg.exp_to:
        expirations = [e for e in expirations if e <= cfg.exp_to]

    if not expirations:
        raise RuntimeError("No expirations remain after filters.")
    return expirations[: cfg.max_expirations]


def fetch_strikes(session: requests.Session, cfg: RunConfig, expiration: str) -> list[float]:
    params = {"expiration": expiration}
    strike_df = _get_endpoint_frame(session, f"/options/strikes/{cfg.underlying}/", params, cfg.token)
    if strike_df.empty:
        return []

    strike_col = _find_col(strike_df, {"strike", "strikes"})
    strikes = pd.to_numeric(strike_df[strike_col], errors="coerce").dropna().astype(float).tolist()
    strikes = sorted(set(strikes))

    if cfg.min_strike is not None:
        strikes = [s for s in strikes if s >= cfg.min_strike]
    if cfg.max_strike is not None:
        strikes = [s for s in strikes if s <= cfg.max_strike]

    if not strikes:
        return []

    if len(strikes) > cfg.max_strikes:
        if cfg.spot is not None:
            strikes = sorted(strikes, key=lambda x: abs(x - cfg.spot))[: cfg.max_strikes]
            strikes = sorted(strikes)
        else:
            mid = len(strikes) // 2
            half = cfg.max_strikes // 2
            lo = max(0, mid - half)
            hi = min(len(strikes), lo + cfg.max_strikes)
            lo = max(0, hi - cfg.max_strikes)
            strikes = strikes[lo:hi]
    return strikes


def build_symbols_from_exp_strikes(session: requests.Session, cfg: RunConfig) -> tuple[pd.DataFrame, list[str]]:
    expirations = fetch_expirations(session, cfg)
    sides = ["call", "put"] if cfg.side == "all" else [cfg.side]

    rows: list[dict] = []
    symbols: list[str] = []
    for expiration in expirations:
        strikes = fetch_strikes(session, cfg, expiration)
        root = choose_option_root(cfg, expiration)
        for strike in strikes:
            for side in sides:
                option_symbol = format_option_symbol(root, expiration, side, strike)
                rows.append(
                    {
                        "underlying": cfg.underlying,
                        "optionRoot": root,
                        "expiration": expiration,
                        "side": side,
                        "strike": strike,
                        "optionSymbol": option_symbol,
                    }
                )
                symbols.append(option_symbol)

    if not symbols:
        raise RuntimeError("No symbols synthesized from expirations/strikes filters.")
    symbols = symbols[: cfg.max_contracts]
    universe_df = pd.DataFrame(rows)
    universe_df = universe_df[universe_df["optionSymbol"].isin(set(symbols))].reset_index(drop=True)
    return universe_df, symbols


def fetch_quote_history_for_symbols(session: requests.Session, symbols: list[str], cfg: RunConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    total = len(symbols)

    for i, symbol in enumerate(symbols, start=1):
        params = {
            "from": cfg.from_date,
            "to": cfg.to_date,
            "dateformat": "timestamp",
        }

        try:
            quote_df = _get_endpoint_frame(session, f"/options/quotes/{symbol}/", params, cfg.token)
            if not quote_df.empty:
                quote_df["requestedSymbol"] = symbol
                frames.append(quote_df)
            print(f"[{i}/{total}] fetched {symbol} rows={len(quote_df)}")
        except Exception as exc:
            print(f"[{i}/{total}] failed {symbol}: {exc}")

        if cfg.sleep_seconds > 0:
            time.sleep(cfg.sleep_seconds)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run(cfg: RunConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    if not cfg.token:
        print("MARKETDATA_TOKEN is not set. Token-less access is usually limited.")

    with requests.Session() as session:
        universe_df: pd.DataFrame
        symbols: list[str]
        universe_name: str

        if cfg.mode == "chain":
            universe_df, symbols = fetch_chain_symbols(session, cfg)
            universe_name = "chain"
        elif cfg.mode == "exp_strikes":
            universe_df, symbols = build_symbols_from_exp_strikes(session, cfg)
            universe_name = "universe"
        else:
            try:
                universe_df, symbols = fetch_chain_symbols(session, cfg)
                universe_name = "chain"
                print("Using chain mode.")
            except Exception as exc:
                print(f"Chain mode failed, falling back to expirations+strikes: {exc}")
                universe_df, symbols = build_symbols_from_exp_strikes(session, cfg)
                universe_name = "universe"

        universe_path = os.path.join(
            cfg.out_dir,
            f"{cfg.underlying}_{universe_name}_{cfg.chain_date}.csv",
        )
        universe_df.to_csv(universe_path, index=False)
        print(f"Saved {universe_name}: {universe_path} rows={len(universe_df)}")
        print(f"Downloading quotes for {len(symbols)} contracts.")

        quotes_df = fetch_quote_history_for_symbols(session, symbols, cfg)
        quotes_path = os.path.join(
            cfg.out_dir,
            f"{cfg.underlying}_quotes_{cfg.from_date}_to_{cfg.to_date}.csv",
        )
        quotes_df.to_csv(quotes_path, index=False)
        print(f"Saved quotes: {quotes_path} rows={len(quotes_df)}")

        greek_cols = [c for c in ["iv", "delta", "gamma", "theta", "vega", "rho"] if c in quotes_df.columns]
        if greek_cols:
            print(f"Greeks columns present: {greek_cols}")
        else:
            print("No greek columns detected in returned quotes.")


def main() -> None:
    cfg = parse_args()
    try:
        run(cfg)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
