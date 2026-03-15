from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    AnalystValuationRequest,
    CrashRiskRequest,
    CrashRiskResponse,
    GlobalIndicesRequest,
    GlobalIndicesResponse,
    HedgedBacktestRequest,
    HedgedBacktestResponse,
    MacroEquityRequest,
    MacroEquityResponse,
    PairCorrelationRequest,
    PairCorrelationResponse,
    SectorCorrelationRequest,
    SectorCorrelationResponse,
    StaticDatasetStatus,
    ValuationRequest,
    ValuationResponse,
    ValuationRow,
)
from .services.crash_risk import run_crash_risk
from .services.global_indices import as_of_date, build_performance_rows, download_index_prices
from .services.hedged_backtest import run_hedged_backtest
from .services.macro_equity import run_macro_equity
from .services.pair_correlation import run_pair_correlation
from .services.sector_correlation import run_sector_correlation
from .services.static_dataset import check_hy_oas_dataset
from .services.valuation import build_analyst_revision_scores, build_valuation_scores

app = FastAPI(title="Research Tools Quant API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPO_ROOT = Path(__file__).resolve().parents[4]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/global-indices", response_model=GlobalIndicesResponse)
def global_indices(payload: GlobalIndicesRequest) -> GlobalIndicesResponse:
    tickers = [t.strip().upper() for t in payload.tickers if t.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="At least one ticker is required")

    px = download_index_prices(tickers, payload.start_date)
    if px.empty:
        raise HTTPException(status_code=404, detail="No data returned from yfinance")

    rows = build_performance_rows(px)
    return GlobalIndicesResponse(as_of=as_of_date(px), rows=rows)


@app.get("/backtest/static-dataset", response_model=StaticDatasetStatus)
def backtest_static_dataset() -> StaticDatasetStatus:
    return StaticDatasetStatus(**check_hy_oas_dataset(REPO_ROOT))


@app.post("/valuation-vs-quality", response_model=ValuationResponse)
def valuation_vs_quality(payload: ValuationRequest) -> ValuationResponse:
    tickers = [t.strip().upper() for t in payload.tickers if t.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="At least one ticker is required")

    rows = build_valuation_scores(
        tickers=tickers,
        min_peers=payload.min_peers,
        highlight=payload.highlight,
    )
    return ValuationResponse(rows=[ValuationRow(**r) for r in rows])


@app.post("/analyst-valuation", response_model=ValuationResponse)
def analyst_valuation(payload: AnalystValuationRequest) -> ValuationResponse:
    tickers = [t.strip().upper() for t in payload.tickers if t.strip()]
    if not tickers:
        raise HTTPException(status_code=400, detail="At least one ticker is required")

    rows = build_valuation_scores(
        tickers=tickers,
        min_peers=payload.min_peers,
        highlight=payload.highlight,
    )
    rev = build_analyst_revision_scores(tickers=tickers, lookback_days=payload.lookback_days)
    for row in rows:
        row["revisions_pct"] = rev.get(row["ticker"])
    return ValuationResponse(rows=[ValuationRow(**r) for r in rows])


@app.post("/pair-correlation", response_model=PairCorrelationResponse)
def pair_correlation(payload: PairCorrelationRequest) -> PairCorrelationResponse:
    try:
        out = run_pair_correlation(
            asset_a=payload.asset_a,
            asset_b=payload.asset_b,
            start_date=payload.start_date,
            win_3m=payload.win_3m,
            win_6m=payload.win_6m,
        )
        return PairCorrelationResponse(**out)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/sector-correlation", response_model=SectorCorrelationResponse)
def sector_correlation(payload: SectorCorrelationRequest) -> SectorCorrelationResponse:
    try:
        out = run_sector_correlation(
            tickers=payload.tickers,
            start_date=payload.start_date,
            window=payload.window,
            method=payload.method,
        )
        return SectorCorrelationResponse(**out)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/crash-risk", response_model=CrashRiskResponse)
def crash_risk(payload: CrashRiskRequest) -> CrashRiskResponse:
    try:
        out = run_crash_risk(
            start_date=payload.start_date,
            horizon=payload.horizon,
            drawdown_threshold=payload.drawdown_threshold,
            n_splits=payload.n_splits,
        )
        return CrashRiskResponse(**out)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/macro-equity", response_model=MacroEquityResponse)
def macro_equity(payload: MacroEquityRequest) -> MacroEquityResponse:
    try:
        out = run_macro_equity(
            start_date=payload.start_date,
            lookback_months=payload.lookback_months,
            min_periods=payload.min_periods,
            min_factors=payload.min_factors,
        )
        return MacroEquityResponse(**out)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/hedged-backtest", response_model=HedgedBacktestResponse)
def hedged_backtest(payload: HedgedBacktestRequest) -> HedgedBacktestResponse:
    try:
        out = run_hedged_backtest(
            repo_root=REPO_ROOT,
            start_date=payload.start_date,
            hedge_h=payload.hedge_h,
            vix_threshold=payload.vix_threshold,
            ema_long_term=payload.ema_long_term,
            ema_short_term=payload.ema_short_term,
            corr_window=payload.corr_window,
            spike_history_window=payload.spike_history_window,
            cross_asset_spike_top_pct=payload.cross_asset_spike_top_pct,
            intra_equity_spike_top_pct=payload.intra_equity_spike_top_pct,
            vvix_spike_top_pct=payload.vvix_spike_top_pct,
            hy_level_bps=payload.hy_level_bps,
            hy_widen_bps=payload.hy_widen_bps,
            hy_lookback_days=payload.hy_lookback_days,
            borrow_rate_annual=payload.borrow_rate_annual,
            slippage_bps=payload.slippage_bps,
            include_costs=payload.include_costs,
        )
        return HedgedBacktestResponse(**out)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
