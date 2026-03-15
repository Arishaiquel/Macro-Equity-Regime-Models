from __future__ import annotations

from pydantic import BaseModel, Field


class GlobalIndicesRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    start_date: str = Field(default="2020-01-01")


class GlobalIndexRow(BaseModel):
    ticker: str
    name: str
    last: float
    monthly_return: float | None
    ytd_return: float | None


class GlobalIndicesResponse(BaseModel):
    as_of: str
    rows: list[GlobalIndexRow]


class StaticDatasetStatus(BaseModel):
    dataset: str
    exists: bool
    min_date: str | None
    max_date: str | None
    rows: int


class ValuationRequest(BaseModel):
    tickers: list[str] = Field(min_length=1)
    highlight: str | None = None
    min_peers: int = 5


class ValuationRow(BaseModel):
    ticker: str
    name: str
    sector: str | None
    industry: str | None
    valuation_pct: float | None
    quality_pct: float | None
    combined_pct: float | None
    cheap_vs_peers: bool
    rich_vs_peers: bool
    cheap_and_quality: bool
    is_highlight: bool
    revisions_pct: float | None = None


class ValuationResponse(BaseModel):
    rows: list[ValuationRow]


class AnalystValuationRequest(ValuationRequest):
    lookback_days: int = 180


class PairCorrelationRequest(BaseModel):
    asset_a: str = "SPY"
    asset_b: str = "XLK"
    start_date: str = "2020-01-01"
    win_3m: int = 63
    win_6m: int = 126


class PairCorrelationResponse(BaseModel):
    asset_a: str
    asset_b: str
    summary: dict
    latest_metrics: dict
    rebased_tail: list[dict]


class SectorCorrelationRequest(BaseModel):
    tickers: list[str] = Field(min_length=2)
    start_date: str = "2018-01-01"
    window: int = 63
    method: str = "pearson"


class SectorCorrelationResponse(BaseModel):
    method: str
    window: int
    latest_date: str
    latest_avg_corr: float
    full_matrix: list[dict]
    latest_matrix: list[dict]
    cohesion_tail: list[dict]


class CrashRiskRequest(BaseModel):
    start_date: str = "2000-01-01"
    horizon: int = 63
    drawdown_threshold: float = 0.15
    n_splits: int = 5


class CrashRiskResponse(BaseModel):
    latest: dict
    metrics_spy: dict
    metrics_qqq: dict
    timeseries_tail: list[dict]


class MacroEquityRequest(BaseModel):
    start_date: str = "1995-01-01"
    lookback_months: int = 60
    min_periods: int = 36
    min_factors: int = 8


class MacroEquityResponse(BaseModel):
    latest: dict
    series_tail: list[dict]


class HedgedBacktestRequest(BaseModel):
    start_date: str = "2016-01-01"
    hedge_h: float = 0.5
    vix_threshold: float = 20.0
    ema_long_term: int = 180
    ema_short_term: int = 40
    corr_window: int = 63
    spike_history_window: int = 256
    cross_asset_spike_top_pct: float = 10.0
    intra_equity_spike_top_pct: float = 10.0
    vvix_spike_top_pct: float = 10.0
    hy_level_bps: float = 350.0
    hy_widen_bps: float = 75.0
    hy_lookback_days: int = 20
    borrow_rate_annual: float = 0.02
    slippage_bps: float = 5.0
    include_costs: bool = False


class HedgedBacktestResponse(BaseModel):
    latest: dict
    overlay_on_pct: float
    unhedged_stats: dict
    hedged_stats: dict
    risk_days_pct: dict
    timeseries_tail: list[dict]
