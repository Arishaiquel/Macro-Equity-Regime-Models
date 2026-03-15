"use client";

import { useMemo, useState } from "react";
import { FunctionCard } from "@/components/FunctionCard";
import { apiGet, apiPost } from "@/lib/api";

type GlobalIndexRow = {
  ticker: string;
  name: string;
  last: number;
  monthly_return: number | null;
  ytd_return: number | null;
};

type GlobalIndicesResponse = {
  as_of: string;
  rows: GlobalIndexRow[];
};

type StaticStatus = {
  dataset: string;
  exists: boolean;
  min_date: string | null;
  max_date: string | null;
  rows: number;
};

type ValuationRow = {
  ticker: string;
  name: string;
  valuation_pct: number | null;
  quality_pct: number | null;
  combined_pct: number | null;
  revisions_pct?: number | null;
  cheap_vs_peers: boolean;
  rich_vs_peers: boolean;
  cheap_and_quality: boolean;
  is_highlight: boolean;
};

type ValuationResponse = { rows: ValuationRow[] };

type PairResponse = {
  asset_a: string;
  asset_b: string;
  summary: Record<string, number | string>;
  latest_metrics: Record<string, number>;
  rebased_tail: Array<Record<string, number | string>>;
};

type SectorResponse = {
  method: string;
  window: number;
  latest_date: string;
  latest_avg_corr: number;
  full_matrix: Array<Record<string, string | number>>;
  latest_matrix: Array<Record<string, string | number>>;
  cohesion_tail: Array<{ date: string; avg_corr: number }>;
};

type CrashResponse = {
  latest: {
    date: string;
    p_crash_spy: number;
    p_crash_qqq: number;
    risk_composite: number;
  };
  metrics_spy: Record<string, number | null>;
  metrics_qqq: Record<string, number | null>;
  timeseries_tail: Array<{ date: string; p_crash_spy: number; p_crash_qqq: number; risk_composite: number }>;
};

type MacroResponse = {
  latest: {
    date: string;
    macro_score: number;
    spy_1m: number | null;
    qqq_1m: number | null;
  };
  series_tail: Array<{ date: string; macro_score: number; spy_1m: number | null; qqq_1m: number | null }>;
};

type HedgedBacktestResponse = {
  latest: Record<string, string | number | boolean | null>;
  overlay_on_pct: number;
  unhedged_stats: Record<string, number | null>;
  hedged_stats: Record<string, number | null>;
  risk_days_pct: Record<string, number>;
  timeseries_tail: Array<Record<string, string | number>>;
};

const defaults = {
  global: "^GSPC,^IXIC,^DJI,^FTSE,^GDAXI,^FCHI,^N225,^HSI,^STOXX50E,^AXJO",
  valuation: "UNH,ABT,CI,HUM,JNJ,PFE,MRK,LLY",
  analyst: "AMZN,MSFT,GOOGL,META,AAPL,ORCL,WMT,COST,TGT,SHOP,NFLX",
  sectors: "XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY",
};

function pct(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "-";
  return `${(x * 100).toFixed(2)}%`;
}

function panel(title: string, children: React.ReactNode) {
  return (
    <section className="mb-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <h2 className="text-lg font-semibold">{title}</h2>
      <div className="mt-4">{children}</div>
    </section>
  );
}

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [globalTickers, setGlobalTickers] = useState(defaults.global);
  const [globalStart, setGlobalStart] = useState("2020-01-01");
  const [globalRes, setGlobalRes] = useState<GlobalIndicesResponse | null>(null);

  const [valTickers, setValTickers] = useState(defaults.valuation);
  const [valHighlight, setValHighlight] = useState("ABT");
  const [valPeers, setValPeers] = useState("5");
  const [valRes, setValRes] = useState<ValuationResponse | null>(null);

  const [anTickers, setAnTickers] = useState(defaults.analyst);
  const [anHighlight, setAnHighlight] = useState("NFLX");
  const [anPeers, setAnPeers] = useState("5");
  const [anLookback, setAnLookback] = useState("180");
  const [anRes, setAnRes] = useState<ValuationResponse | null>(null);

  const [pairA, setPairA] = useState("SPY");
  const [pairB, setPairB] = useState("XLK");
  const [pairStart, setPairStart] = useState("2020-01-01");
  const [pair3m, setPair3m] = useState("63");
  const [pair6m, setPair6m] = useState("126");
  const [pairRes, setPairRes] = useState<PairResponse | null>(null);

  const [secTickers, setSecTickers] = useState(defaults.sectors);
  const [secStart, setSecStart] = useState("2018-01-01");
  const [secWindow, setSecWindow] = useState("63");
  const [secMethod, setSecMethod] = useState("pearson");
  const [secRes, setSecRes] = useState<SectorResponse | null>(null);

  const [crStart, setCrStart] = useState("2000-01-01");
  const [crHorizon, setCrHorizon] = useState("63");
  const [crDD, setCrDD] = useState("0.15");
  const [crSplits, setCrSplits] = useState("5");
  const [crRes, setCrRes] = useState<CrashResponse | null>(null);

  const [mqStart, setMqStart] = useState("1995-01-01");
  const [mqLookback, setMqLookback] = useState("60");
  const [mqMinPeriods, setMqMinPeriods] = useState("36");
  const [mqMinFactors, setMqMinFactors] = useState("8");
  const [mqRes, setMqRes] = useState<MacroResponse | null>(null);

  const [hbStart, setHbStart] = useState("2016-01-01");
  const [hbHedgeH, setHbHedgeH] = useState("0.5");
  const [hbVix, setHbVix] = useState("20");
  const [hbEmaLong, setHbEmaLong] = useState("180");
  const [hbEmaShort, setHbEmaShort] = useState("40");
  const [hbCorrWin, setHbCorrWin] = useState("63");
  const [hbHistWin, setHbHistWin] = useState("256");
  const [hbCrossTop, setHbCrossTop] = useState("10");
  const [hbIntraTop, setHbIntraTop] = useState("10");
  const [hbVvixTop, setHbVvixTop] = useState("10");
  const [hbHyLevel, setHbHyLevel] = useState("350");
  const [hbHyWiden, setHbHyWiden] = useState("75");
  const [hbHyLookback, setHbHyLookback] = useState("20");
  const [hbBorrow, setHbBorrow] = useState("0.02");
  const [hbSlip, setHbSlip] = useState("5");
  const [hbCosts, setHbCosts] = useState(false);
  const [hbRes, setHbRes] = useState<HedgedBacktestResponse | null>(null);

  const [staticStatus, setStaticStatus] = useState<StaticStatus | null>(null);

  const functionCards = useMemo(
    () => [
      ["Valuation vs Quality", "Live", "Peer-relative valuation and quality scoring."],
      ["Analyst Valuation + Revisions", "Live", "Valuation plus analyst revisions proxy."],
      ["Crash Risk (SPY/QQQ)", "Live", "ML crash probability (HistGradientBoostingClassifier)."],
      ["Macro vs Equity", "Live", "Weighted macro score versus index returns."],
      ["Pair Correlation", "Live", "Rolling correlation, beta, TE, and drawdown."],
      ["Sector Correlation", "Live", "Sector heatmap and cohesion analytics."],
      ["Hedged Backtest Overlay", "Static", "Regime overlay backtest using market stress signals."],
      ["Global Indices", "Live", "Major index comparison with MTD and YTD table."],
    ] as const,
    [],
  );

  async function guardedRun(fn: () => Promise<void>) {
    try {
      setLoading(true);
      setError(null);
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto min-h-screen max-w-7xl px-6 py-10">
      <header className="mb-8">
        <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">Research Tools</p>
        <h1 className="text-3xl font-bold text-ink">Vercel Architecture Starter</h1>
        <p className="mt-2 text-sm text-slate-600">Live APIs are now wired across all core model functions.</p>
      </header>

      <section className="mb-8 grid gap-3 md:grid-cols-2 lg:grid-cols-4">
        {functionCards.map(([title, mode, description]) => (
          <FunctionCard key={title} title={title} mode={mode} description={description} />
        ))}
      </section>

      {error && <section className="mb-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error}</section>}

      {panel(
        "Global Indices",
        <>
          <div className="grid gap-3 md:grid-cols-3">
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={globalTickers} onChange={(e) => setGlobalTickers(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" type="date" value={globalStart} onChange={(e) => setGlobalStart(e.target.value)} />
            <button className="rounded-lg bg-signal px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" disabled={loading} onClick={() => guardedRun(async () => {
              const payload = { tickers: globalTickers.split(",").map((x) => x.trim()).filter(Boolean), start_date: globalStart };
              setGlobalRes(await apiPost<GlobalIndicesResponse>("/global-indices", payload));
            })}>Run</button>
          </div>
          {globalRes && (
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full border-collapse text-sm">
                <thead><tr className="border-b border-slate-200 text-left text-slate-500"><th className="py-2 pr-4">Index</th><th className="py-2 pr-4">Ticker</th><th className="py-2 pr-4">Last</th><th className="py-2 pr-4">MTD</th><th className="py-2 pr-4">YTD</th></tr></thead>
                <tbody>{globalRes.rows.map((r) => <tr key={r.ticker} className="border-b border-slate-100"><td className="py-2 pr-4">{r.name}</td><td className="py-2 pr-4">{r.ticker}</td><td className="py-2 pr-4">{r.last.toFixed(2)}</td><td className="py-2 pr-4">{pct(r.monthly_return)}</td><td className="py-2 pr-4">{pct(r.ytd_return)}</td></tr>)}</tbody>
              </table>
            </div>
          )}
        </>,
      )}

      {panel(
        "Valuation vs Quality",
        <>
          <div className="grid gap-3 md:grid-cols-4">
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm md:col-span-2" value={valTickers} onChange={(e) => setValTickers(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={valHighlight} onChange={(e) => setValHighlight(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={valPeers} onChange={(e) => setValPeers(e.target.value)} />
          </div>
          <button className="mt-3 rounded-lg bg-signal px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" disabled={loading} onClick={() => guardedRun(async () => {
            const payload = { tickers: valTickers.split(",").map((x) => x.trim()).filter(Boolean), highlight: valHighlight, min_peers: Number(valPeers) || 5 };
            setValRes(await apiPost<ValuationResponse>("/valuation-vs-quality", payload));
          })}>Run</button>
          {valRes && (
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full border-collapse text-sm">
                <thead><tr className="border-b border-slate-200 text-left text-slate-500"><th className="py-2 pr-4">Ticker</th><th className="py-2 pr-4">Name</th><th className="py-2 pr-4">Valuation</th><th className="py-2 pr-4">Quality</th><th className="py-2 pr-4">Combined</th><th className="py-2 pr-4">Flags</th></tr></thead>
                <tbody>{valRes.rows.map((r) => <tr key={r.ticker} className={`border-b border-slate-100 ${r.is_highlight ? "bg-sky-50" : ""}`}><td className="py-2 pr-4">{r.ticker}</td><td className="py-2 pr-4">{r.name}</td><td className="py-2 pr-4">{pct(r.valuation_pct)}</td><td className="py-2 pr-4">{pct(r.quality_pct)}</td><td className="py-2 pr-4">{pct(r.combined_pct)}</td><td className="py-2 pr-4">{r.cheap_and_quality ? "cheap+quality" : r.cheap_vs_peers ? "cheap" : r.rich_vs_peers ? "rich" : "-"}</td></tr>)}</tbody>
              </table>
            </div>
          )}
        </>,
      )}

      {panel(
        "Analyst Valuation + Revisions",
        <>
          <div className="grid gap-3 md:grid-cols-5">
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm md:col-span-2" value={anTickers} onChange={(e) => setAnTickers(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={anHighlight} onChange={(e) => setAnHighlight(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={anPeers} onChange={(e) => setAnPeers(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={anLookback} onChange={(e) => setAnLookback(e.target.value)} />
          </div>
          <button className="mt-3 rounded-lg bg-signal px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" disabled={loading} onClick={() => guardedRun(async () => {
            const payload = {
              tickers: anTickers.split(",").map((x) => x.trim()).filter(Boolean),
              highlight: anHighlight,
              min_peers: Number(anPeers) || 5,
              lookback_days: Number(anLookback) || 180,
            };
            setAnRes(await apiPost<ValuationResponse>("/analyst-valuation", payload));
          })}>Run</button>
          {anRes && (
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full border-collapse text-sm">
                <thead><tr className="border-b border-slate-200 text-left text-slate-500"><th className="py-2 pr-4">Ticker</th><th className="py-2 pr-4">Combined</th><th className="py-2 pr-4">Revisions</th></tr></thead>
                <tbody>{anRes.rows.map((r) => <tr key={r.ticker} className={`border-b border-slate-100 ${r.is_highlight ? "bg-sky-50" : ""}`}><td className="py-2 pr-4">{r.ticker}</td><td className="py-2 pr-4">{pct(r.combined_pct)}</td><td className="py-2 pr-4">{pct(r.revisions_pct ?? null)}</td></tr>)}</tbody>
              </table>
            </div>
          )}
        </>,
      )}

      {panel(
        "Pair Correlation",
        <>
          <div className="grid gap-3 md:grid-cols-5">
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={pairA} onChange={(e) => setPairA(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={pairB} onChange={(e) => setPairB(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" type="date" value={pairStart} onChange={(e) => setPairStart(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={pair3m} onChange={(e) => setPair3m(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={pair6m} onChange={(e) => setPair6m(e.target.value)} />
          </div>
          <button className="mt-3 rounded-lg bg-signal px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" disabled={loading} onClick={() => guardedRun(async () => {
            const payload = { asset_a: pairA, asset_b: pairB, start_date: pairStart, win_3m: Number(pair3m) || 63, win_6m: Number(pair6m) || 126 };
            setPairRes(await apiPost<PairResponse>("/pair-correlation", payload));
          })}>Run</button>
          {pairRes && (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify(pairRes.summary, null, 2)}</pre>
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify(pairRes.latest_metrics, null, 2)}</pre>
            </div>
          )}
        </>,
      )}

      {panel(
        "Sector Correlation",
        <>
          <div className="grid gap-3 md:grid-cols-4">
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm md:col-span-2" value={secTickers} onChange={(e) => setSecTickers(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" type="date" value={secStart} onChange={(e) => setSecStart(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={secWindow} onChange={(e) => setSecWindow(e.target.value)} />
          </div>
          <select className="mt-3 rounded-lg border border-slate-300 px-3 py-2 text-sm" value={secMethod} onChange={(e) => setSecMethod(e.target.value)}>
            <option value="pearson">pearson</option>
            <option value="spearman">spearman</option>
            <option value="kendall">kendall</option>
          </select>
          <button className="ml-3 rounded-lg bg-signal px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" disabled={loading} onClick={() => guardedRun(async () => {
            const payload = { tickers: secTickers.split(",").map((x) => x.trim()).filter(Boolean), start_date: secStart, window: Number(secWindow) || 63, method: secMethod };
            setSecRes(await apiPost<SectorResponse>("/sector-correlation", payload));
          })}>Run</button>
          {secRes && (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify({ latest_date: secRes.latest_date, latest_avg_corr: secRes.latest_avg_corr, method: secMethod }, null, 2)}</pre>
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify(secRes.cohesion_tail.slice(-20), null, 2)}</pre>
            </div>
          )}
        </>,
      )}

      {panel(
        "Crash Risk (SPY/QQQ)",
        <>
          <div className="grid gap-3 md:grid-cols-4">
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" type="date" value={crStart} onChange={(e) => setCrStart(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={crHorizon} onChange={(e) => setCrHorizon(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={crDD} onChange={(e) => setCrDD(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={crSplits} onChange={(e) => setCrSplits(e.target.value)} />
          </div>
          <button className="mt-3 rounded-lg bg-signal px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" disabled={loading} onClick={() => guardedRun(async () => {
            const payload = { start_date: crStart, horizon: Number(crHorizon) || 63, drawdown_threshold: Number(crDD) || 0.15, n_splits: Number(crSplits) || 5 };
            setCrRes(await apiPost<CrashResponse>("/crash-risk", payload));
          })}>Run</button>
          {crRes && (
            <div className="mt-4 grid gap-4 md:grid-cols-3">
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify(crRes.latest, null, 2)}</pre>
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify(crRes.metrics_spy, null, 2)}</pre>
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify(crRes.metrics_qqq, null, 2)}</pre>
            </div>
          )}
        </>,
      )}

      {panel(
        "Macro vs Equity",
        <>
          <div className="grid gap-3 md:grid-cols-5">
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" type="date" value={mqStart} onChange={(e) => setMqStart(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={mqLookback} onChange={(e) => setMqLookback(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={mqMinPeriods} onChange={(e) => setMqMinPeriods(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={mqMinFactors} onChange={(e) => setMqMinFactors(e.target.value)} />
            <button className="rounded-lg bg-signal px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" disabled={loading} onClick={() => guardedRun(async () => {
              const payload = { start_date: mqStart, lookback_months: Number(mqLookback) || 60, min_periods: Number(mqMinPeriods) || 36, min_factors: Number(mqMinFactors) || 8 };
              setMqRes(await apiPost<MacroResponse>("/macro-equity", payload));
            })}>Run</button>
          </div>
          {mqRes && (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify(mqRes.latest, null, 2)}</pre>
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify(mqRes.series_tail.slice(-20), null, 2)}</pre>
            </div>
          )}
        </>,
      )}

      {panel(
        "Hedged Backtest Overlay",
        <>
          <div className="grid gap-3 md:grid-cols-6">
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" type="date" value={hbStart} onChange={(e) => setHbStart(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbHedgeH} onChange={(e) => setHbHedgeH(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbVix} onChange={(e) => setHbVix(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbEmaLong} onChange={(e) => setHbEmaLong(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbEmaShort} onChange={(e) => setHbEmaShort(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbCorrWin} onChange={(e) => setHbCorrWin(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbHistWin} onChange={(e) => setHbHistWin(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbCrossTop} onChange={(e) => setHbCrossTop(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbIntraTop} onChange={(e) => setHbIntraTop(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbVvixTop} onChange={(e) => setHbVvixTop(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbHyLevel} onChange={(e) => setHbHyLevel(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbHyWiden} onChange={(e) => setHbHyWiden(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbHyLookback} onChange={(e) => setHbHyLookback(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbBorrow} onChange={(e) => setHbBorrow(e.target.value)} />
            <input className="rounded-lg border border-slate-300 px-3 py-2 text-sm" value={hbSlip} onChange={(e) => setHbSlip(e.target.value)} />
            <label className="flex items-center gap-2 rounded-lg border border-slate-300 px-3 py-2 text-sm">
              <input type="checkbox" checked={hbCosts} onChange={(e) => setHbCosts(e.target.checked)} /> Include costs
            </label>
          </div>
          <button className="mt-3 rounded-lg bg-signal px-4 py-2 text-sm font-semibold text-white disabled:opacity-50" disabled={loading} onClick={() => guardedRun(async () => {
            const payload = {
              start_date: hbStart,
              hedge_h: Number(hbHedgeH) || 0.5,
              vix_threshold: Number(hbVix) || 20,
              ema_long_term: Number(hbEmaLong) || 180,
              ema_short_term: Number(hbEmaShort) || 40,
              corr_window: Number(hbCorrWin) || 63,
              spike_history_window: Number(hbHistWin) || 256,
              cross_asset_spike_top_pct: Number(hbCrossTop) || 10,
              intra_equity_spike_top_pct: Number(hbIntraTop) || 10,
              vvix_spike_top_pct: Number(hbVvixTop) || 10,
              hy_level_bps: Number(hbHyLevel) || 350,
              hy_widen_bps: Number(hbHyWiden) || 75,
              hy_lookback_days: Number(hbHyLookback) || 20,
              borrow_rate_annual: Number(hbBorrow) || 0.02,
              slippage_bps: Number(hbSlip) || 5,
              include_costs: hbCosts,
            };
            setHbRes(await apiPost<HedgedBacktestResponse>("/hedged-backtest", payload));
          })}>Run</button>
          {hbRes && (
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify({ latest: hbRes.latest, overlay_on_pct: hbRes.overlay_on_pct }, null, 2)}</pre>
              <pre className="overflow-x-auto rounded-lg bg-slate-50 p-3 text-xs">{JSON.stringify({ unhedged_stats: hbRes.unhedged_stats, hedged_stats: hbRes.hedged_stats, risk_days_pct: hbRes.risk_days_pct }, null, 2)}</pre>
            </div>
          )}
        </>,
      )}

      {panel(
        "Backtest Static Dataset Status",
        <>
          <button className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-ink disabled:opacity-50" disabled={loading} onClick={() => guardedRun(async () => setStaticStatus(await apiGet<StaticStatus>("/backtest/static-dataset")))}>Check HY_OAS.csv</button>
          {staticStatus && (
            <div className="mt-3 text-sm text-slate-700">
              <p>Dataset: {staticStatus.dataset}</p>
              <p>Exists: {staticStatus.exists ? "Yes" : "No"}</p>
              <p>Rows: {staticStatus.rows}</p>
              <p>Date range: {staticStatus.min_date ?? "-"} to {staticStatus.max_date ?? "-"}</p>
            </div>
          )}
        </>,
      )}
    </main>
  );
}
