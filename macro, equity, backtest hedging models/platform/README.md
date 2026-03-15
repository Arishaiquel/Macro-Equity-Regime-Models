# Research Tools - Vercel Architecture Starter

This folder provides a long-term project architecture:

- `apps/web`: Next.js frontend (deploy on Vercel)
- `services/quant-api`: Python FastAPI service for quant/model execution
- `services/worker`: background job worker
- `infra`: local docker-compose for development

## Why this architecture

- Vercel is excellent for frontend and lightweight APIs.
- Python quant workloads are better handled in a dedicated service.
- Background jobs support long-running backtests and historical runs.

## Current endpoints

- `GET /health`
- `POST /global-indices`
  - request: `{ "tickers": ["^GSPC", "^IXIC"], "start_date": "2020-01-01" }`
  - response: as-of date + monthly and YTD performance rows
- `POST /valuation-vs-quality`
  - request: `{ "tickers": ["UNH","ABT","CI"], "highlight": "ABT", "min_peers": 5 }`
- `POST /analyst-valuation`
  - request: `{ "tickers": ["NFLX","AMZN","MSFT"], "lookback_days": 180 }`
- `POST /pair-correlation`
  - request: `{ "asset_a":"SPY", "asset_b":"XLK", "start_date":"2020-01-01", "win_3m":63, "win_6m":126 }`
- `POST /sector-correlation`
  - request: `{ "tickers":["XLB","XLC","XLE"], "start_date":"2018-01-01", "window":63, "method":"pearson" }`
- `POST /crash-risk`
  - request: `{ "start_date":"2000-01-01", "horizon":63, "drawdown_threshold":0.15, "n_splits":5 }`
- `POST /macro-equity`
  - request: `{ "start_date":"1995-01-01", "lookback_months":60, "min_periods":36, "min_factors":8 }`
- `POST /hedged-backtest`
  - request includes EMA, spike %, HY spread level/widen settings, and cost toggles
- `GET /backtest/static-dataset`
  - validates built-in local/static `data/HY_OAS.csv`

## Local run (without Docker)

### 1) Run quant-api

```bash
cd platform/services/quant-api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 2) Run web

```bash
cd platform/apps/web
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

## Deploy plan

1. Deploy `apps/web` to Vercel.
2. Deploy `services/quant-api` to Render/Railway/Fly/Cloud Run.
3. Set `NEXT_PUBLIC_API_BASE_URL` in Vercel to quant-api URL.
4. Keep `HY_OAS.csv` local/static in quant-api environment for backtest validation.

## Next integration tasks

1. Wrap each existing Python model script as service modules/endpoints.
2. Add job queue endpoints for long-running backtests.
3. Persist run metadata/results in Postgres.
