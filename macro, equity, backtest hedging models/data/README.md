# Static dataset contract

This folder stores built-in local datasets for reproducible backtests.

## Required file

- `HY_OAS.csv` is required by `Backtest.py` and the Streamlit `Hedged Backtest Overlay` page.

## App behavior

- Backtest function uses local/static data only.
- No user upload is expected.
- Other 7 functions use live data sources.
