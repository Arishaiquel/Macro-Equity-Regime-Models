import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# US sector ETFs (SPDR)
SECTOR_TICKERS = [
    "XLB",  # Materials
    "XLC",  # Communication Services
    "XLE",  # Energy
    "XLF",  # Financials
    "XLI",  # Industrials
    "XLK",  # Technology
    "XLP",  # Consumer Staples
    "XLRE", # Real Estate
    "XLU",  # Utilities
    "XLV",  # Health Care
    "XLY",  # Consumer Discretionary
]

# -----------------------------
# CONFIG
# -----------------------------
START = "2018-01-01"
WINDOW = 63  # ~3 trading months
METHOD = "pearson"  # choose: pearson, spearman, kendall


def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    return px.dropna(how="all")


def daily_log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff().dropna(how="any")


def rolling_corr_tensor(returns: pd.DataFrame, window: int, method: str) -> dict[pd.Timestamp, pd.DataFrame]:
    corr_map: dict[pd.Timestamp, pd.DataFrame] = {}
    for end_i in range(window - 1, len(returns)):
        window_slice = returns.iloc[end_i - window + 1 : end_i + 1]
        corr_map[returns.index[end_i]] = window_slice.corr(method=method)
    return corr_map


def mean_offdiag_corr(corr_mat: pd.DataFrame) -> float:
    vals = corr_mat.values
    n = vals.shape[0]
    if n < 2:
        return np.nan
    mask = ~np.eye(n, dtype=bool)
    return float(vals[mask].mean())


def plot_latest_heatmap(corr_mat: pd.DataFrame, as_of: pd.Timestamp, method: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr_mat.values, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(range(len(corr_mat.columns)))
    ax.set_yticks(range(len(corr_mat.index)))
    ax.set_xticklabels(corr_mat.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_mat.index)

    ax.set_title(f"{method.title()} rolling correlation ({WINDOW}d) as of {as_of.date()}")
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Correlation")

    plt.tight_layout()
    plt.show()


def plot_all_sector_cohesion(cohesion: pd.Series, method: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    cohesion.plot(ax=ax, lw=2.2, color="tab:blue")
    ax.axhline(0, linestyle="--", alpha=0.4)
    ax.set_title(f"All-sector average rolling {method.title()} correlation ({WINDOW}d)")
    ax.set_ylabel("Average off-diagonal correlation")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    method = METHOD.lower().strip()
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("METHOD must be one of: pearson, spearman, kendall")

    px = download_prices(SECTOR_TICKERS, START)
    if px.empty:
        raise RuntimeError("No price data returned from yfinance.")

    px = px[SECTOR_TICKERS].dropna(how="any")
    rets = daily_log_returns(px)

    full_corr = rets.corr(method=method)
    print("\n=== Full-sample correlation matrix (daily log returns) ===")
    print(full_corr.round(3).to_string())

    corr_map = rolling_corr_tensor(rets, WINDOW, method)
    if not corr_map:
        raise RuntimeError("Not enough data for selected rolling window.")

    latest_date = max(corr_map.keys())
    latest_corr = corr_map[latest_date]
    cohesion = pd.Series(
        {dt: mean_offdiag_corr(mat) for dt, mat in corr_map.items()},
        name="all_sector_avg_corr",
    )

    print(f"\n=== Latest rolling correlation matrix ({WINDOW}d) as of {latest_date.date()} ===")
    print(latest_corr.round(3).to_string())
    print(f"\nAll-sector average rolling correlation ({WINDOW}d): {cohesion.iloc[-1]:.3f}")

    plot_latest_heatmap(latest_corr, latest_date, method)
    plot_all_sector_cohesion(cohesion, method)


if __name__ == "__main__":
    main()
