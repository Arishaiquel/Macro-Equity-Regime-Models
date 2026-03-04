"""
PRICE-ONLY MARKET CRASH RISK MODEL (SPY + QQQ)
------------------------------------------------
Goal:
  Estimate daily probability of a >=15% drawdown occurring within the next 3 months (~63 trading days)
  for:
    - SPY (S&P 500 proxy)
    - QQQ (Nasdaq-100 proxy)

Model type:
  - Supervised classifier trained on forward-looking crash labels
  - Price-only features (returns, vol, downside vol, drawdown, momentum, MA distance, cross-market stress)
  - Time-series cross-validation (walk-forward)
  - Probability calibration (isotonic) so outputs behave like true probabilities

Outputs:
  - out.csv containing:
      p_crash_spy_3m, p_crash_qqq_3m, risk_composite, plus labels for backtesting
  - Metrics printed (base rate, Brier, PR-AUC, ROC-AUC)
  - Optional plots

Install:
  pip install yfinance scikit-learn pandas numpy matplotlib

Run:
  python crash_risk_price_only.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    average_precision_score,
    roc_auc_score,
)


# -----------------------------
# 1) Core definitions
# -----------------------------

HORIZON = 63          # ~3 months of trading days
DD_THRESH = 0.15      # 15% drawdown
START_DATE = "2000-01-01"


def compute_drawdown(px: pd.Series) -> pd.Series:
    peak = px.cummax()
    return 1.0 - (px / peak)


def crash_label_forward(px: pd.Series, horizon: int = HORIZON, dd_thresh: float = DD_THRESH) -> pd.Series:
    """
    y_t = 1 if there exists a date u in (t+1..t+horizon) such that drawdown(u) >= dd_thresh.
    Implementation:
      - Compute drawdown series DD_t (from running peak)
      - For each t, compute max DD over the next 'horizon' days (excluding today)
    """
    dd = compute_drawdown(px)
    # max drawdown over next horizon days (exclude today via shift(-1))
    future_max_dd = dd.shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    y = (future_max_dd >= dd_thresh).astype("float")
    y.name = f"crash_{int(dd_thresh*100)}pct_in_{horizon}d"
    return y


def realized_vol(ret: pd.Series, win: int) -> pd.Series:
    return ret.rolling(win).std() * np.sqrt(252)


def downside_vol(ret: pd.Series, win: int) -> pd.Series:
    neg = ret.where(ret < 0, 0.0)
    return neg.rolling(win).std() * np.sqrt(252)


def rolling_skew(ret: pd.Series, win: int) -> pd.Series:
    return ret.rolling(win).skew()


def rolling_kurt(ret: pd.Series, win: int) -> pd.Series:
    return ret.rolling(win).kurt()


def build_asset_features(px: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Price-only features for one asset.
    """
    r = px.pct_change()
    df = pd.DataFrame(index=px.index)

    # returns
    df[f"{prefix}_ret_1d"] = r
    df[f"{prefix}_ret_5d"] = px.pct_change(5)
    df[f"{prefix}_ret_21d"] = px.pct_change(21)

    # realized vol
    df[f"{prefix}_vol_21d"] = realized_vol(r, 21)
    df[f"{prefix}_vol_63d"] = realized_vol(r, 63)

    # downside risk
    df[f"{prefix}_dvol_21d"] = downside_vol(r, 21)

    # higher moments
    df[f"{prefix}_skew_63d"] = rolling_skew(r, 63)
    df[f"{prefix}_kurt_63d"] = rolling_kurt(r, 63)

    # momentum
    df[f"{prefix}_mom_63d"] = px.pct_change(63)
    df[f"{prefix}_mom_126d"] = px.pct_change(126)

    # drawdown now
    df[f"{prefix}_dd_now"] = compute_drawdown(px)

    # moving average distance
    ma200 = px.rolling(200).mean()
    df[f"{prefix}_dist_ma200"] = (px / ma200) - 1.0

    return df


def prepare_dataset(spy: pd.Series, qqq: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a single feature matrix X shared across both targets,
    and two separate targets y_spy and y_qqq.
    """
    # Labels
    y_spy = crash_label_forward(spy, horizon=HORIZON, dd_thresh=DD_THRESH).rename("y_spy")
    y_qqq = crash_label_forward(qqq, horizon=HORIZON, dd_thresh=DD_THRESH).rename("y_qqq")

    # Features
    X_spy = build_asset_features(spy, "SPY")
    X_qqq = build_asset_features(qqq, "QQQ")

    # Cross-market stress features
    r_spy = spy.pct_change()
    r_qqq = qqq.pct_change()
    cross = pd.DataFrame(index=spy.index)
    cross["corr_63d_spy_qqq"] = r_spy.rolling(63).corr(r_qqq)
    cross["vol_spread_63d_qqq_minus_spy"] = realized_vol(r_qqq, 63) - realized_vol(r_spy, 63)
    cross["mom_spread_63d_qqq_minus_spy"] = qqq.pct_change(63) - spy.pct_change(63)

    X = pd.concat([X_spy, X_qqq, cross], axis=1)

    # Align and drop NaNs (removes early warmup windows and last horizon labels)
    data = pd.concat([X, y_spy, y_qqq], axis=1).dropna()

    X = data.drop(columns=["y_spy", "y_qqq"])
    y_spy = data["y_spy"].astype(int)
    y_qqq = data["y_qqq"].astype(int)

    return X, y_spy, y_qqq


# -----------------------------
# 2) Training: walk-forward + calibration
# -----------------------------

def walkforward_calibrated_probs(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 6,
    seed: int = 42
) -> tuple[pd.Series, dict]:
    """
    Walk-forward training:
      - Fit on past, predict on future
      - Calibrate only if BOTH classes exist in the training window
      - If only one class exists, output constant prob (0.0 or 1.0) for that fold
    """
    base = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=1e-3,
        random_state=seed,
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    probs = pd.Series(index=y.index, dtype=float)

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]

        # --- Critical fix: handle rare-event folds with only one class ---
        classes = np.unique(y_train.values)
        if len(classes) < 2:
            # If training set has no crashes (all 0), predict prob=0.
            # If training set is all 1 (unlikely), predict prob=1.
            const_p = float(classes[0])
            probs.iloc[test_idx] = const_p
            continue

        # Fit base model
        base.fit(X_train, y_train)

        # Calibrate on train only (no leakage)
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(X_train, y_train)

        # Now predict probabilities safely (should have 2 columns)
        p = cal.predict_proba(X_test)
        # be extra defensive
        if p.shape[1] == 1:
            probs.iloc[test_idx] = float(cal.classes_[0])
        else:
            # column for class=1 may not always be at index 1 if classes are [1,0] (rare)
            col_1 = int(np.where(cal.classes_ == 1)[0][0])
            probs.iloc[test_idx] = p[:, col_1]

    valid = probs.dropna().index
    yv = y.loc[valid]
    pv = probs.loc[valid]

    metrics = {
        "base_rate": float(yv.mean()),
        "brier": float(brier_score_loss(yv, pv)),
        "pr_auc": float(average_precision_score(yv, pv)),
        "roc_auc": float(roc_auc_score(yv, pv)),
        "n_obs": int(len(yv)),
    }
    return probs, metrics



# -----------------------------
# 3) Reporting helpers
# -----------------------------

def prob_percentile(p: pd.Series) -> float:
    """Percentile of the last value vs history."""
    last = float(p.dropna().iloc[-1])
    hist = p.dropna().values
    return float((hist <= last).mean() * 100.0)


def make_summary(name: str, p: pd.Series, metrics: dict) -> str:
    p_clean = p.dropna()
    last = float(p_clean.iloc[-1])
    pct = prob_percentile(p)
    return (
        f"{name}\n"
        f"  Latest P(crash in next 3m): {last:.3%}  (percentile vs history: {pct:.1f}th)\n"
        f"  Base rate in OOS: {metrics['base_rate']:.3%}\n"
        f"  Brier: {metrics['brier']:.5f} | PR-AUC: {metrics['pr_auc']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}\n"
        f"  OOS obs: {metrics['n_obs']}\n"
    )

import matplotlib.dates as mdates

import matplotlib.dates as mdates

def plot_probs(out):
    fig, ax = plt.subplots(figsize=(14, 6))

    out["risk_composite"].plot(
        ax=ax,
        color="green",
        lw=2,
        label="Composite crash risk (50/50)"
    )

    ax.set_title("Price-only crash probability (15% drawdown within 3 months)")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Date")

    # --- CLEAN DATE AXIS ---
    # Major ticks: every quarter
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Minor ticks: yearly (for grid reference only)
    ax.xaxis.set_minor_locator(mdates.YearLocator())

    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.15)

    ax.legend()
    plt.tight_layout()
    plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def compute_drawdown(px: pd.Series) -> pd.Series:
    peak = px.cummax()
    return 1.0 - px / peak

def find_bear_bands(px: pd.Series, thresh: float = 0.20):
    dd = compute_drawdown(px)
    in_band = False
    start = None
    bands = []

    for date, d in dd.items():
        if (not in_band) and (d >= thresh):
            in_band = True
            start = date
        elif in_band and (d < thresh):
            bands.append((start, date))
            in_band = False
            start = None

    if in_band:
        bands.append((start, dd.index[-1]))

    return bands

def plot_composite_with_crash_shading(out: pd.DataFrame, spy_prices: pd.Series):
    fig, ax = plt.subplots(figsize=(14, 6))

    out["risk_composite"].plot(ax=ax, lw=2, label="Composite crash risk (50/50)")

    ax.set_title("US Equity Market Crash Risk Model")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Date")

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)

    ax.grid(True, alpha=0.3)

    for start, end in find_bear_bands(spy_prices, thresh=0.20):
        ax.axvspan(start, end, color='red', alpha=0.15)

    ax.legend()
    plt.tight_layout()
    plt.show()






# -----------------------------
# 4) Main
# -----------------------------

def main():
    # Download prices (auto_adjust=True uses adjusted prices)
    px = yf.download(["SPY", "QQQ"], start=START_DATE, auto_adjust=True, progress=False)["Close"]
    spy = px["SPY"].dropna()
    qqq = px["QQQ"].dropna()

    # Build dataset
    X, y_spy, y_qqq = prepare_dataset(spy, qqq)

    # Train & get walk-forward calibrated probabilities
    p_spy, m_spy = walkforward_calibrated_probs(X, y_spy)
    p_qqq, m_qqq = walkforward_calibrated_probs(X, y_qqq)

    # Output table
    out = pd.DataFrame(
        {
            "p_crash_spy_3m": p_spy,
            "p_crash_qqq_3m": p_qqq,
            "y_spy": y_spy,
            "y_qqq": y_qqq,
        }
    ).dropna()

    out["risk_composite"] = 0.5 * out["p_crash_spy_3m"] + 0.5 * out["p_crash_qqq_3m"]

    # Print summaries
    print(make_summary("SPY model", out["p_crash_spy_3m"], m_spy))
    print(make_summary("QQQ model", out["p_crash_qqq_3m"], m_qqq))

    # Save for your “Bloomberg-like” terminal / dashboard
    out.to_csv("crash_risk_price_only_output.csv")
    print("Saved: crash_risk_price_only_output.csv")

    # Optional plot
        # Optional plot: composite risk with historical crash shading
    plot_composite_with_crash_shading(out, spy)

    return out


    return out



if __name__ == "__main__":
    main()

