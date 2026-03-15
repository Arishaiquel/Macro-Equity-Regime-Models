from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


def compute_drawdown(px: pd.Series) -> pd.Series:
    peak = px.cummax()
    return 1.0 - (px / peak)


def crash_label_forward(px: pd.Series, horizon: int, dd_thresh: float) -> pd.Series:
    dd = compute_drawdown(px)
    future_max_dd = dd.shift(-1).rolling(horizon).max().shift(-(horizon - 1))
    y = (future_max_dd >= dd_thresh).astype("float")
    return y


def realized_vol(ret: pd.Series, win: int) -> pd.Series:
    return ret.rolling(win).std() * np.sqrt(252)


def downside_vol(ret: pd.Series, win: int) -> pd.Series:
    neg = ret.where(ret < 0, 0.0)
    return neg.rolling(win).std() * np.sqrt(252)


def build_asset_features(px: pd.Series, prefix: str) -> pd.DataFrame:
    r = px.pct_change()
    df = pd.DataFrame(index=px.index)
    df[f"{prefix}_ret_1d"] = r
    df[f"{prefix}_ret_5d"] = px.pct_change(5)
    df[f"{prefix}_ret_21d"] = px.pct_change(21)
    df[f"{prefix}_vol_21d"] = realized_vol(r, 21)
    df[f"{prefix}_vol_63d"] = realized_vol(r, 63)
    df[f"{prefix}_dvol_21d"] = downside_vol(r, 21)
    df[f"{prefix}_skew_63d"] = r.rolling(63).skew()
    df[f"{prefix}_kurt_63d"] = r.rolling(63).kurt()
    df[f"{prefix}_mom_63d"] = px.pct_change(63)
    df[f"{prefix}_mom_126d"] = px.pct_change(126)
    df[f"{prefix}_dd_now"] = compute_drawdown(px)
    ma200 = px.rolling(200).mean()
    df[f"{prefix}_dist_ma200"] = (px / ma200) - 1.0
    return df


def prepare_dataset(spy: pd.Series, qqq: pd.Series, horizon: int, dd_thresh: float):
    y_spy = crash_label_forward(spy, horizon=horizon, dd_thresh=dd_thresh).rename("y_spy")
    y_qqq = crash_label_forward(qqq, horizon=horizon, dd_thresh=dd_thresh).rename("y_qqq")

    x_spy = build_asset_features(spy, "spy")
    x_qqq = build_asset_features(qqq, "qqq")

    x_cross = pd.DataFrame(index=spy.index)
    r_spy = spy.pct_change()
    r_qqq = qqq.pct_change()
    x_cross["spread_ret_1d"] = r_qqq - r_spy
    x_cross["corr_spy_qqq_63d"] = r_spy.rolling(63).corr(r_qqq)
    x_cross["vol_ratio_qqq_spy_21d"] = realized_vol(r_qqq, 21) / (realized_vol(r_spy, 21) + 1e-12)

    x = pd.concat([x_spy, x_qqq, x_cross], axis=1)
    df = pd.concat([x, y_spy, y_qqq], axis=1).dropna()

    return df[x.columns], df["y_spy"].astype(int), df["y_qqq"].astype(int)


def walk_forward_probs(x: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    base = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )
    tscv = TimeSeriesSplit(n_splits=n_splits)

    prob = pd.Series(index=x.index, dtype=float)
    for train_idx, test_idx in tscv.split(x):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train = y.iloc[train_idx]

        if y_train.nunique() < 2:
            prob.iloc[test_idx] = float(y_train.mean()) if len(y_train) else 0.0
            continue

        base.fit(x_train, y_train)
        cal = CalibratedClassifierCV(base, method="isotonic", cv=3)
        cal.fit(x_train, y_train)
        p = cal.predict_proba(x_test)[:, 1]
        prob.iloc[test_idx] = p

    out = pd.DataFrame({"y": y, "p": prob}).dropna()
    if out.empty:
        raise ValueError("Unable to produce out-of-sample probabilities")

    yy = out["y"].astype(int)
    pp = out["p"].astype(float)

    metrics = {
        "base_rate": float(yy.mean()),
        "brier": float(brier_score_loss(yy, pp)),
        "pr_auc": float(average_precision_score(yy, pp)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(yy, pp))
    except Exception:
        metrics["roc_auc"] = None

    return pp, metrics


def run_crash_risk(
    start_date: str = "2000-01-01",
    horizon: int = 63,
    drawdown_threshold: float = 0.15,
    n_splits: int = 5,
) -> dict:
    px = yf.download(["SPY", "QQQ"], start=start_date, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        raise ValueError("Expected SPY and QQQ price series")

    px = px.dropna(how="any")
    if px.empty:
        raise ValueError("No price data returned for SPY/QQQ")

    x, y_spy, y_qqq = prepare_dataset(px["SPY"], px["QQQ"], horizon, drawdown_threshold)
    p_spy, m_spy = walk_forward_probs(x, y_spy, n_splits=n_splits)
    p_qqq, m_qqq = walk_forward_probs(x, y_qqq, n_splits=n_splits)

    idx = p_spy.index.intersection(p_qqq.index)
    comp = 0.5 * p_spy.reindex(idx) + 0.5 * p_qqq.reindex(idx)

    tail = [
        {
            "date": d.date().isoformat(),
            "p_crash_spy": float(p_spy.reindex([d]).iloc[0]),
            "p_crash_qqq": float(p_qqq.reindex([d]).iloc[0]),
            "risk_composite": float(comp.reindex([d]).iloc[0]),
        }
        for d in idx[-200:]
    ]

    return {
        "latest": {
            "date": idx[-1].date().isoformat(),
            "p_crash_spy": float(p_spy.loc[idx[-1]]),
            "p_crash_qqq": float(p_qqq.loc[idx[-1]]),
            "risk_composite": float(comp.loc[idx[-1]]),
        },
        "metrics_spy": m_spy,
        "metrics_qqq": m_qqq,
        "timeseries_tail": tail,
    }
