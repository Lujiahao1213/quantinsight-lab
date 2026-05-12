"""
Random Forest classifier: probability that Close is higher 5 trading days ahead.
For research and educational use only.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLUMNS: List[str] = [
    "Daily_Return",
    "RSI_14",
    "MACD_DIF",
    "MACD_DEA",
    "MACD_Hist",
    "MA10",
    "MA20",
    "MA50",
    "MA60",
    "Volume_Ratio",
    "Volatility_20",
]

FORWARD_DAYS = 5
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_MA_WINDOW = 20
VOLATILITY_WINDOW = 20


def _rf_failure(
    error: str,
    *,
    test_accuracy: Optional[float] = None,
    n_train: int = 0,
    n_test: int = 0,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": error,
        "prediction_latest": None,
        "probability_up": None,
        "probability_down": None,
        "test_accuracy": _json_float(test_accuracy) if test_accuracy is not None else None,
        "feature_importance": [],
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "recommendation": None,
        "strong_rise_probability": None,
        "sharp_drop_risk": None,
        "strong_rise_accuracy": None,
        "sharp_drop_accuracy": None,
    }


def _json_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, 6)


def _build_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _build_macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def _recommendation(up_probability: float) -> str:
    if up_probability >= 0.65:
        return "Bullish"
    if up_probability >= 0.55:
        return "Slightly Bullish"
    if up_probability >= 0.45:
        return "Neutral"
    return "Bearish"


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns or "Volume" not in df.columns:
        raise ValueError("Close and Volume columns are required for the RF predictor.")

    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.sort_values("Date", ascending=True).reset_index(drop=True)

    close = pd.to_numeric(out["Close"], errors="coerce")
    volume = pd.to_numeric(out["Volume"], errors="coerce").clip(lower=0)

    out["Daily_Return"] = close.pct_change()
    out["Daily_Return"] = out["Daily_Return"].replace([np.inf, -np.inf], np.nan)

    out["RSI_14"] = _build_rsi(close, RSI_PERIOD)

    dif, dea, hist = _build_macd(close)
    out["MACD_DIF"] = dif
    out["MACD_DEA"] = dea
    out["MACD_Hist"] = hist

    out["MA10"] = close.rolling(window=10, min_periods=10).mean()
    out["MA20"] = close.rolling(window=20, min_periods=20).mean()
    out["MA50"] = close.rolling(window=50, min_periods=50).mean()
    out["MA60"] = close.rolling(window=60, min_periods=60).mean()

    vol_ma = volume.rolling(window=VOLUME_MA_WINDOW, min_periods=VOLUME_MA_WINDOW).mean()
    out["Volume_Ratio"] = volume / vol_ma.replace(0, np.nan)

    out["Volatility_20"] = out["Daily_Return"].rolling(
        window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW
    ).std()

    future = close.shift(-FORWARD_DAYS)
    out["Target"] = (future > close).astype("Int64")
    return out


def run_rf_direction_predictor(
    df: pd.DataFrame,
    *,
    n_estimators: int = 150,
    max_depth: Optional[int] = 12,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train RandomForestClassifier to predict Up (1) if Close in +5 sessions is above current Close.

    Returns a JSON-serializable dict with ok flag, metrics, latest prediction, and feature importance.
    """
    prep = _prepare_frame(df)
    modeling = prep.dropna(subset=FEATURE_COLUMNS + ["Target"]).copy()
    modeling["Target"] = modeling["Target"].astype(int)

    min_rows = 80
    if len(modeling) < min_rows:
        return _rf_failure(
            f"Need at least {min_rows} complete rows after building features and targets "
            f"(got {len(modeling)}). Try a longer price history."
        )

    if modeling["Target"].nunique() < 2:
        return _rf_failure("Target has only one class after preparation; cannot train classifier.")

    X = modeling[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = modeling["Target"].to_numpy()

    split_idx = int(len(X) * 0.8)
    if split_idx < 20 or len(X) - split_idx < 5:
        return _rf_failure("Not enough rows for an 80/20 chronological train/test split.")

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    test_accuracy = float(np.mean(y_pred == y_test))

    mask_up = y_test == 1
    mask_dn = y_test == 0
    strong_rise_accuracy = (
        float(np.mean(y_pred[mask_up] == y_test[mask_up])) if np.any(mask_up) else None
    )
    sharp_drop_accuracy = (
        float(np.mean(y_pred[mask_dn] == y_test[mask_dn])) if np.any(mask_dn) else None
    )

    classes = list(clf.classes_)
    idx_down = classes.index(0) if 0 in classes else 0
    idx_up = classes.index(1) if 1 in classes else 1

    # Latest row with full features (may or may not have Target in modeling set)
    feature_only = prep.dropna(subset=FEATURE_COLUMNS)
    if feature_only.empty:
        return _rf_failure(
            "No rows with complete features for prediction.",
            test_accuracy=test_accuracy,
            n_train=int(split_idx),
            n_test=int(len(X_test)),
        )

    x_latest = feature_only[FEATURE_COLUMNS].iloc[[-1]].to_numpy(dtype=np.float64)
    latest_pred = int(clf.predict(x_latest)[0])
    latest_proba = clf.predict_proba(x_latest)[0]
    p_down = float(latest_proba[idx_down])
    p_up = float(latest_proba[idx_up])

    importance = [
        {"feature": name, "importance": _json_float(imp)}
        for name, imp in zip(FEATURE_COLUMNS, clf.feature_importances_)
    ]
    importance.sort(key=lambda r: (r["importance"] or 0), reverse=True)

    as_of = None
    if "Date" in feature_only.columns and pd.notna(feature_only["Date"].iloc[-1]):
        as_of = str(pd.Timestamp(feature_only["Date"].iloc[-1]).date())

    return {
        "ok": True,
        "error": None,
        "prediction_latest": latest_pred,
        "probability_up": _json_float(p_up),
        "probability_down": _json_float(p_down),
        "strong_rise_probability": _json_float(p_up),
        "sharp_drop_risk": _json_float(p_down),
        "test_accuracy": _json_float(test_accuracy),
        "strong_rise_accuracy": _json_float(strong_rise_accuracy),
        "sharp_drop_accuracy": _json_float(sharp_drop_accuracy),
        "feature_importance": importance,
        "n_train_rows": int(split_idx),
        "n_test_rows": int(len(X_test)),
        "recommendation": _recommendation(p_up),
        "as_of_date": as_of,
        "forward_days": FORWARD_DAYS,
        "target_definition": (
            f"Target=1 if Close {FORWARD_DAYS} trading sessions ahead is strictly "
            "greater than current Close; else 0."
        ),
    }
