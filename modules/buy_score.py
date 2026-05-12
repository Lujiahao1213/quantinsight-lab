"""
Rule-based Buy Score from technical indicators (latest bar).
Returns JSON-serializable dicts for Flask/API use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REQUIRED_OHLCV = ("Close", "Volume")

VOLUME_AVG_WINDOW = 20
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def _json_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 6)


def _build_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _build_macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.sort_values("Date", ascending=True).reset_index(drop=True)
    for col in REQUIRED_OHLCV:
        if col not in out.columns:
            raise ValueError(f"DataFrame must include '{col}' for buy score.")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    close = out["Close"]
    out["RSI"] = _build_rsi(close, RSI_PERIOD)
    out["MA10"] = close.rolling(window=10, min_periods=10).mean()
    out["MA20"] = close.rolling(window=20, min_periods=20).mean()
    out["MA50"] = close.rolling(window=50, min_periods=50).mean()
    out["MA60"] = close.rolling(window=60, min_periods=60).mean()
    vol = out["Volume"].fillna(0)
    out["Volume_MA"] = vol.rolling(window=VOLUME_AVG_WINDOW, min_periods=VOLUME_AVG_WINDOW).mean()

    dif, dea, hist = _build_macd(close)
    out["MACD_DIF"] = dif
    out["MACD_DEA"] = dea
    out["MACD_HIST"] = hist

    return out


def _recommendation_for_score(score: float) -> str:
    if score >= 80:
        return "Strong Buy"
    if score >= 60:
        return "Buy"
    if score >= 40:
        return "Neutral"
    if score >= 20:
        return "Weak"
    return "Avoid"


def _append_breakdown(
    breakdown: Dict[str, List[Dict[str, Any]]],
    category: str,
    rule: str,
    points: int,
) -> None:
    breakdown.setdefault(category, []).append(
        {
            "rule": rule,
            "points": points,
            "direction": "positive" if points > 0 else ("negative" if points < 0 else "neutral"),
        }
    )


def _score_rsi(rsi: float, breakdown: Dict[str, List[Dict[str, Any]]]) -> int:
    total = 0
    if rsi > 70:
        total -= 20
        _append_breakdown(breakdown, "rsi", "RSI > 70 (overbought)", -20)
    elif rsi < 30:
        total += 30
        _append_breakdown(breakdown, "rsi", "RSI < 30 (oversold)", 30)
    elif 30 <= rsi < 40:
        total += 20
        _append_breakdown(breakdown, "rsi", "RSI 30-40", 20)
    elif 40 <= rsi <= 60:
        total += 10
        _append_breakdown(breakdown, "rsi", "RSI 40-60", 10)
    return total


def _score_macd(
    dif_c: float,
    dea_c: float,
    dif_p: float,
    dea_p: float,
    hist_c: float,
    hist_p: float,
    breakdown: Dict[str, List[Dict[str, Any]]],
) -> int:
    total = 0
    bull_cross = dif_c > dea_c and dif_p <= dea_p
    bear_cross = dif_c < dea_c and dif_p >= dea_p

    if bull_cross:
        total += 25
        _append_breakdown(breakdown, "macd", "MACD bullish crossover (DIF crosses above DEA)", 25)
    if bear_cross:
        total -= 20
        _append_breakdown(breakdown, "macd", "MACD bearish crossover (DIF crosses below DEA)", -20)

    if dif_c > dea_c:
        total += 15
        _append_breakdown(breakdown, "macd", "DIF > DEA", 15)

    if hist_c > hist_p:
        total += 10
        _append_breakdown(breakdown, "macd", "MACD histogram increasing vs prior bar", 10)

    return total


def _score_ma(
    close: float,
    ma10: float,
    ma20: float,
    ma50: float,
    ma60: float,
    breakdown: Dict[str, List[Dict[str, Any]]],
) -> int:
    total = 0
    if not (np.isnan(ma10) or np.isnan(ma50)) and ma10 > ma50:
        total += 25
        _append_breakdown(breakdown, "moving_average", "MA10 > MA50", 25)

    if not np.isnan(ma20) and close > ma20:
        total += 10
        _append_breakdown(breakdown, "moving_average", "Price > MA20", 10)

    if not np.isnan(ma60) and close < ma60:
        total -= 20
        _append_breakdown(breakdown, "moving_average", "Price below MA60", -20)

    return total


def _score_volume(
    close_c: float,
    close_p: float,
    vol_c: float,
    vol_p: float,
    vol_ma: float,
    breakdown: Dict[str, List[Dict[str, Any]]],
) -> int:
    total = 0
    if not np.isnan(vol_ma) and vol_ma > 0 and vol_c > 1.5 * vol_ma:
        total += 20
        _append_breakdown(breakdown, "volume", "Volume > 1.5× average volume", 20)

    if close_c > close_p and vol_c > vol_p:
        total += 15
        _append_breakdown(breakdown, "volume", "Rising price with rising volume", 15)

    if close_c < close_p and not np.isnan(vol_ma) and vol_c < vol_ma:
        total -= 10
        _append_breakdown(breakdown, "volume", "Falling price with weak volume (below avg)", -10)

    return total


def compute_buy_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute aggregate buy score and recommendation from the latest row of OHLCV data.

    Returns a JSON-friendly dict with buy_score, recommendation, breakdown by category,
    and positive / negative signal lists.
    """
    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        return {
            "ok": False,
            "buy_score": None,
            "recommendation": "N/A",
            "error": f"Missing columns: {', '.join(missing)}",
            "breakdown": {},
            "positive_signals": [],
            "negative_signals": [],
            "indicator_snapshot": {},
        }

    if len(df) < 2:
        return {
            "ok": False,
            "buy_score": None,
            "recommendation": "N/A",
            "error": "Need at least 2 rows to score MACD crossovers and volume trend.",
            "breakdown": {},
            "positive_signals": [],
            "negative_signals": [],
            "indicator_snapshot": {},
        }

    prep = _prepare_frame(df)
    last = prep.iloc[-1]
    prev = prep.iloc[-2]

    rsi = last.get("RSI")
    if pd.isna(rsi):
        return {
            "ok": False,
            "buy_score": None,
            "recommendation": "N/A",
            "error": "RSI is not available yet (insufficient history for RSI period).",
            "breakdown": {},
            "positive_signals": [],
            "negative_signals": [],
            "indicator_snapshot": {},
        }

    breakdown: Dict[str, List[Dict[str, Any]]] = {}
    total = 0
    total += _score_rsi(float(rsi), breakdown)

    dif_c = float(last["MACD_DIF"])
    dea_c = float(last["MACD_DEA"])
    dif_p = float(prev["MACD_DIF"])
    dea_p = float(prev["MACD_DEA"])
    hist_c = float(last["MACD_HIST"])
    hist_p = float(prev["MACD_HIST"])
    if any(map(lambda x: pd.isna(x), (dif_c, dea_c, dif_p, dea_p, hist_c, hist_p))):
        _append_breakdown(breakdown, "macd", "MACD not fully available on last bars", 0)
    else:
        total += _score_macd(dif_c, dea_c, dif_p, dea_p, hist_c, hist_p, breakdown)

    close_c = float(last["Close"])
    close_p = float(prev["Close"])
    ma10, ma20, ma50, ma60 = (
        float(last["MA10"]),
        float(last["MA20"]),
        float(last["MA50"]),
        float(last["MA60"]),
    )
    total += _score_ma(close_c, ma10, ma20, ma50, ma60, breakdown)

    vol_c = float(last["Volume"]) if pd.notna(last["Volume"]) else 0.0
    vol_p = float(prev["Volume"]) if pd.notna(prev["Volume"]) else 0.0
    vol_ma = float(last["Volume_MA"]) if pd.notna(last["Volume_MA"]) else float("nan")
    total += _score_volume(close_c, close_p, vol_c, vol_p, vol_ma, breakdown)

    positive_signals: List[str] = []
    negative_signals: List[str] = []
    for _cat, items in breakdown.items():
        for it in items:
            pts = it["points"]
            label = f"{it['rule']}: {pts:+d}" if pts != 0 else it["rule"]
            if pts > 0:
                positive_signals.append(label)
            elif pts < 0:
                negative_signals.append(label)

    as_of = None
    if "Date" in prep.columns and pd.notna(last.get("Date")):
        as_of = str(pd.Timestamp(last["Date"]).date())

    indicator_snapshot = {
        "rsi": _json_float(rsi),
        "macd_dif": _json_float(last.get("MACD_DIF")),
        "macd_dea": _json_float(last.get("MACD_DEA")),
        "macd_hist": _json_float(last.get("MACD_HIST")),
        "ma10": _json_float(last.get("MA10")),
        "ma20": _json_float(last.get("MA20")),
        "ma50": _json_float(last.get("MA50")),
        "ma60": _json_float(last.get("MA60")),
        "close": _json_float(last.get("Close")),
        "volume": _json_float(last.get("Volume")),
        "volume_ma20": _json_float(last.get("Volume_MA")),
    }

    return {
        "ok": True,
        "buy_score": int(round(total)),
        "recommendation": _recommendation_for_score(total),
        "as_of_date": as_of,
        "breakdown": breakdown,
        "positive_signals": positive_signals,
        "negative_signals": negative_signals,
        "indicator_snapshot": indicator_snapshot,
        "error": None,
    }
