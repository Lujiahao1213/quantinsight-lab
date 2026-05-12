import math

import numpy as np
import pandas as pd


def _safe_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return round(parsed, 6) if math.isfinite(parsed) else "N/A"


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
) -> tuple[pd.DataFrame, dict]:
    if "Close" not in df.columns or "Signal" not in df.columns:
        raise ValueError("Backtest requires Close and Signal columns.")

    out = df.copy()
    out["Market_Return"] = out["Close"].pct_change()
    out["Market_Return"] = (
        out["Market_Return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    shifted_signal = out["Signal"].shift(1).fillna(0).astype(float)
    trade_change = out["Signal"].diff().abs().fillna(out["Signal"].abs()).astype(float)
    out["Strategy_Return"] = (
        shifted_signal * out["Market_Return"] - trade_change * float(transaction_cost)
    )
    out["Strategy_Return"] = (
        out["Strategy_Return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    out["Market_Equity"] = float(initial_capital) * (1 + out["Market_Return"]).cumprod()
    out["Strategy_Equity"] = float(initial_capital) * (1 + out["Strategy_Return"]).cumprod()
    out["Market_Equity"] = (
        out["Market_Equity"].replace([np.inf, -np.inf], np.nan).ffill().fillna(float(initial_capital))
    )
    out["Strategy_Equity"] = (
        out["Strategy_Equity"].replace([np.inf, -np.inf], np.nan).ffill().fillna(float(initial_capital))
    )
    out["Drawdown"] = out["Strategy_Equity"] / out["Strategy_Equity"].cummax() - 1
    out["Drawdown"] = out["Drawdown"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    periods = max(len(out), 1)
    annual_factor = 252
    final_equity = out["Strategy_Equity"].iloc[-1]
    total_return = final_equity / float(initial_capital) - 1
    annualized_return = (
        (final_equity / float(initial_capital)) ** (annual_factor / periods) - 1
        if final_equity > 0
        else np.nan
    )
    annualized_vol = out["Strategy_Return"].std(ddof=0) * math.sqrt(annual_factor)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0.0
    max_drawdown = out["Drawdown"].min()

    active_returns = out.loc[shifted_signal > 0, "Strategy_Return"]
    win_rate = (active_returns > 0).mean() if len(active_returns) > 0 else 0.0
    number_of_trades = int((trade_change > 0).sum())

    metrics = {
        "total_return": _safe_float(total_return),
        "annualized_return": _safe_float(annualized_return),
        "annualized_volatility": _safe_float(annualized_vol),
        "sharpe_ratio": _safe_float(sharpe),
        "max_drawdown": _safe_float(max_drawdown),
        "win_rate": _safe_float(win_rate),
        "number_of_trades": number_of_trades,
    }

    return out, metrics
