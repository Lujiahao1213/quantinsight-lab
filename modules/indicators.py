import pandas as pd
import numpy as np


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("Close column is required to calculate indicators.")

    out = df.copy()
    out["Daily_Return"] = out["Close"].pct_change()
    out["Daily_Return"] = out["Daily_Return"].replace([np.inf, -np.inf], np.nan)
    out["MA_5"] = out["Close"].rolling(window=5, min_periods=1).mean()
    out["MA_20"] = out["Close"].rolling(window=20, min_periods=1).mean()
    out["MA_60"] = out["Close"].rolling(window=60, min_periods=1).mean()
    out["Volatility_20"] = out["Daily_Return"].rolling(window=20, min_periods=1).std()
    return out
