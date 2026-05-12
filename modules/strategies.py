import pandas as pd


def _build_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def apply_strategy(df: pd.DataFrame, strategy_name: str, params: dict) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("Close column is required for strategy signals.")

    out = df.copy()
    out["Signal"] = 0
    strategy = strategy_name.lower()

    if strategy == "ma_crossover":
        short_window = int(params.get("short_window", 5))
        long_window = int(params.get("long_window", 20))
        if short_window <= 0 or long_window <= 0:
            raise ValueError("MA windows must be positive integers.")
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window.")

        out["MA_Short"] = out["Close"].rolling(window=short_window, min_periods=1).mean()
        out["MA_Long"] = out["Close"].rolling(window=long_window, min_periods=1).mean()
        out["Signal"] = (out["MA_Short"] > out["MA_Long"]).astype(int)

    elif strategy == "rsi":
        rsi_period = int(params.get("rsi_period", 14))
        oversold_threshold = float(params.get("oversold_threshold", 30))
        overbought_threshold = float(params.get("overbought_threshold", 70))
        if rsi_period <= 1:
            raise ValueError("rsi_period must be greater than 1.")
        if oversold_threshold >= overbought_threshold:
            raise ValueError("oversold_threshold must be lower than overbought_threshold.")

        out["RSI"] = _build_rsi(out["Close"], rsi_period)
        out.loc[out["RSI"] <= oversold_threshold, "Signal"] = 1
        out.loc[out["RSI"] >= overbought_threshold, "Signal"] = 0
        out["Signal"] = out["Signal"].ffill().fillna(0).astype(int)

    elif strategy == "macd":
        fast_period = int(params.get("fast_period", 12))
        slow_period = int(params.get("slow_period", 26))
        signal_period = int(params.get("signal_period", 9))
        if min(fast_period, slow_period, signal_period) <= 0:
            raise ValueError("MACD periods must be positive integers.")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period.")

        ema_fast = out["Close"].ewm(span=fast_period, adjust=False).mean()
        ema_slow = out["Close"].ewm(span=slow_period, adjust=False).mean()
        out["MACD"] = ema_fast - ema_slow
        out["MACD_Signal"] = out["MACD"].ewm(span=signal_period, adjust=False).mean()
        out["Signal"] = (out["MACD"] > out["MACD_Signal"]).astype(int)

    else:
        raise ValueError(f"Unsupported strategy '{strategy_name}'.")

    return out
