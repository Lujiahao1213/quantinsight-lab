import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_close_price_chart(df: pd.DataFrame) -> str:
    fig = px.line(df, x="Date", y="Close", title="Close Price")
    fig.update_layout(template="plotly_dark")
    return fig.to_html(full_html=False)


def create_volume_chart(df: pd.DataFrame) -> str:
    chart_df = df.copy()
    chart_df["Date"] = pd.to_datetime(chart_df["Date"], errors="coerce")
    chart_df = chart_df.dropna(subset=["Date"])

    if len(chart_df) > 500:
        monthly_df = (
            chart_df.set_index("Date")["Volume"]
            .resample("ME")
            .sum()
            .reset_index()
        )
        fig = px.bar(monthly_df, x="Date", y="Volume", title="Monthly Volume")
    else:
        fig = px.bar(chart_df, x="Date", y="Volume", title="Volume")
    fig.update_layout(template="plotly_dark")
    return fig.to_html(full_html=False)


def create_return_histogram(df: pd.DataFrame) -> str:
    returns = df["Daily_Return"].dropna()
    fig = px.histogram(returns, nbins=40, title="Daily Return Distribution")
    fig.update_layout(template="plotly_dark", xaxis_title="Daily Return")
    return fig.to_html(full_html=False)


def create_moving_average_chart(df: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_5"], name="MA 5"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_20"], name="MA 20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA_60"], name="MA 60"))
    fig.update_layout(title="Close with Moving Averages", template="plotly_dark")
    return fig.to_html(full_html=False)


def create_price_with_signals_chart(df: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", mode="lines"))

    prev_signal = df["Signal"].shift(1).fillna(0)
    buy_points = df[(df["Signal"] == 1) & (prev_signal == 0)]
    sell_points = df[(df["Signal"] == 0) & (prev_signal == 1)]

    fig.add_trace(
        go.Scatter(
            x=buy_points["Date"],
            y=buy_points["Close"],
            mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up", size=10, color="#22c55e"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sell_points["Date"],
            y=sell_points["Close"],
            mode="markers",
            name="Sell",
            marker=dict(symbol="triangle-down", size=10, color="#ef4444"),
        )
    )

    fig.update_layout(title="Price with Buy/Sell Signals", template="plotly_dark")
    return fig.to_html(full_html=False)


def create_strategy_equity_chart(df: pd.DataFrame) -> str:
    fig = px.line(df, x="Date", y="Strategy_Equity", title="Strategy Equity Curve")
    fig.update_layout(template="plotly_dark")
    return fig.to_html(full_html=False)


def create_drawdown_chart(df: pd.DataFrame) -> str:
    fig = px.area(df, x="Date", y="Drawdown", title="Drawdown Curve")
    fig.update_layout(template="plotly_dark", yaxis_tickformat=".2%")
    return fig.to_html(full_html=False)


def create_strategy_vs_market_chart(df: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Market_Equity"], name="Market Equity"))
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Strategy_Equity"], name="Strategy Equity")
    )
    fig.update_layout(title="Strategy vs Market Equity", template="plotly_dark")
    return fig.to_html(full_html=False)


def create_strategy_comparison_equity_chart(df: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA"], name="Moving Average"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD"))
    fig.update_layout(title="Strategy Equity Comparison", template="plotly_dark")
    return fig.to_html(full_html=False)


def create_strategy_comparison_metrics_chart(df: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Strategy"], y=df["Total Return"], name="Total Return"))
    fig.add_trace(go.Bar(x=df["Strategy"], y=df["Sharpe Ratio"], name="Sharpe Ratio"))
    fig.add_trace(go.Bar(x=df["Strategy"], y=df["Max Drawdown"], name="Max Drawdown"))
    fig.update_layout(
        barmode="group",
        title="Strategy Metrics Comparison",
        template="plotly_dark",
    )
    return fig.to_html(full_html=False)
