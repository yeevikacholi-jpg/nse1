"""
Step 4 — Data Cleaning / Processing / Visualisation
=====================================================
Cleans raw OHLCV data, engineers technical features, and produces
interactive Plotly charts for EDA.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from config import CONFIG


# ── Cleaning ──────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna()
    # Remove zero-price rows
    df = df[(df["Close"] > 0) & (df["Volume"] > 0)]
    # Forward-fill any remaining gaps
    df = df.ffill()
    print(f"[feature_engineer] Clean rows: {len(df)}")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Trend indicators
    df["sma_10"]  = ta.trend.sma_indicator(df["Close"], window=10)
    df["sma_30"]  = ta.trend.sma_indicator(df["Close"], window=30)
    df["ema_12"]  = ta.trend.ema_indicator(df["Close"], window=12)
    df["ema_26"]  = ta.trend.ema_indicator(df["Close"], window=26)
    df["macd"]    = ta.trend.macd(df["Close"])
    df["macd_sig"]= ta.trend.macd_signal(df["Close"])

    # Momentum
    df["rsi"]     = ta.momentum.rsi(df["Close"], window=14)

    # Volatility
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    df["bb_upper"]= bb.bollinger_hband()
    df["bb_lower"]= bb.bollinger_lband()
    df["bb_width"]= (df["bb_upper"] - df["bb_lower"]) / df["Close"]

    # Volume
    df["vwap"]    = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3
                     ).cumsum() / df["Volume"].cumsum()
    df["obv"]     = ta.volume.on_balance_volume(df["Close"], df["Volume"])

    # Returns
    df["ret_1d"]  = df["Close"].pct_change(1)
    df["ret_5d"]  = df["Close"].pct_change(5)

    # Target: 1 if next-day return > 0 else 0
    df["target"]  = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna()
    return df


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_charts(df: pd.DataFrame, ticker: str) -> None:
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        vertical_spacing=0.03,
        subplot_titles=(
            f"{ticker} — Candlestick + Bollinger Bands",
            "Volume",
            "RSI",
            "MACD",
        ),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
    ), row=1, col=1)

    for col, color, name in [
        ("bb_upper", "rgba(255,100,100,0.4)", "BB Upper"),
        ("bb_lower", "rgba(100,100,255,0.4)", "BB Lower"),
        ("sma_30",   "orange",                "SMA 30"),
        ("vwap",     "green",                 "VWAP"),
    ]:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df[col], line=dict(color=color, width=1),
            name=name,
        ), row=1, col=1)

    # Volume
    colors = ["#ef5350" if o > c else "#26a69a"
              for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"], marker_color=colors, name="Volume",
    ), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["rsi"], line=dict(color="purple", width=1.2),
        name="RSI",
    ), row=3, col=1)
    for level, color in [(70, "red"), (30, "green")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, row=3, col=1)

    # MACD
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["macd"], line=dict(color="blue", width=1), name="MACD",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["macd_sig"], line=dict(color="orange", width=1), name="Signal",
    ), row=4, col=1)

    fig.update_layout(
        height=800, title=f"{ticker} — Technical Analysis",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation="h", y=1.02),
    )

    fig.write_html("data/chart.html")
    print("[visualise] Chart saved to data/chart.html  — open in a browser.")


def run_feature_engineering() -> pd.DataFrame:
    df_raw   = pd.read_csv(CONFIG["data_path"])
    df_clean = clean_data(df_raw)
    df_feat  = add_features(df_clean)
    df_feat.to_csv("data/features.csv", index=False)
    plot_charts(df_feat, CONFIG["ticker"])
    print(df_feat[["Date","Close","rsi","macd","target"]].tail())
    return df_feat


if __name__ == "__main__":
    run_feature_engineering()
