"""
Step 6 & 7 — Metrics + Backtesting
=====================================
Runs the trained PPO agent on held-out test data, records every trade,
and computes portfolio performance metrics.
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from train_model import TradingEnv, FEATURE_COLS
from config import CONFIG


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(df_test: pd.DataFrame) -> dict:
    model = PPO.load(CONFIG["model_path"])
    env   = TradingEnv(df_test)
    obs, _ = env.reset()

    portfolio_history = []
    trade_log         = []
    action_labels     = {0: "HOLD", 1: "BUY", 2: "SELL"}

    while True:
        action, _ = model.predict(obs, deterministic=True)
        prev_pos  = env.position
        obs, reward, terminated, truncated, _ = env.step(int(action))

        row = df_test.iloc[env.idx - 1]
        portfolio_history.append({
            "date":      row["Date"],
            "price":     row["Close"],
            "portfolio": env.portfolio,
            "action":    action_labels[int(action)],
        })

        # Log actual trade transitions
        if int(action) != 0 and prev_pos != env.position:
            trade_log.append({
                "date":   row["Date"],
                "action": action_labels[int(action)],
                "price":  row["Close"],
                "portfolio_value": env.portfolio,
            })

        if terminated or truncated:
            break

    return {
        "portfolio_history": pd.DataFrame(portfolio_history),
        "trade_log":         pd.DataFrame(trade_log),
        "final_value":       env.portfolio,
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(result: dict) -> dict:
    ph  = result["portfolio_history"]
    prices = ph["price"] if "price" in ph.columns else None

    ret = ph["portfolio"].pct_change().dropna()

    # Strategy metrics (agent path): default to 0 if no action-based portfolio changes
    sharpe      = 0.0
    max_dd      = 0.0
    if not ret.empty and not np.isclose(ret.std(), 0.0):
        sharpe = (ret.mean() / (ret.std() + 1e-8)) * np.sqrt(252)

    rolling_max = ph["portfolio"].cummax() if not ph.empty else pd.Series([])
    if not ph.empty:
        drawdown = (ph["portfolio"] - rolling_max) / (rolling_max + 1e-8)
        max_dd = drawdown.min()

    initial      = CONFIG["initial_capital"]
    final        = result["final_value"]
    total_r      = (final - initial) / initial * 100

    # Benchmark: buy-and-hold from price data (for comparison / fallback)
    buy_hold_r   = 0.0
    bh_drawdown  = 0.0
    bh_sharpe    = 0.0
    if prices is not None and len(prices) > 1:
        bh_ret = prices.pct_change().dropna()
        buy_hold_r = (prices.iloc[-1] / prices.iloc[0] - 1.0) * 100
        if not bh_ret.empty and not np.isclose(bh_ret.std(), 0.0):
            bh_sharpe = (bh_ret.mean() / (bh_ret.std() + 1e-8)) * np.sqrt(252)

        rolling_max_bh = prices.cummax()
        bh_dd = (prices - rolling_max_bh) / (rolling_max_bh + 1e-8)
        bh_drawdown = bh_dd.min() * 100

    trades = result["trade_log"]
    n_round_trades = 0
    win_rate = 0.0
    if "action" in trades.columns and not trades.empty:
        buy_sell = trades[trades["action"].isin(["BUY", "SELL"])].reset_index(drop=True)
        wins = 0
        for i in range(0, len(buy_sell) - 1, 2):
            buy_p = buy_sell.iloc[i]["price"]
            sell_p = buy_sell.iloc[i + 1]["price"]
            if sell_p > buy_p:
                wins += 1
        n_round_trades = len(buy_sell) // 2
        win_rate = (wins / n_round_trades * 100) if n_round_trades > 0 else 0.0

    metrics = {
        "Initial capital (INR)":  f"₹{initial:,.0f}",
        "Final portfolio (INR)":  f"₹{final:,.0f}",
        "Total return (%)":       f"{total_r:.2f}%",
        "Sharpe ratio":           f"{sharpe:.2f}",
        "Max drawdown (%)":       f"{abs(max_dd) * 100:.2f}%",
        "Total trades":           n_round_trades,
        "Win rate (%)":           f"{win_rate:.1f}%",
        "Buy & Hold return (%)": f"{buy_hold_r:.2f}%",
        "Buy & Hold sharpe":     f"{bh_sharpe:.2f}",
        "Buy & Hold drawdown":    f"{abs(bh_drawdown):.2f}%",
    }

    return metrics


if __name__ == "__main__":
    df = pd.read_csv("data/features.csv")
    split    = int(len(df) * CONFIG["train_split"])
    df_test  = df.iloc[split:].copy().reset_index(drop=True)

    result  = backtest(df_test)
    metrics = compute_metrics(result)

    print("\n── Backtest Metrics ──────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<30} {v}")

    result["portfolio_history"].to_csv("data/backtest_results.csv", index=False)
    result["trade_log"].to_csv("data/trade_log.csv", index=False)
    print("\n[backtest] Results saved to data/")
