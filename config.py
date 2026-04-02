"""
Step 1 & 2 — Problem definition + Architecture
================================================
Goal: Predict short-term (intraday) price direction for NSE stocks
      using data scraped from Screener.in, then trade using an RL agent.

Architecture:
  data_collector.py  →  feature_engineer.py  →  visualise.py
       ↓
  environment.py (Gym)  →  train_model.py
       ↓
  backtest.py  →  metrics.py  →  app.py (Flask UI)
"""

CONFIG = {
    "ticker": "reliance",          # Screener.in stock symbol (lowercase company slug)
    "lookback_days": 365,          # 1 year of daily data for training (for daily fetch)
    "resolution": "1d",          # Intraday resolution for API-based data (1m, 5m, 15m, 1h) or "1d"
    "data_source": "yfinance",    # "yfinance" | "alpha_vantage" | "screener"
    "alpha_vantage_key": "YOUR_ALPHA_VANTAGE_API_KEY",  # Set this key for intraday
    "initial_capital": 100_000,    # INR
    "transaction_cost": 0.001,     # 0.1% per trade
    "train_split": 0.8,
    "model_path": "models/ppo_hft.zip",
    "data_path": "data/prices.csv",
    "use_yfinance": True,          # Prefer yfinance direct API for live prices
}
