"""
Step 8 — UI (Flask Web App)
============================
Serves a web dashboard showing:
 - Live portfolio equity curve (Plotly)
 - Trade log table
 - Key metrics cards
 - Controls to re-run backtest
"""

import json
import pandas as pd
from flask import Flask, jsonify, render_template_string, request
from backtest import backtest, compute_metrics
from config import CONFIG
from data_collector import collect_data
from feature_engineer import run_feature_engineering
from train_model import run_training

app = Flask(__name__)

# Keep tracker of last loaded ticker so results endpoint always refreshes on ticker change
_last_ticker_processed = None

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HFT Strategy Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #0f0f13; color: #e2e2e8; }
    header { padding: 1.2rem 2rem; border-bottom: 1px solid #2a2a35; display: flex; align-items: center; gap: 12px; }
    header h1 { font-size: 18px; font-weight: 500; }
    header .ticker { background: #1d9e75; color: #fff; font-size: 12px; padding: 3px 10px; border-radius: 6px; }
    .metrics { display: flex; flex-wrap: wrap; gap: 12px; padding: 1.5rem 2rem 0; }
    .metric-card { background: #1a1a24; border: 1px solid #2a2a35; border-radius: 10px; padding: 14px 18px; min-width: 150px; flex: 1; }
    .metric-card .label { font-size: 11px; color: #888; margin-bottom: 6px; }
    .metric-card .value { font-size: 20px; font-weight: 500; }
    .positive { color: #1d9e75; }
    .negative { color: #e24b4a; }
    .chart-section { padding: 1.5rem 2rem; }
    .chart-section h2 { font-size: 14px; font-weight: 500; color: #888; margin-bottom: 12px; }
    #equity-chart { border-radius: 10px; overflow: hidden; }
    .trade-table-wrap { padding: 0 2rem 2rem; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th { text-align: left; padding: 8px 12px; background: #1a1a24; color: #888; font-weight: 400; border-bottom: 1px solid #2a2a35; }
    td { padding: 8px 12px; border-bottom: 1px solid #1e1e28; }
    tr:hover td { background: #1a1a24; }
    .badge { font-size: 11px; padding: 2px 8px; border-radius: 5px; }
    .buy  { background: #0a3d2e; color: #1d9e75; }
    .sell { background: #3d1212; color: #e24b4a; }
    .controls { padding: 0 2rem 1rem; display: flex; gap: 10px; }
    button { background: #1a1a24; border: 1px solid #2a2a35; color: #e2e2e8; padding: 8px 18px; border-radius: 8px; cursor: pointer; font-size: 13px; }
    button:hover { background: #25253a; }
    footer { text-align: center; padding: 1rem; color: #444; font-size: 11px; border-top: 1px solid #1e1e28; }
  </style>
</head>
<body>
  <header>
    <h1>HFT Strategy Dashboard</h1>
    <span class="ticker" id="ticker-label">{{ ticker }}</span>
  </header>

  <div class="controls" style="padding: 1rem 2rem; gap: 10px;">
    <label for="company-select" style="color:#ccc;font-size:12px;">Select Company:</label>
    <select id="company-select" style="padding: 7px 10px; border-radius:8px; background:#1a1a24; color:#e2e2e8; border:1px solid #2a2a35">
      <option value="reliance">Reliance</option>
      <option value="tcs">TCS</option>
      <option value="infy">Infosys</option>
      <option value="hdfc">HDFC</option>
      <option value="sbin">SBI</option>
      <option value="icicibank">ICICI Bank</option>
    </select>
    <button onclick="loadCompany()">Load Company</button>
    <button onclick="loadData(false)">Refresh</button>
  </div>

  <div class="metrics" id="metrics-area">
    <div class="metric-card"><div class="label">Loading...</div></div>
  </div>

  <div class="chart-section">
    <h2>Candlestick Price Chart</h2>
    <div id="price-chart" style="height:360px;"></div>
  </div>

  <div class="chart-section">
    <h2>Portfolio equity curve</h2>
    <div id="equity-chart" style="height:340px;"></div>
  </div>

  <div class="chart-section" style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
    <div><h2>RSI</h2><div id="rsi-chart" style="height:240px;"></div></div>
    <div><h2>Volume</h2><div id="volume-chart" style="height:240px;"></div></div>
  </div>

  <div class="trade-table-wrap">
    <h2 style="font-size:14px;font-weight:500;color:#888;margin-bottom:12px;">Trade log</h2>
    <table>
      <thead><tr><th>Date</th><th>Action</th><th>Price (₹)</th><th>Portfolio (₹)</th></tr></thead>
      <tbody id="trade-tbody"></tbody>
    </table>
  </div>

  <footer>Simulated backtest results — not financial advice</footer>

  <script>
    async function loadCompany() {
      const ticker = document.getElementById("company-select").value;
      document.getElementById("ticker-label").innerText = ticker.toUpperCase();
      await fetch(`/api/load?ticker=${ticker}`);
      await loadData();
    }

    async function loadData() {
      const selected = document.getElementById("company-select").value;
      const res = await fetch(`/api/results?ticker=${selected}`);
      const data = await res.json();

      // Error handling
      if (!res.ok || data.status === "error") {
        const area = document.getElementById("metrics-area");
        area.innerHTML = `<div class="metric-card"><div class="label">Error</div><div class="value negative">${data.error || "Unable to load data"}</div></div>`;
        document.getElementById("price-chart").innerHTML = "";
        document.getElementById("equity-chart").innerHTML = "";
        document.getElementById("rsi-chart").innerHTML = "";
        document.getElementById("volume-chart").innerHTML = "";
        document.getElementById("trade-tbody").innerHTML = "";
        return;
      }

      // Metrics
      const area = document.getElementById("metrics-area");
      area.innerHTML = "";
      for (const [k, v] of Object.entries(data.metrics)) {
        const cls = v.toString().startsWith("-") ? "negative" : "positive";
        area.innerHTML += `<div class="metric-card"><div class="label">${k}</div><div class="value ${cls}">${v}</div></div>`;
      }

      // Price / candlestick chart
      const chart = data.chart_data;
      const dates = chart.map(r => r.Date);
      Plotly.newPlot("price-chart", [
        { x: dates, y: chart.map(r => r.High), type: "scatter", mode: "lines", name: "High", line: { color: "#5ad" } },
        { x: dates, y: chart.map(r => r.Low), type: "scatter", mode: "lines", name: "Low", line: { color: "#a55" } },
        { x: dates, y: chart.map(r => r.Close), type: "scatter", mode: "lines", name: "Close", line: { color: "#1d9e75" } }
      ], {
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", 
        margin: { t: 5, b: 30, l: 50, r: 5 },
        xaxis: { gridcolor: "#1e1e28" },
        yaxis: { gridcolor: "#1e1e28", tickprefix: "₹" }
      }, { responsive: true, displayModeBar: false });

      // Equity chart
      const ph = data.portfolio_history;
      Plotly.newPlot("equity-chart", [
        { x: ph.map(r => r.date), y: ph.map(r => r.portfolio), type: "scatter", mode: "lines", name: "Portfolio", line: { color: "#1d9e75", width: 2 }, fill: "tozeroy", fillcolor: "rgba(29,158,117,0.08)" },
        { x: ph.map(r => r.date), y: ph.map(r => r.price * data.initial_capital / ph[0]?.price), type: "scatter", mode: "lines", name: "Buy & Hold", line: { color: "#5a5a7a", width: 1.5, dash: "dot" } }
      ], {
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: "#888", size: 11 },
        xaxis: { gridcolor: "#1e1e28", zeroline: false }, yaxis: { gridcolor: "#1e1e28", zeroline: false, tickprefix: "₹" },
        margin: { t: 10, b: 40, l: 70, r: 10 }, legend: { x: 0.01, y: 0.98, bgcolor: "rgba(0,0,0,0)" }, hovermode: "x unified"
      }, { responsive: true, displayModeBar: false });

      // RSI chart
      Plotly.newPlot("rsi-chart", [
        { x: dates, y: chart.map(r => r.rsi), type: "scatter", mode: "lines", name: "RSI", line: { color: "#d19dff" } }
      ], { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", yaxis: { range: [0, 100] }, margin: { t: 5, b: 30, l: 40, r: 5 }}, { responsive: true, displayModeBar: false });

      // Volume chart
      Plotly.newPlot("volume-chart", [
        { x: dates, y: chart.map(r => r.Volume), type: "bar", name: "Volume", marker: { color: "#4f5f9f" } }
      ], { paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { t: 5, b: 30, l: 40, r: 5 }}, { responsive: true, displayModeBar: false });

      // Trade log
      const tbody = document.getElementById("trade-tbody");
      tbody.innerHTML = data.trade_log.map(t => `
        <tr>
          <td>${t.date}</td>
          <td><span class="badge ${t.action.toLowerCase()}">${t.action}</span></td>
          <td>₹${parseFloat(t.price).toFixed(2)}</td>
          <td>₹${parseFloat(t.portfolio_value).toLocaleString("en-IN", {maximumFractionDigits: 0})}</td>
        </tr>`).join("");
    }

    loadData();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        HTML,
        ticker=CONFIG["ticker"],
        initial_capital=CONFIG["initial_capital"],
    )


def normalize_ticker(ticker: str) -> str:
    ticker = ticker.strip().lower()
    if ticker in ["hdfc", "hdfc bank", "hdfc-bank"]:
        return "hdfcbank"
    if ticker in ["rel", "reliance"]:
        return "reliance"
    if ticker in ["infosys", "infy"]:
        return "infy"
    return ticker


@app.route("/api/load")
def api_load():
    ticker = normalize_ticker(request.args.get("ticker", CONFIG["ticker"]))
    CONFIG["ticker"] = ticker
    try:
        collect_data(ticker)
        run_feature_engineering()
        run_training(timesteps=5000)
        return jsonify({"status": "ok", "ticker": ticker})
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/results")
def api_results():
    global _last_ticker_processed

    requested_ticker = normalize_ticker(request.args.get("ticker", CONFIG["ticker"]))
    ticker = requested_ticker

    if ticker != _last_ticker_processed:
        try:
            CONFIG["ticker"] = ticker
            collect_data(ticker)
            run_feature_engineering()
            run_training(timesteps=5000)
            _last_ticker_processed = ticker
        except Exception as exc:
            fallback = "reliance"
            if ticker != fallback:
                try:
                    ticker = fallback
                    CONFIG["ticker"] = ticker
                    collect_data(ticker)
                    run_feature_engineering()
                    run_training(timesteps=5000)
                    _last_ticker_processed = ticker
                except Exception as fallback_exc:
                    return jsonify({"status": "error", "error": f"{exc}; fallback failed: {fallback_exc}"}), 500
            else:
                return jsonify({"status": "error", "error": str(exc)}), 500

    try:
        df      = pd.read_csv("data/features.csv")
        split   = int(len(df) * CONFIG["train_split"])
        df_test = df.iloc[split:].copy().reset_index(drop=True)

        result  = backtest(df_test)
        metrics = compute_metrics(result)

        chart_data = df_test.tail(180).to_dict(orient="records")

        return jsonify({
            "status": "ok",
            "ticker": ticker,
            "initial_capital": CONFIG["initial_capital"],
            "metrics": metrics,
            "portfolio_history": result["portfolio_history"].to_dict(orient="records"),
            "trade_log": result["trade_log"].to_dict(orient="records"),
            "chart_data": chart_data,
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"[app] Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
