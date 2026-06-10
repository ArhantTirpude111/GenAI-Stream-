"""StockPreds REST API (Flask).

A lightweight serving layer so forecasts can be consumed by any client
(React frontend, cron jobs, notebooks) — the Streamlit dashboard remains
the primary UI. Run with:

    python -m api.server          # dev server on :8000
    gunicorn 'api.server:create_app()'   # production

Endpoints
---------
GET  /health                       Liveness probe.
GET  /api/v1/history/<ticker>      Recent OHLCV history as JSON.
POST /api/v1/predict               Train and forecast.
        body: {"ticker": "AAPL", "horizon": 30, "period": "5y"}
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, request

from stockpreds.config import ModelConfig
from stockpreds.data import DataFetchError, fetch_ohlcv
from stockpreds.models import LSTMForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

MAX_HORIZON = 60

# Fewer epochs than the dashboard default: API callers expect a bounded
# response time, and accuracy past ~15 epochs improves only marginally.
API_MODEL_CONFIG = ModelConfig(epochs=15)


def create_app() -> Flask:
    app = Flask("stockpreds")

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "stockpreds-api"})

    @app.get("/api/v1/history/<ticker>")
    def history(ticker: str):
        period = request.args.get("period", "1y")
        try:
            df = fetch_ohlcv(ticker, period=period)
        except DataFetchError as exc:
            return jsonify({"error": str(exc)}), 404
        payload = df.round(4).reset_index()
        payload["Date"] = payload["Date"].dt.strftime("%Y-%m-%d")
        return jsonify({"ticker": ticker.upper(), "rows": len(payload),
                        "data": payload.to_dict(orient="records")})

    @app.post("/api/v1/predict")
    def predict():
        body = request.get_json(silent=True) or {}
        ticker = str(body.get("ticker", "")).strip().upper()
        horizon = int(body.get("horizon", 30))
        period = str(body.get("period", "5y"))

        if not ticker:
            return jsonify({"error": "Field 'ticker' is required."}), 400
        if not 1 <= horizon <= MAX_HORIZON:
            return jsonify({"error": f"'horizon' must be between 1 and {MAX_HORIZON}."}), 400

        try:
            df = fetch_ohlcv(ticker, period=period)
        except DataFetchError as exc:
            return jsonify({"error": str(exc)}), 404

        logger.info("Training forecast model for %s (horizon=%d)", ticker, horizon)
        forecaster = LSTMForecaster(API_MODEL_CONFIG)
        result = forecaster.fit(df["Close"])
        future = forecaster.forecast(horizon)

        return jsonify({
            "ticker": ticker,
            "horizon": horizon,
            "last_close": round(float(df["Close"].iloc[-1]), 4),
            "metrics": {k: round(v, 4) for k, v in result.metrics.items()},
            "forecast": [
                {"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 4)}
                for d, p in future["Forecast"].items()
            ],
        })

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8000, debug=False)
