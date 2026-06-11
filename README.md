# 📈 StockPreds — LSTM Stock Price Forecasting

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

An end-to-end stock price forecasting application. A stacked **LSTM** network is
trained on adjusted closing prices pulled live from the **Yahoo Finance API**,
backtested on a chronologically held-out test set, and served through an
interactive **Streamlit + Plotly** dashboard. A small **Flask** REST API exposes
the same pipeline for programmatic clients.

> ⚠️ **Disclaimer** — educational project, not investment advice.

---

## Features

- **Live market data** — adjusted daily OHLCV for any Yahoo Finance symbol
  (`AAPL`, `RELIANCE.NS`, …) via `yfinance`, with validation and one-hour caching.
- **Technical analysis** — SMA/EMA, RSI (Wilder), MACD, Bollinger Bands, and
  annualised volatility, implemented in pure pandas and unit-tested.
- **LSTM forecasting** — stacked LSTM (Keras/TensorFlow) with dropout, Huber
  loss, early stopping, and learning-rate scheduling. Strictly chronological
  train/val/test splits and train-only scaler fitting prevent look-ahead leakage.
- **Honest evaluation** — RMSE, MAE, MAPE, and directional accuracy reported on
  the most recent ~15% of history, which the model never sees during training.
- **Recursive multi-day forecasts** — with a 95% uncertainty band derived from
  test residuals that widens with the horizon.
- **Interactive dashboard** — candlestick + volume charts, indicator panels,
  live training progress, training curves, and CSV export.
- **REST API** — `POST /api/v1/predict` returns forecasts as JSON for any
  frontend or scheduled job.

## Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │                  stockpreds/                │
 Yahoo Finance ─▶│  data/loader ─▶ features/indicators         │
   (yfinance)    │        │                                    │
                 │        ▼                                    │
                 │  models/trainer (scaling · windowing ·      │
                 │  chronological split · LSTM fit · forecast) │
                 └───────────┬─────────────────┬───────────────┘
                             ▼                 ▼
                     app.py (Streamlit     api/server.py
                      + Plotly UI)          (Flask JSON API)
```

| Path | Purpose |
|---|---|
| `stockpreds/data/` | yFinance download, validation, tidy OHLCV frames |
| `stockpreds/features/` | Technical indicators (pure pandas) |
| `stockpreds/models/` | LSTM architecture + training/forecast pipeline |
| `stockpreds/utils/` | Forecast evaluation metrics |
| `app.py` | Streamlit dashboard (entry point) |
| `api/server.py` | Flask REST API |
| `tests/` | Pytest suite for indicators, metrics, windowing |

## Quick start

```bash
git clone https://github.com/ArhantTirpude111/GenAI-Stream-.git
cd GenAI-Stream-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Open http://localhost:8501, pick a ticker, and click **🚀 Train & Forecast**.
Training runs on CPU in roughly 30–90 seconds depending on history length and
epochs.

### REST API

```bash
python -m api.server   # serves on :8000
```

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "horizon": 30}'
```

```json
{
  "ticker": "AAPL",
  "horizon": 30,
  "last_close": 196.45,
  "metrics": {"RMSE": 4.21, "MAE": 3.37, "MAPE (%)": 1.71, "Directional Accuracy (%)": 52.3},
  "forecast": [{"date": "2026-06-11", "price": 197.12}, ...]
}
```

## Model details

| Aspect | Choice | Rationale |
|---|---|---|
| Architecture | LSTM(96) → LSTM(64) → Dense(1), dropout 0.2 | Enough capacity for daily series without overfitting |
| Input | 60-day sliding window of min-max-scaled closes | Standard univariate setup; window length configurable in the UI |
| Loss | Huber | Robust to gap moves vs. plain MSE |
| Training | Adam, early stopping on val loss, ReduceLROnPlateau | Stable CPU training, converges < 30 epochs |
| Split | Chronological train → val → test (no shuffling) | Shuffled splits leak the future and inflate metrics |
| Scaling | `MinMaxScaler` fitted on the **train slice only** | Avoids look-ahead leakage into evaluation |
| Multi-step | Recursive (predictions fed back in) | Simple and honest; uncertainty band widens with √horizon |

## Testing & linting

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

## Deployment (Streamlit Cloud)

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at `app.py` on the `main` branch.
3. In *Advanced settings*, select **Python 3.11** (or 3.12).
   `requirements.txt` already uses `tensorflow-cpu` to keep the build light.

## License

MIT
