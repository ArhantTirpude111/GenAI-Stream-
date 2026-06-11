"""StockPreds — LSTM stock price forecasting dashboard.

Streamlit entry point. Run locally with:

    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stockpreds.config import DEFAULT_TICKERS, PERIOD_OPTIONS, ModelConfig
from stockpreds.data import DataFetchError, fetch_ohlcv, fetch_ticker_summary
from stockpreds.features import add_technical_indicators
from stockpreds.models import LSTMForecaster

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="StockPreds · LSTM Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#00C49A"
ACCENT_RED = "#FF6B6B"


# --------------------------------------------------------------------------
# Cached data access
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    return fetch_ohlcv(ticker, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def load_summary(ticker: str) -> dict:
    return fetch_ticker_summary(ticker)


# --------------------------------------------------------------------------
# Sidebar — controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📈 StockPreds")
    st.caption("LSTM-based stock price forecasting")

    preset = st.selectbox("Popular tickers", ("Custom…",) + DEFAULT_TICKERS, index=1)
    if preset == "Custom…":
        ticker = st.text_input("Ticker symbol", value="AAPL").strip().upper()
    else:
        ticker = preset

    period_label = st.selectbox("History window", list(PERIOD_OPTIONS), index=2)
    period = PERIOD_OPTIONS[period_label]

    st.divider()
    st.subheader("Forecast settings")
    horizon = st.slider("Forecast horizon (business days)", 5, 60, 30, step=5)

    with st.expander("Model hyperparameters", expanded=False):
        lookback = st.slider("Lookback window (days)", 30, 120, 60, step=10)
        epochs = st.slider("Max epochs", 10, 60, 30, step=5)
        dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
        st.caption(
            "Early stopping monitors validation loss, so the model usually "
            "converges well before max epochs."
        )

    train_clicked = st.button("🚀 Train & Forecast", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "⚠️ Educational project — not investment advice. "
        "Forecasts are statistical extrapolations and ignore news, "
        "earnings, and macro events."
    )

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
if not ticker:
    st.info("Enter a ticker symbol in the sidebar to get started.")
    st.stop()

try:
    with st.spinner(f"Fetching {ticker} data from Yahoo Finance…"):
        raw = load_data(ticker, period)
except DataFetchError as exc:
    st.error(str(exc))
    st.stop()

data = add_technical_indicators(raw)
summary = load_summary(ticker)

# --------------------------------------------------------------------------
# Header & key stats
# --------------------------------------------------------------------------
company = summary.get("longName", ticker)
currency = summary.get("currency", "USD")
st.title(f"{company} ({ticker})")
if summary.get("sector"):
    st.caption(f"{summary['sector']} · {summary.get('industry', '')}")

last_close = float(data["Close"].iloc[-1])
prev_close = float(data["Close"].iloc[-2])
day_change = last_close - prev_close
day_change_pct = day_change / prev_close * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last close", f"{last_close:,.2f} {currency}", f"{day_change_pct:+.2f}%")
c2.metric("52-week high", f"{data['Close'].tail(252).max():,.2f}")
c3.metric("52-week low", f"{data['Close'].tail(252).min():,.2f}")
c4.metric("Ann. volatility", f"{float(data['Volatility_21'].iloc[-1]) * 100:.1f}%")
mcap = summary.get("marketCap")
c5.metric("Market cap", f"{mcap / 1e9:,.1f} B" if mcap else "—")

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_overview, tab_technical, tab_forecast, tab_about = st.tabs(
    ["🕯️ Market Overview", "🔬 Technical Analysis", "🤖 LSTM Forecast", "ℹ️ About"]
)

# ----------------------------------------------------------- Overview tab
with tab_overview:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )
    fig.add_trace(
        go.Candlestick(
            x=data.index, open=data["Open"], high=data["High"],
            low=data["Low"], close=data["Close"], name="OHLC",
            increasing_line_color=ACCENT, decreasing_line_color=ACCENT_RED,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["SMA_50"], name="SMA 50",
                   line=dict(width=1.2, color="#FFD166")),
        row=1, col=1,
    )
    volume_colors = np.where(data["Close"] >= data["Open"], ACCENT, ACCENT_RED)
    fig.add_trace(
        go.Bar(x=data.index, y=data["Volume"], name="Volume",
               marker_color=volume_colors, opacity=0.6),
        row=2, col=1,
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=560, xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    fig.update_yaxes(title_text=f"Price ({currency})", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data (last 250 rows)"):
        st.dataframe(raw.tail(250).sort_index(ascending=False), use_container_width=True)
        st.download_button(
            "Download full history (CSV)",
            raw.to_csv().encode(),
            file_name=f"{ticker}_history.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------- Technical tab
with tab_technical:
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.55, 0.225, 0.225],
        subplot_titles=(
            "Price · Bollinger Bands · Moving Averages", "RSI (14)", "MACD (12, 26, 9)",
        ),
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["BB_Upper"], name="BB Upper",
                   line=dict(width=0.6, color="rgba(160,160,160,0.7)")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["BB_Lower"], name="BB Lower",
                   line=dict(width=0.6, color="rgba(160,160,160,0.7)"),
                   fill="tonexty", fillcolor="rgba(160,160,160,0.12)"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["Close"], name="Close",
                   line=dict(width=1.6, color=ACCENT)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20",
                   line=dict(width=1, color="#FFD166")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["SMA_50"], name="SMA 50",
                   line=dict(width=1, color="#EF8354")),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=data["RSI_14"], name="RSI 14",
                   line=dict(width=1.2, color="#9B5DE5")),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color=ACCENT_RED, opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color=ACCENT, opacity=0.5, row=2, col=1)

    fig.add_trace(
        go.Bar(x=data.index, y=data["MACD_Hist"], name="MACD hist",
               marker_color=np.where(data["MACD_Hist"] >= 0, ACCENT, ACCENT_RED),
               opacity=0.55),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["MACD"], name="MACD",
                   line=dict(width=1.1, color="#00B4D8")),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data["MACD_Signal"], name="Signal",
                   line=dict(width=1.1, color="#FFD166")),
        row=3, col=1,
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=820,
        legend=dict(orientation="h", y=1.03, x=0),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------- Forecast tab
with tab_forecast:
    state_key = f"forecast::{ticker}::{period}"

    if train_clicked:
        config = ModelConfig(lookback=lookback, epochs=epochs, dropout=dropout)
        forecaster = LSTMForecaster(config)

        progress = st.progress(0.0, text="Preparing training data…")

        def on_epoch(epoch: int, total: int, logs: dict) -> None:
            progress.progress(
                epoch / total,
                text=(
                    f"Training epoch {epoch}/{total} — "
                    f"val loss {logs.get('val_loss', float('nan')):.5f}"
                ),
            )

        with st.spinner("Training LSTM — typically 30–90 s on CPU…"):
            result = forecaster.fit(data["Close"], progress_callback=on_epoch)
            future = forecaster.forecast(horizon)
        progress.empty()

        st.session_state[state_key] = (result, future)

    if state_key not in st.session_state:
        st.info(
            "Configure the model in the sidebar and click **🚀 Train & Forecast**. "
            "The LSTM trains on the fly (CPU-friendly, ~1 minute) and is "
            "evaluated on a chronologically held-out test set."
        )
    else:
        result, future = st.session_state[state_key]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"{result.metrics['RMSE']:,.2f}")
        m2.metric("MAE", f"{result.metrics['MAE']:,.2f}")
        m3.metric("MAPE", f"{result.metrics['MAPE (%)']:.2f}%")
        m4.metric("Directional accuracy", f"{result.metrics['Directional Accuracy (%)']:.1f}%")
        st.caption(
            f"Evaluated on the most recent {len(result.y_true)} trading days, "
            "which the model never saw during training."
        )

        # -- Backtest: actual vs predicted on the held-out test window
        fig = go.Figure()
        context = data["Close"].iloc[-(len(result.y_true) * 3):]
        fig.add_trace(go.Scatter(
            x=context.index, y=context, name="Close (history)",
            line=dict(width=1.2, color="rgba(200,200,200,0.55)"),
        ))
        fig.add_trace(go.Scatter(
            x=result.test_dates, y=result.y_true, name="Actual (test)",
            line=dict(width=1.8, color=ACCENT),
        ))
        fig.add_trace(go.Scatter(
            x=result.test_dates, y=result.y_pred, name="Predicted (test)",
            line=dict(width=1.8, color="#FFD166", dash="dash"),
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=420,
            title="Backtest — model predictions vs. actual prices on unseen data",
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(l=10, r=10, t=60, b=10),
            yaxis_title=f"Price ({currency})",
        )
        st.plotly_chart(fig, use_container_width=True)

        # -- Forward forecast with a widening uncertainty band derived from
        #    test-set residuals (grows with sqrt(steps), random-walk style).
        resid_std = float(np.std(result.y_true - result.y_pred))
        steps = np.arange(1, len(future) + 1)
        band = 1.96 * resid_std * np.sqrt(steps)
        upper, lower = future["Forecast"] + band, future["Forecast"] - band

        fig = go.Figure()
        recent = data["Close"].iloc[-120:]
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent, name="Close (recent)",
            line=dict(width=1.4, color="rgba(200,200,200,0.7)"),
        ))
        fig.add_trace(go.Scatter(
            x=future.index, y=upper, name="95% band",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=future.index, y=lower, name="95% band",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(0,196,154,0.15)", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=future.index, y=future["Forecast"], name=f"{len(future)}-day forecast",
            line=dict(width=2.2, color=ACCENT),
        ))
        fig.update_layout(
            template=PLOTLY_TEMPLATE, height=420,
            title=f"{ticker} — {len(future)} business-day forecast",
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(l=10, r=10, t=60, b=10),
            yaxis_title=f"Price ({currency})",
        )
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.download_button(
                "Download forecast (CSV)",
                future.round(2).to_csv().encode(),
                file_name=f"{ticker}_forecast_{len(future)}d.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_b, st.expander("Training curves"):
            hist_df = pd.DataFrame(result.history)
            if {"loss", "val_loss"}.issubset(hist_df.columns):
                st.line_chart(hist_df[["loss", "val_loss"]])

# -------------------------------------------------------------- About tab
with tab_about:
    st.markdown(
        """
        ### How it works

        1. **Data** — Adjusted daily OHLCV is fetched from Yahoo Finance via
           `yfinance` and cached for an hour.
        2. **Model** — A stacked **LSTM** network (Keras/TensorFlow) is trained
           on a sliding window of scaled closing prices. The scaler is fitted
           on the training split only, and the train/validation/test split is
           strictly chronological to prevent look-ahead leakage.
        3. **Evaluation** — The model is backtested on the most recent ~15% of
           history it never saw, reporting RMSE, MAE, MAPE, and directional
           accuracy.
        4. **Forecast** — Future prices are generated recursively; the shaded
           band is a 95% interval derived from test-set residuals and widens
           with the horizon, reflecting compounding uncertainty.

        ### Tech stack
        Python · TensorFlow/Keras · scikit-learn · yFinance · Streamlit · Plotly · Flask (REST API)

        ---
        ⚠️ **Disclaimer** — This is an educational project. Price forecasts from
        historical data alone cannot anticipate news, earnings, or macro shocks.
        Nothing here is investment advice.
        """
    )
