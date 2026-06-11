"""Central configuration for data, features, and model hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the LSTM forecaster.

    Defaults are tuned for daily OHLCV data and a single-feature
    (closing price) univariate forecast. They favour stable training
    on CPU over squeezing out the last basis point of accuracy.
    """

    lookback: int = 60          # trading days fed to the network per sample
    lstm_units: tuple[int, ...] = (96, 64)
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 30
    batch_size: int = 32
    val_fraction: float = 0.10  # chronological tail of the training split
    test_fraction: float = 0.15 # chronological tail of the full series
    early_stopping_patience: int = 6
    seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    """Defaults for fetching market data from Yahoo Finance."""

    period: str = "5y"
    interval: str = "1d"
    min_rows: int = 250  # roughly one trading year; below this an LSTM overfits


DEFAULT_TICKERS: tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "NFLX", "RELIANCE.NS", "TCS.NS", "INFY.NS",
)

PERIOD_OPTIONS: dict[str, str] = {
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "10 Years": "10y",
    "Max": "max",
}
