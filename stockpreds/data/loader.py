"""Market data acquisition via the Yahoo Finance API (yfinance).

All functions return tidy, validated DataFrames so downstream code never
has to deal with yfinance quirks (MultiIndex columns, timezone-aware
indexes, silent empty frames).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from stockpreds.config import DataConfig

logger = logging.getLogger(__name__)

_OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataFetchError(RuntimeError):
    """Raised when market data cannot be retrieved or fails validation."""


def fetch_ohlcv(
    ticker: str,
    period: str = DataConfig.period,
    interval: str = DataConfig.interval,
    min_rows: int = DataConfig.min_rows,
) -> pd.DataFrame:
    """Download adjusted OHLCV history for a ticker.

    Args:
        ticker: Yahoo Finance symbol, e.g. ``"AAPL"`` or ``"RELIANCE.NS"``.
        period: Lookback window accepted by yfinance (``"1y"``, ``"5y"``, ...).
        interval: Bar size (``"1d"`` for daily).
        min_rows: Minimum rows required for the result to be usable.

    Returns:
        DataFrame indexed by naive ``DatetimeIndex`` with columns
        ``Open, High, Low, Close, Volume`` (split/dividend adjusted).

    Raises:
        DataFetchError: If the symbol is unknown, the network call fails,
            or the history is too short to train on.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise DataFetchError("Ticker symbol cannot be empty.")

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:  # yfinance raises a grab-bag of exception types
        raise DataFetchError(f"Failed to download data for '{ticker}': {exc}") from exc

    if df is None or df.empty:
        raise DataFetchError(
            f"No data returned for '{ticker}'. Check that the symbol is valid "
            "on Yahoo Finance (e.g. 'AAPL', 'RELIANCE.NS')."
        )

    # yfinance returns MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[[c for c in _OHLCV_COLUMNS if c in df.columns]].copy()
    df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    if len(df) < min_rows:
        raise DataFetchError(
            f"Only {len(df)} rows of history available for '{ticker}'; "
            f"at least {min_rows} are required to train a reliable model. "
            "Try a longer period."
        )

    logger.info("Fetched %d rows for %s (%s, %s)", len(df), ticker, period, interval)
    return df


def fetch_ticker_summary(ticker: str) -> dict[str, Any]:
    """Best-effort company snapshot (name, sector, market cap, 52w range).

    Returns an empty dict on failure — summary data is cosmetic and must
    never break the forecasting flow.
    """
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        logger.warning("Could not fetch summary info for %s", ticker, exc_info=True)
        return {}

    keys = (
        "longName", "sector", "industry", "marketCap", "currency",
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "trailingPE", "website",
    )
    return {k: info[k] for k in keys if info.get(k) is not None}
