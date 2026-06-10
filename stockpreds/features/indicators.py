"""Technical indicators implemented with pure pandas (no TA-Lib dependency).

Each helper takes a price series and returns a like-indexed series, so they
compose cleanly and are trivial to unit-test. ``add_technical_indicators``
is the one-stop entry point used by the dashboard.
"""

from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return close.rolling(window=window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Bounded in [0, 100]; readings above 70 are conventionally treated as
    overbought and below 30 as oversold.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    # avg_loss == 0 yields rs = inf and therefore RSI = 100, which is the
    # textbook value for an instrument that only gained over the window.
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, and histogram."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper band, middle band (SMA), lower band."""
    mid = sma(close, window)
    std = close.rolling(window=window, min_periods=window).std()
    return mid + num_std * std, mid, mid - num_std * std


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame enriched with standard indicators.

    Added columns: SMA_20, SMA_50, EMA_12, EMA_26, RSI_14, MACD,
    MACD_Signal, MACD_Hist, BB_Upper, BB_Mid, BB_Lower, Daily_Return,
    Volatility_21 (annualised 21-day rolling std of returns).
    """
    out = df.copy()
    close = out["Close"]

    out["SMA_20"] = sma(close, 20)
    out["SMA_50"] = sma(close, 50)
    out["EMA_12"] = ema(close, 12)
    out["EMA_26"] = ema(close, 26)
    out["RSI_14"] = rsi(close, 14)
    out["MACD"], out["MACD_Signal"], out["MACD_Hist"] = macd(close)
    out["BB_Upper"], out["BB_Mid"], out["BB_Lower"] = bollinger_bands(close)
    out["Daily_Return"] = close.pct_change()
    out["Volatility_21"] = out["Daily_Return"].rolling(21).std() * (252 ** 0.5)
    return out
