import numpy as np
import pandas as pd
import pytest

from stockpreds.features.indicators import (
    add_technical_indicators,
    bollinger_bands,
    rsi,
    sma,
)


@pytest.fixture()
def price_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    close = 100 + np.cumsum(rng.normal(0.1, 1.0, n))
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.5, n),
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": rng.integers(1e6, 5e6, n),
        },
        index=idx,
    )


def test_enriched_frame_has_expected_columns(price_frame):
    out = add_technical_indicators(price_frame)
    expected = {
        "SMA_20", "SMA_50", "EMA_12", "EMA_26", "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Mid", "BB_Lower", "Daily_Return", "Volatility_21",
    }
    assert expected.issubset(out.columns)
    assert len(out) == len(price_frame)


def test_enrichment_does_not_mutate_input(price_frame):
    before = price_frame.copy()
    add_technical_indicators(price_frame)
    pd.testing.assert_frame_equal(price_frame, before)


def test_rsi_is_bounded(price_frame):
    values = rsi(price_frame["Close"]).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_rsi_is_100_for_monotonic_gains():
    close = pd.Series(np.arange(1.0, 101.0))
    assert rsi(close).dropna().iloc[-1] == pytest.approx(100.0)


def test_sma_matches_manual_mean(price_frame):
    out = sma(price_frame["Close"], 20)
    manual = price_frame["Close"].iloc[:20].mean()
    assert out.iloc[19] == pytest.approx(manual)
    assert out.iloc[:19].isna().all()


def test_bollinger_band_ordering(price_frame):
    upper, mid, lower = bollinger_bands(price_frame["Close"])
    valid = mid.notna()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()
