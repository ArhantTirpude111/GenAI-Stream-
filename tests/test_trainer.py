import numpy as np
import pytest

from stockpreds.models.trainer import make_windows


def test_make_windows_shapes():
    values = np.arange(100, dtype=float)
    X, y = make_windows(values, lookback=10)
    assert X.shape == (90, 10, 1)
    assert y.shape == (90,)


def test_make_windows_alignment():
    values = np.arange(20, dtype=float)
    X, y = make_windows(values, lookback=5)
    # First window is values[0:5]; its target is values[5].
    np.testing.assert_array_equal(X[0].ravel(), values[:5])
    assert y[0] == values[5]
    # Last window ends at values[-2]; its target is the final value.
    np.testing.assert_array_equal(X[-1].ravel(), values[-6:-1])
    assert y[-1] == values[-1]


def test_make_windows_rejects_short_series():
    with pytest.raises(ValueError, match="lookback"):
        make_windows(np.arange(10, dtype=float), lookback=10)
