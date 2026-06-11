import numpy as np
import pytest

from stockpreds.utils.metrics import (
    directional_accuracy,
    evaluate_forecast,
    mae,
    mape,
    rmse,
)


def test_perfect_forecast_has_zero_error():
    y = np.array([10.0, 11.0, 12.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert mape(y, y) == 0.0
    assert directional_accuracy(y, y) == 100.0


def test_rmse_known_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(2.0 / 3.0))


def test_mape_ignores_zero_targets():
    y_true = np.array([0.0, 100.0])
    y_pred = np.array([5.0, 110.0])
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_directional_accuracy_opposite_directions():
    y_true = np.array([1.0, 2.0, 3.0])   # up, up
    y_pred = np.array([3.0, 2.0, 1.0])   # down, down
    assert directional_accuracy(y_true, y_pred) == 0.0


def test_evaluate_forecast_keys():
    y = np.linspace(100, 110, 20)
    out = evaluate_forecast(y, y + 1.0)
    assert set(out) == {"RMSE", "MAE", "MAPE (%)", "Directional Accuracy (%)"}
