"""Regression metrics for evaluating price forecasts."""

from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent. Ignores zero targets."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of days where the forecast got the *direction* of the move right.

    For trading relevance this often matters more than absolute error: a
    model can have low RMSE while being a coin flip on direction.
    """
    true_dir = np.sign(np.diff(np.asarray(y_true, dtype=float)))
    pred_dir = np.sign(np.diff(np.asarray(y_pred, dtype=float)))
    if len(true_dir) == 0:
        return float("nan")
    return float(np.mean(true_dir == pred_dir) * 100.0)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Bundle of all metrics, keyed by display-friendly names."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
        "Directional Accuracy (%)": directional_accuracy(y_true, y_pred),
    }
