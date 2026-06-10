"""End-to-end training and forecasting pipeline for the LSTM model.

Design notes
------------
* The split is strictly chronological (train -> val -> test) — shuffling
  time series across the split boundary leaks the future into training.
* The ``MinMaxScaler`` is fitted on the *training* slice only, then applied
  to validation/test data, again to avoid look-ahead leakage.
* Multi-step forecasting is recursive: each predicted close is appended to
  the window used to predict the next step. Uncertainty therefore grows
  with the horizon, which the dashboard visualises as a widening band.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from stockpreds.config import ModelConfig
from stockpreds.models.lstm import build_lstm
from stockpreds.utils.metrics import evaluate_forecast

logger = logging.getLogger(__name__)


def make_windows(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice a (n, 1) array into supervised-learning windows.

    Returns ``X`` of shape (n - lookback, lookback, 1) and ``y`` of shape
    (n - lookback,), where ``y[i]`` is the value immediately following
    window ``X[i]``.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    if len(values) <= lookback:
        raise ValueError(
            f"Need more than lookback={lookback} rows to build windows, got {len(values)}."
        )
    X = np.lib.stride_tricks.sliding_window_view(values[:-1, 0], lookback)
    y = values[lookback:, 0]
    return X[..., np.newaxis].astype(np.float32), y.astype(np.float32)


@dataclass
class ForecastResult:
    """Everything the UI/API needs after a training run."""

    test_dates: pd.DatetimeIndex
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]
    history: dict[str, list[float]] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"Actual": self.y_true, "Predicted": self.y_pred}, index=self.test_dates
        )


class LSTMForecaster:
    """Trains an LSTM on a closing-price series and produces forecasts."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self.model = None
        self._close: pd.Series | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        close: pd.Series,
        progress_callback: Callable[[int, int, dict], None] | None = None,
    ) -> ForecastResult:
        """Train on a closing-price series and evaluate on the held-out tail.

        Args:
            close: Closing prices indexed by ``DatetimeIndex``.
            progress_callback: Optional ``f(epoch, total_epochs, logs)`` hook,
                used by the dashboard to drive a progress bar.
        """
        import tensorflow as tf

        cfg = self.config
        tf.keras.utils.set_random_seed(cfg.seed)

        close = close.dropna().astype(float)
        self._close = close
        values = close.to_numpy().reshape(-1, 1)

        n_test = max(int(len(values) * cfg.test_fraction), cfg.lookback + 1)
        train_vals = values[:-n_test]
        # Test windows need `lookback` rows of context preceding the test span.
        test_vals = values[-(n_test + cfg.lookback):]

        scaled_train = self.scaler.fit_transform(train_vals)
        scaled_test = self.scaler.transform(test_vals)

        X_train, y_train = make_windows(scaled_train, cfg.lookback)
        X_test, y_test = make_windows(scaled_test, cfg.lookback)

        n_val = max(int(len(X_train) * cfg.val_fraction), 1)
        X_tr, y_tr = X_train[:-n_val], y_train[:-n_val]
        X_val, y_val = X_train[-n_val:], y_train[-n_val:]

        self.model = build_lstm(cfg.lookback, cfg)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stopping_patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
            ),
        ]
        if progress_callback is not None:
            callbacks.append(_ProgressCallback(cfg.epochs, progress_callback))

        history = self.model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        scaled_pred = self.model.predict(X_test, verbose=0)
        y_pred = self.scaler.inverse_transform(scaled_pred).ravel()
        y_true = self.scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        test_dates = close.index[-len(y_true):]

        metrics = evaluate_forecast(y_true, y_pred)
        logger.info("Test metrics for %d-day holdout: %s", len(y_true), metrics)

        return ForecastResult(
            test_dates=test_dates,
            y_true=y_true,
            y_pred=y_pred,
            metrics=metrics,
            history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        )

    # ------------------------------------------------------------- forecast
    def forecast(self, horizon: int) -> pd.DataFrame:
        """Recursively forecast `horizon` business days past the last close."""
        if self.model is None or self._close is None:
            raise RuntimeError("Call fit() before forecast().")

        cfg = self.config
        window = self.scaler.transform(
            self._close.to_numpy()[-cfg.lookback:].reshape(-1, 1)
        ).ravel()

        preds_scaled: list[float] = []
        for _ in range(horizon):
            x = window[-cfg.lookback:].reshape(1, cfg.lookback, 1)
            next_scaled = float(self.model.predict(x, verbose=0)[0, 0])
            preds_scaled.append(next_scaled)
            window = np.append(window, next_scaled)

        preds = self.scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).ravel()

        future_dates = pd.bdate_range(
            start=self._close.index[-1] + pd.Timedelta(days=1), periods=horizon
        )
        return pd.DataFrame({"Forecast": preds}, index=future_dates)


class _ProgressCallback:
    """Thin adapter turning Keras epoch events into a plain callable."""

    def __new__(cls, total_epochs: int, fn: Callable[[int, int, dict], None]):
        from tensorflow import keras

        class _Inner(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                fn(epoch + 1, total_epochs, logs or {})

        return _Inner()
