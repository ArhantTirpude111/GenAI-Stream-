"""LSTM network architecture for univariate price forecasting.

TensorFlow is imported lazily inside the builder so that modules which
only need data/feature utilities (e.g. the Flask health endpoint or unit
tests) don't pay the multi-second TF import cost.
"""

from __future__ import annotations

from stockpreds.config import ModelConfig


def build_lstm(lookback: int, config: ModelConfig):
    """Compile a stacked-LSTM regressor.

    Architecture: N stacked LSTM blocks (each followed by dropout for
    regularisation) and a single linear output unit predicting the next
    scaled closing price. Huber loss is used instead of MSE so occasional
    gap moves don't dominate the gradient.
    """
    from tensorflow import keras

    layers = keras.layers
    model = keras.Sequential(name="stockpreds_lstm")
    model.add(keras.Input(shape=(lookback, 1)))

    for i, units in enumerate(config.lstm_units):
        is_last = i == len(config.lstm_units) - 1
        model.add(layers.LSTM(units, return_sequences=not is_last))
        model.add(layers.Dropout(config.dropout))

    model.add(layers.Dense(1, name="next_close"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=keras.losses.Huber(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model
