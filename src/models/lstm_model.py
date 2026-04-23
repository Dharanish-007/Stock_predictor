# src/models/lstm_model.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import logging

logger = logging.getLogger(__name__)

class LSTMPredictor:
    def __init__(self, look_back=60):
        self.look_back = look_back # Use past 'look_back' days to predict the next day
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(self.look_back, len(data)):
            X.append(data[i-self.look_back:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def train(self, df, feature_cols=['Close']):
        # Prepare the data
        data = df[feature_cols].values
        scaled_data = self.scaler.fit_transform(data)

        X, y = self._create_sequences(scaled_data)

        # Reshape X for LSTM: (samples, time steps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split into train/test sets (preserving order)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build the LSTM model
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.look_back, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("LSTM model compiled.")

        # Train the model with early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0)

        # Evaluate and return a simple metric
        predictions = self.model.predict(X_test)
        # Calculate a simple directional accuracy (if it's useful for comparison)
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        pred_inv = self.scaler.inverse_transform(predictions)
        accuracy = np.mean((pred_inv[1:] > pred_inv[:-1]) == (y_test_inv[1:] > y_test_inv[:-1]))
        logger.info(f"LSTM training complete. Directional accuracy on test set: {accuracy:.2%}")
        return accuracy

    def predict_next_day(self, last_sequence):
        # 'last_sequence' is the last 'look_back' days of scaled data
        last_sequence_reshaped = last_sequence.reshape(1, self.look_back, 1)
        prediction_scaled = self.model.predict(last_sequence_reshaped, verbose=0)
        return self.scaler.inverse_transform(prediction_scaled)[0,0]