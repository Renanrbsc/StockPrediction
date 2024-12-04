import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Input, Dense
from keras.optimizers import Adam
from tqdm import tqdm


class MarketPredictionModel:
    """
    Handles data preprocessing, LSTM model creation, training, evaluation,
    and future stock price prediction.
    """

    def __init__(
        self,
        lstm_units=256,
        dropout_rate=0.4,
        epochs=300,
        batch_size=64,
    ):
        """
        Initialize the MarketPredictionModel with configurable parameters.

        Args:
            lstm_units (int): Number of units in each LSTM layer.
            dropout_rate (float): Dropout rate for regularization.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each training batch.
        """
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None

    def preprocess_data(self, data, window_size, forecast_days):
        """
        Prepare the stock data for model training.

        Args:
            data (DataFrame): Stock data containing 'Close' prices.

        Returns:
            np.array: Processed feature dataset (X).
            np.array: Labels corresponding to each input sequence (y).
        """
        scaled_data = self.scaler.fit_transform(data[["Close"]])
        X, y = [], []

        for i in range(len(scaled_data) - window_size - forecast_days):
            X.append(scaled_data[i : i + window_size])
            y.append(scaled_data[i + window_size : i + window_size + forecast_days])
        return np.array(X), np.array(y)

    def create_lstm_model(self, input_shape, forecast_days):
        """
        Build and compile an LSTM model for stock price prediction.

        Args:
            input_shape (tuple): Shape of the input data.

        Returns:
            Sequential: Compiled LSTM model.
        """
        model = Sequential(
            [
                Input(shape=input_shape),
                LSTM(self.lstm_units, activation="tanh", return_sequences=True),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units, activation="tanh"),
                Dropout(self.dropout_rate),
                Dense(forecast_days, activation="linear"),
            ]
        )
        model.compile(optimizer=Adam(), loss="mean_squared_error")
        return model

    def train(self, X, y):
        """
        Train the LSTM model with the provided data.

        Args:
            X (np.array): Feature dataset.
            y (np.array): Labels.

        Returns:
            Sequential: Trained LSTM model.
            np.array: Test features (X_test).
            np.array: Test labels (y_test).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model = self.create_lstm_model((X.shape[1], 1), y.shape[1])

        for epoch in tqdm(range(self.epochs), desc="Training Epochs", unit="epoch"):
            self.model.fit(
                X_train,
                y_train,
                epochs=1,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=0,
            )

        return X_test, y_test

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance on test data.

        Args:
            X_test (np.array): Test features.
            y_test (np.array): Test labels.

        Returns:
            float: Mean Squared Error (MSE).
            float: Mean Absolute Error (MAE).
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        predictions = self.model.predict(X_test)
        y_test = np.squeeze(y_test)

        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        r_squared = 1 - (
            np.sum((predictions - y_test) ** 2)
            / np.sum((y_test - np.mean(y_test)) ** 2)
        )
        return mse, mae, r_squared

    def calculate_window_size(self, forecast_days):
        """
        Calculates the data window size based on the forecast duration.

        Args:
            forecast_days (int): Number of forecast days.

        Returns:
            int: Data window size in days.
        """
        try:
            if forecast_days <= 2:
                return 15
            elif forecast_days <= 5:
                return 30
            elif forecast_days <= 15:
                return 60
            else:
                return 90
        except Exception as e:
            print(f"Error while calculating the time window: {e}")
            return None

    def predict_future_prices(self, last_window, forecast_days=3):
        """
        Predicts stock prices for the next specified number of business days.

        Args:
            last_window (np.array): The most recent 'window_size' stock prices.
            forecast_days (int, optional): Number of days to predict. Defaults to 3.

        Returns:
            np.array: Predicted stock prices for the next `forecast_days` business days.

        Raises:
            ValueError: If the model is not trained before making predictions.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions.")

        last_window_scaled = self.scaler.transform(last_window)
        predictions = []

        for _ in range(forecast_days):
            # Predict the next value
            next_pred = self.model.predict(last_window_scaled.reshape(1, -1, 1))[0, 0]

            # Append prediction and update the window
            predictions.append(next_pred)
            last_window_scaled = np.roll(last_window_scaled, -1)
            last_window_scaled[-1] = next_pred

        # Convert predictions back to the original scale
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
