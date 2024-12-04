# controller.py
from model.model_prediction import MarketPredictionModel
from model.model_extract import StockDataExtractor
from view.view import MarketView
import pandas as pd


class MarketController:
    """
    Controls the market prediction process.
    Handles user input, stock data retrieval, model execution, and result presentation.
    """

    def __init__(self):
        """
        Initialize the MarketController with model instances for extraction and prediction, and a view.

        Args:
            future_days (int): Number of days to predict into the future. Defaults to 10.
        """
        self.data_extractor = StockDataExtractor()
        self.model = MarketPredictionModel()
        self.view = MarketView()

    def execute(self):
        """
        Execute the market prediction workflow: retrieve user input,
        fetch stock data, predict prices, and present the results.
        """
        # Retrieve stock symbol from user
        stock_symbol = self.view.get_user_input()
        forecast_days = self.view.get_forecast_days_input()
        window_size = self.model.calculate_window_size(forecast_days)

        # Fetch stock data using the data extractor
        try:
            stock_data = self.data_extractor.fetch_stock_data(stock_symbol)
        except ValueError as e:
            self.view.display_error(str(e))
            return

        # Retrieve the most recent closing price
        try:
            real_price = float(stock_data["Close"].iloc[-1].values[0])
            print(f"Latest closing price: ${real_price:.2f}")
        except (IndexError, KeyError):
            self.view.display_error("Stock data is not sufficient for predictions.")
            return

        # Predict next week's stock prices and calculate projected return
        (
            predicted_dates,
            predicted_prices,
            mse,
            mae,
            r_squared,
            projected_return,
            predicted_price,
        ) = self.predict_and_calculate(stock_data, window_size, forecast_days)

        # Prepare and present the prediction results
        results = {
            "dates": predicted_dates,
            "prices": predicted_prices,
            "mse": mse,
            "mae": mae,
            "r_squared": r_squared,
            "real_price": real_price,
            "predicted_price": predicted_price,
            "projected_return": projected_return,
        }

        self.view.display_results(results)

    def predict_and_calculate(
        self, stock_data: dict, window_size: int, forecast_days: int
    ):
        """
        Generate stock price predictions and calculate metrics.

        Args:
            stock_data (DataFrame): Stock data containing historical 'Close' prices.

        Returns:
            tuple: Predicted dates, prices, MSE, MAE, R-squared, projected return, and predicted close price.
        """
        # Preprocess data and train the prediction model
        X, y = self.model.preprocess_data(stock_data, window_size, forecast_days)
        try:
            X_test, y_test = self.model.train(X, y)
        except ValueError as e:
            self.view.display_error(
                "Model training failed. Ensure sufficient data is provided."
            )
            raise e

        # Evaluate the model
        mse, mae, r_squared = self.model.evaluate(X_test, y_test)

        # Predict stock prices for the future days
        recent_prices = stock_data["Close"].values[-window_size:].reshape(-1, 1)
        next_predictions = self.model.predict_future_prices(
            recent_prices, forecast_days
        )
        next_dates = pd.date_range(
            start=stock_data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq="B",
        )

        # Calculate expected return rate and projected earnings
        expected_return_rate = (
            next_predictions[-1] - stock_data["Close"].iloc[-1]
        ) / stock_data["Close"].iloc[-1]

        projected_earnings = 1000 * float(expected_return_rate.iloc[0])
        predicted_close_price = float(next_predictions[-1])

        return (
            next_dates,
            next_predictions.flatten(),
            mse,
            mae,
            r_squared,
            projected_earnings,
            predicted_close_price,
        )
