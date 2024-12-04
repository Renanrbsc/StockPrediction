import yfinance as yf
from datetime import datetime


class StockDataExtractor:
    """
    A class to handle the extraction of stock data from Yahoo Finance.
    """

    def __init__(self, start_date="2022-01-01"):
        """
        Initialize the StockDataExtractor with a start date.
        """
        self.start_date = start_date

    def fetch_stock_data(self, stock_symbol):
        """
        Fetch historical stock data for a given stock symbol from Yahoo Finance.

        Args:
            stock_symbol (str): The stock symbol to retrieve data for.

        Returns:
            DataFrame: A DataFrame containing historical stock data.
        """
        # Get today's date to use as the end date
        current_date = datetime.today().strftime("%Y-%m-%d")

        # Download the stock data using yfinance
        stock_data = yf.download(stock_symbol, start=self.start_date, end=current_date)

        if stock_data.empty:
            raise ValueError("No data available for the specified stock symbol.")

        return stock_data
