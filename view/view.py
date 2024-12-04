class MarketView:
    """
    Handles user interaction and displays results for the market prediction process.
    """

    @staticmethod
    def get_user_input():
        """
        Prompt the user for a stock ticker symbol.

        Returns:
            str: The stock ticker symbol entered by the user.
        """
        print("\033[94m" + "\nStock Prediction System" + "\033[0m")
        print("-" * 40)
        return input("\033[92mEnter the stock ticker symbol: \033[0m").strip().upper()

    @staticmethod
    def get_forecast_days_input():
        """
        Prompt the user for a number of forecast days.

        Returns:
            int: The number of forecast day entered by the user.
        """
        print("-" * 40)
        return int(input("\033[92mEnter the forecast days you need: \033[0m").strip())

    @staticmethod
    def display_results(results):
        """
        Displays the results of the market prediction process by delegating
        specific tasks to specialized methods.

        Args:
            results (dict): A dictionary containing the results to display.
                Expected keys:
                    - 'dates': List of predicted dates.
                    - 'prices': List of predicted stock prices.
                    - 'mse': Mean Squared Error of the model.
                    - 'mae': Mean Absolute Error of the model.
                    - 'r_squared': R-squared value of the model.
                    - 'real_price': Actual closing price of the stock.
                    - 'predicted_price': Predicted price of the stock.
                    - 'projected_return': Projected return for an investment of $1,000.
        """
        print("\n\033[94m" + "Market Prediction Results" + "\033[0m")
        print("-" * 40)

        # Display predicted prices
        MarketView._display_predicted_prices(results["dates"], results["prices"])

        # Display model performance metrics
        MarketView._display_model_performance(
            results["mse"], results["mae"], results["r_squared"]
        )

        # Display real vs predicted prices
        MarketView._display_real_vs_predicted(
            results["real_price"], results["predicted_price"]
        )

        # Display recommendation based on predicted variation
        MarketView._display_recommendation(
            results["real_price"], results["predicted_price"]
        )

        # Display investment analysis
        MarketView._display_investment_analysis(results["projected_return"])

        print("-" * 40)
        print("\033[94mThank you for using the Stock Prediction System!\033[0m\n")

    @staticmethod
    def _display_predicted_prices(dates, prices):
        """
        Display the predicted stock prices.

        Args:
            dates (list): List of predicted dates.
            prices (list): List of predicted stock prices.
        """
        print("\033[93mPredicted prices for the next week:\033[0m")
        for date, price in zip(dates, prices):
            print(f"  {date.strftime('%d/%m/%Y')}: \033[92m${price:.2f}\033[0m")

    @staticmethod
    def _display_model_performance(mse, mae, r_squared):
        """
        Display the model's performance metrics.

        Args:
            mse (float): Mean Squared Error of the model.
            mae (float): Mean Absolute Error of the model.
            r_squared (float): R-squared value of the model.
        """
        print("\n\033[93mModel Performance:\033[0m")
        print(f"  \033[96mMean Squared Error (MSE):\033[0m {mse:.4f}")
        print(f"  \033[96mMean Absolute Error (MAE):\033[0m {mae:.4f}")
        print(f"  \033[96mR-Squared (R²):\033[0m {r_squared:.4f}")

    @staticmethod
    def _display_real_vs_predicted(real_price, predicted_price):
        """
        Display the comparison between real and predicted stock prices.

        Args:
            real_price (float): The actual closing price of the stock.
            predicted_price (float): The predicted price of the stock.
        """
        variation_percentage = ((predicted_price - real_price) / real_price) * 100
        print("\n\033[93mReal vs Predicted Price:\033[0m")
        print(f"  \033[96mActual Closing Price:\033[0m ${real_price:.2f}")
        print(f"  \033[96mPredicted Price:\033[0m ${predicted_price:.2f}")
        print(
            f"  \033[96mPredicted Price Variation:\033[0m {variation_percentage:.2f}%"
        )

    @staticmethod
    def _display_recommendation(real_price, predicted_price):
        """
        Display a recommendation based on the predicted variation.

        Args:
            real_price (float): The actual closing price of the stock.
            predicted_price (float): The predicted price of the stock.
        """
        variation_percentage = ((predicted_price - real_price) / real_price) * 100
        print("\n\033[93mRecommendation:\033[0m")
        if variation_percentage > 0:
            print(
                "  \033[92mConsider buying:\033[0m The model predicts an upward trend."
            )
        elif variation_percentage < 0:
            print(
                "  \033[91mConsider selling or holding:\033[0m The model predicts a downward trend."
            )
        else:
            print("  \033[96mHold:\033[0m The model predicts price stability.")

    @staticmethod
    def _display_investment_analysis(projected_return):
        """
        Display the projected return for an investment of $1,000.

        Args:
            projected_return (float): Projected return for a $1,000 investment.
        """
        print("\n\033[93mInvestment Analysis:\033[0m")
        print(
            f"  If you invest \033[92m$1,000.00\033[0m, the projected return is approximately \033[92m${projected_return:.2f}\033[0m."
        )
