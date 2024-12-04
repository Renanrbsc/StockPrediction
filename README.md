# StockPrediction

This repository contains a Python-based implementation of an LSTM (Long Short-Term Memory) algorithm for predicting future stock prices. The model leverages historical investment data collected from Yahoo Finance to forecast stock prices for a specified number of future days.

Key Features
    Data Collection: Automatically fetches historical stock data using Yahoo Finance API.
    Preprocessing: Includes data cleaning, normalization, and scaling for LSTM model compatibility.
    Model Training: Utilizes TensorFlow/Keras to build and train an LSTM model tailored for time series forecasting.
    Prediction: Predicts future stock prices for N business days, allowing flexible forecasting horizons.
    Evaluation: Generates performance metrics and visualizations to assess prediction accuracy.

Applications
    Investment Analysis: Offers insights into potential future stock movements.

Requirements
    Python 3.8+
    TensorFlow
    NumPy, Pandas, Matplotlib
    Scikit-learn
    Yahoo Finance API or equivalent library (e.g., yfinance)

How It Works
    Fetch historical data for a specific stock ticker from Yahoo Finance.
    Preprocess data into a format suitable for LSTM modeling.
    Train the LSTM model on historical data.
    Generate predictions for future stock prices and visualize the results.

Repository Structure
    controller/: Scripts for fetching and preprocessing stock data.
    model/: Implementation of the LSTM architecture and training pipeline.
    view/: Create a terminal visualization with prints and inputs.