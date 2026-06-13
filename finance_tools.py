"""Data layer: fetches daily stock prices from Alpha Vantage and computes moving averages.

This module owns all external market-data access. It returns a pandas DataFrame with the
raw OHLCV columns plus computed SMA20/SMA50 columns, using a (df, error) tuple so callers
can handle failures without exceptions crossing layer boundaries.
"""
import os
import pandas as pd
import requests


class FinanceDataTool:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_KEY")

    def fetch_stock_data(self, symbol: str) -> tuple:
        """Returns (DataFrame with SMA columns, None) on success or (None, error_str) on failure."""
        if not self.api_key:
            return None, "Alpha Vantage API key is required. Set ALPHA_VANTAGE_KEY env var."

        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "compact",
        }
        try:
            response = requests.get("https://www.alphavantage.co/query", params=params, timeout=10)
            data = response.json()
        except Exception as e:
            return None, f"Network error: {e}"

        if "Time Series (Daily)" not in data:
            if "Error Message" in data:
                return None, f"API Error: {data['Error Message']}"
            if "Note" in data:
                return None, f"API rate limit reached: {data['Note']}"
            return None, f"Unexpected API response: {data}"

        try:
            df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. volume": "Volume",
            }, inplace=True)
            df["SMA20"] = df["Close"].rolling(window=20).mean()
            df["SMA50"] = df["Close"].rolling(window=50).mean()
            return df, None
        except Exception as e:
            return None, f"Error processing data: {e}"
