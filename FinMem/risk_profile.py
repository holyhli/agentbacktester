"""
Risk Profile Calculator for FinMem
Computes risk metrics for assets based on market and news data.
"""
import numpy as np
import pandas as pd

class RiskProfiler:
    def __init__(self):
        pass

    def calculate(self, price_series, news_sentiment_series=None):
        """Calculate risk metrics (volatility, drawdown, sentiment, etc.)."""
        risk = {}
        if price_series is not None and len(price_series) > 1:
            returns = np.diff(price_series) / price_series[:-1]
            risk['volatility'] = np.std(returns)
            risk['max_drawdown'] = self.max_drawdown(price_series)
        if news_sentiment_series is not None:
            risk['avg_sentiment'] = np.mean(news_sentiment_series)
        # TODO: Add more metrics as needed
        return risk

    @staticmethod
    def max_drawdown(prices):
        """Calculate maximum drawdown."""
        prices = np.array(prices)
        roll_max = np.maximum.accumulate(prices)
        drawdowns = (prices - roll_max) / roll_max
        return np.min(drawdowns) 