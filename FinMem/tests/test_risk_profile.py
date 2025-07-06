import pytest
from FinMem.risk_profile import RiskProfiler
import numpy as np

def test_risk_profiler():
    profiler = RiskProfiler()
    prices = np.array([100, 105, 102, 110])
    sentiments = np.array([0.1, 0.2, -0.1, 0.0])
    risk = profiler.calculate(prices, sentiments)
    assert 'volatility' in risk
    assert 'max_drawdown' in risk
    assert 'avg_sentiment' in risk 