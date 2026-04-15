import pandas as pd
import numpy as np
import pytest
from trading_bot.strategies.momentum import rank_by_momentum, build_equal_weight_portfolio

def test_rank_by_momentum_returns_list():
    prices = pd.DataFrame({
        "AAPL": np.linspace(100, 200, 100),
        "MSFT": np.linspace(200, 300, 100),
        "GOOGL": np.linspace(150, 250, 100),
    })
    result = rank_by_momentum(prices, short_window=20, long_window=50, top_n=2)
    assert isinstance(result, list)

def test_rank_by_momentum_returns_top_n():
    prices = pd.DataFrame({
        "AAPL": np.linspace(100, 200, 100),
        "MSFT": np.linspace(200, 300, 100),
        "GOOGL": np.linspace(150, 250, 100),
        "NVDA": np.linspace(300, 500, 100),
    })
    result = rank_by_momentum(prices, short_window=20, long_window=50, top_n=2)
    assert len(result) == 2

def test_build_equal_weight_portfolio():
    symbols = ["AAPL", "MSFT"]
    weights = build_equal_weight_portfolio(symbols, exposure=1.0)
    assert len(weights) == 2
    assert all(w == 0.5 for w in weights.values())