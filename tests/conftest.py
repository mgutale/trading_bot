"""
Pytest Fixtures for Trading Bot Tests

Provides sample data and test utilities.
"""

import pandas as pd
import numpy as np
import pytest
from typing import Dict


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Generate sample OHLCV price data for testing."""
    np.random.seed(42)
    n_days = 1000

    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="B")

    # Generate realistic price series with trend and noise
    base_price = 100
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily return ~8% annual, 20% vol
    price_series = base_price * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'open': price_series * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': price_series * (1 + np.random.uniform(0, 0.03, n_days)),
        'low': price_series * (1 - np.random.uniform(0, 0.03, n_days)),
        'close': price_series,
        'volume': np.random.randint(1000000, 10000000, n_days),
    }, index=dates)

    return df


@pytest.fixture
def sample_multi_stock_data(sample_prices) -> Dict[str, pd.DataFrame]:
    """Generate sample data for multiple stocks."""
    symbols = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    data = {}

    np.random.seed(42)
    for i, symbol in enumerate(symbols):
        # Create slightly different data for each symbol
        df = sample_prices.copy()
        df['close'] = df['close'] * (1 + i * 0.1)  # Different price levels
        data[symbol] = df

    return data


@pytest.fixture
def sample_returns(sample_prices) -> pd.Series:
    """Generate sample returns series."""
    return sample_prices['close'].pct_change().dropna()


@pytest.fixture
def sample_multi_returns(sample_multi_stock_data) -> pd.DataFrame:
    """Generate sample returns for multiple stocks."""
    returns_dict = {}
    for symbol, df in sample_multi_stock_data.items():
        returns_dict[symbol] = df['close'].pct_change()
    return pd.DataFrame(returns_dict).dropna()


@pytest.fixture
def strategy_config():
    """Sample strategy configuration matching production defaults."""
    return {
        'n_states': 4,
        'momentum_short': 71,
        'momentum_long': 83,
        'top_n_stocks': 5,
        'rebalance_frequency': 11,
        'stop_loss_pct': 0.053,
        'regime_exposure': {
            'strong_bull': 1.0,
            'weak_bull': 0.75,
            'weak_bear': 0.25,
            'strong_bear': 0.0,
        }
    }


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory for data loader."""
    cache_dir = tmp_path / "data_cache"
    cache_dir.mkdir()
    return str(cache_dir)
