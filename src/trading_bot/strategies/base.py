"""
Base Strategy Interface

Abstract base class for all trading strategies.
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def fit(self, benchmark_data: pd.DataFrame) -> "BaseStrategy":
        """Fit the strategy to benchmark data."""
        ...

    @abstractmethod
    def generate_signals(
        self,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate trading signals. Returns DataFrame with regime and returns columns."""
        ...

    @abstractmethod
    def backtest(
        self,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame
    ) -> Dict:
        """Run backtest and return performance metrics."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config=None, **overrides) -> "BaseStrategy":
        """Construct strategy from StrategyConfig with optional overrides."""
        ...