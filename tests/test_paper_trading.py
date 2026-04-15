"""
Tests for trading strategy and position management.

Tests cover the core strategy (HMM + momentum + stop loss),
live trading readiness, and position management.
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLiveTradingReadiness:
    """Tests to verify live trading readiness."""

    def test_minimum_historical_data_requirement(self):
        """Test that minimum historical data requirement is enforced."""
        from trading_bot.strategies.hybrid import HybridHMMStopLoss

        # Create minimal data that should fail
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        prices = pd.DataFrame({
            'open': np.linspace(100, 105, 50),
            'high': np.linspace(102, 107, 50),
            'low': np.linspace(98, 103, 50),
            'close': np.linspace(101, 106, 50),
            'volume': np.random.randint(1000000, 10000000, 50)
        }, index=dates)

        strat = HybridHMMStopLoss(n_states=4)

        # Should raise error with insufficient data
        with pytest.raises(ValueError, match="Need at least 63 days"):
            strat.fit(prices)

    def test_sufficient_historical_data(self):
        """Test with sufficient historical data (63+ days)."""
        from trading_bot.strategies.hybrid import HybridHMMStopLoss

        # Create sufficient data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            'open': np.linspace(100, 110, 100),
            'high': np.linspace(102, 112, 100),
            'low': np.linspace(98, 108, 100),
            'close': np.linspace(101, 111, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

        strat = HybridHMMStopLoss(n_states=4)

        # Should work with sufficient data
        strat.fit(prices)
        assert strat.fitted

    def test_regime_detection_works(self):
        """Test that regime detection produces valid results."""
        from trading_bot.strategies.hybrid import HybridHMMStopLoss

        # Create realistic price data
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
        returns = np.random.normal(0.0005, 0.02, 252)
        prices = 100 * np.cumprod(1 + returns)

        prices_df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, 252)),
            'high': prices * (1 + np.random.uniform(0, 0.03, 252)),
            'low': prices * (1 - np.random.uniform(0, 0.03, 252)),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 252)
        }, index=dates)

        strat = HybridHMMStopLoss(n_states=4)
        strat.fit(prices_df)

        # Predict regimes
        regimes = strat.hmm_detector.predict(prices_df)
        labeled = strat._label_regimes_walkforward(regimes, prices_df)

        # Verify regime labels are valid (4-state)
        valid_regimes = ['strong_bull', 'weak_bull', 'weak_bear', 'strong_bear']
        for regime in labeled.unique():
            assert regime in valid_regimes or pd.isna(regime)


class TestPositionManagement:
    """Tests for position management."""

    def test_position_creation_and_trailing_stop(self):
        """Test Position dataclass and trailing stop logic."""
        from trading_bot.strategy_engine import Position

        pos = Position(
            symbol="AAPL",
            qty=10,
            entry_price=150.0,
            current_price=160.0,
            stop_loss_pct=0.05,
        )

        # Verify initial state
        assert pos.symbol == "AAPL"
        assert pos.qty == 10
        assert pos.entry_price == 150.0
        assert pos.stop_price == pytest.approx(142.5)  # 150 * 0.95

        # Verify unrealized P&L
        assert pos.unrealized_pnl == 100.0  # (160 - 150) * 10
        assert pos.unrealized_pnl_pct == pytest.approx(0.0667, abs=0.01)

    def test_trailing_stop_moves_up_only(self):
        """Test that trailing stop only moves up, never down."""
        from trading_bot.strategy_engine import Position

        pos = Position(symbol="AAPL", qty=10, entry_price=150.0, current_price=160.0,
                        stop_loss_pct=0.05)

        # Initial stop at 5% below entry
        assert pos.stop_price == pytest.approx(142.5)  # 150 * 0.95

        # Price goes up, stop should move up
        pos.update_price(170.0)
        pos.update_trailing_stop(0.05)
        assert pos.stop_price == pytest.approx(161.5)  # 170 * 0.95

        # Price drops, stop should NOT move down
        pos.update_price(155.0)
        pos.update_trailing_stop(0.05)
        assert pos.stop_price == pytest.approx(161.5)  # Still at 170 * 0.95

    def test_position_manager(self):
        """Test PositionManager buy/sell and portfolio tracking."""
        from trading_bot.strategy_engine import PositionManager

        pm = PositionManager(stop_loss_pct=0.05)

        # Buy a position
        pm.set_equity(10000)
        trade = pm.buy("AAPL", 10, 150.0)
        assert trade is not None
        assert trade["action"] == "BUY"
        assert trade["symbol"] == "AAPL"
        assert pm.position_count() == 1

        # Update price and check stop
        pm.update_prices({"AAPL": 155.0})
        pm.update_trailing_stops()
        pos = pm.get_position("AAPL")
        assert pos.unrealized_pnl == 50.0  # (155 - 150) * 10

        # Sell the position
        sell_trade = pm.sell("AAPL")
        assert sell_trade is not None
        assert sell_trade["action"] == "SELL"
        assert pm.position_count() == 0