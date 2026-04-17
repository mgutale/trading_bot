"""
Dashboard Data Manager

Unified data access layer for the analytics dashboard.
Supports backtest mode only.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging

from trading_bot.analytics.dashboard.models import (
    PerformanceStats,
    RiskMetrics,
    AttributionEntry,
    DashboardConfig,
)
from trading_bot.analytics.dashboard.metrics_engine import MetricsEngine

logger = logging.getLogger(__name__)


class DashboardDataManager:
    """
    Data access layer for dashboard components.

    Usage:
        manager = DashboardDataManager.from_backtest(
            results=backtest_results,
            stock_data=stock_data,
            benchmark_data=benchmark_data,
            initial_capital=10000.0
        )

        stats = manager.get_performance_stats()
        stocks = manager.get_stock_performance()
        equity = manager.get_equity_curve()
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        benchmark_symbol: str = "SPY",
    ):
        """
        Initialize the data manager.

        Args:
            initial_capital: Starting portfolio value
            benchmark_symbol: Benchmark ticker for comparison
        """
        self.initial_capital = initial_capital
        self.benchmark_symbol = benchmark_symbol

        # Data storage
        self._results: Optional[Dict] = None
        self._stock_data: Optional[Dict[str, pd.DataFrame]] = None
        self._benchmark_data: Optional[pd.DataFrame] = None
        self._metrics_engine: Optional[MetricsEngine] = None

        # Cached data
        self._cached_stats: Optional[PerformanceStats] = None
        self._cached_risk: Optional[RiskMetrics] = None
        self._cached_stocks: Optional[List[AttributionEntry]] = None

    @classmethod
    def from_backtest(
        cls,
        results: Dict[str, Any],
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        initial_capital: float = 10000.0,
        benchmark_symbol: str = "SPY",
    ) -> 'DashboardDataManager':
        """
        Create data manager from backtest results.

        Args:
            results: Backtest results from HybridHMMStopLoss.backtest()
            stock_data: Dict of symbol -> DataFrame with price data
            benchmark_data: DataFrame with benchmark prices
            initial_capital: Starting portfolio value
            benchmark_symbol: Benchmark ticker

        Returns:
            DashboardDataManager instance
        """
        manager = cls(
            initial_capital=initial_capital,
            benchmark_symbol=benchmark_symbol,
        )

        manager._results = results
        manager._stock_data = stock_data
        manager._benchmark_data = benchmark_data

        # Initialize metrics engine with enriched trades
        if 'trade_log' in results and len(results.get('trade_log', [])) > 0:
            from trading_bot.analytics.dashboard.trade_enricher import enrich_trade_log

            signals = results.get('signals', pd.DataFrame())
            raw_trade_log = results.get('trade_log', [])
            turnover_log = results.get('turnover_log', [])

            if len(signals) > 0 and len(raw_trade_log) > 0:
                enriched_trades = enrich_trade_log(
                    raw_trade_log=raw_trade_log,
                    signals=signals,
                    stock_data=stock_data,
                    benchmark_data=benchmark_data,
                    turnover_log=turnover_log,
                    initial_capital=initial_capital,
                    stop_loss_pct=results.get('stop_loss_pct', 0.025),
                )

                manager._metrics_engine = MetricsEngine(
                    enriched_trades=enriched_trades,
                    signals=signals,
                    initial_capital=initial_capital,
                    benchmark_data=benchmark_data,
                )

                # Pre-compute and cache metrics
                manager._cached_stats = manager._metrics_engine.get_performance_stats()
                manager._cached_risk = manager._metrics_engine.get_risk_metrics()
                manager._cached_stocks = manager._metrics_engine.get_all_stock_performance()

        return manager

    def get_performance_stats(self) -> PerformanceStats:
        """
        Get portfolio performance statistics.

        Returns:
            PerformanceStats object
        """
        if self._cached_stats is not None:
            return self._cached_stats

        if self._metrics_engine is not None:
            return self._metrics_engine.get_performance_stats()

        return self._empty_performance_stats()

    def _empty_performance_stats(self) -> PerformanceStats:
        """Return empty stats when no data available."""
        return PerformanceStats(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            buy_trades=0,
            sell_trades=0,
            stop_loss_trades=0,
        )

    def get_risk_metrics(self) -> RiskMetrics:
        """
        Get portfolio risk metrics.

        Returns:
            RiskMetrics object
        """
        if self._cached_risk is not None:
            return self._cached_risk

        if self._metrics_engine is not None:
            return self._metrics_engine.get_risk_metrics()

        return self._empty_risk_metrics()

    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics when no data available."""
        return RiskMetrics(
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            volatility=0.0,
            downside_deviation=0.0,
            max_drawdown=0.0,
        )

    def get_stock_performance(self) -> List[AttributionEntry]:
        """
        Get per-stock performance metrics.

        Returns:
            List of AttributionEntry sorted by contribution
        """
        if self._cached_stocks is not None:
            return self._cached_stocks

        if self._metrics_engine is not None:
            return self._metrics_engine.get_all_stock_performance()

        return []

    def get_stock_detail(self, symbol: str) -> Optional[AttributionEntry]:
        """
        Get detailed performance for a specific stock.

        Args:
            symbol: Stock ticker

        Returns:
            AttributionEntry with full metrics, or None if not found
        """
        if self._metrics_engine is not None:
            return self._metrics_engine.get_stock_performance(symbol)

        return None

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve data for charting.

        Returns:
            DataFrame with date, equity, benchmark, daily_return columns
        """
        if self._metrics_engine is not None:
            return self._metrics_engine.get_equity_curve()

        return pd.DataFrame()

    def get_drawdown_series(self) -> pd.Series:
        """
        Get drawdown series for charting.

        Returns:
            Series of drawdown values
        """
        if self._metrics_engine is not None:
            return self._metrics_engine.get_drawdown_series()

        return pd.Series(dtype=float)

    def get_regime_breakdown(self) -> List[Dict[str, Any]]:
        """
        Get performance breakdown by market regime.

        Returns:
            List of regime performance dicts
        """
        if self._metrics_engine is not None:
            breakdown = self._metrics_engine.get_regime_breakdown()
            return [
                {
                    'regime': bp.regime,
                    'days': bp.days,
                    'total_return': bp.total_return,
                    'avg_return': bp.avg_return,
                    'win_rate': bp.win_rate,
                    'trades': bp.trades,
                    'exposure': bp.exposure,
                }
                for bp in breakdown
            ]

        return []

    def get_trade_journal(self) -> pd.DataFrame:
        """
        Get trade history for display.

        Returns:
            DataFrame with trade details
        """
        if self._metrics_engine is not None:
            return self._metrics_engine.get_trade_journal()

        return pd.DataFrame()

    def get_current_regime(self) -> str:
        """
        Get current market regime.

        Returns:
            Regime label string
        """
        if self._results is not None:
            signals = self._results.get('signals', pd.DataFrame())
            if 'regime' in signals.columns and len(signals) > 0:
                return str(signals['regime'].iloc[-1])

        return 'unknown'

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get complete portfolio summary.

        Returns:
            Dict with all portfolio data
        """
        stats = self.get_performance_stats()
        risk = self.get_risk_metrics()

        equity = self.initial_capital * (1 + stats.total_return)

        return {
            'equity': equity,
            'cash': 0.0,
            'total_return': stats.total_return,
            'daily_return': 0.0,
            'current_regime': self.get_current_regime(),
            'stats': stats,
            'risk': risk,
        }
