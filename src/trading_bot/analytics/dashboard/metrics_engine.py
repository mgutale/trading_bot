"""
Metrics Engine

Computes all analytics metrics from enriched trades and signals.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import logging

from trading_bot.analytics.dashboard.models import (
    AttributionEntry,
    EnrichedTrade,
    PerformanceStats,
    PortfolioSnapshot,
    RegimePerformance,
    RiskMetrics,
)

logger = logging.getLogger(__name__)


class MetricsEngine:
    """
    Computes all analytics metrics from enriched trades and signals.

    Usage:
        engine = MetricsEngine(
            enriched_trades=enriched_trades,
            signals=signals,
            initial_capital=10000.0
        )
        stats = engine.get_performance_stats()
        risk = engine.get_risk_metrics()
    """

    def __init__(
        self,
        enriched_trades: List[EnrichedTrade],
        signals: pd.DataFrame,
        initial_capital: float = 10000.0,
        benchmark_data: pd.DataFrame = None,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize the metrics engine.

        Args:
            enriched_trades: List of EnrichedTrade objects
            signals: DataFrame with daily returns, regime, benchmark columns
            initial_capital: Starting portfolio value
            benchmark_data: DataFrame with benchmark prices (optional)
            risk_free_rate: Risk-free rate for Sharpe/Sortino calculation (default 0)
        """
        self.enriched_trades = enriched_trades or []
        self.signals = signals
        self.initial_capital = initial_capital
        self.benchmark_data = benchmark_data
        self.risk_free_rate = risk_free_rate

        # Pre-compute common series
        self._returns = signals.get('strategy_return', pd.Series(dtype=float)) if signals is not None else pd.Series(dtype=float)
        self._returns_net = signals.get('strategy_return_net', self._returns) if signals is not None else pd.Series(dtype=float)
        self._cumulative = (1 + self._returns_net).cumprod() - 1 if len(self._returns_net) > 0 else pd.Series(dtype=float)
        self._benchmark_returns = signals.get('benchmark_return', pd.Series(dtype=float)) if signals is not None else pd.Series(dtype=float)
        self._benchmark_cumulative = signals.get('benchmark_cumulative', pd.Series(dtype=float)) if signals is not None else pd.Series(dtype=float)

    def get_performance_stats(self) -> PerformanceStats:
        """Calculate comprehensive performance statistics."""
        if len(self._returns_net) == 0:
            return self._empty_performance_stats()

        returns = self._returns_net.dropna()

        # Basic metrics
        total_return = float(self._cumulative.iloc[-1]) if len(self._cumulative) > 0 else 0.0
        # Annualized return using geometric compounding: (1 + total_return)^(252/n_days) - 1
        n_days = len(returns)
        if n_days > 0 and total_return > -1:
            annualized_return = float((1 + total_return) ** (252 / n_days) - 1)
        else:
            annualized_return = 0.0
        volatility = float(returns.std() * np.sqrt(252)) if len(returns) > 0 else 0.0

        # Risk-adjusted metrics
        excess_return = annualized_return - self.risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else 0.0

        # Sortino (downside deviation)
        negative_returns = returns[returns < 0]
        downside_std = float(negative_returns.std() * np.sqrt(252)) if len(negative_returns) > 0 else 0.0
        sortino = excess_return / downside_std if downside_std > 0 else 0.0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0
        max_dd_date = drawdowns.idxmin() if len(drawdowns) > 0 else None

        # Win rate
        win_rate = float((returns > 0).mean()) if len(returns) > 0 else 0.0

        # Profit factor
        gross_profit = float(returns[returns > 0].sum()) if len(returns) > 0 else 0.0
        gross_loss = abs(float(returns[returns < 0].sum())) if len(returns) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        # Trade statistics
        buy_trades = len([t for t in self.enriched_trades if t.side == 'buy'])
        sell_trades = len([t for t in self.enriched_trades if t.side == 'sell'])
        stop_loss_trades = len([t for t in self.enriched_trades if t.reason == 'stop_loss'])
        total_trades = buy_trades + sell_trades

        # Holding period statistics
        closed_trades = [t for t in self.enriched_trades if t.is_closed and t.holding_period_days is not None]
        avg_holding = float(np.mean([t.holding_period_days for t in closed_trades])) if closed_trades else None

        # Win/loss statistics
        winning_trades = [t for t in closed_trades if (t.realized_pnl or 0) > 0]
        losing_trades = [t for t in closed_trades if (t.realized_pnl or 0) <= 0]

        avg_win_pct = None
        avg_loss_pct = None
        largest_win_pct = None
        largest_loss_pct = None
        if winning_trades:
            win_pcts = [t.realized_pnl_pct for t in winning_trades if t.realized_pnl_pct is not None]
            if win_pcts:
                avg_win_pct = float(np.mean(win_pcts))
                largest_win_pct = float(max(win_pcts))
        if losing_trades:
            loss_pcts = [t.realized_pnl_pct for t in losing_trades if t.realized_pnl_pct is not None]
            if loss_pcts:
                avg_loss_pct = float(np.mean(loss_pcts))
                largest_loss_pct = float(min(loss_pcts))

        # Average trade P&L
        all_pnls = [t.realized_pnl for t in closed_trades if t.realized_pnl is not None]
        avg_trade_pnl = float(np.mean(all_pnls)) if all_pnls else None

        return PerformanceStats(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_date=max_dd_date,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            stop_loss_trades=stop_loss_trades,
            avg_holding_days=avg_holding,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            largest_win_pct=largest_win_pct,
            largest_loss_pct=largest_loss_pct,
            avg_trade_pnl=avg_trade_pnl,
        )

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

    def get_risk_metrics(self, var_confidence: float = 0.95) -> RiskMetrics:
        """
        Calculate risk metrics (VaR, CVaR, etc.).

        Args:
            var_confidence: Confidence level for VaR (default 0.95 = 95%)
        """
        if len(self._returns_net) == 0:
            return self._empty_risk_metrics()

        returns = self._returns_net.dropna()

        # Value at Risk (VaR)
        var_95 = float(np.percentile(returns, (1 - 0.95) * 100))
        var_99 = float(np.percentile(returns, (1 - 0.99) * 100))

        # Conditional VaR (CVaR) - average of losses beyond VaR
        cvar_95 = float(returns[returns <= var_95].mean()) if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = float(returns[returns <= var_99].mean()) if len(returns[returns <= var_99]) > 0 else var_99

        # Volatility
        volatility = float(returns.std() * np.sqrt(252))

        # Downside deviation
        negative_returns = returns[returns < 0]
        downside_deviation = float(negative_returns.std() * np.sqrt(252)) if len(negative_returns) > 0 else 0.0

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_dd = float(drawdowns.min())
        max_dd_date = drawdowns.idxmin() if len(drawdowns) > 0 else None

        # Average drawdown
        avg_drawdown = float(drawdowns[drawdowns < 0].mean()) if len(drawdowns[drawdowns < 0]) > 0 else 0.0
        drawdown_days = int((drawdowns < 0).sum())

        # Recovery days (days from max drawdown to new high)
        recovery_days = None
        if max_dd_date is not None:
            post_max_idx = drawdowns.index.get_loc(max_dd_date)
            if post_max_idx < len(drawdowns) - 1:
                post_max_drawdowns = drawdowns.iloc[post_max_idx:]
                new_high_idx = post_max_drawdowns[post_max_drawdowns >= 1.0].index
                if len(new_high_idx) > 0:
                    recovery_days = (new_high_idx[0] - max_dd_date).days

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_drawdown=max_dd,
            max_drawdown_date=max_dd_date,
            recovery_days=recovery_days,
            avg_drawdown=avg_drawdown,
            drawdown_days=drawdown_days,
        )

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

    def get_regime_breakdown(self) -> List[RegimePerformance]:
        """
        Calculate performance breakdown by market regime.

        Returns list of RegimePerformance, one per regime.
        """
        if len(self._returns_net) == 0 or 'regime' not in self.signals.columns:
            return []

        returns = self._returns_net
        regimes = self.signals['regime']

        breakdown = []
        for regime in ['strong_bull', 'weak_bull', 'weak_bear', 'strong_bear']:
            mask = regimes == regime
            regime_returns = returns[mask]

            if len(regime_returns) == 0:
                continue

            regime_trades = [t for t in self.enriched_trades if t.regime_at_entry == regime]
            exposure = self._get_regime_exposure(regime)

            total_ret = float(regime_returns.sum())
            avg_ret = float(regime_returns.mean())

            win_rate = float((regime_returns > 0).mean()) if len(regime_returns) > 0 else 0.0

            breakdown.append(RegimePerformance(
                regime=regime,
                days=int(mask.sum()),
                total_return=total_ret,
                avg_return=avg_ret,
                win_rate=win_rate,
                trades=len(regime_trades),
                exposure=exposure,
            ))

        return breakdown

    def _get_regime_exposure(self, regime: str) -> float:
        """Get the exposure setting for a regime."""
        exposure_map = {
            'strong_bull': 1.0,
            'weak_bull': 0.75,
            'weak_bear': 0.25,
            'strong_bear': 0.0,
        }
        return exposure_map.get(regime, 0.5)

    def get_attribution(self) -> List[AttributionEntry]:
        """
        Calculate stock-level attribution (contribution to returns).

        Returns list of AttributionEntry sorted by contribution.
        """
        if not self.enriched_trades:
            return []

        # Group by symbol
        symbol_data = {}
        for trade in self.enriched_trades:
            if trade.symbol not in symbol_data:
                symbol_data[trade.symbol] = {
                    'trades': [],
                    'total_pnl': 0.0,
                    'realized_pnl': 0.0,
                    'holding_days': [],
                }
            symbol_data[trade.symbol]['trades'].append(trade)
            if trade.realized_pnl is not None:
                symbol_data[trade.symbol]['realized_pnl'] += trade.realized_pnl
            if trade.holding_period_days is not None:
                symbol_data[trade.symbol]['holding_days'].append(trade.holding_period_days)

        # Calculate total P&L for percentage
        total_pnl = sum(d['realized_pnl'] for d in symbol_data.values())
        if total_pnl == 0:
            total_pnl = 1.0  # Avoid division by zero

        attribution = []
        for symbol, data in symbol_data.items():
            trades = data['trades']
            realized = data['realized_pnl']
            holdings = data['holding_days']

            # Find best/worst trade
            trade_pcts = [t.realized_pnl_pct for t in trades if t.realized_pnl_pct is not None]
            best_trade = max(trade_pcts) if trade_pcts else None
            worst_trade = min(trade_pcts) if trade_pcts else None

            attribution.append(AttributionEntry(
                symbol=symbol,
                total_return=realized,
                contribution_pct=realized / total_pnl * 100 if total_pnl != 0 else 0.0,
                trades=len(trades),
                avg_holding_days=float(np.mean(holdings)) if holdings else 0.0,
                best_trade_pct=best_trade,
                worst_trade_pct=worst_trade,
            ))

        # Sort by contribution descending
        attribution.sort(key=lambda x: abs(x.contribution_pct), reverse=True)
        return attribution

    def get_stock_performance(self, symbol: str) -> Optional[AttributionEntry]:
        """
        Calculate comprehensive performance metrics for a single stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            AttributionEntry with full metrics, or None if no trades found
        """
        if not self.enriched_trades:
            return None

        # Filter trades for this symbol
        stock_trades = [t for t in self.enriched_trades if t.symbol == symbol]
        if not stock_trades:
            return None

        closed_trades = [t for t in stock_trades if t.is_closed and t.realized_pnl is not None]
        if not closed_trades:
            # Only open positions
            return AttributionEntry(
                symbol=symbol,
                total_return=sum(t.unrealized_pnl for t in stock_trades),
                contribution_pct=0.0,
                trades=len(stock_trades),
                avg_holding_days=0.0,
            )

        # Win/loss analysis
        winning_trades = [t for t in closed_trades if (t.realized_pnl or 0) > 0]
        losing_trades = [t for t in closed_trades if (t.realized_pnl or 0) <= 0]

        total_pnl = sum(t.realized_pnl for t in closed_trades)
        gross_profit = sum(t.realized_pnl for t in winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(t.realized_pnl for t in losing_trades)) if losing_trades else 0.0

        win_pcts = [t.realized_pnl_pct for t in winning_trades if t.realized_pnl_pct is not None]
        loss_pcts = [t.realized_pnl_pct for t in losing_trades if t.realized_pnl_pct is not None]

        # Calculate Sharpe ratio for this stock
        stock_returns = pd.Series([t.realized_pnl_pct for t in closed_trades if t.realized_pnl_pct is not None])
        if len(stock_returns) > 1 and stock_returns.std() > 0:
            sharpe = (stock_returns.mean() * 252) / (stock_returns.std() * np.sqrt(252))
        else:
            sharpe = 0.0

        # Calculate max drawdown for this stock
        if len(stock_returns) > 0:
            cumulative = (1 + stock_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = cumulative / rolling_max - 1
            max_dd = float(drawdowns.min()) if len(drawdowns) > 0 else 0.0
        else:
            max_dd = 0.0

        # Volatility
        volatility = float(stock_returns.std() * np.sqrt(252)) if len(stock_returns) > 0 else 0.0

        # Alpha vs benchmark
        alphas = []
        for t in closed_trades:
            if t.realized_pnl_pct is not None and t.benchmark_return is not None:
                alphas.append(t.realized_pnl_pct - t.benchmark_return)
        alpha = float(np.mean(alphas)) if alphas else 0.0

        # Regime breakdown
        regime_breakdown = {}
        for regime in ['strong_bull', 'weak_bull', 'weak_bear', 'strong_bear']:
            regime_trades = [t for t in closed_trades if t.regime_at_entry == regime]
            if regime_trades:
                regime_wins = [t for t in regime_trades if (t.realized_pnl or 0) > 0]
                regime_pnls = [t.realized_pnl for t in regime_trades if t.realized_pnl is not None]
                regime_breakdown[regime] = {
                    'trades': len(regime_trades),
                    'wins': len(regime_wins),
                    'win_rate': len(regime_wins) / len(regime_trades) if regime_trades else 0.0,
                    'total_pnl': sum(regime_pnls) if regime_pnls else 0.0,
                    'avg_return': float(np.mean(regime_pnls)) if regime_pnls else 0.0,
                }

        # Reason breakdown
        reason_breakdown = {}
        for reason in ['stop_loss', 'rebalance', 'regime_exit', 'take_profit', 'momentum']:
            reason_trades = [t for t in closed_trades if t.reason == reason]
            if reason_trades:
                reason_wins = [t for t in reason_trades if (t.realized_pnl or 0) > 0]
                reason_pnls = [t.realized_pnl for t in reason_trades if t.realized_pnl is not None]
                reason_breakdown[reason] = {
                    'trades': len(reason_trades),
                    'wins': len(reason_wins),
                    'win_rate': len(reason_wins) / len(reason_trades) if reason_trades else 0.0,
                    'total_pnl': sum(reason_pnls) if reason_pnls else 0.0,
                }

        # Total P&L for contribution calculation
        total_portfolio_pnl = sum(
            t.realized_pnl for t in self.enriched_trades
            if t.is_closed and t.realized_pnl is not None
        )
        if total_portfolio_pnl == 0:
            total_portfolio_pnl = 1.0

        holdings = [t.holding_period_days for t in stock_trades if t.holding_period_days is not None]
        trade_pcts = [t.realized_pnl_pct for t in closed_trades if t.realized_pnl_pct is not None]

        return AttributionEntry(
            symbol=symbol,
            total_return=total_pnl,
            contribution_pct=total_pnl / total_portfolio_pnl * 100,
            trades=len(stock_trades),
            avg_holding_days=float(np.mean(holdings)) if holdings else 0.0,
            best_trade_pct=max(trade_pcts) if trade_pcts else None,
            worst_trade_pct=min(trade_pcts) if trade_pcts else None,
            # Win/loss stats
            win_rate=len(winning_trades) / len(closed_trades) if closed_trades else 0.0,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0,
            avg_win_pct=float(np.mean(win_pcts)) if win_pcts else None,
            avg_loss_pct=float(np.mean(loss_pcts)) if loss_pcts else None,
            largest_win_pct=max(win_pcts) if win_pcts else None,
            largest_loss_pct=min(loss_pcts) if loss_pcts else None,
            # Risk metrics
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            volatility=volatility,
            # Alpha
            alpha_vs_benchmark=alpha,
            # Breakdowns
            regime_breakdown=regime_breakdown,
            reason_breakdown=reason_breakdown,
        )

    def get_all_stock_performance(self) -> List[AttributionEntry]:
        """
        Get comprehensive performance for all stocks.

        Returns:
            List of AttributionEntry sorted by contribution percentage (descending)
        """
        if not self.enriched_trades:
            return []

        # Get unique symbols
        symbols = set(t.symbol for t in self.enriched_trades)
        performances = []

        for symbol in symbols:
            perf = self.get_stock_performance(symbol)
            if perf is not None:
                performances.append(perf)

        # Sort by absolute contribution descending
        performances.sort(key=lambda x: abs(x.contribution_pct), reverse=True)
        return performances

    def get_regime_stock_matrix(self) -> pd.DataFrame:
        """
        Generate a matrix of stock returns by regime.

        Returns:
            DataFrame with stocks as rows, regimes as columns, values = average returns
        """
        if not self.enriched_trades:
            return pd.DataFrame()

        # Collect data
        data = []
        for trade in self.enriched_trades:
            if trade.is_closed and trade.realized_pnl_pct is not None:
                data.append({
                    'symbol': trade.symbol,
                    'regime': trade.regime_at_entry,
                    'return': trade.realized_pnl_pct,
                })

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Pivot to create matrix
        matrix = df.pivot_table(
            values='return',
            index='symbol',
            columns='regime',
            aggfunc='mean',
            fill_value=0.0
        )

        return matrix

    def get_stock_rankings_history(self) -> pd.DataFrame:
        """
        Track how stock rankings changed over time.

        Returns:
            DataFrame with date, symbol, rank, momentum_score columns
        """
        if not self.signals or 'momentum_scores' not in self.signals.columns:
            return pd.DataFrame()

        # Extract momentum scores over time if available
        # This requires momentum_scores to be stored in signals during backtest
        # For now, return empty DataFrame as placeholder
        return pd.DataFrame()

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Generate equity curve data for charting.

        Returns DataFrame with timestamp index, equity, benchmark columns.
        """
        if len(self._returns_net) == 0:
            return pd.DataFrame()

        dates = self._returns_net.index
        equity = self.initial_capital * (1 + self._cumulative)
        benchmark = self.initial_capital * (1 + self._benchmark_cumulative)

        df = pd.DataFrame({
            'equity': equity.values,
            'benchmark': benchmark.values if len(self._benchmark_cumulative) > 0 else equity.values,
            'daily_return': self._returns_net.values,
            'regime': self.signals['regime'].values if 'regime' in self.signals.columns else ['unknown'] * len(dates),
        }, index=dates)

        return df

    def get_drawdown_series(self) -> pd.Series:
        """Calculate drawdown series for charting."""
        if len(self._returns_net) == 0:
            return pd.Series(dtype=float)

        cumulative = (1 + self._returns_net).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max - 1)

        return drawdown

    def get_trade_journal(self) -> pd.DataFrame:
        """
        Format trade history for display.

        Returns DataFrame with trade details.
        """
        if not self.enriched_trades:
            return pd.DataFrame()

        rows = []
        for t in self.enriched_trades:
            rows.append({
                'date': t.entry_date,
                'symbol': t.symbol,
                'side': t.side.upper(),
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'qty': t.qty,
                'pnl': t.realized_pnl if t.is_closed else t.unrealized_pnl,
                'pnl_pct': t.realized_pnl_pct if t.is_closed else None,
                'holding_days': t.holding_period_days,
                'regime': t.regime_at_entry,
                'reason': t.reason,
                'status': 'Closed' if t.is_closed else 'Open',
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('date', ascending=False)
        return df

    def get_portfolio_snapshots(self) -> List[PortfolioSnapshot]:
        """
        Generate daily portfolio snapshots.

        Returns list of PortfolioSnapshot for equity curve charting.
        """
        if len(self._returns_net) == 0:
            return []

        snapshots = []
        cumulative = (1 + self._returns_net).cumprod()
        benchmark_cumulative = (1 + self._benchmark_cumulative).cumprod() if len(self._benchmark_cumulative) > 0 else cumulative

        for i, (date, row) in enumerate(self.signals.iterrows()):
            if i >= len(cumulative) or i >= len(benchmark_cumulative):
                continue

            regime = row.get('regime', 'unknown') if 'regime' in row else 'unknown'

            snapshots.append(PortfolioSnapshot(
                timestamp=date,
                equity=float(self.initial_capital * (1 + cumulative.iloc[i])),
                cash=0.0,  # Would need live data
                positions_value=0.0,  # Would need live data
                daily_return=float(self._returns_net.iloc[i]) if i < len(self._returns_net) else 0.0,
                cumulative_return=float(cumulative.iloc[i]),
                benchmark_cumulative=float(benchmark_cumulative.iloc[i]),
                current_regime=regime,
                regime_numeric=int(row.get('regime_numeric', 0)) if 'regime_numeric' in row else 0,
                positions={},  # Populated from live data
                exposure=1.0,
            ))

        return snapshots


def safe_compute(func):
    """
    Decorator for safe metric computation.

    Returns None or default value on error instead of crashing.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Error computing {func.__name__}: {e}")
            return None
    return wrapper
