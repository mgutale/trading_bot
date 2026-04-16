"""
Dashboard Data Models

Enhanced data structures for the analytics dashboard.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Optional
from enum import Enum
from collections import defaultdict


class TradeReason(Enum):
    """Reason for trade execution."""
    BUY = "buy"
    SELL = "sell"
    STOP_LOSS = "stop_loss"
    REBALANCE = "rebalance"
    REGIME_EXIT = "regime_exit"
    TAKE_PROFIT = "take_profit"
    MOMENTUM = "momentum"
    OPEN_POSITION = "open_position"


@dataclass
class EnrichedTrade:
    """Complete trade record with entry/exit prices, P&L, and context.

    This is the core data structure for the trade journal and attribution analysis.
    """
    entry_date: datetime
    exit_date: Optional[datetime]
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    exit_price: Optional[float]  # None if position still open
    qty: float
    entry_stop_price: float
    realized_pnl: Optional[float]  # Realized when closed
    realized_pnl_pct: Optional[float]
    unrealized_pnl: float  # Current unrealized (for open positions)
    total_pnl: float  # realized + unrealized
    holding_period_days: Optional[int]
    regime_at_entry: str
    regime_at_exit: Optional[str]
    reason: str  # stop_loss, rebalance, regime_exit, etc.
    momentum_score: Optional[float] = None
    benchmark_return: Optional[float] = None
    entry_benchmark: Optional[float] = None  # Benchmark value at entry

    @property
    def is_closed(self) -> bool:
        """Whether the position has been closed."""
        return self.exit_date is not None and self.exit_price is not None

    @property
    def pnl_display(self) -> str:
        """Formatted P&L for display."""
        if self.realized_pnl is not None:
            return f"${self.realized_pnl:+,.2f}"
        return f"${self.unrealized_pnl:+,.2f} (unrealized)"

    @property
    def reason_display(self) -> str:
        """Human-readable reason."""
        reason_map = {
            'stop_loss': 'Stop Loss',
            'rebalance': 'Rebalance',
            'regime_exit': 'Regime Exit',
            'take_profit': 'Take Profit',
            'momentum': 'Momentum Signal',
            'open_position': 'Open Position',
        }
        return reason_map.get(self.reason, self.reason.replace('_', ' ').title())


@dataclass
class PortfolioSnapshot:
    """Single point-in-time portfolio state.

    Used for equity curve and drawdown charting.
    """
    timestamp: datetime
    equity: float
    cash: float
    positions_value: float
    daily_return: float
    cumulative_return: float
    benchmark_cumulative: float
    current_regime: str
    regime_numeric: int
    positions: Dict[str, Dict]  # symbol -> position details
    exposure: float

    @property
    def total_value(self) -> float:
        """Total portfolio value."""
        return self.equity

    @property
    def positions_count(self) -> int:
        """Number of open positions."""
        return len(self.positions)


@dataclass
class DashboardConfig:
    """Configuration for dashboard behavior."""
    update_interval_seconds: int = 5  # For live mode
    max_trade_history: int = 500
    initial_capital: float = 10000.0
    benchmark_symbol: str = "SPY"
    theme: str = "light"  # light or dark
    show_rationale: bool = True
    refresh_interval: int = 10  # seconds between refreshes


@dataclass
class PerformanceStats:
    """Aggregated performance statistics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    buy_trades: int
    sell_trades: int
    stop_loss_trades: int
    max_drawdown_date: Optional[datetime] = None
    avg_holding_days: Optional[float] = None
    avg_win_pct: Optional[float] = None
    avg_loss_pct: Optional[float] = None
    largest_win_pct: Optional[float] = None
    largest_loss_pct: Optional[float] = None
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    avg_trade_pnl: Optional[float] = None


@dataclass
class RiskMetrics:
    """Risk metrics (VaR, CVaR, etc.)."""
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    cvar_95: float  # Conditional VaR at 95%
    cvar_99: float  # Conditional VaR at 99%
    volatility: float
    downside_deviation: float
    max_drawdown: float
    max_drawdown_date: Optional[datetime] = None
    recovery_days: Optional[int] = None
    avg_drawdown: float = 0.0
    drawdown_days: int = 0


@dataclass
class RegimePerformance:
    """Performance breakdown by market regime."""
    regime: str
    days: int
    total_return: float
    avg_return: float
    win_rate: float
    trades: int
    exposure: float


@dataclass
class AttributionEntry:
    """Stock-level attribution data."""
    symbol: str
    total_return: float
    contribution_pct: float  # % of portfolio return
    trades: int
    avg_holding_days: float
    best_trade_pct: Optional[float] = None
    worst_trade_pct: Optional[float] = None


@dataclass
class LiveComparisonData:
    """Comparison between live trading and backtest expectations."""
    live_equity: float
    backtest_equity: float
    live_return: float
    backtest_return: float
    live_daily_pnl: float
    backtest_expected_daily: float
    tracking_error: float
    correlation: float
    regime: str
    live_positions: int
    backtest_expected_positions: int


def create_enriched_trade(
    entry_date: datetime,
    symbol: str,
    side: str,
    entry_price: float,
    exit_date: Optional[datetime],
    exit_price: Optional[float],
    qty: float,
    entry_stop_price: float,
    regime_at_entry: str,
    reason: str,
    regime_at_exit: Optional[str] = None,
    momentum_score: Optional[float] = None,
) -> EnrichedTrade:
    """Factory function to create an EnrichedTrade with calculated fields."""
    if exit_date and exit_price is not None:
        realized_pnl = (exit_price - entry_price) * qty
        realized_pnl_pct = (exit_price - entry_price) / entry_price
        holding_days = (exit_date - entry_date).days if isinstance(entry_date, datetime) else None
        total_pnl = realized_pnl
        unrealized_pnl = 0.0
    else:
        realized_pnl = None
        realized_pnl_pct = None
        holding_days = None
        total_pnl = 0.0
        unrealized_pnl = 0.0

    return EnrichedTrade(
        entry_date=entry_date,
        exit_date=exit_date,
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        qty=qty,
        entry_stop_price=entry_stop_price,
        realized_pnl=realized_pnl,
        realized_pnl_pct=realized_pnl_pct,
        unrealized_pnl=unrealized_pnl,
        total_pnl=total_pnl,
        holding_period_days=holding_days,
        regime_at_entry=regime_at_entry,
        regime_at_exit=regime_at_exit,
        reason=reason,
        momentum_score=momentum_score,
    )


def trades_to_dataframe(trades: List[EnrichedTrade]) -> 'pd.DataFrame':
    """Convert list of EnrichedTrade to pandas DataFrame for display."""
    import pandas as pd

    if not trades:
        return pd.DataFrame()

    rows = []
    for t in trades:
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
            'reason': t.reason_display,
            'status': 'Closed' if t.is_closed else 'Open',
        })

    return pd.DataFrame(rows)
