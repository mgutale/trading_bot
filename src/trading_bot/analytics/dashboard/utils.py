"""
Dashboard Utilities

Helper functions for the analytics dashboard.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def format_currency(value: float, show_sign: bool = True) -> str:
    """Format a value as currency."""
    if show_sign:
        return f"${value:+,.2f}"
    return f"${value:,.2f}"


def format_percent(value: float, decimals: int = 2, show_sign: bool = True) -> str:
    """Format a value as percentage."""
    sign = "+" if show_sign and value > 0 else ""
    return f"{sign}{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number with commas."""
    if decimals == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def calculate_drawdown(equity_series: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from equity curve.

    Args:
        equity_series: Series of portfolio values

    Returns:
        Series of drawdown values (negative = drawdown)
    """
    cumulative = equity_series / equity_series.iloc[0] if equity_series.iloc[0] != 0 else equity_series
    rolling_max = cumulative.expanding().max()
    drawdown = cumulative / rolling_max - 1
    return drawdown


def calculate_max_drawdown(equity_series: pd.Series) -> tuple:
    """
    Calculate maximum drawdown and its date.

    Args:
        equity_series: Series of portfolio values

    Returns:
        Tuple of (max_drawdown, max_drawdown_date)
    """
    drawdown = calculate_drawdown(equity_series)
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin() if len(drawdown) > 0 else None
    return max_dd, max_dd_date


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_return = returns.mean() - risk_free_rate / periods_per_year
    return excess_return / returns.std() * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    negative_returns = returns[returns < 0]
    downside_std = negative_returns.std() * np.sqrt(periods_per_year)

    if downside_std == 0:
        return 0.0

    excess_return = returns.mean() - risk_free_rate / periods_per_year
    return excess_return / downside_std * periods_per_year


def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Series of returns
        confidence: Confidence level (default 0.95)

    Returns:
        VaR value (negative for loss)
    """
    if len(returns) == 0:
        return 0.0

    return np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Calculate Conditional VaR (CVaR / Expected Shortfall).

    Args:
        returns: Series of returns
        confidence: Confidence level (default 0.95)

    Returns:
        CVaR value (negative for loss)
    """
    if len(returns) == 0:
        return 0.0

    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()


def get_trade_statistics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics from a list of trades.

    Args:
        trades: List of trade dicts

    Returns:
        Dict with trade statistics
    """
    if not trades:
        return {
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'win_rate': 0.0,
            'avg_pnl': 0.0,
        }

    closed_trades = [t for t in trades if t.get('exit_date')]
    winning_trades = [t for t in closed_trades if (t.get('realized_pnl') or 0) > 0]
    losing_trades = [t for t in closed_trades if (t.get('realized_pnl') or 0) <= 0]

    total_pnl = sum(t.get('realized_pnl', 0) or 0 for t in closed_trades)
    win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0

    return {
        'total_trades': len(trades),
        'closed_trades': len(closed_trades),
        'open_trades': len(trades) - len(closed_trades),
        'buy_trades': len([t for t in trades if t.get('side') == 'buy']),
        'sell_trades': len([t for t in trades if t.get('side') == 'sell']),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': total_pnl / len(closed_trades) if closed_trades else 0,
        'avg_win': sum(t.get('realized_pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0,
        'avg_loss': sum(t.get('realized_pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0,
    }


def get_period_summary(returns: pd.Series, period: str = 'M') -> pd.DataFrame:
    """
    Get returns summary by period (month, quarter, year).

    Args:
        returns: Series of returns
        period: 'M' for month, 'Q' for quarter, 'Y' for year

    Returns:
        DataFrame with period returns
    """
    if len(returns) == 0:
        return pd.DataFrame()

    cumulative = (1 + returns).cumprod()

    if period == 'M':
        grouped = cumulative.resample('ME').last()
    elif period == 'Q':
        grouped = cumulative.resample('QE').last()
    elif period == 'Y':
        grouped = cumulative.resample('YE').last()
    else:
        return pd.DataFrame()

    period_returns = grouped.pct_change().fillna(0)

    return pd.DataFrame({
        'period': period_returns.index,
        'return': period_returns.values,
    })


def merge_snapshots(
    snapshots: List[Dict[str, Any]],
    max_points: int = 500
) -> List[Dict[str, Any]]:
    """
    Merge snapshots to reduce number of points for charting.

    Args:
        snapshots: List of snapshot dicts
        max_points: Maximum number of points to keep

    Returns:
        Downsampled list of snapshots
    """
    if len(snapshots) <= max_points:
        return snapshots

    # Simple uniform sampling
    step = len(snapshots) / max_points
    indices = [int(i * step) for i in range(max_points)]

    return [snapshots[i] for i in indices]


def validate_data(data: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate that required keys are present in data.

    Args:
        data: Dict to validate
        required_keys: List of required keys

    Returns:
        True if valid, False otherwise
    """
    for key in required_keys:
        if key not in data:
            logger.warning(f"Missing required key: {key}")
            return False
    return True


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if b is zero."""
    if b == 0:
        return default
    return a / b


def safe_mean(values: List[float], default: float = 0.0) -> float:
    """Safely calculate mean, returning default if list is empty."""
    if not values:
        return default
    return sum(values) / len(values)
