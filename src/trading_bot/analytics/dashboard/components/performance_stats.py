"""
Performance Stats Component

Key performance metrics cards (Sharpe, Sortino, Max DD, etc.).
"""

from typing import Dict, Any
from dash import html
import plotly.graph_objects as go


def create_stats_cards(stats: Dict[str, Any]) -> html.Div:
    """
    Create performance statistics cards.

    Args:
        stats: Dict with performance metrics

    Returns:
        Dash HTML component
    """
    metrics = [
        ('Total Return', stats.get('total_return', 0), '.1%', False),
        ('Annualized Return', stats.get('annualized_return', 0), '.1%', False),
        ('Sharpe Ratio', stats.get('sharpe_ratio', 0), '.2f', False),
        ('Sortino Ratio', stats.get('sortino_ratio', 0), '.2f', False),
        ('Max Drawdown', stats.get('max_drawdown', 0), '.1%', True),
        ('Win Rate', stats.get('win_rate', 0), '.1%', False),
        ('Profit Factor', stats.get('profit_factor', 0), '.2f', False),
        ('Total Trades', stats.get('total_trades', 0), 'd', False),
    ]

    cards = []
    for name, value, fmt, invert in metrics:
        display = _format_value(value, fmt)
        is_positive = _is_positive_value(value, name, invert)
        css_class = 'positive' if is_positive else 'negative'

        cards.append(html.Div([
            html.Span(name, className='metric-label'),
            html.Br(),
            html.Span(display, className=f'metric-value {css_class}'),
        ], className='metric-card'))

    return html.Div(cards, className='stats-cards')


def _format_value(value, fmt: str) -> str:
    """Format a metric value for display."""
    if fmt == 'd':
        return f"{int(value):,}"
    elif '.1%' in fmt:
        return f"{value*100:.1f}%"
    elif '.2%' in fmt:
        return f"{value*100:.2f}%"
    elif '%' in fmt:
        return f"{value*100:.2f}%"
    elif '.2f' in fmt or '.1f' in fmt:
        return f"{value:.2f}"
    else:
        return str(value)


def _is_positive_value(value: float, name: str, invert: bool) -> bool:
    """Determine if a metric value is considered positive."""
    if name in ['Sharpe Ratio', 'Sortino Ratio', 'Profit Factor', 'Win Rate', 'Total Trades']:
        return value > 0
    if invert:
        return value < 0
    return value > 0


def create_detailed_stats_table(stats: Dict[str, Any]) -> html.Table:
    """
    Create a detailed statistics table.

    Args:
        stats: Dict with performance metrics

    Returns:
        Dash HTML table
    """
    rows = [
        ('Total Return', f"{stats.get('total_return', 0):.2%}"),
        ('Annualized Return', f"{stats.get('annualized_return', 0):.2%}"),
        ('Volatility', f"{stats.get('volatility', 0):.2%}"),
        ('Sharpe Ratio', f"{stats.get('sharpe_ratio', 0):.2f}"),
        ('Sortino Ratio', f"{stats.get('sortino_ratio', 0):.2f}"),
        ('Max Drawdown', f"{stats.get('max_drawdown', 0):.2%}"),
        ('Win Rate', f"{stats.get('win_rate', 0):.2%}"),
        ('Profit Factor', f"{stats.get('profit_factor', 0):.2f}"),
        ('Total Trades', f"{stats.get('total_trades', 0)}"),
        ('Buy Trades', f"{stats.get('buy_trades', 0)}"),
        ('Sell Trades', f"{stats.get('sell_trades', 0)}"),
        ('Stop Loss Trades', f"{stats.get('stop_loss_trades', 0)}"),
    ]

    if 'avg_holding_days' in stats and stats['avg_holding_days']:
        rows.append(('Avg Holding Period', f"{stats['avg_holding_days']:.1f} days"))
    if 'avg_trade_pnl' in stats and stats['avg_trade_pnl']:
        rows.append(('Avg Trade P&L', f"${stats['avg_trade_pnl']:+,.2f}"))

    return html.Table([
        html.Tr([html.Td(name, style={'padding': '8px'}), html.Td(value, style={'padding': '8px', 'fontWeight': 'bold'})])
        for name, value in rows
    ], className='detailed-stats-table')


def create_metric_gauge(name: str, value: float, max_value: float = 1.0, unit: str = '%') -> go.Figure:
    """
    Create a gauge chart for a single metric.

    Args:
        name: Metric name
        value: Current value
        max_value: Maximum value for scaling
        unit: Unit suffix

    Returns:
        Plotly Figure
    """
    # Normalize value to 0-100 range
    normalized = min(value / max_value * 100, 100) if max_value > 0 else 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': '#1f77b4'},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': '#2ca02c'},
                {'range': [max_value * 0.5, max_value * 0.75], 'color': '#ff7f0e'},
                {'range': [max_value * 0.75, max_value], 'color': '#d62728'},
            ],
        },
        title={'text': name},
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig
