"""
Live Comparison Component

Live trading vs backtest comparison display.
"""

from typing import Dict, Any, Optional
from dash import html
import plotly.graph_objects as go


def create_live_comparison(
    live_data: Dict[str, Any],
    backtest_data: Optional[Dict[str, Any]] = None
) -> html.Div:
    """
    Create live vs backtest comparison display.

    Args:
        live_data: Current live trading data
        backtest_data: Expected backtest metrics for comparison

    Returns:
        Dash HTML component
    """
    if not live_data:
        return html.Div([])

    live_return = live_data.get('total_return', 0)
    backtest_return = backtest_data.get('total_return', 0) if backtest_data else 0
    difference = live_return - backtest_return

    live_color = '#2ca02c' if live_return >= 0 else '#d62728'
    diff_color = '#2ca02c' if difference >= 0 else '#d62728'

    cards = [
        _create_comparison_card('Live Return', f"{live_return:.2%}", live_color),
        _create_comparison_card('Backtest Return', f"{backtest_return:.2%}", '#1f77b4'),
        _create_comparison_card('Difference', f"{difference:+.2%}", diff_color),
        _create_comparison_card('Live Positions', str(live_data.get('positions', 0)), '#1f77b4'),
    ]

    return html.Div([
        html.H4("Live vs Backtest", style={'marginBottom': '10px'}),
        html.Div(cards, className='comparison-cards')
    ], className='live-comparison')


def _create_comparison_card(name: str, value: str, color: str = '#1f77b4') -> html.Div:
    """Create a single comparison card."""
    return html.Div([
        html.Span(name, className='comparison-label'),
        html.Br(),
        html.Span(value, className='comparison-value', style={'color': color}),
    ], className='comparison-card')


def create_tracking_error_chart(
    live_returns: list,
    backtest_returns: list
) -> go.Figure:
    """
    Create tracking error analysis chart.

    Args:
        live_returns: List of live daily returns
        backtest_returns: List of backtest daily returns

    Returns:
        Plotly Figure
    """
    if not live_returns or not backtest_returns:
        return _empty_figure("No data")

    # Align lengths
    min_len = min(len(live_returns), len(backtest_returns))
    live_returns = live_returns[:min_len]
    backtest_returns = backtest_returns[:min_len]

    # Calculate cumulative returns
    live_cumulative = _cumulative_returns(live_returns)
    backtest_cumulative = _cumulative_returns(backtest_returns)

    # Tracking difference
    tracking_diff = [l - b for l, b in zip(live_cumulative, backtest_cumulative)]

    fig = go.Figure()

    # Cumulative returns
    fig.add_trace(go.Scatter(
        y=live_cumulative,
        name='Live',
        line=dict(color='#1f77b4', width=2),
    ))
    fig.add_trace(go.Scatter(
        y=backtest_cumulative,
        name='Backtest',
        line=dict(color='#17becf', width=2, dash='dash'),
    ))

    fig.update_layout(
        title=dict(text='Live vs Backtest Cumulative Returns', font=dict(size=16, color='#ffffff')),
        height=300,
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis=dict(title='Cumulative Return', tickformat='.1%', gridcolor='#404040', tickcolor='#404040'),
        xaxis=dict(title='Trading Days', gridcolor='#404040', tickcolor='#404040'),
    )

    return fig


def create_live_equity_chart(
    live_snapshots: list,
    backtest_snapshots: Optional[list] = None
) -> go.Figure:
    """
    Create live equity curve with backtest comparison.

    Args:
        live_snapshots: List of portfolio snapshots during live trading
        backtest_snapshots: Corresponding backtest snapshots for comparison

    Returns:
        Plotly Figure
    """
    if not live_snapshots:
        return _empty_figure("No live data")

    # Extract equity values
    live_dates = [s.get('timestamp') for s in live_snapshots]
    live_equity = [s.get('equity', 0) for s in live_snapshots]

    fig = go.Figure()

    # Live equity curve
    fig.add_trace(go.Scatter(
        x=live_dates,
        y=live_equity,
        name='Live',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>Equity: $%{y:,.0f}<extra></extra>'
    ))

    # Backtest comparison if available
    if backtest_snapshots:
        backtest_dates = [s.get('timestamp') for s in backtest_snapshots]
        backtest_equity = [s.get('equity', 0) for s in backtest_snapshots]

        fig.add_trace(go.Scatter(
            x=backtest_dates,
            y=backtest_equity,
            name='Backtest',
            line=dict(color='#17becf', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>Equity: $%{y:,.0f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text='Live Trading vs Backtest', font=dict(size=16, color='#ffffff')),
        height=350,
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis=dict(title='Portfolio Value ($)', tickformat='$,.0f', gridcolor='#404040', tickcolor='#404040'),
        xaxis=dict(title='Date', gridcolor='#404040', tickcolor='#404040'),
        hovermode='x unified',
    )

    return fig


def create_regime_comparison(
    live_regime: str,
    expected_regime: str
) -> html.Div:
    """
    Create regime comparison display.

    Args:
        live_regime: Current regime in live trading
        expected_regime: Expected regime from backtest

    Returns:
        Dash HTML component
    """
    match = live_regime == expected_regime
    status_color = '#2ca02c' if match else '#ff7f0e'
    status_text = 'Matching' if match else 'Diverged'

    return html.Div([
        html.H4("Regime Alignment", style={'marginBottom': '10px'}),
        html.Div([
            html.Span(f"Live Regime: ", style={'fontWeight': 'bold'}),
            html.Span(live_regime.replace('_', ' ').title()),
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span(f"Expected Regime: ", style={'fontWeight': 'bold'}),
            html.Span(expected_regime.replace('_', ' ').title()),
        ], style={'marginBottom': '5px'}),
        html.Div([
            html.Span("Status: ", style={'fontWeight': 'bold'}),
            html.Span(status_text, style={'color': status_color}),
        ]),
    ], className='regime-comparison')


def _cumulative_returns(returns: list) -> list:
    """Calculate cumulative returns from daily returns."""
    cumulative = [1.0]
    for r in returns:
        cumulative.append(cumulative[-1] * (1 + r))
    return cumulative[1:]  # Remove initial 1.0


def _empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray")
    )
    fig.update_layout(template='plotly_white', height=250)
    return fig
