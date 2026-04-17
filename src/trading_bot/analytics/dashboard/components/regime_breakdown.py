"""
Regime Breakdown Component

Performance breakdown by market regime.
"""

from typing import List, Dict, Any
from dash import html
import plotly.graph_objects as go

from trading_bot.analytics.dashboard.theme import DEFAULT_THEME
from trading_bot.analytics.dashboard.chart_utils import create_figure, create_bar_chart, empty_figure


def create_regime_breakdown(breakdown: List[Dict[str, Any]]) -> html.Div:
    """
    Create regime performance breakdown display.

    Args:
        breakdown: List of regime performance dicts

    Returns:
        Dash HTML component
    """
    if not breakdown:
        return html.Div([
            html.H4("Regime Breakdown"),
            html.P("No regime data available", style={'color': DEFAULT_THEME.text_secondary}),
        ])

    items = []
    for bp in breakdown:
        regime = bp.get('regime', 'unknown')
        color = DEFAULT_THEME.regime_colors.get(regime, DEFAULT_THEME.text_secondary)
        name = _format_regime_name(regime)
        return_color = _get_return_color(bp.get('total_return', 0))

        items.append(html.Div([
            html.Div([
                html.Span(name, style={'fontWeight': 'bold', 'color': color}),
                html.Span(f" ({bp.get('days', 0)} days)", style={'color': DEFAULT_THEME.text_secondary, 'fontSize': '12px'}),
            ], style={'marginBottom': '5px'}),
            html.Div([
                html.Span(f"Return: ", style={'fontSize': '12px'}),
                html.Span(f"{bp.get('total_return', 0):.2%}", style={'fontWeight': 'bold', 'color': return_color}),
                html.Span(f" | Avg: {bp.get('avg_return', 0):.3%}", style={'fontSize': '12px'}),
            ], style={'marginBottom': '3px'}),
            html.Div([
                html.Span(f"Win Rate: {bp.get('win_rate', 0):.1%}", style={'fontSize': '12px'}),
                html.Span(f" | Trades: {bp.get('trades', 0)}", style={'fontSize': '12px'}),
                html.Span(f" | Exposure: {bp.get('exposure', 0):.0%}", style={'fontSize': '12px'}),
            ], style={'marginBottom': '10px'}),
        ], className='regime-breakdown-item'))

    return html.Div([
        html.H4("Performance by Regime", style={'marginBottom': '10px'}),
        html.Div(items, className='regime-breakdown-list'),
    ], className='regime-breakdown')


def create_regime_comparison_chart(breakdown: List[Dict[str, Any]]) -> go.Figure:
    """
    Create bar chart comparing performance across regimes.

    Args:
        breakdown: List of regime performance dicts

    Returns:
        Plotly Figure
    """
    if not breakdown:
        return empty_figure("No regime data", theme=DEFAULT_THEME)

    regimes = [_format_regime_name(bp.get('regime', '')) for bp in breakdown]
    returns = [bp.get('total_return', 0) * 100 for bp in breakdown]
    colors = [DEFAULT_THEME.regime_colors.get(bp.get('regime', ''), DEFAULT_THEME.text_secondary) for bp in breakdown]

    fig = create_bar_chart(
        x=regimes,
        y=returns,
        colors=colors,
        title="",
        y_suffix="%",
    )

    fig.update_layout(
        height=300,
        yaxis=dict(title='Total Return (%)', gridcolor=DEFAULT_THEME.border_default, tickcolor=DEFAULT_THEME.border_default),
        xaxis=dict(title='Regime', gridcolor=DEFAULT_THEME.border_default, tickcolor=DEFAULT_THEME.border_default),
        margin=dict(l=60, r=20, t=20, b=60),
    )

    fig.update_xaxes(tickangle=-45)

    return fig


def create_regime_returns_heatmap(regime_history: List, returns_history: List) -> go.Figure:
    """
    Create a heatmap showing returns by regime over time.

    Args:
        regime_history: List of regime labels
        returns_history: List of corresponding returns

    Returns:
        Plotly Figure
    """
    if not regime_history or not returns_history:
        return empty_figure("No data", theme=DEFAULT_THEME)

    # Create DataFrame for pivot
    import pandas as pd
    df = pd.DataFrame({'regime': regime_history, 'return': returns_history})

    if hasattr(df.index, 'year'):
        df['year'] = df.index.year
        df['month'] = df.index.month
    else:
        df['year'] = range(len(df))
        df['month'] = range(len(df))

    # Pivot
    pivot = df.pivot_table(values='return', index='regime', columns='month', aggfunc='sum')

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f'Month {i}' for i in pivot.columns],
        y=pivot.index,
        colorscale=[
            [0.0, DEFAULT_THEME.danger],
            [0.5, DEFAULT_THEME.warning],
            [1.0, DEFAULT_THEME.success],
        ],
        zmid=0,
        text=pivot.values,
        texttemplate='%{text:.1%}',
        textfont={"color": DEFAULT_THEME.text_primary},
    ))

    fig.update_layout(
        title=dict(text='Returns by Regime and Month', font=dict(size=16)),
        height=350,
        template='plotly_dark',
        plot_bgcolor=DEFAULT_THEME.bg_primary,
        paper_bgcolor=DEFAULT_THEME.bg_primary,
    )

    return fig


def create_regime_duration_chart(regime_duration: Dict[str, int]) -> go.Figure:
    """
    Create a chart showing average duration in each regime.

    Args:
        regime_duration: Dict mapping regime -> total days

    Returns:
        Plotly Figure
    """
    if not regime_duration:
        return empty_figure("No regime data", theme=DEFAULT_THEME)

    regimes = [_format_regime_name(r) for r in regime_duration.keys()]
    days = list(regime_duration.values())
    colors = [DEFAULT_THEME.regime_colors.get(r, DEFAULT_THEME.text_secondary) for r in regime_duration.keys()]

    fig = create_bar_chart(
        x=regimes,
        y=days,
        colors=colors,
        title="",
        y_suffix=" days",
    )

    fig.update_layout(
        height=300,
        yaxis=dict(title='Total Days', gridcolor=DEFAULT_THEME.border_default),
        xaxis=dict(title='Regime', gridcolor=DEFAULT_THEME.border_default),
        margin=dict(l=60, r=20, t=20, b=60),
    )

    fig.update_xaxes(tickangle=-45)

    return fig


def _format_regime_name(regime: str) -> str:
    """Format regime name for display."""
    return regime.replace('_', ' ').title()


def _get_regime_color(regime: str) -> str:
    """Get the color for a regime."""
    return DEFAULT_THEME.regime_colors.get(regime, DEFAULT_THEME.text_secondary)


def _get_return_color(return_val: float) -> str:
    """Get color based on return value."""
    return color_for_return(return_val)


def _empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    return empty_figure(message, theme=DEFAULT_THEME)
