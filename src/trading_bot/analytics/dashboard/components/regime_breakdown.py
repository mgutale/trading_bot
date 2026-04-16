"""
Regime Breakdown Component

Performance breakdown by market regime.
"""

from typing import List, Dict, Any
from dash import html
import plotly.graph_objects as go


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
            html.P("No regime data available", style={'color': 'gray'})
        ])

    regime_colors = {
        'strong_bull': '#006400',
        'weak_bull': '#2ca02c',
        'weak_bear': '#ff7f0e',
        'strong_bear': '#8b0000',
    }

    items = []
    for bp in breakdown:
        regime = bp.get('regime', 'unknown')
        color = regime_colors.get(regime, '#808080')
        name = _format_regime_name(regime)

        items.append(html.Div([
            html.Div([
                html.Span(name, style={'fontWeight': 'bold', 'color': color}),
                html.Span(f" ({bp.get('days', 0)} days)", style={'color': 'gray', 'fontSize': '12px'}),
            ], style={'marginBottom': '5px'}),
            html.Div([
                html.Span(f"Return: ", style={'fontSize': '12px'}),
                html.Span(f"{bp.get('total_return', 0):.2%}", style={'fontWeight': 'bold', 'color': _get_return_color(bp.get('total_return', 0))}),
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
        html.Div(items, className='regime-breakdown-list')
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
        return _empty_figure("No regime data")

    regimes = [_format_regime_name(bp.get('regime', '')) for bp in breakdown]
    returns = [bp.get('total_return', 0) * 100 for bp in breakdown]
    colors = [_get_regime_color(bp.get('regime', '')) for bp in breakdown]

    fig = go.Figure(data=[go.Bar(
        x=regimes,
        y=returns,
        marker_color=colors,
        text=[f'{r:.1f}%' for r in returns],
        textposition='outside',
    )])

    fig.update_layout(
        title=dict(text='Returns by Market Regime', font=dict(size=16, color='#ffffff')),
        height=300,
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        yaxis=dict(title='Total Return (%)', gridcolor='#404040', tickcolor='#404040'),
        xaxis=dict(title='Regime', gridcolor='#404040', tickcolor='#404040'),
        showlegend=False,
    )

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
        return _empty_figure("No data")

    # Create DataFrame for pivot
    import pandas as pd
    df = pd.DataFrame({'regime': regime_history, 'return': returns_history})
    df['year'] = df.index.year if hasattr(df.index, 'year') else range(len(df))
    df['month'] = df.index.month if hasattr(df.index, 'month') else range(len(df))

    # Pivot
    pivot = df.pivot_table(values='return', index='regime', columns='month', aggfunc='sum')

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f'Month {i}' for i in pivot.columns],
        y=pivot.index,
        colorscale=[
            [0.0, '#d62728'],
            [0.5, '#ffdd00'],
            [1.0, '#2ca02c']
        ],
        zmid=0,
        text=pivot.values,
        texttemplate='%{text:.1%}',
        textfont={"color": "white"},
    ))

    fig.update_layout(
        title=dict(text='Returns by Regime and Month', font=dict(size=16)),
        height=350,
        template='plotly_dark',
        plot_bgcolor='#21262d',
        paper_bgcolor='#21262d',
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
        return _empty_figure("No regime data")

    regimes = [_format_regime_name(r) for r in regime_duration.keys()]
    days = list(regime_duration.values())
    colors = [_get_regime_color(r) for r in regime_duration.keys()]

    fig = go.Figure(data=[go.Bar(
        x=regimes,
        y=days,
        marker_color=colors,
        text=[f'{d} days' for d in days],
        textposition='outside',
    )])

    fig.update_layout(
        title=dict(text='Time Spent in Each Regime', font=dict(size=16)),
        height=300,
        template='plotly_dark',
        plot_bgcolor='#21262d',
        paper_bgcolor='#21262d',
        yaxis=dict(title='Total Days', gridcolor='#eee'),
        xaxis=dict(title='Regime', gridcolor='#eee'),
        showlegend=False,
    )

    return fig


def _format_regime_name(regime: str) -> str:
    """Format regime name for display."""
    return regime.replace('_', ' ').title()


def _get_regime_color(regime: str) -> str:
    """Get the color for a regime."""
    colors = {
        'strong_bull': '#006400',
        'weak_bull': '#2ca02c',
        'weak_bear': '#ff7f0e',
        'strong_bear': '#8b0000',
    }
    return colors.get(regime, '#808080')


def _get_return_color(return_val: float) -> str:
    """Get color based on return value."""
    if return_val > 0:
        return '#2ca02c'
    elif return_val < 0:
        return '#d62728'
    return '#808080'


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
