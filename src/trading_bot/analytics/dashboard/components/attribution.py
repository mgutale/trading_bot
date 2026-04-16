"""
Attribution Component

Stock-level contribution analysis.
"""

from typing import List, Dict, Any
from dash import html
import plotly.graph_objects as go


def create_attribution_chart(attribution: List[Dict[str, Any]]) -> go.Figure:
    """
    Create stock attribution bar chart.

    Args:
        attribution: List of attribution dicts with symbol, contribution_pct, total_return

    Returns:
        Plotly Figure
    """
    if not attribution:
        return _empty_figure("No attribution data")

    symbols = [a.get('symbol', 'Unknown') for a in attribution]
    contributions = [a.get('contribution_pct', 0) for a in attribution]
    returns = [a.get('total_return', 0) for a in attribution]

    # Color by return (green for positive, red for negative)
    colors = ['#2ca02c' if r >= 0 else '#d62728' for r in returns]

    fig = go.Figure(data=[go.Bar(
        x=symbols,
        y=contributions,
        marker_color=colors,
        text=[f'{c:.1f}%' for c in contributions],
        textposition='outside',
    )])

    fig.update_layout(
        title=dict(text='Stock Contribution to Returns', font=dict(size=16, color='#ffffff')),
        height=350,
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        yaxis=dict(title='Contribution (%)', gridcolor='#404040', tickcolor='#404040'),
        xaxis=dict(title='Symbol', gridcolor='#404040', tickcolor='#404040'),
        showlegend=False,
    )

    return fig


def create_attribution_table(attribution: List[Dict[str, Any]]) -> html.Div:
    """
    Create stock attribution summary table.

    Args:
        attribution: List of attribution dicts

    Returns:
        Dash HTML component
    """
    if not attribution:
        return html.Div([
            html.H4("Stock Attribution"),
            html.P("No attribution data", style={'color': 'gray'})
        ])

    rows = []
    for a in attribution:
        symbol = a.get('symbol', 'Unknown')
        total_return = a.get('total_return', 0)
        contribution = a.get('contribution_pct', 0)
        trades = a.get('trades', 0)
        avg_holding = a.get('avg_holding_days', 0)
        best = a.get('best_trade_pct')
        worst = a.get('worst_trade_pct')

        rows.append(html.Tr([
            html.Td(symbol, style={'fontWeight': 'bold'}),
            html.Td(f"${total_return:+,.2f}"),
            html.Td(f"{contribution:.1f}%"),
            html.Td(str(trades)),
            html.Td(f"{avg_holding:.1f}d"),
            html.Td(f"{best:.2%}" if best else "-"),
            html.Td(f"{worst:.2%}" if worst else "-"),
        ]))

    table = html.Table([
        html.Thead(
            html.Tr([
                html.Th('Symbol'),
                html.Th('P&L'),
                html.Th('Contribution'),
                html.Th('Trades'),
                html.Th('Avg Hold'),
                html.Th('Best'),
                html.Th('Worst'),
            ])
        ),
        html.Tbody(rows)
    ], className='attribution-table', style={'width': '100%', 'borderCollapse': 'collapse'})

    return html.Div([
        html.H4("Stock Attribution", style={'marginBottom': '10px'}),
        table
    ], className='attribution-panel')


def create_sector_allocation(positions: List[Dict[str, Any]]) -> go.Figure:
    """
    Create sector allocation pie chart.

    Note: This requires sector data which isn't currently in positions.
    Placeholder for future enhancement.

    Args:
        positions: List of position dicts

    Returns:
        Plotly Figure
    """
    if not positions:
        return _empty_figure("No positions")

    symbols = [p.get('symbol', 'Unknown') for p in positions]
    values = [p.get('market_value', 0) for p in positions]

    fig = go.Figure(data=[go.Pie(
        labels=symbols,
        values=values,
        hole=0.3,
        textinfo='label+percent',
    )])

    fig.update_layout(
        title=dict(text='Portfolio Allocation', font=dict(size=16)),
        height=300,
        template='plotly_dark',
        plot_bgcolor='#21262d',
        paper_bgcolor='#21262d',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
    )

    return fig


def create_returns_waterfall(
    initial_capital: float,
    final_capital: float,
    contributions: List[Dict[str, Any]]
) -> go.Figure:
    """
    Create a waterfall chart showing how returns accumulated.

    Args:
        initial_capital: Starting value
        final_capital: Ending value
        contributions: List of attribution dicts

    Returns:
        Plotly Figure
    """
    if not contributions:
        return _empty_figure("No data")

    # Build waterfall data
    symbols = [c.get('symbol', '') for c in contributions]
    pnl_values = [c.get('total_return', 0) for c in contributions]

    # Add initial and final
    x_labels = ['Start'] + symbols + ['End']
    y_values = [0] + pnl_values + [sum(pnl_values)]

    fig = go.Figure(data=[go.Waterfall(
        x=x_labels,
        y=y_values,
        measure=['relative'] * (len(x_labels)),
        increasing=dict(marker=dict(color='#2ca02c')),
        decreasing=dict(marker=dict(color='#d62728')),
        totals=dict(marker=dict(color='#1f77b4')),
    )])

    fig.update_layout(
        title=dict(text='Returns Waterfall', font=dict(size=16)),
        height=400,
        template='plotly_dark',
        plot_bgcolor='#21262d',
        paper_bgcolor='#21262d',
        showlegend=False,
    )

    return fig


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
