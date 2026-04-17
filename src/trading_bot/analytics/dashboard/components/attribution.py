"""
Attribution Component

Stock-level contribution analysis.
"""

from typing import List, Dict, Any
from dash import html
import plotly.graph_objects as go

from trading_bot.analytics.dashboard.theme import DEFAULT_THEME
from trading_bot.analytics.dashboard.chart_utils import create_figure, create_bar_chart, empty_figure


def create_attribution_chart(attribution: List[Dict[str, Any]]) -> go.Figure:
    """
    Create stock attribution bar chart.

    Args:
        attribution: List of attribution dicts with symbol, contribution_pct, total_return

    Returns:
        Plotly Figure
    """
    if not attribution:
        return empty_figure("No attribution data", theme=DEFAULT_THEME)

    symbols = [a.get('symbol', 'Unknown') for a in attribution]
    contributions = [a.get('contribution_pct', 0) for a in attribution]
    returns = [a.get('total_return', 0) for a in attribution]

    # Color by return (green for positive, red for negative)
    colors = [DEFAULT_THEME.success if r >= 0 else DEFAULT_THEME.danger for r in returns]

    fig = create_bar_chart(
        x=symbols,
        y=contributions,
        colors=colors,
        title="",
        y_suffix="%",
    )

    fig.update_layout(
        height=350,
        yaxis=dict(title='Contribution (%)', gridcolor=DEFAULT_THEME.border_default, tickcolor=DEFAULT_THEME.border_default),
        xaxis=dict(title='Symbol', gridcolor=DEFAULT_THEME.border_default, tickcolor=DEFAULT_THEME.border_default),
        margin=dict(l=60, r=20, t=20, b=80),
    )

    fig.update_xaxes(tickangle=-45)

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
            html.P("No attribution data", style={'color': DEFAULT_THEME.text_secondary}),
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

        return_color = DEFAULT_THEME.success if total_return >= 0 else DEFAULT_THEME.danger

        rows.append(html.Tr([
            html.Td(symbol, style={'fontWeight': 'bold'}),
            html.Td(html.Span(f"${total_return:+,.2f}", style={'color': return_color})),
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
        html.Tbody(rows),
    ], className='attribution-table')

    return html.Div([
        html.H4("Stock Attribution", style={'marginBottom': '10px'}),
        table,
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
        return empty_figure("No positions", theme=DEFAULT_THEME)

    symbols = [p.get('symbol', 'Unknown') for p in positions]
    values = [p.get('market_value', 0) for p in positions]

    fig = go.Figure(data=[go.Pie(
        labels=symbols,
        values=values,
        hole=0.3,
        textinfo='label+percent',
        marker=dict(colors=DEFAULT_THEME.chart_palette[:len(symbols)]),
    )])

    fig.update_layout(
        title=dict(text='Portfolio Allocation', font=dict(size=16)),
        height=300,
        template='plotly_dark',
        plot_bgcolor=DEFAULT_THEME.bg_primary,
        paper_bgcolor=DEFAULT_THEME.bg_primary,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
    )

    return fig


def create_returns_waterfall(
    initial_capital: float,
    final_capital: float,
    contributions: List[Dict[str, Any]],
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
        return empty_figure("No data", theme=DEFAULT_THEME)

    # Build waterfall data
    symbols = [c.get('symbol', '') for c in contributions]
    pnl_values = [c.get('total_return', 0) for c in contributions]

    # Add initial and final
    x_labels = ['Start'] + symbols + ['End']
    y_values = [initial_capital] + pnl_values + [final_capital]

    fig = go.Figure(data=[go.Waterfall(
        x=x_labels,
        y=y_values,
        measure=['absolute'] + ['relative'] * len(symbols) + ['absolute'],
        increasing=dict(marker=dict(color=DEFAULT_THEME.success)),
        decreasing=dict(marker=dict(color=DEFAULT_THEME.danger)),
        totals=dict(marker=dict(color=DEFAULT_THEME.primary)),
    )])

    fig.update_layout(
        title=dict(text='Returns Waterfall', font=dict(size=16)),
        height=400,
        template='plotly_dark',
        plot_bgcolor=DEFAULT_THEME.bg_primary,
        paper_bgcolor=DEFAULT_THEME.bg_primary,
        showlegend=False,
    )

    return fig
