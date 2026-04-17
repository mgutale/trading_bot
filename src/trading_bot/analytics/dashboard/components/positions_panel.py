"""
Positions Panel Component

Current positions with P&L, stop loss levels, and management.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from dash import html, dash_table, dcc

from trading_bot.analytics.dashboard.theme import DEFAULT_THEME
from trading_bot.analytics.dashboard.chart_utils import color_for_return, empty_figure


def create_positions_table(
    positions: List[Dict[str, Any]],
    current_prices: Dict[str, float] = None,
    initial_capital: float = 10000.0,
) -> html.Div:
    """
    Create current positions display table.

    Args:
        positions: List of position dicts with symbol, qty, entry_price, etc.
        current_prices: Dict of symbol -> current price
        initial_capital: For calculating position value percentage

    Returns:
        Dash HTML component
    """
    current_prices = current_prices or {}

    if not positions:
        return html.Div([
            html.H4("Current Positions"),
            html.P("No open positions", style={'color': DEFAULT_THEME.text_secondary, 'fontStyle': 'italic'}),
        ], className="positions-panel")

    rows = []
    for pos in positions:
        symbol = pos.get('symbol', 'Unknown')
        qty = pos.get('qty', 0)
        entry_price = pos.get('entry_price', 0)
        current_price = current_prices.get(symbol, pos.get('current_price', entry_price))
        stop_price = pos.get('stop_price', 0)
        take_profit = pos.get('take_profit_price', 0)

        if current_price <= 0:
            current_price = entry_price

        pnl = (current_price - entry_price) * qty
        pnl_pct = (current_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
        market_value = current_price * qty

        pnl_color = color_for_return(pnl)

        rows.append(html.Tr([
            html.Td(symbol, style={'fontWeight': 'bold'}),
            html.Td(f"{qty:.0f}"),
            html.Td(f"${entry_price:.2f}"),
            html.Td(f"${current_price:.2f}"),
            html.Td(f"${market_value:,.0f}"),
            html.Td(html.Span(f"${pnl:+,.2f}", style={'color': pnl_color})),
            html.Td(html.Span(f"{pnl_pct:+.2f}%", style={'color': pnl_color})),
            html.Td(f"${stop_price:.2f}" if stop_price > 0 else "-"),
        ]))

    table = html.Table([
        html.Thead(
            html.Tr([
                html.Th('Symbol'),
                html.Th('Qty'),
                html.Th('Entry'),
                html.Th('Current'),
                html.Th('Mkt Value'),
                html.Th('P&L'),
                html.Th('P&L %'),
                html.Th('Stop'),
            ])
        ),
        html.Tbody(rows),
    ], className='positions-table')

    return html.Div([
        html.H4("Current Positions", style={'marginBottom': '10px'}),
        table,
    ], className="positions-panel")


def create_position_summary(positions: List[Dict[str, Any]]) -> html.Div:
    """
    Create a summary card showing total positions value and P&L.

    Args:
        positions: List of position dicts

    Returns:
        Dash HTML component
    """
    if not positions:
        return html.Div([])

    total_value = sum(p.get('market_value', p.get('current_price', 0) * p.get('qty', 0)) for p in positions)
    total_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
    total_pnl_pct = sum(p.get('unrealized_pnl_pct', 0) for p in positions)

    pnl_color = color_for_return(total_pnl)

    return html.Div([
        html.Div([
            html.Span(f"Total Value: ", style={'fontWeight': 'bold'}),
            html.Span(f"${total_value:,.2f}"),
        ]),
        html.Div([
            html.Span(f"Total P&L: ", style={'fontWeight': 'bold'}),
            html.Span(html.Span(f"${total_pnl:+,.2f} ({total_pnl_pct/len(positions):+.2f}%)", style={'color': pnl_color})),
        ]),
    ], className="position-summary", style={'marginBottom': '15px', 'fontSize': '16px'})


def create_positions_chart(positions: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a pie chart showing position allocation.

    Args:
        positions: List of position dicts

    Returns:
        Plotly Figure
    """
    if not positions:
        return empty_figure("No positions", theme=DEFAULT_THEME)

    labels = [p.get('symbol', 'Unknown') for p in positions]
    values = [p.get('market_value', p.get('current_price', 0) * p.get('qty', 0)) for p in positions]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=DEFAULT_THEME.chart_palette[:len(labels)]),
        textinfo='label+percent',
        hole=0.3,
    )])

    fig.update_layout(
        title=dict(text="Position Allocation", font=dict(size=16, color=DEFAULT_THEME.text_primary)),
        height=300,
        plot_bgcolor=DEFAULT_THEME.bg_primary,
        paper_bgcolor=DEFAULT_THEME.bg_primary,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
    )

    return fig
