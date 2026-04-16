"""
Positions Panel Component

Current positions with P&L, stop loss levels, and management.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from dash import html, dash_table, dcc


def create_positions_table(
    positions: List[Dict[str, Any]],
    current_prices: Dict[str, float] = None,
    initial_capital: float = 10000.0
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
            html.P("No open positions", style={'color': 'gray', 'fontStyle': 'italic'})
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

        pnl_color = '#2ca02c' if pnl >= 0 else '#d62728'

        rows.append(html.Tr([
            html.Td(symbol, style={'fontWeight': 'bold'}),
            html.Td(f"{qty:.0f}"),
            html.Td(f"${entry_price:.2f}"),
            html.Td(f"${current_price:.2f}"),
            html.Td(f"${market_value:,.0f}"),
            html.Td(f"${pnl:+,.2f}", style={'color': pnl_color}),
            html.Td(f"{pnl_pct:+.2f}%", style={'color': pnl_color}),
            html.Td(f"${stop_price:.2f}" if stop_price > 0 else "-"),
        ]))

    table = dash_table.DataTable(
        columns=[
            {'name': 'Symbol', 'id': 'symbol'},
            {'name': 'Qty', 'id': 'qty'},
            {'name': 'Entry', 'id': 'entry_price'},
            {'name': 'Current', 'id': 'current_price'},
            {'name': 'Mkt Value', 'id': 'market_value'},
            {'name': 'P&L', 'id': 'pnl'},
            {'name': 'P&L %', 'id': 'pnl_pct'},
            {'name': 'Stop', 'id': 'stop_price'},
        ],
        data=positions,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '8px',
            'fontSize': '14px',
        },
        style_header={
            'backgroundColor': '#161b22',
            'fontWeight': 'bold',
            'color': '#8b949e',
            'border': '1px solid #30363d',
        },
        style_data={
            'backgroundColor': '#21262d',
            'color': '#f0f6fc',
            'border': '1px solid #30363d',
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{pnl} > 0'},
                'color': '#2ca02c',
            },
            {
                'if': {'filter_query': '{pnl} < 0'},
                'color': '#d62728',
            },
        ],
    )

    return html.Div([
        html.H4("Current Positions", style={'marginBottom': '10px'}),
        table
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

    pnl_color = '#2ca02c' if total_pnl >= 0 else '#d62728'

    return html.Div([
        html.Div([
            html.Span(f"Total Value: ", style={'fontWeight': 'bold'}),
            html.Span(f"${total_value:,.2f}"),
        ]),
        html.Div([
            html.Span(f"Total P&L: ", style={'fontWeight': 'bold'}),
            html.Span(f"${total_pnl:+,.2f} ({total_pnl_pct/len(positions):+.2f}%)", style={'color': pnl_color}),
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
        return _empty_figure("No positions")

    labels = [p.get('symbol', 'Unknown') for p in positions]
    values = [p.get('market_value', p.get('current_price', 0) * p.get('qty', 0)) for p in positions]

    colors = ['#1f77b4', '#17becf', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='label+percent',
        hole=0.3,
    )])

    fig.update_layout(
        title=dict(text="Position Allocation", font=dict(size=16, color='#ffffff')),
        height=300,
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
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
