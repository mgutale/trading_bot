"""
Trade Journal Component

Interactive trade history with filtering by symbol, reason, P&L.
"""

import pandas as pd
from typing import List, Dict, Any
from datetime import datetime
from dash import html, dash_table, dcc

from trading_bot.analytics.dashboard.theme import DEFAULT_THEME
from trading_bot.analytics.dashboard.chart_utils import color_for_return


def create_trade_journal_table(
    trades: List[Dict[str, Any]] = None,
    max_rows: int = 100,
) -> html.Div:
    """
    Create interactive trade journal table.

    Args:
        trades: List of trade dicts
        max_rows: Maximum rows to display

    Returns:
        Dash HTML component
    """
    if not trades:
        return html.Div([
            html.H4("Trade Journal"),
            html.P("No trades to display", style={'color': DEFAULT_THEME.text_secondary, 'fontStyle': 'italic'}),
        ], className="trade-journal")

    # Prepare data for table
    rows = []
    for t in trades[:max_rows]:
        entry_date = t.get('entry_date')
        if isinstance(entry_date, datetime):
            entry_date = entry_date.strftime('%Y-%m-%d')

        exit_date = t.get('exit_date')
        if isinstance(exit_date, datetime):
            exit_date = exit_date.strftime('%Y-%m-%d')

        pnl = t.get('realized_pnl', t.get('pnl', 0))
        pnl_display = f"${pnl:+,.2f}" if pnl is not None else "-"

        pnl_pct = t.get('realized_pnl_pct', t.get('pnl_pct'))
        pnl_pct_display = f"{pnl_pct:+.2f}%" if pnl_pct is not None else "-"

        holding = t.get('holding_period_days', t.get('holding_days'))
        holding_display = f"{holding}d" if holding is not None else "-"

        reason_display = _format_reason(t.get('reason', ''))
        regime_display = _format_regime(t.get('regime_at_entry', ''))

        rows.append({
            'date': entry_date,
            'symbol': t.get('symbol', 'Unknown'),
            'side': t.get('side', 'buy').upper(),
            'entry_price': f"${t.get('entry_price', 0):.2f}",
            'exit_price': f"${t.get('exit_price', 0):.2f}" if t.get('exit_price') else "-",
            'qty': f"{t.get('qty', 0):.0f}",
            'pnl': pnl_display,
            'pnl_pct': pnl_pct_display,
            'holding': holding_display,
            'regime': regime_display,
            'reason': reason_display,
            'status': 'Closed' if t.get('exit_date') or t.get('is_closed') else 'Open',
        })

    df = pd.DataFrame(rows)

    # Get unique symbols and reasons for filters
    symbols = sorted(set(t.get('symbol', '') for t in trades))
    reasons = sorted(set(t.get('reason', '') for t in trades))

    table = dash_table.DataTable(
        columns=[{'name': i.title(), 'id': i} for i in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto', 'maxHeight': '400px'},
        style_cell={
            'textAlign': 'center',
            'padding': '8px',
            'fontSize': '13px',
            'backgroundColor': DEFAULT_THEME.bg_card,
            'color': DEFAULT_THEME.text_primary,
            'border': f'1px solid {DEFAULT_THEME.border_default}',
        },
        style_header={
            'backgroundColor': DEFAULT_THEME.bg_secondary,
            'fontWeight': 'bold',
            'color': DEFAULT_THEME.text_primary,
            'border': f'1px solid {DEFAULT_THEME.border_default}',
        },
        style_data_conditional=[
            {
                'if': {'filter_query': '{side} = "BUY"'},
                'color': DEFAULT_THEME.success,
                'fontWeight': 'bold',
            },
            {
                'if': {'filter_query': '{side} = "SELL"'},
                'color': DEFAULT_THEME.danger,
                'fontWeight': 'bold',
            },
            {
                'if': {'filter_query': '{pnl} contains "+"'},
                'color': DEFAULT_THEME.success,
            },
            {
                'if': {'filter_query': '{pnl} contains "-"'},
                'color': DEFAULT_THEME.danger,
            },
            {
                'if': {'filter_query': '{status} = "Open"'},
                'backgroundColor': DEFAULT_THEME.bg_secondary,
            },
        ],
        sort_action='native',
        filter_action='native',
        page_size=20,
        id='trade-journal-table',
    )

    # Filter dropdowns
    symbol_options = [{'label': s, 'value': s} for s in symbols if s]
    reason_options = [{'label': _format_reason(r), 'value': r} for r in reasons if r]

    return html.Div([
        html.H4("Trade Journal", style={'marginBottom': '10px'}),
        html.Details([
            html.Summary("Filters"),
            html.Div([
                dcc.Dropdown(
                    id='trade-symbol-filter',
                    options=symbol_options,
                    placeholder="Filter by symbol",
                    style={'width': '200px', 'marginRight': '10px'},
                ),
                dcc.Dropdown(
                    id='trade-reason-filter',
                    options=reason_options,
                    placeholder="Filter by reason",
                    style={'width': '200px'},
                ),
            ], style={'display': 'flex', 'marginTop': '10px'}),
        ]),
        table,
    ], className="trade-journal")


def _format_reason(reason: str) -> str:
    """Format reason for display."""
    reason_map = {
        'stop_loss': 'Stop Loss',
        'rebalance': 'Rebalance',
        'regime_exit': 'Regime Exit',
        'take_profit': 'Take Profit',
        'momentum': 'Momentum',
        'open_position': 'Open',
        'buy': 'Buy',
        'sell': 'Sell',
    }
    return reason_map.get(reason, reason.replace('_', ' ').title())


def _format_regime(regime: str) -> str:
    """Format regime for display."""
    regime_map = {
        'strong_bull': 'Strong Bull',
        'weak_bull': 'Weak Bull',
        'weak_bear': 'Weak Bear',
        'strong_bear': 'Strong Bear',
        'unknown': 'Unknown',
    }
    return regime_map.get(regime, regime.replace('_', ' ').title())


def create_trade_statistics(trades: List[Dict[str, Any]]) -> html.Div:
    """
    Create trade statistics summary.

    Args:
        trades: List of trade dicts

    Returns:
        Dash HTML component
    """
    if not trades:
        return html.Div([])

    closed_trades = [t for t in trades if t.get('exit_date') or t.get('is_closed')]
    open_trades = [t for t in trades if not t.get('exit_date') and not t.get('is_closed')]

    winning_trades = [t for t in closed_trades if (t.get('realized_pnl', 0) or 0) > 0]
    losing_trades = [t for t in closed_trades if (t.get('realized_pnl', 0) or 0) <= 0]

    total_pnl = sum(t.get('realized_pnl', 0) or 0 for t in closed_trades)
    avg_pnl = total_pnl / len(closed_trades) if closed_trades else 0
    avg_win = sum(t.get('realized_pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t.get('realized_pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0

    holding_periods = [t.get('holding_period_days', t.get('holding_days', 0)) for t in closed_trades if t.get('holding_period_days') or t.get('holding_days')]
    avg_holding = sum(holding_periods) / len(holding_periods) if holding_periods else 0

    stats = [
        ("Total Trades", len(trades)),
        ("Closed", len(closed_trades)),
        ("Open", len(open_trades)),
        ("Win Rate", f"{len(winning_trades)/len(closed_trades)*100:.1f}%" if closed_trades else "0%"),
        ("Avg P&L", f"${avg_pnl:+,.2f}"),
        ("Avg Win", f"${avg_win:+,.2f} ({len(winning_trades)} trades)"),
        ("Avg Loss", f"${avg_loss:+,.2f} ({len(losing_trades)} trades)"),
        ("Avg Holding", f"{avg_holding:.1f} days"),
    ]

    return html.Div([
        html.H4("Trade Statistics", style={'marginBottom': '10px'}),
        html.Table([
            html.Tr([html.Td(s[0], style={'padding': '5px'}), html.Td(str(s[1]), style={'padding': '5px', 'fontWeight': 'bold'})])
            for s in stats
        ], style={'width': '100%', 'fontSize': '14px'}),
    ], className="trade-statistics", style={'marginTop': '20px'})
