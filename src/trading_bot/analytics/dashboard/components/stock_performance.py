"""
Stock Performance Component

Comprehensive per-stock analytics with summary cards, charts, and drill-down detail view.
"""

from typing import List, Dict, Any, Optional
from dash import html, dcc, dash_table
import plotly.graph_objects as go
import pandas as pd

from trading_bot.analytics.dashboard.theme import DEFAULT_THEME
from trading_bot.analytics.dashboard.chart_utils import (
    create_figure,
    apply_dark_theme,
    empty_figure,
    create_bar_chart,
    create_scatter_plot,
    create_heatmap,
    color_for_return,
)
from trading_bot.analytics.dashboard.models import AttributionEntry


def create_stock_performance_panel(attribution: List[AttributionEntry]) -> html.Div:
    """
    Create comprehensive stock performance display.

    Args:
        attribution: List of AttributionEntry with per-stock metrics

    Returns:
        Dash HTML component with summary cards and charts
    """
    if not attribution:
        return html.Div([
            html.H4("Stock Performance"),
            html.P("No stock data available", style={'color': DEFAULT_THEME.text_secondary})
        ], className="stock-performance-panel")

    return html.Div([
        html.H4("Stock Performance", style={'marginBottom': '16px'}),

        # Summary cards
        html.Div(
            create_stock_summary_cards(attribution),
            className="stock-cards-grid",
            style={'marginBottom': '24px'}
        ),

        # Main charts row
        html.Div([
            # Contribution chart
            html.Div([
                html.H5("Contribution to Returns", style={'marginBottom': '12px'}),
                dcc.Graph(figure=create_contribution_chart(attribution)),
            ], className="chart-card", style={'flex': '1', 'minWidth': '400px'}),

            # Risk/return scatter
            html.Div([
                html.H5("Risk vs Return", style={'marginBottom': '12px'}),
                dcc.Graph(figure=create_risk_return_scatter(attribution)),
            ], className="chart-card", style={'flex': '1', 'minWidth': '400px'}),
        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '24px', 'flexWrap': 'wrap'}),

        # Detailed table
        html.Div([
            html.H5("Detailed Metrics", style={'marginBottom': '12px'}),
            create_stock_metrics_table(attribution),
        ], className="chart-card"),
    ], className="stock-performance-panel")


def create_stock_summary_cards(attribution: List[AttributionEntry]) -> List[html.Div]:
    """
    Create summary cards for top-level stock metrics.

    Args:
        attribution: List of AttributionEntry

    Returns:
        List of card components
    """
    if not attribution:
        return []

    # Aggregate stats
    total_stocks = len(attribution)
    profitable_stocks = sum(1 for a in attribution if a.total_return > 0)
    avg_win_rate = sum(a.win_rate for a in attribution) / total_stocks if total_stocks > 0 else 0
    best_contributor = max(attribution, key=lambda x: x.contribution_pct) if attribution else None

    cards = [
        _create_summary_card("Total Stocks", str(total_stocks), DEFAULT_THEME.primary),
        _create_summary_card("Profitable", str(profitable_stocks), DEFAULT_THEME.success),
        _create_summary_card("Avg Win Rate", f"{avg_win_rate:.1%}", DEFAULT_THEME.info),
    ]

    if best_contributor:
        cards.append(
            _create_summary_card(
                "Top Contributor",
                f"{best_contributor.symbol} ({best_contributor.contribution_pct:.1f}%)",
                color_for_return(best_contributor.total_return)
            )
        )

    return cards


def _create_summary_card(label: str, value: str, color: str) -> html.Div:
    """Create a single summary card."""
    return html.Div([
        html.Span(label, className="stock-card-label"),
        html.Br(),
        html.Span(value, className="stock-card-value", style={'color': color}),
    ], className="stock-card")


def create_contribution_chart(attribution: List[AttributionEntry]) -> go.Figure:
    """
    Create bar chart showing each stock's contribution to portfolio returns.

    Args:
        attribution: List of AttributionEntry

    Returns:
        Plotly Figure
    """
    if not attribution:
        return empty_figure("No attribution data")

    symbols = [a.symbol for a in attribution]
    contributions = [a.contribution_pct for a in attribution]
    colors = [color_for_return(a.total_return) for a in attribution]

    fig = create_bar_chart(
        x=symbols,
        y=contributions,
        title="",
        y_suffix="%",
        colors=colors,
    )

    fig.update_layout(
        height=350,
        yaxis=dict(title="Contribution (%)", tickformat=".1f%"),
        xaxis=dict(title="Stock"),
        margin=dict(l=60, r=20, t=20, b=80),
    )

    # Rotate x labels for readability
    fig.update_xaxes(tickangle=-45)

    return fig


def create_risk_return_scatter(attribution: List[AttributionEntry]) -> go.Figure:
    """
    Create scatter plot of risk vs return for each stock.

    Args:
        attribution: List of AttributionEntry

    Returns:
        Plotly Figure
    """
    if not attribution:
        return empty_figure("No data")

    # Filter stocks with valid metrics
    valid = [a for a in attribution if a.volatility > 0]
    if not valid:
        return empty_figure("No risk/return data available")

    x_vals = [a.volatility * 100 for a in valid]  # Volatility as %
    y_vals = [a.total_return * 100 for a in valid]  # Return as $
    labels = [a.symbol for a in valid]
    colors = [color_for_return(a.total_return) for a in valid]

    fig = create_scatter_plot(
        x=x_vals,
        y=y_vals,
        labels=labels,
        color_by=y_vals,
        title="",
        x_label="Volatility (%)",
        y_label="Total Return ($)",
    )

    fig.update_layout(
        height=350,
        margin=dict(l=60, r=20, t=20, b=40),
    )

    return fig


def create_stock_metrics_table(attribution: List[AttributionEntry]) -> dash_table.DataTable:
    """
    Create detailed metrics table for all stocks.

    Args:
        attribution: List of AttributionEntry

    Returns:
        Dash DataTable
    """
    if not attribution:
        return dash_table.DataTable(columns=[], data=[])

    # Build data rows
    rows = []
    for a in attribution:
        rows.append({
            'symbol': a.symbol,
            'total_return': f"${a.total_return:+,.2f}",
            'contribution': f"{a.contribution_pct:+.1f}%",
            'trades': str(a.trades),
            'win_rate': f"{a.win_rate:.1%}",
            'profit_factor': f"{a.profit_factor:.2f}" if a.profit_factor != float('inf') else "∞",
            'sharpe': f"{a.sharpe_ratio:.2f}",
            'max_dd': f"{a.max_drawdown:.1%}",
            'avg_hold': f"{a.avg_holding_days:.1f}d",
            'alpha': f"{a.alpha_vs_benchmark:.2%}",
        })

    df = pd.DataFrame(rows)

    # Define conditional formatting
    style_data_conditional = [
        {
            'if': {'filter_query': '{contribution} > 0'},
            'color': DEFAULT_THEME.success,
            'fontWeight': 'bold',
        },
        {
            'if': {'filter_query': '{contribution} < 0'},
            'color': DEFAULT_THEME.danger,
            'fontWeight': 'bold',
        },
        {
            'if': {'filter_query': '{win_rate} > 0.5'},
            'color': DEFAULT_THEME.success,
        },
        {
            'if': {'filter_query': '{win_rate} < 0.5'},
            'color': DEFAULT_THEME.danger,
        },
    ]

    return dash_table.DataTable(
        columns=[
            {'name': 'Symbol', 'id': 'symbol', 'type': 'string', 'fixed': {'row': 0, 'column': 0}},
            {'name': 'Total Return', 'id': 'total_return', 'type': 'string'},
            {'name': 'Contribution', 'id': 'contribution', 'type': 'string'},
            {'name': 'Trades', 'id': 'trades', 'type': 'numeric'},
            {'name': 'Win Rate', 'id': 'win_rate', 'type': 'string'},
            {'name': 'Profit Factor', 'id': 'profit_factor', 'type': 'string'},
            {'name': 'Sharpe', 'id': 'sharpe', 'type': 'string'},
            {'name': 'Max DD', 'id': 'max_dd', 'type': 'string'},
            {'name': 'Avg Hold', 'id': 'avg_hold', 'type': 'string'},
            {'name': 'Alpha', 'id': 'alpha', 'type': 'string'},
        ],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'padding': '10px',
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
        style_data_conditional=style_data_conditional,
        sort_action='native',
        filter_action='native',
        page_size=15,
        row_selectable='radio',
        selected_rows=[0] if rows else [],
        id='stock-metrics-table',
    )


def create_stock_detail_modal(selected_stock: Optional[AttributionEntry]) -> html.Div:
    """
    Create detailed drill-down view for a single stock.

    Args:
        selected_stock: AttributionEntry for the selected stock

    Returns:
        Dash HTML component with detailed analysis
    """
    if selected_stock is None:
        return html.Div([
            html.H4("Stock Detail"),
            html.P("Select a stock to view details", style={'color': DEFAULT_THEME.text_secondary})
        ], className="stock-detail-panel")

    return html.Div([
        # Header with symbol and key metrics
        html.Div([
            html.H3(selected_stock.symbol, style={'display': 'inline-block', 'marginRight': '16px'}),
            html.Span(
                f"${selected_stock.total_return:+,.2f}",
                style={'color': color_for_return(selected_stock.total_return), 'fontSize': '20px', 'fontWeight': 'bold'}
            ),
            html.Span(f" | {selected_stock.contribution_pct:+.1f}% of portfolio", style={'color': DEFAULT_THEME.text_secondary}),
        ], style={'marginBottom': '20px', 'paddingBottom': '12px', 'borderBottom': f'1px solid {DEFAULT_THEME.border_default}'}),

        # Metrics grid
        html.Div([
            _create_detail_metric("Trades", str(selected_stock.trades)),
            _create_detail_metric("Win Rate", f"{selected_stock.win_rate:.1%}"),
            _create_detail_metric("Profit Factor", f"{selected_stock.profit_factor:.2f}" if selected_stock.profit_factor != float('inf') else "∞"),
            _create_detail_metric("Sharpe Ratio", f"{selected_stock.sharpe_ratio:.2f}"),
            _create_detail_metric("Max Drawdown", f"{selected_stock.max_drawdown:.1%}"),
            _create_detail_metric("Volatility", f"{selected_stock.volatility:.1%}"),
            _create_detail_metric("Avg Hold", f"{selected_stock.avg_holding_days:.1f} days"),
            _create_detail_metric("Alpha", f"{selected_stock.alpha_vs_benchmark:.2%}"),
        ], className="metrics-grid", style={'marginBottom': '24px'}),

        # Best/Worst trades
        html.Div([
            html.Div([
                html.H5("Best Trade", style={'color': DEFAULT_THEME.success}),
                html.P(f"{selected_stock.best_trade_pct:.2%}" if selected_stock.best_trade_pct else "N/A"),
            ], className="detail-card"),
            html.Div([
                html.H5("Worst Trade", style={'color': DEFAULT_THEME.danger}),
                html.P(f"{selected_stock.worst_trade_pct:.2%}" if selected_stock.worst_trade_pct else "N/A"),
            ], className="detail-card"),
        ], style={'display': 'flex', 'gap': '16px', 'marginBottom': '24px'}),

        # Regime breakdown
        html.Div([
            html.H5("Performance by Regime", style={'marginBottom': '12px'}),
            create_regime_breakdown_mini(selected_stock.regime_breakdown),
        ], className="chart-card", style={'marginBottom': '16px'}),

        # Reason breakdown
        html.Div([
            html.H5("Performance by Exit Reason", style={'marginBottom': '12px'}),
            create_reason_breakdown_mini(selected_stock.reason_breakdown),
        ], className="chart-card"),
    ], className="stock-detail-panel")


def _create_detail_metric(label: str, value: str) -> html.Div:
    """Create a detail metric card."""
    return html.Div([
        html.Span(label, className="detail-label"),
        html.Br(),
        html.Span(value, className="detail-value"),
    ], className="detail-metric")


def create_regime_breakdown_mini(regime_breakdown: Dict[str, Dict]) -> html.Div:
    """
    Create mini regime breakdown display.

    Args:
        regime_breakdown: Dict of regime -> stats

    Returns:
        Dash HTML component
    """
    if not regime_breakdown:
        return html.P("No regime data", style={'color': DEFAULT_THEME.text_secondary})

    items = []
    for regime, stats in regime_breakdown.items():
        color = DEFAULT_THEME.regime_colors.get(regime, DEFAULT_THEME.text_secondary)
        items.append(html.Div([
            html.Span(
                regime.replace('_', ' ').title(),
                style={'color': color, 'fontWeight': 'bold', 'marginRight': '8px'}
            ),
            html.Span(f"{stats.get('trades', 0)} trades"),
            html.Span(f" | Win: {stats.get('win_rate', 0):.1%}"),
            html.Span(f" | P&L: ${stats.get('total_pnl', 0):+,.0f}"),
        ], style={'marginBottom': '8px'}))

    return html.Div(items)


def create_reason_breakdown_mini(reason_breakdown: Dict[str, Dict]) -> html.Div:
    """
    Create mini reason breakdown display.

    Args:
        reason_breakdown: Dict of reason -> stats

    Returns:
        Dash HTML component
    """
    if not reason_breakdown:
        return html.P("No reason data", style={'color': DEFAULT_THEME.text_secondary})

    items = []
    for reason, stats in reason_breakdown.items():
        items.append(html.Div([
            html.Span(
                reason.replace('_', ' ').title(),
                style={'fontWeight': 'bold', 'marginRight': '8px'}
            ),
            html.Span(f"{stats.get('trades', 0)} trades"),
            html.Span(f" | Win: {stats.get('win_rate', 0):.1%}"),
            html.Span(f" | P&L: ${stats.get('total_pnl', 0):+,.0f}"),
        ], style={'marginBottom': '8px'}))

    return html.Div(items)


def create_holdings_scatter(attribution: List[AttributionEntry]) -> go.Figure:
    """
    Create scatter plot of holding period vs return.

    Args:
        attribution: List of AttributionEntry

    Returns:
        Plotly Figure
    """
    if not attribution:
        return empty_figure("No data")

    x_vals = [a.avg_holding_days for a in attribution]
    y_vals = [a.total_return for a in attribution]
    labels = [a.symbol for a in attribution]

    fig = create_scatter_plot(
        x=x_vals,
        y=y_vals,
        labels=labels,
        color_by=y_vals,
        title="Holding Period vs Return",
        x_label="Avg Holding Period (days)",
        y_label="Total Return ($)",
    )

    return fig


def create_win_rate_chart(attribution: List[AttributionEntry]) -> go.Figure:
    """
    Create bar chart of win rates by stock.

    Args:
        attribution: List of AttributionEntry

    Returns:
        Plotly Figure
    """
    if not attribution:
        return empty_figure("No data")

    symbols = [a.symbol for a in attribution]
    win_rates = [a.win_rate * 100 for a in attribution]
    colors = [DEFAULT_THEME.success if wr >= 50 else DEFAULT_THEME.danger for wr in win_rates]

    fig = create_bar_chart(
        x=symbols,
        y=win_rates,
        title="Win Rate by Stock",
        y_suffix="%",
        colors=colors,
    )

    fig.update_layout(
        height=300,
        yaxis=dict(title="Win Rate (%)", range=[0, 100]),
        margin=dict(l=60, r=20, t=20, b=80),
    )

    fig.update_xaxes(tickangle=-45)

    return fig
