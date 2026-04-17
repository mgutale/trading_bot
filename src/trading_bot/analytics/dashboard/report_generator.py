"""
Report Generator

Generates static HTML reports from backtest results.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

from trading_bot.analytics.dashboard.data_manager import DashboardDataManager
from trading_bot.analytics.dashboard.theme import DEFAULT_THEME


def generate_backtest_report(
    results: Dict[str, Any],
    stock_data: Dict[str, pd.DataFrame],
    benchmark_data: pd.DataFrame,
    initial_capital: float = 10000.0,
    output_path: str = "results/backtest_report.html",
) -> Optional[str]:
    """
    Generate a static HTML report from backtest results.

    Args:
        results: Backtest results from HybridHMMStopLoss.backtest()
        stock_data: Dict of symbol -> DataFrame with price data
        benchmark_data: DataFrame with benchmark prices
        initial_capital: Starting portfolio value
        output_path: Path to save HTML report

    Returns:
        Path to generated HTML file, or None if failed
    """
    try:
        # Create data manager
        manager = DashboardDataManager.from_backtest(
            results=results,
            stock_data=stock_data,
            benchmark_data=benchmark_data,
            initial_capital=initial_capital,
        )

        # Get all data
        stats = manager.get_performance_stats()
        risk = manager.get_risk_metrics()
        stocks = manager.get_stock_performance()
        equity = manager.get_equity_curve()
        regime_breakdown = manager.get_regime_breakdown()
        trades = manager.get_trade_journal()

        # Convert stats to dict for components
        stats_dict = {
            'total_return': stats.total_return,
            'annualized_return': stats.annualized_return,
            'sharpe_ratio': stats.sharpe_ratio,
            'sortino_ratio': stats.sortino_ratio,
            'max_drawdown': stats.max_drawdown,
            'win_rate': stats.win_rate,
            'profit_factor': stats.profit_factor,
            'total_trades': stats.total_trades,
            'buy_trades': stats.buy_trades,
            'sell_trades': stats.sell_trades,
            'stop_loss_trades': stats.stop_loss_trades,
        }

        risk_dict = {
            'var_95': risk.var_95,
            'var_99': risk.var_99,
            'cvar_95': risk.cvar_95,
            'cvar_99': risk.cvar_99,
            'volatility': risk.volatility,
            'max_drawdown': risk.max_drawdown,
        }

        # Generate HTML
        html_content = _build_html_report(
            stats_dict=stats_dict,
            risk_dict=risk_dict,
            equity=equity,
            stocks=stocks,
            regime_breakdown=regime_breakdown,
            trades=trades,
            results=results,
            initial_capital=initial_capital,
        )

        # Write to file
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    except Exception as e:
        import logging
        logging.error(f"Failed to generate report: {e}")
        return None


def _build_html_report(
    stats_dict: Dict[str, Any],
    risk_dict: Dict[str, Any],
    equity: pd.DataFrame,
    stocks: list,
    regime_breakdown: list,
    trades: pd.DataFrame,
    results: Dict[str, Any],
    initial_capital: float,
) -> str:
    """
    Build complete HTML report.

    Args:
        stats_dict: Dict with performance metrics
        risk_dict: Dict with risk metrics
        equity: DataFrame with equity curve
        stocks: List of AttributionEntry
        regime_breakdown: List of regime performance dicts
        trades: DataFrame with trade history
        results: Backtest results dict
        initial_capital: Starting capital

    Returns:
        Complete HTML string
    """
    # Convert Plotly figures to div
    from trading_bot.analytics.dashboard.components import create_equity_chart
    equity_chart = create_equity_chart(equity) if len(equity) > 0 else None
    equity_div = _plotly_to_div(equity_chart) if equity_chart else '<div class="empty-state">No equity data available</div>'

    # Calculate summary values
    total_return = results.get('total_return', 0)
    portfolio_value = initial_capital * (1 + total_return)
    benchmark_return = results.get('benchmark_return', 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Backtest Report</title>
    <style>
        :root {{
            --bg-primary: {DEFAULT_THEME.bg_primary};
            --bg-secondary: {DEFAULT_THEME.bg_secondary};
            --bg-card: {DEFAULT_THEME.bg_card};
            --text-primary: {DEFAULT_THEME.text_primary};
            --text-secondary: {DEFAULT_THEME.text_secondary};
            --success: {DEFAULT_THEME.success};
            --danger: {DEFAULT_THEME.danger};
            --warning: {DEFAULT_THEME.warning};
            --primary: {DEFAULT_THEME.primary};
            --border-default: {DEFAULT_THEME.border_default};
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 24px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        .header {{
            background: var(--bg-card);
            border: 1px solid var(--border-default);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}

        .header h1 {{
            font-size: 28px;
            margin-bottom: 8px;
        }}

        .header-meta {{
            display: flex;
            gap: 24px;
            color: var(--text-secondary);
            font-size: 14px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .summary-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-default);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}

        .summary-card .label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
        }}

        .summary-card .value {{
            font-size: 24px;
            font-weight: bold;
            margin-top: 8px;
        }}

        .value.positive {{ color: var(--success); }}
        .value.negative {{ color: var(--danger); }}

        .section {{
            background: var(--bg-card);
            border: 1px solid var(--border-default);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}

        .section h2 {{
            font-size: 20px;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-default);
        }}

        .stats-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
        }}

        .metric-card {{
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}

        .metric-label {{
            font-size: 12px;
            color: var(--text-secondary);
            text-transform: uppercase;
        }}

        .metric-value {{
            font-size: 20px;
            font-weight: bold;
            margin-top: 8px;
        }}

        .chart-container {{
            width: 100%;
            height: 600px;
        }}

        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}

        @media (max-width: 900px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}
        }}

        .empty-state {{
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
        }}

        /* Table styles */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        .data-table th {{
            background: var(--bg-secondary);
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid var(--border-default);
        }}

        .data-table td {{
            padding: 12px;
            border-bottom: 1px solid var(--border-default);
        }}

        .data-table tr:hover {{
            background: var(--bg-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Trading Bot Backtest Report</h1>
            <div class="header-meta">
                <span>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
                <span>Initial Capital: ${initial_capital:,.0f}</span>
                <span>Final Value: ${portfolio_value:,.2f}</span>
                <span>Benchmark Return: {benchmark_return:.1%}</span>
            </div>
        </div>

        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card">
                <div class="label">Total Return</div>
                <div class="value {'positive' if total_return > 0 else 'negative'}">{total_return:.1%}</div>
            </div>
            <div class="summary-card">
                <div class="label">Annualized Return</div>
                <div class="value {'positive' if results.get('annualized_return', 0) > 0 else 'negative'}">{results.get('annualized_return', 0):.1%}</div>
            </div>
            <div class="summary-card">
                <div class="label">Sharpe Ratio</div>
                <div class="value">{results.get('sharpe_ratio', 0):.2f}</div>
            </div>
            <div class="summary-card">
                <div class="label">Max Drawdown</div>
                <div class="value negative">{results.get('max_drawdown', 0):.1%}</div>
            </div>
            <div class="summary-card">
                <div class="label">Total Trades</div>
                <div class="value">{results.get('num_trades', 0)}</div>
            </div>
            <div class="summary-card">
                <div class="label">Win Rate</div>
                <div class="value {'positive' if results.get('win_rate', 0) > 0.5 else 'negative'}">{results.get('win_rate', 0):.1%}</div>
            </div>
        </div>

        <!-- Performance Stats -->
        <div class="section">
            <h2>Performance Statistics</h2>
            <div class="stats-cards">
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {'positive' if stats_dict['total_return'] > 0 else 'negative'}">{stats_dict['total_return']:.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Annualized</div>
                    <div class="metric-value {'positive' if stats_dict['annualized_return'] > 0 else 'negative'}">{stats_dict['annualized_return']:.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{stats_dict['sharpe_ratio']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sortino Ratio</div>
                    <div class="metric-value">{stats_dict['sortino_ratio']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">{stats_dict['max_drawdown']:.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value {'positive' if stats_dict['win_rate'] > 0.5 else 'negative'}">{stats_dict['win_rate']:.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">{stats_dict['profit_factor']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">{stats_dict['total_trades']}</div>
                </div>
            </div>
        </div>

        <!-- Equity Curve -->
        <div class="section">
            <h2>Equity Curve</h2>
            <div class="chart-container">
                {equity_div}
            </div>
        </div>

        <!-- Stock Performance -->
        <div class="section">
            <h2>Stock Performance</h2>
            <p style="color: var(--text-secondary); margin-bottom: 16px;">Comprehensive analysis of each stock's contribution to portfolio returns.</p>
            {_build_stock_performance_table(stocks)}
        </div>

        <!-- Risk Metrics -->
        <div class="section">
            <h2>Risk Metrics</h2>
            <div class="stats-cards">
                <div class="metric-card">
                    <div class="metric-label">VaR (95%)</div>
                    <div class="metric-value negative">{risk_dict['var_95']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">VaR (99%)</div>
                    <div class="metric-value negative">{risk_dict['var_99']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">CVaR (95%)</div>
                    <div class="metric-value negative">{risk_dict['cvar_95']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Volatility</div>
                    <div class="metric-value">{risk_dict['volatility']:.2%}</div>
                </div>
            </div>
        </div>

        <!-- Regime Breakdown -->
        <div class="section">
            <h2>Performance by Market Regime</h2>
            {_build_regime_breakdown(regime_breakdown)}
        </div>

        <!-- Trade Journal Summary -->
        <div class="section">
            <h2>Trade Journal Summary</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Total Trades</td>
                        <td>{results.get('num_trades', 0)}</td>
                    </tr>
                    <tr>
                        <td>Buy Trades</td>
                        <td>{results.get('buy_trades', 0)}</td>
                    </tr>
                    <tr>
                        <td>Sell Trades</td>
                        <td>{results.get('sell_trades', 0)}</td>
                    </tr>
                    <tr>
                        <td>Stop Loss Exits</td>
                        <td>{results.get('stop_loss_trades', 0)}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Full Trade Journal -->
        <div class="section">
            <h2>Full Trade Journal</h2>
            {_build_trade_journal_table(trades)}
        </div>
    </div>
</body>
</html>
"""


def _plotly_to_div(fig) -> str:
    """Convert Plotly figure to HTML div with full embedding."""
    if fig is None:
        return '<div class="empty-state">No chart data</div>'

    import plotly.io as pio
    # Use full embedding with inline Plotly JS for reliable offline viewing
    return pio.to_html(fig, full_html=False, include_plotlyjs=True)


def _build_stock_performance_table(stocks: list) -> str:
    """Build HTML table for stock performance."""
    if not stocks:
        return '<div class="empty-state">No stock performance data</div>'

    rows = []
    for stock in stocks:
        symbol = stock.symbol if hasattr(stock, 'symbol') else stock.get('symbol', 'Unknown')
        total_return = stock.total_return if hasattr(stock, 'total_return') else stock.get('total_return', 0)
        contribution = stock.contribution_pct if hasattr(stock, 'contribution_pct') else stock.get('contribution_pct', 0)
        trades = stock.trades if hasattr(stock, 'trades') else stock.get('trades', 0)
        win_rate = stock.win_rate if hasattr(stock, 'win_rate') else stock.get('win_rate', 0)
        profit_factor = stock.profit_factor if hasattr(stock, 'profit_factor') else stock.get('profit_factor', 0)
        sharpe = stock.sharpe_ratio if hasattr(stock, 'sharpe_ratio') else stock.get('sharpe_ratio', 0)
        avg_holding = stock.avg_holding_days if hasattr(stock, 'avg_holding_days') else stock.get('avg_holding_days', 0)

        return_class = 'positive' if total_return >= 0 else 'negative'
        win_class = 'positive' if win_rate >= 0.5 else 'negative'

        rows.append(f'''
            <tr>
                <td style="font-weight: bold;">{symbol}</td>
                <td class="{return_class}">${total_return:+,.2f}</td>
                <td class="{return_class}">{contribution:.1f}%</td>
                <td>{trades}</td>
                <td class="{win_class}">{win_rate:.1%}</td>
                <td>{profit_factor:.2f}</td>
                <td>{sharpe:.2f}</td>
                <td>{avg_holding:.1f}d</td>
            </tr>
        ''')

    return f'''
        <table class="data-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>P&L</th>
                    <th>Contribution</th>
                    <th>Trades</th>
                    <th>Win Rate</th>
                    <th>Profit Factor</th>
                    <th>Sharpe</th>
                    <th>Avg Hold</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    '''


def _build_regime_breakdown(breakdown: list) -> str:
    """Build HTML table for regime breakdown."""
    if not breakdown:
        return '<div class="empty-state">No regime data available</div>'

    regime_names = {
        'strong_bull': 'Strong Bull',
        'weak_bull': 'Weak Bull',
        'weak_bear': 'Weak Bear',
        'strong_bear': 'Strong Bear',
    }

    regime_colors = {
        'strong_bull': '#006400',
        'weak_bull': '#2ca02c',
        'weak_bear': '#ff7f0e',
        'strong_bear': '#8b0000',
    }

    rows = []
    for bp in breakdown:
        regime = bp.get('regime', 'unknown') if isinstance(bp, dict) else getattr(bp, 'regime', 'unknown')
        name = regime_names.get(regime, regime.replace('_', ' ').title())
        color = regime_colors.get(regime, '#808080')
        days = bp.get('days', 0) if isinstance(bp, dict) else getattr(bp, 'days', 0)
        total_return = bp.get('total_return', 0) if isinstance(bp, dict) else getattr(bp, 'total_return', 0)
        avg_return = bp.get('avg_return', 0) if isinstance(bp, dict) else getattr(bp, 'avg_return', 0)
        win_rate = bp.get('win_rate', 0) if isinstance(bp, dict) else getattr(bp, 'win_rate', 0)
        trades = bp.get('trades', 0) if isinstance(bp, dict) else getattr(bp, 'trades', 0)
        exposure = bp.get('exposure', 0) if isinstance(bp, dict) else getattr(bp, 'exposure', 0)

        return_class = 'positive' if total_return >= 0 else 'negative'

        rows.append(f'''
            <tr>
                <td style="color: {color}; font-weight: bold;">{name}</td>
                <td>{days}</td>
                <td class="{return_class}">{total_return:.1%}</td>
                <td class="{return_class}">{avg_return:.2%}</td>
                <td>{win_rate:.1%}</td>
                <td>{trades}</td>
                <td>{exposure:.0%}</td>
            </tr>
        ''')

    return f'''
        <table class="data-table">
            <thead>
                <tr>
                    <th>Regime</th>
                    <th>Days</th>
                    <th>Total Return</th>
                    <th>Avg Return</th>
                    <th>Win Rate</th>
                    <th>Trades</th>
                    <th>Exposure</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    '''


def _build_trade_journal_table(trades_df) -> str:
    """Build HTML table for full trade journal."""
    import pandas as pd

    if trades_df is None or len(trades_df) == 0:
        return '<div class="empty-state">No trade journal data</div>'

    # Limit to first 100 trades for readability
    if len(trades_df) > 100:
        trades_df = trades_df.head(100)
        note = '<p style="color: var(--text-secondary); font-size: 12px; margin-top: 10px;">Showing first 100 trades only.</p>'
    else:
        note = ''

    rows = []
    for _, row in trades_df.iterrows():
        pnl = row.get('pnl', 0) or 0
        pnl_class = 'positive' if pnl >= 0 else 'negative'
        side = row.get('side', 'BUY').upper()
        side_class = 'side-buy' if side == 'BUY' else 'side-sell'
        status = row.get('status', 'Closed')
        status_style = 'color: var(--primary);' if status == 'Open' else ''

        pnl_display = f"${pnl:+,.2f}" if pnl is not None else '-'
        pnl_pct = row.get('pnl_pct')
        pnl_pct_display = f"{pnl_pct:+.2f}%" if pnl_pct is not None else '-'

        rows.append(f'''
            <tr>
                <td>{row.get('date', '-')}</td>
                <td style="font-weight: bold;">{row.get('symbol', 'Unknown')}</td>
                <td class="{side_class}">{side}</td>
                <td>${row.get('entry_price', 0):.2f}</td>
                <td>${row.get('exit_price', 0):.2f}</td>
                <td>{row.get('qty', 0):.0f}</td>
                <td class="{pnl_class}">{pnl_display}</td>
                <td class="{pnl_class}">{pnl_pct_display}</td>
                <td>{row.get('holding_days', '-')}</td>
                <td>{row.get('regime', '-')}</td>
                <td>{row.get('reason', '-')}</td>
                <td style="{status_style}">{status}</td>
            </tr>
        ''')

    return f'''
        <table class="data-table">
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>Qty</th>
                    <th>P&L ($)</th>
                    <th>P&L (%)</th>
                    <th>Hold Days</th>
                    <th>Regime</th>
                    <th>Reason</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        {note}
        <style>
            .side-buy {{ color: {DEFAULT_THEME.success}; font-weight: bold; }}
            .side-sell {{ color: {DEFAULT_THEME.danger}; font-weight: bold; }}
        </style>
    '''
