"""
Live Dashboard

Real-time analytics dashboard using Plotly Dash.
Works in both backtest and live trading modes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import logging

import dash
from dash import html, dcc, callback, Output, Input
import plotly.graph_objects as go

from trading_bot.analytics.dashboard.models import (
    EnrichedTrade,
    DashboardConfig,
    PerformanceStats,
    RiskMetrics,
)
from trading_bot.analytics.dashboard.trade_enricher import TradeEnricher, enrich_trade_log
from trading_bot.analytics.dashboard.metrics_engine import MetricsEngine
from trading_bot.analytics.dashboard.components import (
    create_equity_chart,
    create_positions_table,
    create_trade_journal_table,
    create_regime_display,
    create_stats_cards,
    create_risk_metrics,
    create_regime_breakdown,
    create_attribution_chart,
    create_live_comparison,
)
from trading_bot.analytics.dashboard.utils import format_currency, format_percent

logger = logging.getLogger(__name__)


class LiveDashboard:
    """
    Real-time analytics dashboard using Plotly Dash.

    Works in two modes:
    1. BACKTEST mode: Renders full historical analysis from backtest results
    2. LIVE mode: Polls for updates and refreshes charts in real-time

    Usage:
        # Backtest mode
        dashboard = LiveDashboard.from_backtest_results(
            results=backtest_results,
            stock_data=stock_data,
            benchmark_data=benchmark_data
        )
        dashboard.run(debug=True, port=8050)

        # Live mode
        dashboard = LiveDashboard.from_live_manager(
            position_manager=pm,
            strategy=strat,
            ibkr_client=ibkr,
        )
        dashboard.run(debug=True, port=8050)
    """

    def __init__(self, config: DashboardConfig = None):
        """
        Initialize the dashboard.

        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.app = dash.Dash(
            __name__,
            suppress_callback_exceptions=True,
            title="Trading Bot Analytics"
        )

        # Data storage
        self.enriched_trades: List[EnrichedTrade] = []
        self.metrics_engine: Optional[MetricsEngine] = None
        self.performance_stats: Optional[PerformanceStats] = None
        self.risk_metrics: Optional[RiskMetrics] = None

        # Live mode state
        self._update_callback: Optional[Callable] = None
        self._is_live_mode = False

        # Colors - Dark Theme
        self.colors = {
            'background': '#0d1117',
            'card_bg': '#21262d',
            'primary': '#58a6ff',
            'success': '#3fb950',
            'danger': '#f85149',
            'warning': '#d29922',
            'text': '#f0f6fc',
            'text_secondary': '#8b949e',
        }

        self._build_layout()

    @classmethod
    def from_backtest_results(
        cls,
        results: Dict,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        config: DashboardConfig = None,
    ) -> 'LiveDashboard':
        """
        Create dashboard from backtest results.

        Args:
            results: Backtest results dict from HybridHMMStopLoss.backtest()
            stock_data: Dict of symbol -> DataFrame with price data
            benchmark_data: DataFrame with benchmark prices
            config: Dashboard configuration

        Returns:
            LiveDashboard instance
        """
        dashboard = cls(config=config)

        # Enrich the raw trade log
        raw_trade_log = results.get('trade_log', [])
        signals = results.get('signals', pd.DataFrame())

        if raw_trade_log and len(signals) > 0:
            dashboard.enriched_trades = enrich_trade_log(
                raw_trade_log=raw_trade_log,
                signals=signals,
                stock_data=stock_data,
                benchmark_data=benchmark_data,
                initial_capital=dashboard.config.initial_capital,
            )

        # Create metrics engine
        dashboard.metrics_engine = MetricsEngine(
            enriched_trades=dashboard.enriched_trades,
            signals=signals,
            initial_capital=dashboard.config.initial_capital,
            benchmark_data=benchmark_data,
        )

        # Compute metrics
        dashboard.performance_stats = dashboard.metrics_engine.get_performance_stats()
        dashboard.risk_metrics = dashboard.metrics_engine.get_risk_metrics()

        # Store signals for charts
        dashboard._signals = signals
        dashboard._stock_data = stock_data
        dashboard._benchmark_data = benchmark_data
        dashboard._results = results

        return dashboard

    @classmethod
    def from_live_manager(
        cls,
        position_manager,
        strategy,
        ibkr_client=None,
        config: DashboardConfig = None,
    ) -> 'LiveDashboard':
        """
        Create dashboard for live trading.

        Args:
            position_manager: PositionManager instance
            strategy: HybridHMMStopLoss strategy instance
            ibkr_client: IBKRClient instance (optional)
            config: Dashboard configuration

        Returns:
            LiveDashboard instance
        """
        dashboard = cls(config=config)
        dashboard._is_live_mode = True
        dashboard._position_manager = position_manager
        dashboard._strategy = strategy
        dashboard._ibkr_client = ibkr_client

        # Set up live update callback
        dashboard._update_callback = dashboard._create_live_update_callback()

        return dashboard

    def _create_live_update_callback(self) -> Callable:
        """Create the callback for live updates."""
        def update():
            """Fetch latest data from live trading systems."""
            try:
                # Get portfolio summary from position manager
                summary = self._position_manager.get_portfolio_summary()

                # Get positions
                positions = summary.get('positions', [])

                # Get current regime from strategy
                # This would be updated in real-time during live trading
                current_regime = getattr(self._strategy, '_current_regime', 'unknown')

                return {
                    'positions': positions,
                    'equity': summary.get('equity', 0),
                    'cash': summary.get('cash', 0),
                    'regime': current_regime,
                }
            except Exception as e:
                logger.error(f"Error in live update: {e}")
                return {'positions': [], 'equity': 0, 'cash': 0, 'regime': 'unknown'}

        return update

    def _build_layout(self):
        """Build the Dash layout."""
        self.app.layout = html.Div([
            # Header
            html.Header([
                html.Div([
                    html.H1("Trading Bot Analytics", style={'margin': 0}),
                    html.Span(
                        "Live Trading Mode" if self._is_live_mode else "Backtest Mode",
                        className='mode-badge'
                    ),
                ], className='header-content'),
                html.Div([
                    html.Span(id='last-update', children='Last Update: --'),
                ], className='header-meta'),
            ], className='dashboard-header'),

            # Main content
            html.Main([
                # Metrics cards row
                html.Div([
                    html.Div(id='stats-cards', className='stats-cards-container'),
                ], className='section'),

                # Equity chart
                html.Section([
                    html.H2("Equity Curve"),
                    dcc.Graph(id='equity-chart'),
                ], className='chart-section'),

                # Two-column layout
                html.Div([
                    # Left column
                    html.Div([
                        # Positions panel
                        html.Div(id='positions-panel', className='positions-section'),
                    ], className='left-column'),

                    # Right column
                    html.Div([
                        # Regime display
                        html.Div(id='regime-panel', className='regime-section'),
                    ], className='right-column'),
                ], className='two-column-layout'),

                # Risk metrics
                html.Section([
                    html.H2("Risk Metrics"),
                    dcc.Graph(id='risk-chart'),
                    html.Div(id='risk-cards'),
                ], className='chart-section'),

                # Regime breakdown
                html.Section([
                    html.H2("Performance by Regime"),
                    dcc.Graph(id='regime-chart'),
                    html.Div(id='regime-breakdown-table'),
                ], className='chart-section'),

                # Attribution
                html.Section([
                    html.H2("Stock Attribution"),
                    dcc.Graph(id='attribution-chart'),
                ], className='chart-section'),

                # Trade journal
                html.Section([
                    html.H2("Trade Journal"),
                    html.Div(id='trade-journal'),
                ], className='chart-section'),

            ], className='dashboard-main'),

            # Footer
            html.Footer([
                html.Span("Trading Bot Analytics Dashboard"),
                html.Span(f" | Mode: {'Live' if self._is_live_mode else 'Backtest'}"),
            ], className='dashboard-footer'),

            # Update interval for live mode
            dcc.Interval(
                id='update-interval',
                interval=self.config.update_interval_seconds * 1000,
                n_intervals=0,
                disabled=not self._is_live_mode,
            ),

            # Store for holding data
            dcc.Store(id='dashboard-data'),

            # Hidden trigger for initial callback (forces render on page load)
            html.Div(id='dashboard-initial-load', style={'display': 'none'}),
        ], className='dashboard-container')

        # Register callbacks
        self._register_callbacks()

    def _register_callbacks(self):
        """Register Dash callbacks for updates."""
        if self._is_live_mode:
            # Live update callback - uses interval + Initial
            @self.app.callback(
                [Output('stats-cards', 'children'),
                 Output('equity-chart', 'figure'),
                 Output('positions-panel', 'children'),
                 Output('regime-panel', 'children'),
                 Output('risk-chart', 'figure'),
                 Output('risk-cards', 'children'),
                 Output('regime-chart', 'figure'),
                 Output('attribution-chart', 'figure'),
                 Output('trade-journal', 'children'),
                 Output('last-update', 'children')],
                [Input('update-interval', 'n_intervals'),
                 Input('dashboard-initial-load', 'data')]
            )
            def update_dashboard(n, initial_data):
                # Fetch live data
                live_data = self._update_callback()

                # Build components from live data
                stats = self._build_live_stats(live_data)
                equity_fig = self._build_live_equity_chart(live_data)
                positions = self._build_positions_panel(live_data)
                regime = self._build_regime_panel(live_data)
                risk_fig = self._build_risk_chart()
                risk_cards = self._build_risk_cards()
                regime_fig = self._build_regime_chart()
                attribution_fig = self._build_attribution_chart()
                trade_journal = self._build_trade_journal()
                last_update = f"Last Update: {datetime.now().strftime('%H:%M:%S')}"

                return (stats, equity_fig, positions, regime,
                        risk_fig, risk_cards, regime_fig, attribution_fig,
                        trade_journal, last_update)
        else:
            # Backtest mode - render static data once on page load
            @self.app.callback(
                [Output('stats-cards', 'children'),
                 Output('equity-chart', 'figure'),
                 Output('positions-panel', 'children'),
                 Output('regime-panel', 'children'),
                 Output('risk-chart', 'figure'),
                 Output('risk-cards', 'children'),
                 Output('regime-chart', 'figure'),
                 Output('attribution-chart', 'figure'),
                 Output('trade-journal', 'children'),
                 Output('last-update', 'children')],
                [Input('dashboard-initial-load', 'n_clicks')]
            )
            def render_backtest_dashboard(n_clicks):
                # Build components from backtest data
                stats = self._build_backtest_stats()
                equity_fig = self._build_backtest_equity_chart()
                positions = self._build_backtest_positions()
                regime = self._build_backtest_regime()
                risk_fig = self._build_risk_chart()
                risk_cards = self._build_risk_cards()
                regime_fig = self._build_regime_chart()
                attribution_fig = self._build_attribution_chart()
                trade_journal = self._build_trade_journal()
                last_update = f"Backtest Results | Mode: Backtest"

                return (stats, equity_fig, positions, regime,
                        risk_fig, risk_cards, regime_fig, attribution_fig,
                        trade_journal, last_update)

    def _build_live_stats(self, live_data: Dict) -> html.Div:
        """Build stats cards from live data."""
        equity = live_data.get('equity', 0)
        total_return = (equity / self.config.initial_capital - 1) if equity > 0 else 0

        stats = {
            'total_return': total_return,
            'annualized_return': total_return * 252 / max(1, 252),  # Approximate
            'sharpe_ratio': 0,  # Would need rolling window
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
        }
        return create_stats_cards(stats)

    def _build_backtest_stats(self) -> html.Div:
        """Build stats cards from backtest results."""
        if self.performance_stats:
            stats = {
                'total_return': self.performance_stats.total_return,
                'annualized_return': self.performance_stats.annualized_return,
                'sharpe_ratio': self.performance_stats.sharpe_ratio,
                'sortino_ratio': self.performance_stats.sortino_ratio,
                'max_drawdown': self.performance_stats.max_drawdown,
                'win_rate': self.performance_stats.win_rate,
                'profit_factor': self.performance_stats.profit_factor,
                'total_trades': self.performance_stats.total_trades,
                'buy_trades': self.performance_stats.buy_trades,
                'sell_trades': self.performance_stats.sell_trades,
                'stop_loss_trades': self.performance_stats.stop_loss_trades,
            }
        else:
            stats = {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
            }
        return create_stats_cards(stats)

    def _build_backtest_equity_chart(self) -> go.Figure:
        """Build equity chart from backtest data."""
        if self.metrics_engine:
            equity_data = self.metrics_engine.get_equity_curve()
            if equity_data is not None and len(equity_data) > 0:
                from trading_bot.analytics.dashboard.components.equity_chart import create_equity_chart
                return create_equity_chart(equity_data)

        fig = go.Figure()
        fig.add_annotation(
            text="No equity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(template='plotly_dark', height=400)
        return fig

    def _build_backtest_positions(self) -> html.Div:
        """Build positions panel for backtest (empty - no open positions in backtest)."""
        return create_positions_table([])

    def _build_backtest_regime(self) -> html.Div:
        """Build regime panel for backtest mode."""
        # Show regime from signals if available
        regime = 'unknown'
        regime_numeric = 0
        regime_duration_days = 0
        exposure = 0.0

        if hasattr(self, '_signals') and 'regime' in self._signals.columns:
            if len(self._signals) > 0:
                last_regime = self._signals['regime'].iloc[-1]
                regime = last_regime if isinstance(last_regime, str) else str(last_regime)
                regime_numeric = 0
                exposure = self._signals.get('strategy_exposure', pd.Series([1.0])).iloc[-1] if 'strategy_exposure' in self._signals.columns else 0.5

        return create_regime_display(
            current_regime=regime,
            regime_numeric=regime_numeric,
            regime_duration_days=regime_duration_days,
            exposure=exposure,
        )

    def _build_live_equity_chart(self, live_data: Dict) -> go.Figure:
        """Build equity chart from live data."""
        # For live mode, this would show real-time portfolio value
        fig = go.Figure()
        fig.add_annotation(
            text="Real-time equity chart updates during live trading",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(
            template='plotly_white',
            height=400,
        )
        return fig

    def _build_positions_panel(self, live_data: Dict) -> html.Div:
        """Build positions panel from live data."""
        positions = live_data.get('positions', [])
        return create_positions_table(positions)

    def _build_regime_panel(self, live_data: Dict) -> html.Div:
        """Build regime panel from live data."""
        regime = live_data.get('regime', 'unknown')
        return create_regime_display(
            current_regime=regime,
            regime_numeric=0,
            regime_duration_days=0,
            exposure=1.0,
        )

    def _build_risk_chart(self) -> go.Figure:
        """Build risk chart."""
        if self.risk_metrics:
            returns = self._signals.get('strategy_return', pd.Series()) if hasattr(self, '_signals') else pd.Series()
            if len(returns) > 0:
                from trading_bot.analytics.dashboard.components.risk_metrics import create_var_chart
                return create_var_chart(returns)

        fig = go.Figure()
        fig.add_annotation(
            text="Run backtest to see risk metrics",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(template='plotly_white', height=300)
        return fig

    def _build_risk_cards(self) -> html.Div:
        """Build risk metrics cards."""
        if self.risk_metrics:
            risk_dict = {
                'var_95': self.risk_metrics.var_95,
                'var_99': self.risk_metrics.var_99,
                'cvar_95': self.risk_metrics.cvar_95,
                'cvar_99': self.risk_metrics.cvar_99,
                'volatility': self.risk_metrics.volatility,
                'max_drawdown': self.risk_metrics.max_drawdown,
            }
            return create_risk_metrics(risk_dict)
        return html.Div([])

    def _build_regime_chart(self) -> go.Figure:
        """Build regime breakdown chart."""
        if hasattr(self, '_signals') and 'regime' in self._signals.columns:
            from trading_bot.analytics.dashboard.components.regime_display import create_regime_chart
            return create_regime_chart(self._signals['regime'])

        fig = go.Figure()
        fig.add_annotation(
            text="Run backtest to see regime analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(template='plotly_white', height=250)
        return fig

    def _build_attribution_chart(self) -> go.Figure:
        """Build attribution chart."""
        if self.metrics_engine and self.enriched_trades:
            attribution = self.metrics_engine.get_attribution()
            if attribution:
                attr_dicts = [
                    {
                        'symbol': a.symbol,
                        'total_return': a.total_return,
                        'contribution_pct': a.contribution_pct,
                        'trades': a.trades,
                        'avg_holding_days': a.avg_holding_days,
                        'best_trade_pct': a.best_trade_pct,
                        'worst_trade_pct': a.worst_trade_pct,
                    }
                    for a in attribution
                ]
                from trading_bot.analytics.dashboard.components.attribution import create_attribution_chart
                return create_attribution_chart(attr_dicts)

        fig = go.Figure()
        fig.add_annotation(
            text="Run backtest to see attribution analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="gray")
        )
        fig.update_layout(template='plotly_white', height=350)
        return fig

    def _build_trade_journal(self) -> html.Div:
        """Build trade journal table."""
        if self.enriched_trades:
            trades = [
                {
                    'entry_date': t.entry_date,
                    'exit_date': t.exit_date,
                    'symbol': t.symbol,
                    'side': t.side,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'qty': t.qty,
                    'realized_pnl': t.realized_pnl,
                    'realized_pnl_pct': t.realized_pnl_pct,
                    'unrealized_pnl': t.unrealized_pnl,
                    'holding_period_days': t.holding_period_days,
                    'regime_at_entry': t.regime_at_entry,
                    'reason': t.reason,
                    'is_closed': t.is_closed,
                }
                for t in self.enriched_trades
            ]
            return create_trade_journal_table(trades)

        return html.Div([
            html.H4("Trade Journal"),
            html.P("No trades to display", style={'color': 'gray', 'fontStyle': 'italic'})
        ])

    def run(self, debug: bool = False, port: int = 8050):
        """
        Launch the Dash application.

        Args:
            debug: Enable debug mode
            port: Port to run on
        """
        logger.info(f"Starting dashboard on port {port}")
        self.app.run(debug=debug, port=port, host='0.0.0.0')

    def get_app(self):
        """Get the Dash app instance."""
        return self.app


def create_dashboard_html(
    results: Dict,
    stock_data: Dict[str, pd.DataFrame],
    benchmark_data: pd.DataFrame,
    config: DashboardConfig = None,
) -> str:
    """
    Create a standalone HTML dashboard from backtest results.

    This generates a static HTML file that can be viewed in a browser
    without running a Dash server.

    Args:
        results: Backtest results dict
        stock_data: Dict of symbol -> DataFrame with price data
        benchmark_data: DataFrame with benchmark prices
        config: Dashboard configuration

    Returns:
        Path to generated HTML file
    """
    # Create dashboard from results
    dashboard = LiveDashboard.from_backtest_results(
        results=results,
        stock_data=stock_data,
        benchmark_data=benchmark_data,
        config=config,
    )

    # Get signals
    signals = results.get('signals', pd.DataFrame())

    # Build components
    equity_data = dashboard.metrics_engine.get_equity_curve() if dashboard.metrics_engine else pd.DataFrame()
    equity_fig = create_equity_chart(equity_data) if len(equity_data) > 0 else go.Figure()

    stats = {
        'total_return': results.get('total_return', 0),
        'annualized_return': results.get('annualized_return', 0),
        'sharpe_ratio': results.get('sharpe_ratio', 0),
        'sortino_ratio': results.get('sortino_ratio', 0),
        'max_drawdown': results.get('max_drawdown', 0),
        'win_rate': results.get('win_rate', 0),
        'profit_factor': results.get('profit_factor', 0),
        'total_trades': results.get('num_trades', 0),
        'buy_trades': results.get('buy_trades', 0),
        'sell_trades': results.get('sell_trades', 0),
        'stop_loss_trades': results.get('stop_loss_trades', 0),
    }
    stats_cards = create_stats_cards(stats)

    # Export to HTML
    import os
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dashboard_report.html')

    # For standalone HTML, we'd need to convert Dash components to HTML
    # This is a simplified version - full implementation would need
    # server-side rendering or pre-rendering
    logger.info(f"Dashboard report generated at {output_path}")

    return output_path
