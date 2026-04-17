"""
Dashboard Components

Individual chart and display components for the analytics dashboard.
"""

from .equity_chart import create_equity_chart
from .positions_panel import create_positions_table
from .trade_journal import create_trade_journal_table
from .regime_display import create_regime_display
from .performance_stats import create_stats_cards
from .risk_metrics import create_risk_metrics
from .regime_breakdown import create_regime_breakdown
from .attribution import create_attribution_chart
from .stock_performance import (
    create_stock_performance_panel,
    create_stock_detail_modal,
    create_contribution_chart,
    create_risk_return_scatter,
    create_stock_metrics_table,
)

__all__ = [
    'create_equity_chart',
    'create_positions_table',
    'create_trade_journal_table',
    'create_regime_display',
    'create_stats_cards',
    'create_risk_metrics',
    'create_regime_breakdown',
    'create_attribution_chart',
    'create_stock_performance_panel',
    'create_stock_detail_modal',
    'create_contribution_chart',
    'create_risk_return_scatter',
    'create_stock_metrics_table',
]
