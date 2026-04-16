"""
Analytics Dashboard

Real-time analytics dashboard for trading bot.
"""

from trading_bot.analytics.dashboard.live_dashboard import (
    LiveDashboard,
    DashboardConfig,
    create_dashboard_html,
)

__all__ = ['LiveDashboard', 'DashboardConfig', 'create_dashboard_html']
