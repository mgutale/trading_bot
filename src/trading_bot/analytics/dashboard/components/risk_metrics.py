"""
Risk Metrics Component

VaR, CVaR, volatility, and other risk metrics display.
"""

from typing import Dict, Any
from dash import html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_bot.analytics.dashboard.theme import DEFAULT_THEME
from trading_bot.analytics.dashboard.chart_utils import create_figure, apply_dark_theme, empty_figure


def create_risk_metrics(risk: Dict[str, Any]) -> html.Div:
    """
    Create risk metrics display.

    Args:
        risk: Dict with risk metrics (var_95, cvar_95, volatility, etc.)

    Returns:
        Dash HTML component
    """
    var_95 = risk.get('var_95', 0)
    var_99 = risk.get('var_99', 0)
    cvar_95 = risk.get('cvar_95', 0)
    cvar_99 = risk.get('cvar_99', 0)
    volatility = risk.get('volatility', 0)
    max_dd = risk.get('max_drawdown', 0)

    cards = [
        create_risk_card('VaR (95%)', f"{var_95:.2%}", DEFAULT_THEME.danger),
        create_risk_card('VaR (99%)', f"{var_99:.2%}", DEFAULT_THEME.danger),
        create_risk_card('CVaR (95%)', f"{cvar_95:.2%}", DEFAULT_THEME.danger),
        create_risk_card('CVaR (99%)', f"{cvar_99:.2%}", DEFAULT_THEME.danger),
        create_risk_card('Volatility', f"{volatility:.2%}", DEFAULT_THEME.warning),
        create_risk_card('Max Drawdown', f"{max_dd:.2%}", DEFAULT_THEME.danger),
    ]

    return html.Div([
        html.H4("Risk Metrics", style={'marginBottom': '10px'}),
        html.Div(cards, className='risk-cards-grid'),
    ], className='risk-metrics')


def create_risk_card(name: str, value: str, color: str = None) -> html.Div:
    """Create a single risk metric card."""
    color = color or DEFAULT_THEME.text_primary
    return html.Div([
        html.Span(name, className='risk-label'),
        html.Br(),
        html.Span(value, className='risk-value', style={'color': color}),
    ], className='risk-card')


def create_var_chart(returns, var_confidence: float = 0.95) -> go.Figure:
    """
    Create VaR chart showing the distribution of returns with VaR markers.

    Args:
        returns: Series of returns
        var_confidence: VaR confidence level

    Returns:
        Plotly Figure
    """
    if len(returns) == 0:
        return empty_figure("No return data", theme=DEFAULT_THEME)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Returns Distribution', 'VaR Analysis'),
    )

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns',
            marker_color=DEFAULT_THEME.primary,
            opacity=0.7,
        ),
        row=1, col=1,
    )

    # VaR lines
    var_val = returns.quantile(1 - var_confidence)
    cvar_val = returns[returns <= var_val].mean()

    fig.add_vline(
        x=var_val,
        line=dict(color=DEFAULT_THEME.danger, width=2, dash='dash'),
        row=1, col=1,
        annotation_text=f'VaR {int(var_confidence*100)}%: {var_val:.2%}',
    )
    fig.add_vline(
        x=cvar_val,
        line=dict(color='#8b0000', width=2, dash='dot'),
        row=1, col=1,
        annotation_text=f'CVaR: {cvar_val:.2%}',
    )

    # VaR bar chart
    var_95 = returns.quantile(0.05)
    var_99 = returns.quantile(0.01)

    fig.add_trace(
        go.Bar(
            x=['VaR 95%', 'VaR 99%', 'CVaR 95%', 'CVaR 99%'],
            y=[var_95, var_99, returns[returns <= var_95].mean(), returns[returns <= var_99].mean()],
            marker_color=[DEFAULT_THEME.warning, DEFAULT_THEME.danger, DEFAULT_THEME.warning, DEFAULT_THEME.danger],
            name='Risk Metrics',
        ),
        row=1, col=2,
    )

    fig.update_layout(
        height=350,
        showlegend=False,
        plot_bgcolor=DEFAULT_THEME.bg_primary,
        paper_bgcolor=DEFAULT_THEME.bg_primary,
    )

    fig.update_xaxes(title_text='Return', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=1, col=1)
    fig.update_xaxes(title_text='Metric', row=1, col=2)
    fig.update_yaxes(title_text='Value', row=1, col=2, tickformat='.1%')

    return fig


def create_drawdown_chart(drawdown_series) -> go.Figure:
    """
    Create drawdown chart with annotations.

    Args:
        drawdown_series: Series of drawdown values

    Returns:
        Plotly Figure
    """
    if len(drawdown_series) == 0:
        return empty_figure("No drawdown data", theme=DEFAULT_THEME)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=drawdown_series.index,
            y=drawdown_series,
            name='Drawdown',
            fill='tozeroy',
            fillcolor=f'{DEFAULT_THEME.danger}4d',
            line=dict(color=DEFAULT_THEME.danger, width=1),
        ),
    )

    # Mark max drawdown
    max_dd_idx = drawdown_series.idxmin()
    max_dd = drawdown_series.min()

    fig.add_trace(
        go.Scatter(
            x=[max_dd_idx],
            y=[max_dd],
            mode='markers+text',
            marker=dict(size=12, color=DEFAULT_THEME.danger, symbol='circle'),
            text=[f'Max DD: {max_dd:.1%}'],
            textposition='top center',
        ),
    )

    fig.update_layout(
        title=dict(text='Drawdown Analysis', font=dict(size=16)),
        height=300,
        hovermode='x unified',
        template='plotly_dark',
        plot_bgcolor=DEFAULT_THEME.bg_primary,
        paper_bgcolor=DEFAULT_THEME.bg_primary,
        yaxis=dict(tickformat='.0%', gridcolor=DEFAULT_THEME.border_default),
        xaxis=dict(gridcolor=DEFAULT_THEME.border_default),
    )

    return fig


def create_concentration_risk(positions: list) -> go.Figure:
    """
    Create a chart showing position concentration risk.

    Args:
        positions: List of position dicts

    Returns:
        Plotly Figure
    """
    if not positions:
        return empty_figure("No positions", theme=DEFAULT_THEME)

    values = [p.get('market_value', 0) for p in positions]
    total = sum(values)
    symbols = [p.get('symbol', 'Unknown') for p in positions]

    # Herfindahl index (concentration measure)
    weights = [v / total for v in values if total > 0]
    hhi = sum(w ** 2 for w in weights)

    # Bar chart of position sizes
    fig = go.Figure(data=[go.Bar(
        x=symbols,
        y=values,
        marker_color=DEFAULT_THEME.primary,
    )])

    # Add horizontal line at equal weight
    n = len(positions)
    equal_weight = total / n if n > 0 else 0
    fig.add_hline(
        y=equal_weight,
        line=dict(color=DEFAULT_THEME.success, width=2, dash='dash'),
        annotation_text=f'Equal Weight: ${equal_weight:,.0f}',
    )

    fig.update_layout(
        title=dict(text=f'Position Concentration (HHI: {hhi:.3f})', font=dict(size=16)),
        height=300,
        template='plotly_dark',
        plot_bgcolor=DEFAULT_THEME.bg_primary,
        paper_bgcolor=DEFAULT_THEME.bg_primary,
        yaxis=dict(title='Market Value ($)', gridcolor=DEFAULT_THEME.border_default),
        xaxis=dict(title='Symbol', gridcolor=DEFAULT_THEME.border_default),
    )

    return fig
