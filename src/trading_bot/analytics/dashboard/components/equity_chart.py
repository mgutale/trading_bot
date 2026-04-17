"""
Equity Chart Component

Portfolio value over time vs benchmark.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from trading_bot.analytics.dashboard.theme import DEFAULT_THEME
from trading_bot.analytics.dashboard.chart_utils import create_figure, apply_dark_theme


def create_equity_chart(
    equity_data: pd.DataFrame,
    initial_capital: float = 10000.0,
    title: str = "Portfolio Value vs Benchmark",
) -> go.Figure:
    """
    Create equity curve chart comparing portfolio vs benchmark.

    Args:
        equity_data: DataFrame with date, equity, benchmark columns
        initial_capital: Starting portfolio value
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if equity_data is None or len(equity_data) == 0:
        return _empty_figure("No equity data available")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Portfolio Value ($)', 'Monthly Returns'),
    )

    # Equity curve
    dates = equity_data['date'] if 'date' in equity_data.columns else equity_data.index
    equity = equity_data['equity'] if 'equity' in equity_data.columns else equity_data.iloc[:, 0]
    benchmark = equity_data.get('benchmark', equity) if isinstance(equity_data, pd.DataFrame) else equity

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=equity,
            name="Portfolio",
            line=dict(color=DEFAULT_THEME.primary, width=2.5),
            hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>',
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=benchmark,
            name="Benchmark (SPY)",
            line=dict(color=DEFAULT_THEME.success, width=2.5, dash='dash'),
            hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>',
        ),
        row=1, col=1,
    )

    # Monthly returns heatmap
    if 'equity' in equity_data.columns:
        # Use equity values to calculate monthly returns
        monthly_data = _get_monthly_returns(equity_data['equity'])

        fig.add_trace(
            go.Heatmap(
                z=monthly_data['values'].T,
                x=monthly_data['years'],
                y=monthly_data['months'],
                colorscale=[
                    [0.0, DEFAULT_THEME.danger],
                    [0.49, DEFAULT_THEME.danger],
                    [0.5, DEFAULT_THEME.warning],
                    [0.51, DEFAULT_THEME.success],
                    [1.0, DEFAULT_THEME.success],
                ],
                zmid=0,
                hovertemplate='Month: %{y} %{x}<br>Return: %{z:.1%}<extra></extra>',
                showscale=True,
                colorbar=dict(title='Return', thickness=10),
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color=DEFAULT_THEME.text_primary)),
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor=DEFAULT_THEME.bg_primary,
        paper_bgcolor=DEFAULT_THEME.bg_primary,
        font=dict(color=DEFAULT_THEME.text_primary),
    )

    fig.update_yaxes(title_text='Portfolio Value ($)', row=1, col=1, tickformat='$,.0f', gridcolor=DEFAULT_THEME.border_default, tickcolor=DEFAULT_THEME.border_default)
    fig.update_yaxes(title_text='Return', row=2, col=1, tickformat='.0%', gridcolor=DEFAULT_THEME.border_default, tickcolor=DEFAULT_THEME.border_default)
    fig.update_xaxes(gridcolor=DEFAULT_THEME.border_default, tickcolor=DEFAULT_THEME.border_default)

    return fig


def _get_monthly_returns(equity: pd.Series) -> Dict:
    """
    Calculate monthly returns for heatmap.

    Args:
        equity: Series of portfolio values (not cumulative returns)

    Returns:
        Dict with values matrix, years list, and month labels
    """
    # Resample to month-end values
    monthly_equity = equity.resample('ME').last()

    # Calculate monthly returns as pct change of month-end values
    monthly_returns = monthly_equity.pct_change().fillna(0)

    df = pd.DataFrame({
        'date': monthly_returns.index,
        'return': monthly_returns.values,
    })
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    pivot = df.pivot(index='year', columns='month', values='return')

    return {
        'values': pivot.values,
        'years': [str(y) for y in pivot.index],
        'months': [month_labels[m-1] for m in pivot.columns],
    }


def create_drawdown_chart(
    drawdown_data: pd.Series,
    title: str = "Drawdown Analysis",
) -> go.Figure:
    """
    Create drawdown chart.

    Args:
        drawdown_data: Series of drawdown values (negative = drawdown)
        title: Chart title

    Returns:
        Plotly Figure object
    """
    if drawdown_data is None or len(drawdown_data) == 0:
        return _empty_figure("No drawdown data available")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=drawdown_data.index,
            y=drawdown_data,
            name='Drawdown',
            fill='tozeroy',
            fillcolor=f'{DEFAULT_THEME.danger}4d',  # 30% opacity
            line=dict(color=DEFAULT_THEME.danger, width=1),
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.1%}<extra></extra>',
        ),
    )

    # Mark max drawdown
    max_dd_idx = drawdown_data.idxmin()
    max_dd = drawdown_data.min()

    fig.add_trace(
        go.Scatter(
            x=[max_dd_idx],
            y=[max_dd],
            mode='markers',
            marker=dict(size=10, color=DEFAULT_THEME.danger, symbol='circle'),
            name=f'Max Drawdown: {max_dd:.1%}',
            hovertemplate='Max DD: {y:.1%}<extra></extra>',
        ),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=300,
        hovermode='x unified',
        template='plotly_dark',
        plot_bgcolor=DEFAULT_THEME.bg_primary,
        paper_bgcolor=DEFAULT_THEME.bg_primary,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis=dict(tickformat='.0%', gridcolor=DEFAULT_THEME.border_default),
        xaxis=dict(gridcolor=DEFAULT_THEME.border_default),
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
        font=dict(size=16, color=DEFAULT_THEME.text_secondary),
    )
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor=DEFAULT_THEME.bg_primary,
        paper_bgcolor=DEFAULT_THEME.bg_primary,
        height=300,
    )
    return fig
