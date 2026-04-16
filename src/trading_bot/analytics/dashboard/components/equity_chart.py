"""
Equity Chart Component

Portfolio value over time vs benchmark.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_equity_chart(
    equity_data: pd.DataFrame,
    initial_capital: float = 10000.0,
    title: str = "Portfolio Value vs Benchmark"
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
        subplot_titles=('Portfolio Value ($)', 'Daily Returns')
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
            line=dict(color="#4da6ff", width=2.5),
            hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=benchmark,
            name="Benchmark (SPY)",
            line=dict(color="#4cd964", width=2.5, dash='dash'),
            hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Daily returns heatmap
    if 'daily_return' in equity_data.columns:
        daily_returns = equity_data['daily_return'].fillna(0)
        monthly_returns = _get_monthly_returns(daily_returns)

        fig.add_trace(
            go.Heatmap(
                z=monthly_returns['values'].T,
                x=monthly_returns['years'],
                y=monthly_returns['months'],
                colorscale=[
                    [0.0, '#d62728'],
                    [0.49, '#d62728'],
                    [0.5, '#ffdd00'],
                    [0.51, '#2ca02c'],
                    [1.0, '#2ca02c']
                ],
                zmid=0,
                hovertemplate='Month: %{y} %{x}<br>Return: %{z:.1%}<extra></extra>',
                showscale=True,
                colorbar=dict(title='Return', thickness=10)
            ),
            row=2, col=1
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#ffffff')),
        height=600,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        font=dict(color='#ffffff'),
    )

    fig.update_yaxes(title_text='Portfolio Value ($)', row=1, col=1, tickformat='$,.0f', gridcolor='#404040', tickcolor='#404040')
    fig.update_yaxes(title_text='Return', row=2, col=1, tickformat='.0%', gridcolor='#404040', tickcolor='#404040')
    fig.update_xaxes(gridcolor='#404040', tickcolor='#404040')

    return fig


def _get_monthly_returns(cumulative: pd.Series) -> Dict:
    """Calculate monthly returns for heatmap."""
    returns = cumulative.pct_change().fillna(0)

    monthly = returns.resample('ME').sum()

    df = pd.DataFrame({
        'date': monthly.index,
        'return': monthly.values
    })
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    pivot = df.pivot(index='year', columns='month', values='return')

    return {
        'values': pivot.values,
        'years': [str(y) for y in pivot.index],
        'months': [month_labels[m-1] for m in pivot.columns]
    }


def create_drawdown_chart(
    drawdown_data: pd.Series,
    title: str = "Drawdown Analysis"
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
            fillcolor='rgba(214, 39, 40, 0.3)',
            line=dict(color='#d62728', width=1),
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.1%}<extra></extra>'
        )
    )

    # Mark max drawdown
    max_dd_idx = drawdown_data.idxmin()
    max_dd = drawdown_data.min()

    fig.add_trace(
        go.Scatter(
            x=[max_dd_idx],
            y=[max_dd],
            mode='markers',
            marker=dict(size=10, color='#d62728', symbol='circle'),
            name=f'Max Drawdown: {max_dd:.1%}',
            hovertemplate='Max DD: {y:.1%}<extra></extra>'
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=300,
        hovermode='x unified',
        template='plotly_dark',
        plot_bgcolor='#21262d',
        paper_bgcolor='#21262d',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        yaxis=dict(tickformat='.0%', gridcolor='#eee'),
        xaxis=dict(gridcolor='#eee'),
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
    fig.update_layout(
        template='plotly_dark',
        plot_bgcolor='#21262d',
        paper_bgcolor='#21262d',
        height=300,
    )
    return fig
