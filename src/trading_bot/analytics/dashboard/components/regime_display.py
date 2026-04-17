"""
Regime Display Component

Current market regime and historical regime timeline.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import plotly.graph_objects as go
from dash import html

from trading_bot.analytics.dashboard.theme import DEFAULT_THEME
from trading_bot.analytics.dashboard.chart_utils import empty_figure


def create_regime_display(
    current_regime: str = 'unknown',
    regime_numeric: int = 0,
    regime_duration_days: int = 0,
    exposure: float = 1.0,
    regime_history: pd.Series = None,
    title: str = "Market Regime",
) -> html.Div:
    """
    Create current regime display with exposure indicator.

    Args:
        current_regime: Current regime label
        regime_numeric: Numeric regime value (0-3)
        regime_duration_days: Days in current regime
        exposure: Current exposure percentage
        regime_history: Series of regime labels over time
        title: Component title

    Returns:
        Dash HTML component
    """
    color = DEFAULT_THEME.regime_colors.get(current_regime, DEFAULT_THEME.text_secondary)
    name = _format_regime_name(current_regime)

    # Exposure bar
    exposure_pct = int(exposure * 100) if exposure else 0

    return html.Div([
        html.H4(title, style={'marginBottom': '10px'}),
        html.Div([
            html.Div([
                html.Span("Current Regime: ", style={'fontWeight': 'bold'}),
                html.Span(name, style={'color': color, 'fontWeight': 'bold', 'fontSize': '18px'}),
            ], style={'marginBottom': '5px'}),
            html.Div([
                html.Span("Duration: ", style={'fontWeight': 'bold'}),
                html.Span(f"{regime_duration_days} days"),
            ], style={'marginBottom': '5px'}),
            html.Div([
                html.Span("Exposure: ", style={'fontWeight': 'bold'}),
                html.Span(f"{exposure_pct}%"),
            ], style={'marginBottom': '10px'}),
            # Exposure bar
            html.Div([
                html.Div(
                    style={
                        'width': f'{exposure_pct}%',
                        'backgroundColor': color,
                        'height': '20px',
                        'borderRadius': DEFAULT_THEME.radius_sm,
                        'transition': 'width 0.3s ease',
                    }
                ),
            ], style={
                'backgroundColor': DEFAULT_THEME.bg_secondary,
                'borderRadius': DEFAULT_THEME.radius_sm,
                'width': '100%',
                'height': '20px',
            }),
        ], className="regime-display-card"),
    ], className="regime-display")


def create_regime_chart(regime_history: pd.Series) -> go.Figure:
    """
    Create historical regime visualization.

    Args:
        regime_history: Series of regime labels indexed by date

    Returns:
        Plotly Figure
    """
    if regime_history is None or len(regime_history) == 0:
        return empty_figure("No regime data", theme=DEFAULT_THEME)

    numeric_map = {'strong_bull': 3, 'weak_bull': 2, 'weak_bear': 1, 'strong_bear': 0}
    regime_names = {0: 'Strong Bear', 1: 'Weak Bear', 2: 'Weak Bull', 3: 'Strong Bull'}

    regimes_numeric = regime_history.map(lambda x: numeric_map.get(x, 1))

    fig = go.Figure()

    # Add colored line segments for each regime
    for regime_num in [0, 1, 2, 3]:
        mask = regimes_numeric == regime_num
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        # Find contiguous segments
        segments = []
        start = indices[0]
        prev = indices[0]
        for i in indices[1:]:
            if i != prev + 1:
                segments.append((start, prev))
                start = i
            prev = i
        segments.append((start, prev))

        # Add each segment as a thick line
        for idx, (seg_start, seg_end) in enumerate(segments):
            x_start = regime_history.index[seg_start]
            x_end = regime_history.index[min(seg_end + 1, len(regime_history) - 1)]
            y_val = regime_num + 1

            fig.add_trace(
                go.Scatter(
                    x=[x_start, x_end],
                    y=[y_val, y_val],
                    mode='lines',
                    line=dict(color=DEFAULT_THEME.regime_colors.get(list(numeric_map.keys())[regime_num], DEFAULT_THEME.text_secondary), width=12),
                    name=regime_names[regime_num],
                    legendgroup=regime_names[regime_num],
                    showlegend=(idx == 0),
                    hovertemplate=f'Regime: {regime_names[regime_num]}<br>From: {x_start.strftime("%Y-%m-%d")}<br>To: {x_end.strftime("%Y-%m-%d")}<extra></extra>',
                ),
            )

    fig.update_layout(
        title=dict(text="Market Regime History", font=dict(size=16, color=DEFAULT_THEME.text_primary)),
        height=250,
        yaxis=dict(
            range=[0.5, 4.5],
            showticklabels=True,
            tickvals=[1, 2, 3, 4],
            ticktext=['Strong Bear', 'Weak Bear', 'Weak Bull', 'Strong Bull'],
            tickcolor=DEFAULT_THEME.border_default,
            gridcolor=DEFAULT_THEME.border_default,
        ),
        plot_bgcolor=DEFAULT_THEME.bg_primary,
        paper_bgcolor=DEFAULT_THEME.bg_primary,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis=dict(gridcolor=DEFAULT_THEME.border_default, tickcolor=DEFAULT_THEME.border_default),
    )

    fig.update_xaxes(title_text='Date', tickcolor=DEFAULT_THEME.border_default)
    fig.update_yaxes(title_text='Market Regime', gridcolor=DEFAULT_THEME.border_default, tickcolor=DEFAULT_THEME.border_default)

    return fig


def create_regime_summary(regime_history: pd.Series) -> html.Div:
    """
    Create regime distribution summary.

    Args:
        regime_history: Series of regime labels

    Returns:
        Dash HTML component
    """
    if regime_history is None or len(regime_history) == 0:
        return html.Div([])

    regime_counts = regime_history.value_counts()
    total = len(regime_history)

    regime_labels = {
        'strong_bull': 'Strong Bull',
        'weak_bull': 'Weak Bull',
        'weak_bear': 'Weak Bear',
        'strong_bear': 'Strong Bear',
    }

    items = []
    for regime, label in regime_labels.items():
        count = regime_counts.get(regime, 0)
        pct = count / total * 100 if total > 0 else 0
        color = DEFAULT_THEME.regime_colors.get(regime, DEFAULT_THEME.text_secondary)
        items.append(html.Div([
            html.Span(f"{label}: ", style={'fontWeight': 'bold', 'color': color}),
            html.Span(f"{count} days ({pct:.1f}%)"),
        ], style={'marginBottom': '5px'}))

    return html.Div([
        html.H4("Regime Distribution", style={'marginBottom': '10px'}),
        html.Div(items),
    ], className="regime-summary")


def _format_regime_name(regime: str) -> str:
    """Format regime name for display."""
    return regime.replace('_', ' ').title()


def _empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    return empty_figure(message, theme=DEFAULT_THEME)
