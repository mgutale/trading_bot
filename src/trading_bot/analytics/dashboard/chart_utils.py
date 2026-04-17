"""
Chart Utilities

Helper functions for consistent Plotly chart styling across the dashboard.
All charts should use these utilities for unified appearance.
"""

import plotly.graph_objects as go
from typing import Optional, List, Dict, Any
import numpy as np

from trading_bot.analytics.dashboard.theme import DashboardTheme, DEFAULT_THEME


def create_figure(
    title: str = "",
    height: int = 400,
    theme: Optional[DashboardTheme] = None,
    **layout_kwargs
) -> go.Figure:
    """
    Create a Plotly figure with consistent theme settings.

    Args:
        title: Chart title
        height: Chart height in pixels
        theme: Theme to use (defaults to DEFAULT_THEME)
        **layout_kwargs: Additional layout arguments

    Returns:
        Configured Plotly Figure
    """
    theme = theme or DEFAULT_THEME

    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color=theme.text_primary),
            x=0,
            xanchor='left',
        ),
        height=height,
        plot_bgcolor=theme.bg_primary,
        paper_bgcolor=theme.bg_primary,
        font=dict(family=theme.font_family, size=12, color=theme.text_primary),
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor=theme.bg_card,
            bordercolor=theme.border_default,
            borderwidth=1,
        ),
        **layout_kwargs
    )

    return fig


def apply_dark_theme(
    fig: go.Figure,
    theme: Optional[DashboardTheme] = None,
    grid_color: Optional[str] = None,
) -> go.Figure:
    """
    Apply dark theme styling to an existing figure.

    Args:
        fig: Plotly Figure to style
        theme: Theme to use (defaults to DEFAULT_THEME)
        grid_color: Override for grid line color

    Returns:
        Styled Plotly Figure
    """
    theme = theme or DEFAULT_THEME
    grid_color = grid_color or theme.border_default

    fig.update_layout(
        plot_bgcolor=theme.bg_primary,
        paper_bgcolor=theme.bg_primary,
        font=dict(color=theme.text_primary),
    )

    # Update all axes
    fig.update_xaxes(
        gridcolor=grid_color,
        linecolor=grid_color,
        tickcolor=grid_color,
        title_font=dict(color=theme.text_secondary),
        tickfont=dict(color=theme.text_secondary),
    )

    fig.update_yaxes(
        gridcolor=grid_color,
        linecolor=grid_color,
        tickcolor=grid_color,
        title_font=dict(color=theme.text_secondary),
        tickfont=dict(color=theme.text_secondary),
    )

    return fig


def empty_figure(
    message: str,
    theme: Optional[DashboardTheme] = None,
    height: int = 250,
) -> go.Figure:
    """
    Create an empty figure with a centered message.

    Args:
        message: Text to display
        theme: Theme to use
        height: Figure height

    Returns:
        Plotly Figure with centered message
    """
    theme = theme or DEFAULT_THEME

    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color=theme.text_secondary),
    )
    fig.update_layout(
        height=height,
        plot_bgcolor=theme.bg_primary,
        paper_bgcolor=theme.bg_primary,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig


def color_for_return(value: float, theme: Optional[DashboardTheme] = None) -> str:
    """
    Get color based on return value.

    Args:
        value: Return value (positive = gain, negative = loss)
        theme: Theme to use

    Returns:
        Color hex string
    """
    theme = theme or DEFAULT_THEME

    if value > 0:
        return theme.success
    elif value < 0:
        return theme.danger
    return theme.text_secondary


def create_bar_chart(
    x: List[Any],
    y: List[float],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title: str = "",
    y_suffix: str = "",
    theme: Optional[DashboardTheme] = None,
    **kwargs
) -> go.Figure:
    """
    Create a styled bar chart.

    Args:
        x: X-axis values
        y: Y-axis values
        labels: Optional bar labels
        colors: Optional bar colors (auto-colors by return if not provided)
        title: Chart title
        y_suffix: Suffix for y-axis tick labels (e.g., '%', '$')
        theme: Theme to use
        **kwargs: Additional bar trace arguments

    Returns:
        Plotly Figure
    """
    theme = theme or DEFAULT_THEME
    fig = create_figure(title=title, height=theme.chart_height_md, theme=theme)

    # Auto-color by return values if not specified
    if colors is None and y_suffix == '%':
        colors = [color_for_return(v, theme) for v in y]
    elif colors is None:
        colors = [theme.primary] * len(y)

    fig.add_trace(go.Bar(
        x=x,
        y=y,
        marker_color=colors,
        text=labels if labels else [f'{v:.1f}{y_suffix}' for v in y],
        textposition='outside',
        hovertemplate='%{x}<br>%{y:.2f}' + y_suffix + '<extra></extra>',
        **kwargs
    ))

    fig.update_layout(showlegend=False)
    apply_dark_theme(fig, theme)

    return fig


def create_line_chart(
    x: List[Any],
    y: List[float],
    name: str = "Series",
    color: Optional[str] = None,
    title: str = "",
    y_suffix: str = "",
    theme: Optional[DashboardTheme] = None,
    fill: Optional[str] = None,
    dash: Optional[str] = None,
    **kwargs
) -> go.Figure:
    """
    Create a styled line chart.

    Args:
        x: X-axis values
        y: Y-axis values
        name: Series name for legend
        color: Line color
        title: Chart title
        y_suffix: Suffix for y-axis tick labels
        theme: Theme to use
        fill: Fill option (e.g., 'tozeroy')
        dash: Dash style (e.g., 'dot', 'dash')
        **kwargs: Additional scatter trace arguments

    Returns:
        Plotly Figure
    """
    theme = theme or DEFAULT_THEME
    color = color or theme.primary

    fig = create_figure(title=title, height=theme.chart_height_md, theme=theme)

    hovertemplate = '%{x}<br>%{y:.2f}' + y_suffix + '<extra></extra>'

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        name=name,
        line=dict(color=color, width=theme.chart_line_width, dash=dash),
        fill=fill,
        hovertemplate=hovertemplate,
        **kwargs
    ))

    apply_dark_theme(fig, theme)
    return fig


def create_multi_line_chart(
    series: List[Dict[str, Any]],
    title: str = "",
    y_suffix: str = "",
    theme: Optional[DashboardTheme] = None,
    **kwargs
) -> go.Figure:
    """
    Create a chart with multiple lines.

    Args:
        series: List of dicts with keys: name, x, y, color, dash
        title: Chart title
        y_suffix: Suffix for y-axis tick labels
        theme: Theme to use
        **kwargs: Additional layout arguments

    Returns:
        Plotly Figure
    """
    theme = theme or DEFAULT_THEME
    fig = create_figure(title=title, height=theme.chart_height_md, theme=theme, **kwargs)

    for i, s in enumerate(series):
        color = s.get('color') or theme.get_chart_color(i)
        dash = s.get('dash')

        fig.add_trace(go.Scatter(
            x=s['x'],
            y=s['y'],
            name=s.get('name', f'Series {i}'),
            line=dict(color=color, width=theme.chart_line_width, dash=dash),
            fill=s.get('fill'),
            hovertemplate='%{x}<br>%{y:.2f}' + y_suffix + '<extra></extra>',
        ))

    apply_dark_theme(fig, theme)
    return fig


def create_heatmap(
    z: List[List[float]],
    x: Optional[List[str]] = None,
    y: Optional[List[str]] = None,
    title: str = "",
    colorscale: Optional[List] = None,
    zmid: float = 0,
    theme: Optional[DashboardTheme] = None,
    **kwargs
) -> go.Figure:
    """
    Create a styled heatmap.

    Args:
        z: 2D array of values
        x: X-axis labels
        y: Y-axis labels
        title: Chart title
        colorscale: Color scale (defaults to red-yellow-green)
        zmid: Center value for diverging colorscale
        theme: Theme to use
        **kwargs: Additional heatmap arguments

    Returns:
        Plotly Figure
    """
    theme = theme or DEFAULT_THEME

    if colorscale is None:
        colorscale = [
            [0.0, theme.danger],
            [0.5, theme.warning],
            [1.0, theme.success],
        ]

    fig = create_figure(title=title, height=theme.chart_height_md, theme=theme)

    fig.add_trace(go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=colorscale,
        zmid=zmid,
        text=z,
        texttemplate='%{text:.1%}',
        textfont={"color": theme.text_primary},
        hovertemplate='%{y}<br>%{x}: %{z:.2%}<extra></extra>',
        **kwargs
    ))

    apply_dark_theme(fig, theme)
    return fig


def create_scatter_plot(
    x: List[float],
    y: List[float],
    labels: Optional[List[str]] = None,
    color_by: Optional[List[float]] = None,
    title: str = "",
    x_label: str = "X",
    y_label: str = "Y",
    theme: Optional[DashboardTheme] = None,
    **kwargs
) -> go.Figure:
    """
    Create a scatter plot with optional coloring.

    Args:
        x: X-axis values
        y: Y-axis values
        labels: Point labels
        color_by: Values to color points by
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        theme: Theme to use
        **kwargs: Additional scatter arguments

    Returns:
        Plotly Figure
    """
    theme = theme or DEFAULT_THEME
    fig = create_figure(title=title, height=theme.chart_height_md, theme=theme)

    if color_by is not None:
        # Color by value
        colors = [color_for_return(v, theme) for v in color_by]
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                color=colors,
                size=10,
                line=dict(width=1, color=theme.bg_card),
            ),
            text=labels,
            hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
            **kwargs
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            marker=dict(
                color=theme.primary,
                size=10,
                line=dict(width=1, color=theme.bg_card),
            ),
            text=labels,
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
            **kwargs
        ))

    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    apply_dark_theme(fig, theme)

    return fig
