"""
Backtest Visualization and HTML Report Generator

Creates powerful charts and interactive HTML reports for strategy performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


class BacktestVisualizer:
    """
    Creates professional charts and HTML reports for backtest results.
    """

    def __init__(self, results: Dict, strategy_name: str = "Hybrid HMM + 5% Stop Loss"):
        self.results = results
        self.strategy_name = strategy_name
        self.signals = results.get('signals', pd.DataFrame())
        self.colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17becf',
            'strong_bull': '#006400',    # Dark green
            'weak_bull': '#2ca02c',      # Green
            'weak_bear': '#ff7f0e',      # Orange
            'strong_bear': '#8b0000'     # Dark red
        }

    def create_equity_curve(self, initial_capital: float = 10000) -> go.Figure:
        """Create cumulative returns chart comparing strategy vs benchmark."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Portfolio Value ($)', 'Monthly Returns')
        )

        # Strategy cumulative return
        if 'cumulative_return_net' in self.signals.columns:
            strat_cum = self.signals['cumulative_return_net']
        else:
            strat_cum = self.signals['cumulative_return']

        # Benchmark cumulative
        bench_cum = self.signals['benchmark_cumulative']

        # Convert to actual portfolio values
        strat_value = initial_capital * (1 + strat_cum)
        bench_value = initial_capital * (1 + bench_cum)

        # Plot cumulative returns as portfolio value
        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=strat_value,
                name=self.strategy_name,
                line=dict(color=self.colors['primary'], width=2),
                hovertemplate='Date: %{x}<br>Portfolio Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=bench_value,
                name='Benchmark (SPY)',
                line=dict(color=self.colors['info'], width=2, dash='dash'),
                hovertemplate='Date: %{x}<br>Portfolio Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Monthly returns heatmap data
        monthly_returns = self._get_monthly_returns(strat_cum)

        fig.add_trace(
            go.Heatmap(
                z=monthly_returns['values'].T,
                x=monthly_returns['years'],
                y=monthly_returns['months'],
                colorscale=[
                    [0.0, '#d62728'],      # Red for most negative
                    [0.49, '#d62728'],     # Keep red up to just below zero
                    [0.5, '#ffdd00'],      # Yellow exactly at zero
                    [0.51, '#2ca02c'],     # Green immediately above zero
                    [1.0, '#2ca02c']       # Green for most positive
                ],
                zmid=0,
                hovertemplate='Month: %{y} %{x}<br>Return: %{z:.1%}<extra></extra>',
                showscale=True,
                colorbar=dict(title='Return', thickness=10)
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f'{self.strategy_name} - Equity Curve (${initial_capital:,} Initial)',
            height=700,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            hovermode='x unified',
            template='plotly_white'
        )

        fig.update_yaxes(title_text='Portfolio Value ($)', row=1, col=1, tickformat='$,.0f')
        fig.update_yaxes(title_text='Month', row=2, col=1)

        return fig

    def _get_monthly_returns(self, cumulative: pd.Series) -> Dict:
        """Calculate monthly returns for heatmap."""
        returns = cumulative.pct_change().fillna(0)
        monthly = returns.resample('ME').sum()

        # Pivot to get years x months
        df = pd.DataFrame({
            'date': monthly.index,
            'return': monthly.values
        })
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # Create month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Pivot table - years on y-axis (rows), months on x-axis (columns)
        pivot = df.pivot(index='year', columns='month', values='return')

        return {
            'values': pivot.values,
            'years': [str(y) for y in pivot.index],
            'months': [month_labels[m-1] for m in pivot.columns]
        }

    def create_drawdown_chart(self) -> go.Figure:
        """Create drawdown analysis chart."""
        # Calculate drawdown
        cumulative = (1 + self.signals['strategy_return']).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative / rolling_max - 1)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.signals.index,
                y=drawdown,
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.3)',
                line=dict(color=self.colors['danger'], width=1),
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.1%}<extra></extra>'
            )
        )

        # Mark max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd = drawdown.min()

        fig.add_trace(
            go.Scatter(
                x=[max_dd_idx],
                y=[max_dd],
                mode='markers',
                marker=dict(size=10, color=self.colors['danger'], symbol='circle'),
                name=f'Max Drawdown: {max_dd:.1%}',
                hovertemplate='Max DD: {y:.1%}<extra></extra>'
            )
        )

        fig.update_layout(
            title='Drawdown Analysis',
            height=300,
            hovermode='x unified',
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        fig.update_yaxes(title_text='Drawdown', tickformat='.0%', gridcolor='#eee')
        fig.update_xaxes(gridcolor='#eee')

        return fig

    def create_regime_chart(self) -> go.Figure:
        """Create market regime visualization (4-State HMM)."""
        if 'regime' not in self.signals.columns:
            return go.Figure()

        # Map regime names to numeric values and colors
        regime_map = {'strong_bull': 3, 'weak_bull': 2, 'weak_bear': 1, 'strong_bear': 0}
        regime_colors = {
            0: self.colors['strong_bear'],    # Dark red
            1: self.colors['weak_bear'],      # Orange
            2: self.colors['weak_bull'],      # Green
            3: self.colors['strong_bull']     # Dark green
        }
        regime_names = {0: 'Strong Bear', 1: 'Weak Bear', 2: 'Weak Bull', 3: 'Strong Bull'}

        regimes_numeric = self.signals['regime_numeric'].map(regime_map)

        fig = go.Figure()

        # Add colored line segments for each regime
        for regime_num, color in regime_colors.items():
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
                x_start = self.signals.index[seg_start]
                x_end = self.signals.index[min(seg_end + 1, len(self.signals) - 1)]
                y_val = regime_num + 1  # Offset for visibility (1-4)

                fig.add_trace(
                    go.Scatter(
                        x=[x_start, x_end],
                        y=[y_val, y_val],
                        mode='lines',
                        line=dict(color=color, width=12),
                        name=regime_names[regime_num],
                        legendgroup=regime_names[regime_num],
                        showlegend=(idx == 0),  # Only show legend for first segment of each regime
                        hovertemplate=f'Regime: {regime_names[regime_num]}<br>From: {x_start.strftime("%Y-%m-%d")}<br>To: {x_end.strftime("%Y-%m-%d")}<extra></extra>'
                    )
                )

        fig.update_layout(
            title='Market Regime Detection (4-State HMM)',
            height=250,
            yaxis=dict(
                range=[0.5, 4.5],
                showticklabels=True,
                tickvals=[1, 2, 3, 4],
                ticktext=['Strong Bear', 'Weak Bear', 'Weak Bull', 'Strong Bull']
            ),
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Market Regime')

        return fig

    def create_returns_distribution(self) -> go.Figure:
        """Create returns distribution histogram."""
        returns = self.signals['strategy_return']

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Daily Returns Distribution', 'Cumulative Returns by Year')
        )

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                marker_color=self.colors['primary'],
                opacity=0.7
            ),
            row=1, col=1
        )

        # Add normal distribution curve
        mu, sigma = returns.mean(), returns.std()
        if sigma > 0 and len(returns) > 1:
            x_norm = np.linspace(returns.min(), returns.max(), 100)
            y_norm = len(returns) * (x_norm.max() - x_norm.min()) / 50 * \
                     (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)

            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    name='Normal Fit',
                    line=dict(color=self.colors['danger'], width=2)
                ),
                row=1, col=1
            )

        # Annual returns
        annual = returns.groupby(returns.index.year).sum()
        colors = [self.colors['success'] if v > 0 else self.colors['danger'] for v in annual.values]

        fig.add_trace(
            go.Bar(
                x=annual.index.astype(str),
                y=annual.values,
                marker_color=colors,
                name='Annual',
                hovertemplate='Year: %{x}<br>Return: %{y:.1%}<extra></extra>'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            showlegend=True,
            template='plotly_white'
        )

        fig.update_xaxes(title_text='Return', row=1, col=1)
        fig.update_yaxes(title_text='Frequency', row=1, col=1)
        fig.update_xaxes(title_text='Year', row=1, col=2)
        fig.update_yaxes(title_text='Return', row=1, col=2, tickformat='.0%')

        return fig

    def create_trade_analysis(self) -> go.Figure:
        """Create trade analysis chart."""
        trade_log = self.results.get('trade_log', [])

        if not trade_log:
            return go.Figure()

        # Count trades by type
        buy_count = len([t for t in trade_log if t[2] == 'buy'])
        sell_count = len([t for t in trade_log if t[2] == 'sell'])
        stop_loss_count = len([t for t in trade_log if t[3] == 'stop_loss'])
        rebalance_count = len([t for t in trade_log if t[3] == 'rebalance'])

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'bar'}, {'type': 'pie'}]],
            subplot_titles=('Trade Count', 'Trade Type Distribution')
        )

        # Bar chart
        fig.add_trace(
            go.Bar(
                x=['Buys', 'Sells', 'Stop Loss', 'Rebalances'],
                y=[buy_count, sell_count, stop_loss_count, rebalance_count],
                marker_color=[
                    self.colors['success'],
                    self.colors['danger'],
                    self.colors['danger'],
                    self.colors['warning']
                ],
                name='Trades'
            ),
            row=1, col=1
        )

        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=['Buys', 'Sells', 'Stop Loss', 'Rebalances'],
                values=[buy_count, sell_count, stop_loss_count, rebalance_count],
                marker_colors=[
                    self.colors['success'],
                    self.colors['danger'],
                    self.colors['danger'],
                    self.colors['warning']
                ],
                name='Distribution'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            showlegend=False,
            template='plotly_white'
        )

        fig.update_yaxes(title_text='Count', row=1, col=1)

        return fig

    def create_full_report(self, save_path: str = "results/backtest_report.html") -> str:
        """Create comprehensive HTML report with all charts."""
        # Create all charts
        equity_fig = self.create_equity_curve()
        drawdown_fig = self.create_drawdown_chart()
        regime_fig = self.create_regime_chart()
        returns_fig = self.create_returns_distribution()
        trade_fig = self.create_trade_analysis()

        # Download Plotly JS for offline embedding (self-contained HTML)
        import urllib.request
        plotly_js_tag = ""
        try:
            plotly_js = urllib.request.urlopen(
                "https://cdn.plot.ly/plotly-2.35.2.min.js", timeout=15
            ).read().decode('utf-8')
            plotly_js_tag = f"<script>{plotly_js}</script>"
        except Exception:
            plotly_js_tag = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

        # Convert to HTML (no CDN needed - we embed it above)
        equity_html = equity_fig.to_html(full_html=False, include_plotlyjs=False)
        drawdown_html = drawdown_fig.to_html(full_html=False, include_plotlyjs=False)
        regime_html = regime_fig.to_html(full_html=False, include_plotlyjs=False)
        returns_html = returns_fig.to_html(full_html=False, include_plotlyjs=False)
        trade_html = trade_fig.to_html(full_html=False, include_plotlyjs=False)

        # Get metrics
        metrics = self._get_metrics_table()

        # Build HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.strategy_name} - Backtest Report</title>
    {plotly_js_tag}
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background: linear-gradient(135deg, #1f77b4 0%, #17becf 100%);
            color: white;
            padding: 40px 20px;
            margin-bottom: 30px;
            border-radius: 0 0 20px 20px;
        }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        header p {{ opacity: 0.9; font-size: 1.1em; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-card h3 {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .metric-card .value {{
            font-size: 2.2em;
            font-weight: bold;
            color: #1f77b4;
        }}
        .metric-card .value.positive {{ color: #2ca02c; }}
        .metric-card .value.negative {{ color: #d62728; }}
        .chart-section {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .chart-section h2 {{
            margin-bottom: 20px;
            color: #333;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 10px;
        }}
        .chart-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: #fafafa;
            border-radius: 10px;
            padding: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #555;
        }}
        tr:hover {{ background: #f8f9fa; }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .chart-row {{ grid-template-columns: 1fr; }}
            header h1 {{ font-size: 1.8em; }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>{self.strategy_name}</h1>
            <p>Backtest Report | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
    </header>

    <div class="container">
        <div class="metrics-grid">
            {metrics}
        </div>

        <div class="chart-section">
            <h2>Equity Curve</h2>
            {equity_html}
        </div>

        <div class="chart-section">
            <h2>Drawdown Analysis</h2>
            {drawdown_html}
        </div>

        <div class="chart-section">
            <h2>Market Regimes</h2>
            {regime_html}
        </div>

        <div class="chart-row">
            <div class="chart-container">
                <div class="chart-section">
                    <h2>Returns Distribution</h2>
                    {returns_html}
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-section">
                    <h2>Trade Analysis</h2>
                    {trade_html}
                </div>
            </div>
        </div>

        <div class="chart-section">
            <h2>Performance Summary</h2>
            {self._get_performance_summary()}
        </div>
    </div>

    <div class="footer">
        <p>Generated by Trading Bot - Hybrid HMM + 5% Stop Loss Strategy</p>
        <p>Disclaimer: Past performance does not guarantee future results. Trading involves substantial risk.</p>
    </div>
</body>
</html>
"""

        # Save to file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return save_path

    def _get_metrics_table(self) -> str:
        """Generate metrics cards HTML."""
        metrics = [
            ('Total Return', self.results['total_return'], '.1%', False),
            ('Annualized Return', self.results['annualized_return'], '.1%', False),
            ('Sharpe Ratio', self.results['sharpe_ratio'], '.2f', False),
            ('Sortino Ratio', self.results['sortino_ratio'], '.2f', False),
            ('Max Drawdown', self.results['max_drawdown'], '.1%', True),
            ('Win Rate', self.results['win_rate'], '.1%', False),
            ('Profit Factor', self.results['profit_factor'], '.2f', False),
            ('Total Trades', self.results['num_trades'], 'd', False),
        ]

        html = ''
        for name, value, fmt, invert in metrics:
            if fmt == 'd':
                display = f"{value:,}"
            elif '%' in fmt:
                display = f"{value*100:.1f}%" if '.1' in fmt else f"{value*100:.2f}%"
            else:
                display = f"{value:.2f}"

            # Determine if positive or negative
            is_positive = (value > 0 and not invert) or (value < 0 and invert) or name in ['Sharpe Ratio', 'Sortino Ratio', 'Profit Factor', 'Win Rate', 'Total Trades']
            css_class = 'positive' if is_positive else 'negative'

            html += f"""
            <div class="metric-card">
                <h3>{name}</h3>
                <div class="value {css_class}">{display}</div>
            </div>
            """

        return html

    def _get_performance_summary(self) -> str:
        """Generate performance summary table."""
        rows = [
            ('Backtest Period', f"{self.signals.index[0].strftime('%Y-%m-%d')} to {self.signals.index[-1].strftime('%Y-%m-%d')}"),
            ('Total Trading Days', f"{self.results['total_days']:,}"),
            ('Benchmark Return', f"{self.results['benchmark_return']:.1%}"),
            ('Outperformance', f"{self.results['total_return'] - self.results['benchmark_return']:.1%}"),
            ('Walk-forward Training', 'Yes' if self.results.get('uses_walkforward') else 'No'),
            ('Transaction Costs', 'Applied' if self.results.get('transaction_costs_applied') else 'None'),
            ('Stop Loss Trades', f"{self.results.get('stop_loss_trades', 0):,}"),
            ('Rebalance Frequency', f"{self.results.get('rebalance_frequency_days', 21)} days"),
        ]

        html = """
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
        """

        for name, value in rows:
            html += f"""
                <tr>
                    <td>{name}</td>
                    <td>{value}</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        return html
