"""
Command-Line Interface for Trading Bot

Hybrid HMM + 5% Stop Loss Strategy
"""

import os
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from trading_bot.config import TradingBotConfig
from trading_bot.data.loader import DataLoader
from trading_bot.logging import setup_logging
from trading_bot.strategies.hybrid import HybridHMMStopLoss
from trading_bot.strategies.universes import BENCHMARK_SYMBOL, TECH_UNIVERSE

console = Console()


@click.group()
@click.option("--config", "-c", default=None, help="Path to config YAML file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx, config: Optional[str], verbose: bool):
    """Trading Bot - Hybrid HMM + 5% Stop Loss Strategy.

    \b
    A production-grade algorithmic trading system featuring:
    - Hidden Markov Model (HMM) for market regime detection
    - Momentum-based stock selection
    - Stop loss and take profit risk management
    - Equal weight position sizing

    \b
    Examples:
      trading-bot backtest --years 5
      trading-bot analyze --symbol SPY
      trading-bot optimize --trials 100
    """
    ctx.ensure_object(dict)
    level = "DEBUG" if verbose else "INFO"
    ctx.obj["logger"] = setup_logging(level=level)
    ctx.obj["config"] = TradingBotConfig.from_yaml(config) if config else TradingBotConfig()


@main.command()
@click.option("--years", "-y", default=5, type=float, help="Number of years to backtest")
@click.option("--capital", "-c", default=5000, help="Initial capital")
@click.option("--report", "-r", is_flag=True, help="Generate static HTML report in results/")
@click.pass_context
def backtest(ctx, years: int, capital: float, report: bool):
    """Run backtest for the Hybrid HMM + 5% Stop Loss strategy."""
    logger = ctx.obj["logger"]
    config = ctx.obj["config"]

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    strat_config = config.strategy.__class__.for_daily()
    console.print(f"\n[bold blue]Hybrid HMM + 5% Stop Loss - Backtest (Daily)[/bold blue]")
    console.print(f"Period: {start_date} to {end_date}")
    console.print(f"Initial Capital: ${capital:,.0f}\n")

    loader = DataLoader()
    logger.info("Loading Daily market data...")

    tech_data = loader.get_multiple_symbols(TECH_UNIVERSE, timeframe="1Day", start=start_date, end=end_date)
    benchmark_data = loader.get_historical_data(BENCHMARK_SYMBOL, timeframe="1Day", start=start_date, end=end_date)

    console.print(f"Loaded {len(tech_data)} stocks")
    console.print(f"Loaded benchmark ({BENCHMARK_SYMBOL}): {len(benchmark_data)} days")
    console.print()

    # Run strategy (params from StrategyConfig, single source of truth)
    strat = HybridHMMStopLoss.from_config(
        strat_config,
        use_walkforward=True,
        transaction_costs=True,
    )
    results = strat.backtest(tech_data, benchmark_data)

    # Display results
    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Return", f"{results['total_return']:.1%}")
    table.add_row("Annualized Return", f"{results['annualized_return']:.1%}")
    table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    table.add_row("Sortino Ratio", f"{results['sortino_ratio']:.2f}")
    table.add_row("Max Drawdown", f"{results['max_drawdown']:.1%}")
    table.add_row("Win Rate", f"{results['win_rate']:.1%}")
    table.add_row("Profit Factor", f"{results['profit_factor']:.2f}")
    table.add_row("Total Days", f"{results['total_days']:,}")

    # Show trade statistics
    table.add_row("Total Trades", f"{results['num_trades']}")
    table.add_row("Buy Trades", f"{results['buy_trades']}")
    table.add_row("Sell Trades", f"{results['sell_trades']}")
    table.add_row("Stop Loss Exits", f"{results['stop_loss_trades']}")
    table.add_row("Rebalance Frequency", f"{results['rebalance_frequency_days']} days")

    # Show transaction cost info
    table.add_row("Walk-forward Mode", "Yes" if results.get('uses_walkforward') else "No")
    table.add_row("Transaction Costs", "Applied" if results.get('transaction_costs_applied') else "None")
    if results.get('cost_drag_annual', 0) != 0:
        table.add_row("Annual Cost Drag", f"{results['cost_drag_annual']:.2%}")

    console.print(table)

    # Calculate portfolio value
    portfolio_value = capital * (1 + results['total_return'])
    console.print(f"\n[bold]Portfolio Value:[/bold] ${portfolio_value:,.2f}")
    console.print(f"[bold]Profit/Loss:[/bold] ${portfolio_value - capital:,.2f}")

    # Benchmark comparison
    console.print(f"\n[bold]Benchmark ({BENCHMARK_SYMBOL}) Return:[/bold] {results['benchmark_return']:.1%}")
    console.print(f"[bold]Outperformance:[/bold] {results['total_return'] - results['benchmark_return']:.1%}")

    if report:
        console.print("\n[bold blue]Generating HTML Report...[/bold blue]")
        try:
            from trading_bot.analytics.dashboard.report_generator import generate_backtest_report

            os.makedirs("results", exist_ok=True)
            report_path = generate_backtest_report(
                results=results,
                stock_data=tech_data,
                benchmark_data=benchmark_data,
                initial_capital=capital,
                output_path="results/backtest_report.html",
            )

            if report_path:
                console.print(f"[green]Report saved:[/green] {report_path}")
                console.print("[dim]Open in browser to view full analytics dashboard.[/dim]")
                webbrowser.open(f"file://{report_path}")
        except Exception as e:
            console.print(f"[red]Failed to generate report: {e}[/red]")


@main.command()
@click.option("--symbol", "-s", default="SPY", help="Symbol to analyze")
@click.pass_context
def analyze(ctx, symbol: str):
    """Analyze market regimes for a symbol."""
    logger = ctx.obj["logger"]

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    console.print(f"\n[bold blue]Regime Analysis: {symbol} (Daily)[/bold blue]\n")

    loader = DataLoader()
    data = loader.get_historical_data(symbol, timeframe="1Day", start=start_date, end=end_date)

    console.print(f"Loaded {len(data)} days of data")

    strat = HybridHMMStopLoss()
    strat.fit(data)
    regimes_numeric, state_means_by_period = strat.hmm_detector.predict_walkforward(data)
    labeled = strat._label_regimes_walkforward(regimes_numeric, data, state_means_by_period)

    table = Table(title="Regime Distribution")
    table.add_column("Regime", style="cyan")
    table.add_column("Days", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    for label in ["strong_bull", "weak_bull", "weak_bear", "strong_bear"]:
        count = (labeled == label).sum()
        pct = count / len(labeled) * 100
        table.add_row(label.upper().replace('_', ' '), f"{count:,}", f"{pct:.1f}%")

    console.print(table)

@main.command()
@click.option("--trials", "-n", default=100, help="Number of optimization trials")
@click.option("--years", "-y", default=2, help="Years of data for optimization")
@click.option("--objective", "-o", default="sharpe",
              type=click.Choice(["sharpe", "sortino", "calmar", "composite"]),
              help="Objective to maximize")
@click.option("--timeout", "-t", default=None, type=int, help="Max optimization time (seconds)")
@click.option("--pruner", default="median", type=click.Choice(["median", "hyperband"]),
              help="Pruning strategy")
@click.option("--report", "-r", is_flag=True, help="Generate HTML optimization report")
@click.option("--walk-forward", "-w", is_flag=True, help="Run walk-forward validation after optimization")
@click.pass_context
def optimize(ctx, trials: int, years: int, objective: str, timeout: Optional[int],
             pruner: str, report: bool, walk_forward: bool):
    """Optimize momentum and risk parameters using Optuna.

    \b
    Optimizes:
    - momentum_short (21-126 days)
    - momentum_long (63-252 days)
    - stop_loss_pct (2-10%)
    - rebalance_frequency (5-42 days)

    \b
    Examples:
      trading-bot optimize --trials 100 --years 2
      trading-bot optimize -n 50 -o sortino --report
      trading-bot optimize --trials 100 --walk-forward
    """
    logger = ctx.obj["logger"]

    # Check Optuna availability
    try:
        from trading_bot.optimization.optuna_optimizer import (
            OptunaOptimizer,
            OptunaOptimizationResult,
            create_optimization_report
        )
    except ImportError:
        console.print("[red]ERROR: Optuna is required for optimization.[/red]")
        console.print("Install with: pip install optuna")
        return

    console.print(f"\n[bold blue]Optuna Parameter Optimization[/bold blue]")
    console.print(f"Objective: Maximize {objective}")
    console.print(f"Trials: {trials}")
    console.print(f"Data: Last {years} year(s)")

    # Load data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    loader = DataLoader()
    console.print(f"\n[bold]Loading data ({start_date} to {end_date})...[/bold]")

    from trading_bot.strategies.universes import TECH_UNIVERSE, BENCHMARK_SYMBOL

    tech_data = loader.get_multiple_symbols(TECH_UNIVERSE, timeframe="1Day",
                                             start=start_date, end=end_date)
    benchmark_data = loader.get_historical_data(BENCHMARK_SYMBOL, timeframe="1Day",
                                                 start=start_date, end=end_date)

    console.print(f"Loaded {len(tech_data)} instruments")
    console.print(f"Loaded benchmark: {len(benchmark_data)} days")

    # Create optimizer
    optimizer = OptunaOptimizer(
        objective=objective,
        pruner_type=pruner,
        minimize_drawdown=True,
        max_drawdown_penalty=0.15
    )

    # Run optimization
    console.print(f"\n[bold]Starting optimization...[/bold]")
    result = optimizer.optimize_momentum_params(
        strategy_class=HybridHMMStopLoss,
        stock_data=tech_data,
        benchmark_data=benchmark_data,
        n_trials=trials,
        timeout=timeout,
    )

    # Display results
    console.print(f"\n[bold green]Optimization Complete![/bold green]")

    results_table = Table(title="Best Parameters")
    results_table.add_column("Parameter", style="cyan")
    results_table.add_column("Value", style="green")

    for param, value in result.best_params.items():
        if 'pct' in param:
            results_table.add_row(param, f"{value:.2%}")
        else:
            results_table.add_row(param, f"{value:,}")

    console.print(results_table)

    # Performance metrics
    perf_table = Table(title="Performance with Optimized Parameters")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="green")

    perf_table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    perf_table.add_row("Sortino Ratio", f"{result.sortino_ratio:.2f}")
    perf_table.add_row("Total Return", f"{result.total_return:.1%}")
    perf_table.add_row("Max Drawdown", f"{result.max_drawdown:.1%}")
    perf_table.add_row("Best Score", f"{result.best_score:.4f}")

    console.print(perf_table)

    # Optimization stats
    stats_table = Table(title="Optimization Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="yellow")

    stats_table.add_row("Total Trials", f"{result.n_trials}")
    stats_table.add_row("Completed", f"{result.complete_trials}")
    stats_table.add_row("Pruned", f"{result.pruned_trials}")
    stats_table.add_row("Optimization Time", f"{result.optimization_time:.1f}s")

    console.print(stats_table)

    # Walk-forward validation
    if walk_forward:
        console.print(f"\n[bold]Running walk-forward validation...[/bold]")
        wf_results = optimizer.walk_forward_validation(
            strategy_class=HybridHMMStopLoss,
            best_params=result.best_params,
            stock_data=tech_data,
            benchmark_data=benchmark_data,
            n_splits=5
        )

    # Parameter importance analysis
    console.print(f"\n[bold]Analyzing parameter importance...[/bold]")
    importance = optimizer.analyze_parameter_importance(result)

    if 'recommendations' in importance:
        importance_table = Table(title="Parameter Importance")
        importance_table.add_column("Parameter", style="cyan")
        importance_table.add_column("Importance", style="yellow")
        importance_table.add_column("Best Value", style="green")

        for param, imp in importance.get('importance', {}).items():
            best_val = result.best_params.get(param, 'N/A')
            if 'pct' in param:
                best_val_str = f"{best_val:.2%}" if isinstance(best_val, float) else best_val
            else:
                best_val_str = f"{best_val:,}" if isinstance(best_val, (int, float)) else best_val

            importance_table.add_row(
                param,
                f"{imp:.1%}",
                best_val_str
            )

        console.print(importance_table)

        # Print recommendations
        console.print(f"\n[bold]Recommendations:[/bold]")
        for rec in importance['recommendations']:
            console.print(f"  {rec}")

    if report:
        console.print(f"\n[bold blue]Generating HTML optimization report...[/bold blue]")
        try:
            os.makedirs("results", exist_ok=True)
            report_path = create_optimization_report(
                result=result,
                importance_analysis=importance,
                walk_forward=wf_results if walk_forward else None,
                output_path="results/optimization_report.html",
            )

            if report_path:
                console.print(f"[green]Report saved:[/green] {report_path}")
                console.print("[dim]Open in browser to view interactive charts.[/dim]")
                webbrowser.open(f"file://{report_path}")
        except Exception as e:
            console.print(f"[red]Failed to generate report: {e}[/red]")


if __name__ == "__main__":
    main()
