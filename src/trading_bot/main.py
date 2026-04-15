"""
Command-Line Interface for Trading Bot

Hybrid HMM + 5% Stop Loss Strategy
"""

import os
from datetime import datetime, timedelta
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from trading_bot.logging import setup_logging
from trading_bot.config import TradingBotConfig
from trading_bot.data.loader import DataLoader
from trading_bot.strategies.hybrid import HybridHMMStopLoss

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
    - Sector Momentum Rotation for stock selection
    - 5% Stop Loss for risk management per position
    - Equal weight position sizing (10% per stock)

    \b
    Examples:
      trading-bot backtest --years 5
      trading-bot live --dry-run
    """
    ctx.ensure_object(dict)
    level = "DEBUG" if verbose else "INFO"
    ctx.obj["logger"] = setup_logging(level=level)
    ctx.obj["config"] = TradingBotConfig.from_yaml(config) if config else TradingBotConfig()


@main.command()
@click.option("--years", "-y", default=5, type=float, help="Number of years to backtest")
@click.option("--capital", "-c", default=5000, help="Initial capital")
@click.option("--report", "-r", is_flag=True, help="Generate HTML report with charts")
@click.pass_context
def backtest(ctx, years: int, capital: float, report: bool):
    """Run backtest for the Hybrid HMM + 5% Stop Loss strategy."""
    logger = ctx.obj["logger"]
    config = ctx.obj["config"]

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365 * years)).strftime("%Y-%m-%d")

    # Get strategy config for daily trading
    strat_config = config.strategy.__class__.for_daily()
    console.print(f"\n[bold blue]Hybrid HMM + 5% Stop Loss - Backtest (Daily)[/bold blue]")

    console.print(f"Period: {start_date} to {end_date}")
    console.print(f"Initial Capital: ${capital:,.0f}\n")

    # Load data
    loader = DataLoader()
    logger.info("Loading Daily market data...")

    from trading_bot.strategies.universes import TECH_UNIVERSE, BENCHMARK_SYMBOL

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

    # Generate HTML report
    if report:
        console.print("\n[bold blue]Generating HTML Report with Charts...[/bold blue]")
        try:
            from trading_bot.analytics.visualizer import BacktestVisualizer
            import os
            os.makedirs("results", exist_ok=True)

            visualizer = BacktestVisualizer(results, strategy_name="Hybrid HMM + 5% Stop Loss")
            report_path = visualizer.create_full_report(save_path="results/backtest_report.html")
            console.print(f"[green]HTML Report saved:[/green] {report_path}")
            console.print("[dim]Open this file in your browser to view interactive charts.[/dim]")
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

    # Use the HMM detector from the strategy (params from StrategyConfig)
    strat = HybridHMMStopLoss()
    strat.fit(data)
    regimes_numeric, state_means_by_period = strat.hmm_detector.predict_walkforward(data)
    labeled = strat._label_regimes_walkforward(regimes_numeric, data, state_means_by_period)

    # Display regime distribution
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
@click.option("--dry-run", is_flag=True, help="Simulate trades without executing")
@click.option("--lookback", "-l", default=126, help="Period to fetch (days)")
@click.option("--daily", "-d", is_flag=True, help="Run daily at market open")
@click.option("--time", "-t", default="09:30", help="Daily run time (HH:MM format, default 09:30)")
@click.option("--telegram", is_flag=True, help="Send daily reports via Telegram")
@click.pass_context
def live(ctx, dry_run: bool, lookback: int, daily: bool, time: str, telegram: bool):
    """Run live trading with Interactive Brokers.

    \b
    Requirements:
      - TWS or IB Gateway running
      - IBKR_HOST environment variable (default: 127.0.0.1)
      - IBKR_PORT environment variable (default: 7497 for paper, 7496 for live)
      - IBKR_CLIENT_ID environment variable (default: 1)

    The algorithm fetches historical data upfront to initialize
    the HMM model, then trades based on current regime.

    \b
    Options:
      --daily     Run automatically every trading day
      --time      Set custom run time (default: 09:30)
      --telegram  Send daily reports via Telegram
    """
    logger = ctx.obj["logger"]
    config = ctx.obj["config"]

    min_required = 63  # Minimum 3 months for HMM training

    # Ensure minimum historical data
    if lookback < min_required:
        console.print(f"[red]ERROR: Minimum {min_required} days required for HMM training.[/red]")
        console.print(f"Use --lookback {min_required} or higher.")
        return

    # Daily mode - schedule recurring runs
    if daily:
        console.print(f"\n[bold blue]Daily Trading Mode: Hybrid HMM + 5% Stop Loss[/bold blue]")
        console.print(f"Scheduled to run at {time} ET every trading day")
        if telegram:
            console.print("[bold green]Telegram notifications enabled[/bold green]")
        if dry_run:
            console.print("[yellow]DRY RUN MODE - No real trades[/yellow]\n")
        else:
            console.print("[red]LIVE MODE - Real trades will be executed[/red]\n")

        # Run once immediately, then schedule
        console.print("[bold]Running initial execution...[/bold]\n")
        _run_trading_cycle(ctx, dry_run, lookback, logger, telegram=telegram)

        # Schedule daily runs with Telegram reports
        console.print(f"\n[bold green]Scheduled daily runs at {time} ET[/bold green]")
        console.print("[dim]Bot will run automatically every trading day at market open.[/dim]")
        if telegram:
            console.print("[dim]Daily reports will be sent via Telegram.[/dim]")
        console.print("[dim]Dashboard available at: logs/dashboard.html[/dim]")
        console.print("[dim]Press Ctrl+C to stop the scheduler.[/dim]\n")

        import schedule
        import time as time_module

        def scheduled_job():
            console.print(f"\n[bold blue]=== Scheduled Run: {datetime.now().strftime('%Y-%m-%d %H:%M')} ===[/bold blue]")
            _run_trading_cycle(ctx, dry_run, lookback, logger, telegram=telegram)

        # Ensure time format is HH:MM:SS for schedule library
        time_formatted = time if len(time.split(':')) == 3 else f"{time}:00"
        schedule.every().day.at(time_formatted).do(scheduled_job)

        try:
            while True:
                schedule.run_pending()
                time_module.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            console.print("\n[yellow]Scheduler stopped by user.[/yellow]")
            return

        return

    # Single run mode (existing behavior)
    console.print(f"\n[bold blue]Live Trading: Hybrid HMM + 5% Stop Loss (Daily)[/bold blue]")
    console.print(f"Historical data: {lookback} days for HMM initialization")
    if dry_run:
        console.print("[yellow]DRY RUN MODE - No real trades[/yellow]\n")
    else:
        console.print("[red]LIVE MODE - Real trades will be executed[/red]\n")

    # Import IBKR client
    try:
        from trading_bot.core.ibkr_client import IBKRClient
        client = IBKRClient()
        account = client.get_account()
        console.print(f"Account: {account.get('account_number', 'N/A')}")
        console.print(f"Equity: ${float(account.get('equity', 0)):,.2f}")
        console.print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}\n")
    except ConnectionError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Ensure TWS or IB Gateway is running and IBKR_HOST/IBKR_PORT are set in .env")
        return
    except Exception as e:
        console.print(f"[red]Failed to connect to IBKR: {e}[/red]")
        return

    # Fetch historical data for HMM initialization
    from trading_bot.data.loader import DataLoader, TECH_UNIVERSE, BENCHMARK_SYMBOL

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")

    console.print(f"\n[bold]Fetching historical data ({start_date} to {end_date})...[/bold]")
    console.print(f"[dim]This may take a moment while downloading {lookback} days of market data.[/dim]")
    loader = DataLoader()

    # Load benchmark for regime detection
    benchmark_data = loader.get_historical_data(BENCHMARK_SYMBOL, timeframe="1Day", start=start_date, end=end_date)
    console.print(f"Loaded {BENCHMARK_SYMBOL}: {len(benchmark_data)} days")

    # Load instruments for momentum ranking
    tech_data = loader.get_multiple_symbols(TECH_UNIVERSE, timeframe="1Day", start=start_date, end=end_date)
    console.print(f"Loaded {len(tech_data)} instruments")

    # Verify we have enough data
    if len(benchmark_data) < min_required:
        console.print(f"[red]ERROR: Only {len(benchmark_data)} days loaded. Need at least {min_required}.[/red]")
        return

    # Get strategy config for daily trading
    strat_config = config.strategy.__class__.for_daily()

    # Initialize strategy using from_config (single source of truth)
    strat = HybridHMMStopLoss.from_config(strat_config, use_walkforward=True, transaction_costs=False)

    # Pre-fit the HMM model on historical data
    console.print("\n[bold]Fitting HMM model...[/bold]")
    strat.fit(benchmark_data)
    console.print("[green]HMM model fitted and ready to trade![/green]")

    # Get current regime using walk-forward prediction
    regimes_numeric, state_means_by_period = strat.hmm_detector.predict_walkforward(benchmark_data)
    labeled = strat._label_regimes_walkforward(regimes_numeric, benchmark_data, state_means_by_period)
    current_regime = labeled.iloc[-1]
    current_regime_numeric = regimes_numeric.iloc[-1]

    console.print(f"\n[bold]Current Market Regime:[/bold] {current_regime.upper()} (State {current_regime_numeric})")

    # Get momentum rankings (no current_idx needed - live data ends at current date)
    top_stocks = strat._get_momentum_ranking(tech_data)
    console.print(f"[bold]Top Momentum Stocks:[/bold] {', '.join(top_stocks[:5])}")

    # Determine exposure
    exposure = strat.regime_exposure.get(current_regime, 0.5)
    console.print(f"\n[bold]Target Exposure:[/bold] {exposure:.0%}")

    if exposure <= 0:
        console.print("[yellow]Market in BEAR regime - holding cash[/yellow]")
        console.print("\n[dim]No trades will be placed until regime changes to Bull or Neutral.[/dim]")
    else:
        console.print(f"[green]Ready to trade {len(top_stocks)} instruments with {exposure:.0%} exposure[/green]")
        console.print("\n[bold]Recommended Actions:[/bold]")
        for i, stock in enumerate(top_stocks[:5], 1):
            console.print(f"  {i}. Buy {stock} (equal weight)")

        if not dry_run:
            console.print("\n[yellow]To execute trades, implement the order logic below:[/yellow]")
            console.print("[dim]# Example: Place orders for top stocks[/dim]")
            console.print("[dim]for symbol in top_stocks[:10]:[/dim]")
            console.print("[dim]    qty = calculate_position_size(symbol)[/dim]")
            console.print("[dim]    client.submit_order(symbol, qty, 'buy')[/dim]")

    # Log performance snapshot (live trading only)
    if not dry_run:
        pass


def _run_trading_cycle(ctx, dry_run: bool, lookback: int, logger, telegram: bool = False):
    """Execute a single trading cycle (called by daily scheduler)."""
    config = ctx.obj["config"]
    min_required = 63

    # Import IBKR client
    try:
        from trading_bot.core.ibkr_client import IBKRClient
        client = IBKRClient()
        account = client.get_account()
        console.print(f"Account: {account.get('account_number', 'N/A')}")
        console.print(f"Equity: ${float(account.get('equity', 0)):,.2f}")
        console.print(f"Buying Power: ${float(account.get('buying_power', 0)):,.2f}\n")
    except ConnectionError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("Ensure TWS or IB Gateway is running and IBKR_HOST/IBKR_PORT are set in .env")
        return
    except Exception as e:
        console.print(f"[red]Failed to connect to IBKR: {e}[/red]")
        return

    # Fetch historical data
    from trading_bot.data.loader import DataLoader, TECH_UNIVERSE, BENCHMARK_SYMBOL

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")

    console.print(f"\n[bold]Fetching historical data ({start_date} to {end_date})...[/bold]")
    loader = DataLoader()

    benchmark_data = loader.get_historical_data(BENCHMARK_SYMBOL, timeframe="1Day", start=start_date, end=end_date)
    console.print(f"Loaded {BENCHMARK_SYMBOL}: {len(benchmark_data)} days")

    tech_data = loader.get_multiple_symbols(TECH_UNIVERSE, timeframe="1Day", start=start_date, end=end_date)
    console.print(f"Loaded {len(tech_data)} instruments")

    if len(benchmark_data) < min_required:
        console.print(f"[red]ERROR: Only {len(benchmark_data)} days loaded. Need at least {min_required}.[/red]")
        return

    # Get strategy config and initialize using from_config (single source of truth)
    strat_config = config.strategy.__class__.for_daily()
    strat = HybridHMMStopLoss.from_config(strat_config, use_walkforward=True, transaction_costs=False)

    # Fit HMM model
    console.print("\n[bold]Fitting HMM model...[/bold]")
    strat.fit(benchmark_data)
    console.print("[green]HMM model fitted![/green]")

    # Get current regime using walk-forward prediction
    regimes_numeric, state_means_by_period = strat.hmm_detector.predict_walkforward(benchmark_data)
    labeled = strat._label_regimes_walkforward(regimes_numeric, benchmark_data, state_means_by_period)
    current_regime = labeled.iloc[-1]
    current_regime_numeric = regimes_numeric.iloc[-1]

    console.print(f"\n[bold]Current Market Regime:[/bold] {current_regime.upper()} (State {current_regime_numeric})")

    # Get momentum rankings
    top_stocks = strat._get_momentum_ranking(tech_data)
    console.print(f"[bold]Top Momentum Stocks:[/bold] {', '.join(top_stocks[:5])}")

    # Determine exposure
    exposure = strat.regime_exposure.get(current_regime, 0.5)
    console.print(f"\n[bold]Target Exposure:[/bold] {exposure:.0%}")

    if exposure <= 0:
        console.print("[yellow]Market in BEAR regime - holding cash[/yellow]")
        console.print("[dim]No trades will be placed until regime changes to Bull or Neutral.[/dim]")
    else:
        console.print(f"[green]Ready to trade {len(top_stocks)} instruments with {exposure:.0%} exposure[/green]")
        console.print("\n[bold]Recommended Actions:[/bold]")
        for i, stock in enumerate(top_stocks[:5], 1):
            console.print(f"  {i}. Buy {stock} (equal weight)")

        # Execute trades in LIVE mode (not dry-run)
        if not dry_run:
            console.print("\n[bold red]Executing LIVE trades...[/bold red]")

            # Initialize Telegram notifier before trading
            notifier = None
            if telegram:
                from trading_bot.telegram_notifier import TelegramNotifier
                notifier = TelegramNotifier()

            # Get current prices and calculate positions
            equity = float(account.get('equity', 0))
            target_value = (equity * exposure) / 5  # Equal weight across 5 stocks

            trades_executed = 0
            for stock in top_stocks[:5]:
                try:
                    # Get current price
                    price_data = tech_data[stock].iloc[-1]
                    price = float(price_data['close'])

                    # Calculate quantity (round down to whole shares)
                    qty = int(target_value / price)

                    if qty > 0:
                        # Submit order
                        order = client.submit_order(
                            symbol=stock,
                            qty=qty,
                            side='buy',
                            order_type='market',
                            time_in_force='day'
                        )

                        trade_value = qty * price
                        console.print(f"  [green]✓ BUY {qty} {stock} @ ${price:.2f} = ${trade_value:,.2f}[/green]")

                        # Send Telegram notification for this trade
                        if telegram:
                            notifier.send_trade_notification(
                                action="BUY",
                                symbol=stock,
                                qty=qty,
                                price=price,
                                value=trade_value
                            )

                        trades_executed += 1
                    else:
                        console.print(f"  [yellow]⚠ Skipping {stock} - insufficient funds for 1 share (${price:.2f})[/yellow]")

                except Exception as e:
                    console.print(f"  [red]✗ Failed to buy {stock}: {e}[/red]")
                    if telegram:
                        notifier.send_error_notification(f"Failed to buy {stock}: {e}")

            console.print(f"\n[bold green]Executed {trades_executed} trades[/bold green]")
        else:
            trades_executed = 0

    # Send Telegram notification
    if telegram:
        try:
            # Generate HTML report path for link
            report_url = os.environ.get("REPORT_URL", "http://localhost:8080/results")

            # Send daily report
            notifier.send_daily_report(
                account_data=account,
                regime=current_regime,
                top_stocks=top_stocks,
                trades_executed=trades_executed if not dry_run else 0
            )

            # Send follow-up with report link
            caption = f"📊 Full Report: {report_url}"
            notifier.send_message(caption, parse_mode="")

            console.print("[green]Telegram notifications sent[/green]")
        except Exception as e:
            console.print(f"[yellow]Telegram notification failed: {e}[/yellow]")


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

    # Generate HTML report
    if report:
        console.print(f"\n[bold blue]Generating HTML optimization report...[/bold blue]")
        try:
            import os
            os.makedirs("results", exist_ok=True)

            report_path = create_optimization_report(
                result=result,
                importance_analysis=importance,
                walk_forward=wf_results if walk_forward else None,
                output_path="results/optimization_report.html"
            )

            if report_path:
                console.print(f"[green]Report saved:[/green] {report_path}")
                console.print("[dim]Open in browser to view interactive charts.[/dim]")

                # Auto-open in browser
                import webbrowser
                webbrowser.open(f"file://{report_path}")
        except Exception as e:
            console.print(f"[red]Failed to generate report: {e}[/red]")


if __name__ == "__main__":
    main()
