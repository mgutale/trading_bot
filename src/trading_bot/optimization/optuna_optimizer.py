"""
Optuna-based Hyperparameter Optimization

Optimizes trading strategy parameters using Tree-structured Parzen Estimator (TPE).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import time
import warnings

warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


@dataclass
class OptunaOptimizationResult:
    """Result of Optuna optimization."""
    best_params: Dict[str, Any]
    best_score: float
    sharpe_ratio: float
    sortino_ratio: float
    total_return: float
    max_drawdown: float
    n_trials: int
    complete_trials: int
    pruned_trials: int
    optimization_time: float
    study: Optional[Any] = None


class OptunaOptimizer:
    """
    Optuna-based optimizer for trading strategy parameters.

    Uses TPE (Tree-structured Parzen Estimator) for efficient
    hyperparameter search with automatic pruning.
    """

    def __init__(
        self,
        objective: str = "sharpe",
        pruner_type: str = "median",
        minimize_drawdown: bool = True,
        max_drawdown_penalty: float = 0.15,
        seed: int = 42
    ):
        self.objective = objective
        self.pruner_type = pruner_type
        self.minimize_drawdown = minimize_drawdown
        self.max_drawdown_penalty = max_drawdown_penalty
        self.seed = seed

        # Initialize sampler and pruner
        self.sampler = TPESampler(seed=seed)
        if pruner_type == "hyperband":
            self.pruner = optuna.pruners.HyperbandPruner()
        else:
            self.pruner = MedianPruner()

    def _create_momentum_search_space(self, trial: "optuna.Trial") -> Dict[str, Any]:
        """Define search space for momentum and risk parameters."""
        return {
            'momentum_short': trial.suggest_int('momentum_short', 21, 126),
            'momentum_long': trial.suggest_int('momentum_long', 63, 252),
            'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.02, 0.10, log=True),
            'rebalance_frequency': trial.suggest_int('rebalance_frequency', 5, 42),
        }

    def _calculate_objective(self, results: Dict) -> float:
        """Calculate objective value from backtest results."""
        sharpe = results.get('sharpe_ratio', 0)
        sortino = results.get('sortino_ratio', 0)
        max_dd = abs(results.get('max_drawdown', 0))
        total_return = results.get('total_return', 0)

        if self.objective == "sharpe":
            obj = sharpe
        elif self.objective == "sortino":
            obj = sortino
        elif self.objective == "calmar":
            obj = total_return / max_dd if max_dd > 0 else 0
        elif self.objective == "composite":
            obj = sharpe * 0.4 + sortino * 0.3 + total_return * 0.2 + (1 - max_dd) * 0.1
        else:
            obj = sharpe

        # Apply drawdown penalty
        if self.minimize_drawdown and max_dd > self.max_drawdown_penalty:
            penalty = (max_dd - self.max_drawdown_penalty) * 5
            obj -= penalty

        return obj

    def optimize_momentum_params(
        self,
        strategy_class: Callable,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        n_trials: int = 100,
        timeout: Optional[int] = None,
    ) -> OptunaOptimizationResult:
        """
        Optimize momentum and risk management parameters.

        Parameters optimized:
        - momentum_short: Short-term momentum period (21-126 days)
        - momentum_long: Long-term momentum period (63-252 days)
        - stop_loss_pct: Stop loss percentage (2-10%)
        - rebalance_frequency: Rebalance frequency (5-42 days)
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")

        start_time = time.time()

        def objective(trial: optuna.Trial) -> float:
            params = self._create_momentum_search_space(trial)

            try:
                strategy = strategy_class(
                    n_states=4,
                    momentum_short=params['momentum_short'],
                    momentum_long=params['momentum_long'],
                    top_n_stocks=5,  # Aligned with live trading
                    rebalance_frequency=params['rebalance_frequency'],
                    stop_loss_pct=params['stop_loss_pct'],
                    use_walkforward=True,
                    transaction_costs=True,
                )

                results = strategy.backtest(stock_data, benchmark_data)
                obj_value = self._calculate_objective(results)

                # Store results in trial user attrs
                trial.set_user_attr('sharpe', results.get('sharpe_ratio', 0))
                trial.set_user_attr('sortino', results.get('sortino_ratio', 0))
                trial.set_user_attr('total_return', results.get('total_return', 0))
                trial.set_user_attr('max_drawdown', results.get('max_drawdown', 0))

                return obj_value

            except Exception as e:
                print(f"Trial failed: {e}")
                return -np.inf

        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner,
        )

        # Run optimization
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        optimization_time = time.time() - start_time

        # Get best params
        best_params = study.best_params
        best_trial = study.best_trial

        # Run one more backtest with best params to get full metrics
        best_strategy = strategy_class(
            n_states=4,
            momentum_short=best_params['momentum_short'],
            momentum_long=best_params['momentum_long'],
            top_n_stocks=5,  # Aligned with live trading
            rebalance_frequency=best_params['rebalance_frequency'],
            stop_loss_pct=best_params['stop_loss_pct'],
            use_walkforward=True,
            transaction_costs=True,
        )
        best_results = best_strategy.backtest(stock_data, benchmark_data)

        return OptunaOptimizationResult(
            best_params=best_params,
            best_score=best_trial.value,
            sharpe_ratio=best_results.get('sharpe_ratio', 0),
            sortino_ratio=best_results.get('sortino_ratio', 0),
            total_return=best_results.get('total_return', 0),
            max_drawdown=best_results.get('max_drawdown', 0),
            n_trials=len(study.trials),
            complete_trials=len([t for t in study.trials if t.state.is_complete()]),
            pruned_trials=len([t for t in study.trials if t.state.is_pruned()]),
            optimization_time=optimization_time,
            study=study,
        )

    def analyze_parameter_importance(
        self,
        result: OptunaOptimizationResult
    ) -> Dict[str, Any]:
        """Analyze which parameters matter most."""
        if not OPTUNA_AVAILABLE or result.study is None:
            return {}

        try:
            from optuna.importance import get_param_importances
            importance = get_param_importances(result.study)

            return {
                'importance': importance,
                'recommendations': self._generate_recommendations(importance, result.best_params)
            }
        except Exception:
            return {}

    def _generate_recommendations(
        self,
        importance: Dict[str, float],
        best_params: Dict[str, Any]
    ) -> list:
        """Generate optimization recommendations."""
        recs = []

        # Find most important params
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        for param, imp in sorted_importance[:2]:
            if imp > 0.3:
                recs.append(f"{param} is CRITICAL ({imp:.1%} importance) - use optimized value: {best_params.get(param, 'N/A')}")

        return recs

    def walk_forward_validation(
        self,
        strategy_class: Callable,
        best_params: Dict[str, Any],
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """Validate parameters using walk-forward analysis."""
        dates = benchmark_data.index
        n_dates = len(dates)
        split_size = n_dates // n_splits

        is_scores = []
        oos_scores = []

        for i in range(n_splits - 1):
            # In-sample period
            is_start = i * split_size
            is_end = (i + 1) * split_size
            oos_start = is_end
            oos_end = min((i + 2) * split_size, n_dates)

            is_dates = dates[is_start:is_end]
            oos_dates = dates[oos_start:oos_end]

            # Fit on IS, evaluate on OOS
            is_benchmark = benchmark_data.loc[is_dates]
            oos_benchmark = benchmark_data.loc[oos_dates]

            try:
                strategy = strategy_class(
                    n_states=4,
                    momentum_short=best_params['momentum_short'],
                    momentum_long=best_params['momentum_long'],
                    top_n_stocks=5,  # Aligned with live trading
                    rebalance_frequency=best_params['rebalance_frequency'],
                    stop_loss_pct=best_params['stop_loss_pct'],
                    use_walkforward=True,
                    transaction_costs=True,
                )
                strategy.fit(is_benchmark)

                # OOS evaluation
                oos_signals = strategy.generate_signals(stock_data, oos_benchmark)
                oos_returns = oos_signals['strategy_return']
                oos_sharpe = oos_returns.mean() / oos_returns.std() * np.sqrt(252) if oos_returns.std() > 0 else 0
                oos_scores.append(oos_sharpe)

            except Exception:
                continue

        return {
            'oos_scores': oos_scores,
            'avg_oos_sharpe': np.mean(oos_scores) if oos_scores else 0,
            'oos_std': np.std(oos_scores) if oos_scores else 0,
        }


def create_optimization_report(
    result: OptunaOptimizationResult,
    importance_analysis: Dict[str, Any],
    walk_forward: Optional[Dict] = None,
    output_path: str = "results/optimization_report.html"
) -> str:
    """Generate HTML optimization report."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    html = f"""<!DOCTYPE html>
<html>
<head><title>Optimization Report</title></head>
<body>
<h1>Optuna Optimization Report</h1>

<h2>Best Parameters</h2>
<ul>
    <li>momentum_short: {result.best_params.get('momentum_short', 'N/A')}</li>
    <li>momentum_long: {result.best_params.get('momentum_long', 'N/A')}</li>
    <li>stop_loss_pct: {result.best_params.get('stop_loss_pct', 'N/A'):.2%}</li>
    <li>rebalance_frequency: {result.best_params.get('rebalance_frequency', 'N/A')} days</li>
</ul>

<h2>Performance Metrics</h2>
<ul>
    <li>Sharpe Ratio: {result.sharpe_ratio:.2f}</li>
    <li>Sortino Ratio: {result.sortino_ratio:.2f}</li>
    <li>Total Return: {result.total_return:.1%}</li>
    <li>Max Drawdown: {result.max_drawdown:.1%}</li>
</ul>

<h2>Optimization Stats</h2>
<ul>
    <li>Total Trials: {result.n_trials}</li>
    <li>Complete: {result.complete_trials}</li>
    <li>Pruned: {result.pruned_trials}</li>
    <li>Time: {result.optimization_time:.1f}s</li>
</ul>

<h2>Parameter Importance</h2>
<ul>
"""

    if importance_analysis and 'importance' in importance_analysis:
        for param, imp in importance_analysis['importance'].items():
            html += f"<li>{param}: {imp:.1%}</li>\n"

    if walk_forward:
        html += f"""
<h2>Walk-Forward Validation</h2>
<ul>
    <li>Avg OOS Sharpe: {walk_forward.get('avg_oos_sharpe', 0):.2f}</li>
    <li>OOS Std Dev: {walk_forward.get('oos_std', 0):.2f}</li>
</ul>
"""

    html += """
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path
