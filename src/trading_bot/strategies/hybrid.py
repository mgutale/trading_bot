"""
Hybrid HMM + Sector Momentum with Trailing Stop Loss

Strategy featuring:
- 4-state HMM regime detection (Strong Bull, Weak Bull, Weak Bear, Strong Bear)
- Walk-forward training to avoid look-ahead bias
- Momentum-based stock selection (top N of 15 tech stocks)
- Trailing stop loss (default 5.3%) for risk management
- Transaction cost modeling (spread + slippage + commission)
- Equal-weight position sizing

All parameters sourced from StrategyConfig (single source of truth).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from trading_bot.ml.markov_regime import MarkovRegimeDetector
from trading_bot.strategies.universes import TECH_UNIVERSE, SURVIVORSHIP_ADJUSTED_UNIVERSE
from trading_bot.config import StrategyConfig
from trading_bot.strategies.regime_exposure import REGIME_EXPOSURE
from trading_bot.strategies.base import BaseStrategy

# Transaction cost defaults - read from StrategyConfig single source of truth
_defaults = StrategyConfig()


class HybridHMMStopLoss(BaseStrategy):
    """
    Hybrid HMM + Momentum Strategy with Trailing Stop Loss.

    Components:
    1. HMM Regime Detection: 4-state Gaussian HMM classifies market regimes
       based on SPY returns. States labeled by mean return ranking.
    2. Momentum Selection: Ranks 15 tech stocks by momentum,
       holds top N with equal weighting.
    3. Trailing Stop Loss: stop that moves up only, protecting profits.
    4. Regime-Based Exposure: 100%/75%/25%/0% based on market regime.

    All default parameters come from StrategyConfig (the single source of truth).
    Inherits from BaseStrategy.
    """

    def __init__(
        self,
        n_states: int = _defaults.n_states,
        momentum_short: int = _defaults.momentum_short,
        momentum_long: int = _defaults.momentum_long,
        top_n_stocks: int = _defaults.top_n_stocks,
        rebalance_frequency: int = _defaults.rebalance_frequency,
        stop_loss_pct: float = _defaults.stop_loss_pct,
        regime_exposure: Optional[Dict[str, float]] = None,
        universe: Optional[List[str]] = None,
        universe_method: str = "static",
        use_walkforward: bool = True,
        transaction_costs: bool = True,
        spread_pct: float = _defaults.spread_pct,
        slippage_pct: float = _defaults.slippage_pct,
        commission_pct: float = _defaults.commission_pct,
    ):
        self.n_states = n_states
        self.momentum_short = momentum_short
        self.momentum_long = momentum_long
        self.top_n_stocks = top_n_stocks
        self.rebalance_frequency = rebalance_frequency
        self.stop_loss_pct = stop_loss_pct
        self.universe = universe or (
            SURVIVORSHIP_ADJUSTED_UNIVERSE if universe_method == "survivorship_adjusted"
            else TECH_UNIVERSE
        )
        self.use_walkforward = use_walkforward
        self.transaction_costs = transaction_costs
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct

        self.regime_exposure = regime_exposure or dict(REGIME_EXPOSURE)

        self.hmm_detector = MarkovRegimeDetector(
            n_states=n_states,
            min_training_days=63,
            retrain_frequency=21
        )
        self._stop_prices = {}
        self._entry_prices = {}
        self._trade_log = []
        self.fitted = False

    @classmethod
    def from_config(cls, config: StrategyConfig = None, **overrides) -> 'HybridHMMStopLoss':
        """Construct strategy from StrategyConfig (single source of truth).

        Args:
            config: StrategyConfig instance. Uses defaults if None.
            **overrides: Override any config parameter.
        """
        if config is None:
            config = StrategyConfig()
        params = {
            'n_states': config.n_states,
            'momentum_short': config.momentum_short,
            'momentum_long': config.momentum_long,
            'top_n_stocks': config.top_n_stocks,
            'rebalance_frequency': config.rebalance_frequency,
            'stop_loss_pct': config.stop_loss_pct,
            'regime_exposure': dict(config.regime_exposure),
            'spread_pct': config.spread_pct,
            'slippage_pct': config.slippage_pct,
            'commission_pct': config.commission_pct,
            'universe_method': config.universe_method,
        }
        params.update(overrides)
        return cls(**params)

    def fit(self, benchmark_data: pd.DataFrame) -> 'HybridHMMStopLoss':
        """Fit HMM to benchmark data.

        Only fits the HMM model. Regime labeling happens incrementally
        during backtest when using walk-forward mode.
        """
        self.hmm_detector.fit(benchmark_data)

        # Store state means from HMM model (learned from training data, no look-ahead)
        if hasattr(self.hmm_detector.model, 'means_'):
            state_means = self.hmm_detector.model.means_.flatten()
            for state in range(self.n_states):
                self.hmm_detector._state_means[state] = state_means[state] if state < len(state_means) else 0.0

        # Don't store pre-computed labels - they will be generated incrementally
        # during backtest when using walk-forward mode
        self.hmm_detector._state_labels = None

        self.fitted = True
        return self

    def _get_momentum_ranking(
        self,
        stock_data: Dict[str, pd.DataFrame],
        current_idx: int = None
    ) -> List[str]:
        """
        Rank stocks by momentum and return top N.

        Args:
            stock_data: Dictionary of symbol -> DataFrame with price data
            current_idx: Index position for current date in backtest.
                        If None, uses end of data (live trading where data ends at current date).

        Uses prices.iloc[current_idx] to avoid look-ahead bias.
        """
        momentum_scores = {}

        for symbol in self.universe:
            if symbol not in stock_data or len(stock_data[symbol]) < self.momentum_long:
                continue

            prices = stock_data[symbol]['close']

            # Use current_idx to avoid look-ahead bias
            if current_idx is not None and current_idx >= self.momentum_long:
                current_price = prices.iloc[current_idx]
                short_price = prices.iloc[current_idx - self.momentum_short]
                long_price = prices.iloc[current_idx - self.momentum_long]
                mom_short = (current_price / short_price) - 1
                mom_long = (current_price / long_price) - 1
            elif current_idx is None:
                # Live trading: last price is current (no look-ahead since data ends at now)
                mom_short = (prices.iloc[-1] / prices.iloc[-self.momentum_short]) - 1 if len(prices) > self.momentum_short else 0
                mom_long = (prices.iloc[-1] / prices.iloc[-self.momentum_long]) - 1 if len(prices) > self.momentum_long else 0
            else:
                # Not enough data at this point in backtest
                mom_short = 0
                mom_long = 0

            momentum_scores[symbol] = mom_short + mom_long

        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in ranked[:self.top_n_stocks]]

    def generate_signals(
        self,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate signals with stop loss logic."""
        if not self.fitted:
            self.fit(benchmark_data)

        # CRITICAL: Filter stock_data to match benchmark_data date range
        # This prevents look-ahead bias from stock data extending beyond benchmark
        bench_start = benchmark_data.index[0]
        bench_end = benchmark_data.index[-1]

        # Filter each stock to match benchmark date range
        stock_data_aligned = {}
        for symbol in stock_data:
            stock_df = stock_data[symbol]
            mask = (stock_df.index >= bench_start) & (stock_df.index <= bench_end)
            filtered_stock = stock_df.loc[mask]

            if len(filtered_stock) < len(benchmark_data):
                logger.warning(
                    f"{symbol}: Filtered from {len(stock_df)} to {len(filtered_stock)} rows "
                    f"to match benchmark ({len(benchmark_data)})"
                )

            stock_data_aligned[symbol] = filtered_stock

        # Use aligned stock data for backtesting
        stock_data = stock_data_aligned

        # Get HMM regimes - use walk-forward to avoid look-ahead bias
        if self.use_walkforward:
            # Capture both regimes and state_means_by_period for consistent labeling
            regimes_numeric, state_means_by_period = self.hmm_detector.predict_walkforward(benchmark_data)
            # Label regimes using period-specific state means to avoid look-ahead bias
            regimes_labeled = self._label_regimes_walkforward(regimes_numeric, benchmark_data, state_means_by_period)
        else:
            regimes_numeric = self.hmm_detector.predict(benchmark_data)
            regimes_labeled = self.hmm_detector._label_regimes(regimes_numeric, benchmark_data)

        # Get returns for all stocks
        returns_dict = {}
        price_data = {}
        low_data = {}   # Intraday low for stop loss trigger checks
        open_data = {}  # Open price for gap-down stop loss fills
        for symbol in self.universe:
            if symbol in stock_data:
                returns_dict[symbol] = stock_data[symbol]['close'].pct_change()
                price_data[symbol] = stock_data[symbol]['close']
                low_data[symbol] = stock_data[symbol]['low']
                open_data[symbol] = stock_data[symbol]['open']

        returns_df = pd.DataFrame(returns_dict)

        # Calculate strategy returns with stop loss
        strategy_returns = []
        active_positions = {}  # symbol -> entry_price
        position_weights = {}  # symbol -> weight
        self._trade_log = []  # Track all trades: (date, symbol, side, reason)
        self._turnover_log = []  # Track turnover for transaction costs: (date, turnover_pct)
        rebalance_exposure = 0.75  # Track exposure at last rebalance for scaling

        for i, idx in enumerate(regimes_labeled.index):
            # Get regime exposure
            regime = regimes_labeled.iloc[i]
            exposure = self.regime_exposure.get(regime, 0.5)

            # regimes_labeled.index starts at benchmark_data.index[1]
            # (after pct_change drops first row). So regime position i maps to
            # stock_data position i+1. This offset ensures momentum uses the
            # correct price at the current backtest date.
            stock_pos = i + 1

            # STEP 1: Calculate daily return for CURRENTLY held positions
            # BEFORE any rebalancing (no T+0 return attribution).
            # New positions added today start earning from tomorrow.
            # IMPORTANT: Returns are always calculated for held positions,
            # regardless of regime. You can't wish away losses by changing
            # exposure labels - if you hold stocks, you earn/lose their returns.
            daily_ret = 0.0
            stopped_out = []

            for symbol, entry_price in list(active_positions.items()):
                if symbol not in price_data or idx not in price_data[symbol].index:
                    continue

                current_price = price_data[symbol].loc[idx]
                current_low = low_data[symbol].loc[idx] if symbol in low_data and idx in low_data[symbol].index else current_price
                stop_price = self._stop_prices.get(symbol, entry_price * (1 - self.stop_loss_pct))

                # Check stop loss against intraday LOW price,
                # not just the close. A stop loss order triggers if the
                # low breaches the stop, even if the close recovers.
                if current_low <= stop_price:
                    # Stop loss triggered - exit position
                    # Gap-down penalty: if stock opens below stop price,
                    # the fill is at the open (worse than the stop level).
                    # Otherwise, fill at the stop price.
                    current_open = open_data[symbol].loc[idx] if symbol in open_data and idx in open_data[symbol].index else stop_price
                    if current_open < stop_price:
                        # Gapped down below stop - fill at open price (worse)
                        fill_price = current_open
                    else:
                        # Stop triggered during the day - fill at stop price
                        fill_price = stop_price
                    stopped_out.append(symbol)
                    self._trade_log.append((idx, symbol, 'sell', 'stop_loss'))
                    stopped_weight = position_weights.get(symbol, 0)
                    self._turnover_log.append((idx, stopped_weight))
                    # Realize loss using the actual fill price
                    actual_loss = (fill_price - entry_price) / entry_price
                    daily_ret += actual_loss * position_weights.get(symbol, 1.0/len(active_positions))
                else:
                    # Update trailing stop (only moves up)
                    new_stop = current_price * (1 - self.stop_loss_pct)
                    self._stop_prices[symbol] = max(stop_price, new_stop)

                    # Add daily return weighted by position
                    try:
                        ret = returns_df.loc[idx, symbol]
                        if not np.isnan(ret):
                            weight = position_weights.get(symbol, 1.0/len(active_positions)) if active_positions else 0
                            daily_ret += ret * weight
                    except (KeyError, IndexError):
                        pass  # Skip if price data unavailable for this symbol/date

            # Record the daily return (calculated BEFORE any rebalancing)
            # Scale by current exposure vs rebalance-day exposure.
            # If rebalance happened at 100% exposure but regime now says 25%,
            # we should have 25% of the position, not 100%. Scale accordingly.
            # If no positions are held, scaling doesn't matter (daily_ret is 0).
            if active_positions and rebalance_exposure > 0:
                exposure_scale = min(exposure / rebalance_exposure, 1.0)
            else:
                exposure_scale = 1.0  # No positions or no prior rebalance
            strategy_returns.append(daily_ret * exposure_scale)

            # Remove stopped out positions (after return calculation)
            for symbol in stopped_out:
                if symbol in active_positions:
                    del active_positions[symbol]
                if symbol in position_weights:
                    del position_weights[symbol]

            # STEP 2: Rebalance AFTER return calculation.
            # Positions changed here start earning returns from T+1.
            if i % self.rebalance_frequency == 0:
                # Track exposure at rebalance for scaling returns between rebalances
                rebalance_exposure = exposure
                # Use stock_pos (i+1) for correct index alignment
                new_holdings = self._get_momentum_ranking(stock_data, current_idx=stock_pos)

                # Calculate turnover BEFORE making changes (for transaction costs)
                old_holdings = set(active_positions.keys())
                old_weights = dict(position_weights)
                new_holdings_set = set(new_holdings)

                # Close existing positions if needed
                if exposure <= 0:
                    # Exit all positions - turnover = sum of all old weights
                    for symbol in list(active_positions.keys()):
                        self._trade_log.append((idx, symbol, 'sell', 'regime_exit'))
                    turnover = sum(old_weights.values())
                    self._turnover_log.append((idx, turnover))
                    active_positions.clear()
                    position_weights.clear()
                else:
                    # Equal weight per position
                    weight_per_stock = exposure / len(new_holdings) if new_holdings else 0
                    position_weights_target = {s: weight_per_stock for s in new_holdings}

                    # Remove stocks no longer in top N
                    for symbol in list(active_positions.keys()):
                        if symbol not in new_holdings:
                            self._trade_log.append((idx, symbol, 'sell', 'rebalance'))
                            del active_positions[symbol]
                            del position_weights[symbol]

                    # Add/update positions
                    for symbol in new_holdings:
                        if symbol not in active_positions and symbol in price_data:
                            active_positions[symbol] = price_data[symbol].loc[idx]
                            self._stop_prices[symbol] = price_data[symbol].loc[idx] * (1 - self.stop_loss_pct)
                            position_weights[symbol] = position_weights_target.get(symbol, 0.1)
                            self._trade_log.append((idx, symbol, 'buy', 'rebalance'))
                        elif symbol in active_positions:
                            # Update weight for existing position (rebalancing)
                            position_weights[symbol] = position_weights_target.get(symbol, 0)

                    # Calculate turnover: sum of absolute weight changes / 2
                    all_symbols = old_holdings | new_holdings_set
                    turnover = 0.0
                    for symbol in all_symbols:
                        old_w = old_weights.get(symbol, 0)
                        new_w = position_weights.get(symbol, 0)
                        turnover += abs(new_w - old_w)
                    turnover = turnover / 2  # Divide by 2 because each trade has two sides
                    self._turnover_log.append((idx, turnover))

        # Create signals DataFrame
        signals = pd.DataFrame(index=regimes_labeled.index)
        signals['regime'] = regimes_labeled
        signals['regime_numeric'] = regimes_numeric
        signals['strategy_return'] = strategy_returns
        signals['cumulative_return'] = (1 + signals['strategy_return']).cumprod() - 1

        # Benchmark
        benchmark_returns = benchmark_data['close'].pct_change()
        signals['benchmark_return'] = benchmark_returns.reindex(signals.index).fillna(0)
        signals['benchmark_cumulative'] = (1 + signals['benchmark_return']).cumprod() - 1

        return signals

    def _label_regimes_walkforward(
        self,
        regimes_numeric: pd.Series,
        benchmark_data: pd.DataFrame,
        state_means_by_period: Dict[int, Dict[int, float]] = None
    ) -> pd.Series:
        """
        Label walk-forward regimes using period-specific state means.

        Uses state_means_by_period from predict_walkforward() to label
        regimes consistently even as the HMM model is retrained multiple times.

        Each walk-forward iteration has its own state means, which are used to label
        regimes for that specific period. This ensures:
        1. No look-ahead bias (only uses training data up to that point)
        2. Consistent labeling across retraining iterations (states ranked by means)

        Labeling scheme (4-State HMM):
        - State with highest mean return -> 'strong_bull'
        - Second highest -> 'weak_bull'
        - Third highest -> 'weak_bear'
        - Lowest -> 'strong_bear'

        Args:
            regimes_numeric: Series of numeric regime predictions (0, 1, 2, 3)
            benchmark_data: DataFrame with benchmark price data
            state_means_by_period: Dict mapping train_end_index (int position) -> {state -> mean_return}
                                   Returned from predict_walkforward()

        Returns:
            Series of labeled regimes ('strong_bull', 'weak_bull', etc.)
        """
        # Create labels series with same index as regimes
        labels_series = pd.Series(index=regimes_numeric.index, dtype=object)

        # If we have state_means_by_period, use period-specific labeling
        if state_means_by_period and len(state_means_by_period) > 0:
            # Get the index as a list to convert between position and date
            regimes_index = regimes_numeric.index
            sorted_periods = sorted(state_means_by_period.keys())

            for i, train_end_pos in enumerate(sorted_periods):
                state_means = state_means_by_period[train_end_pos]

                # Determine prediction range for this model (in positions)
                predict_start_pos = train_end_pos
                predict_end_pos = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else len(regimes_numeric)

                # Convert positions to actual index labels (dates)
                if predict_start_pos >= len(regimes_index):
                    continue

                # Get the actual date/index at these positions
                predict_start_date = regimes_index[predict_start_pos]
                predict_end_date = regimes_index[min(predict_end_pos, len(regimes_index)) - 1]

                # Get regime predictions for this period using iloc
                period_regimes = regimes_numeric.iloc[predict_start_pos:predict_end_pos]

                if len(period_regimes) == 0:
                    continue

                # Rank states by their mean for this period's model
                sorted_states = sorted(state_means.items(), key=lambda x: x[1], reverse=True)

                # Assign labels based on ranking
                period_labels = {}
                if len(sorted_states) >= 4:
                    period_labels[sorted_states[0][0]] = 'strong_bull'
                    period_labels[sorted_states[1][0]] = 'weak_bull'
                    period_labels[sorted_states[2][0]] = 'weak_bear'
                    period_labels[sorted_states[3][0]] = 'strong_bear'
                elif len(sorted_states) == 3:
                    period_labels[sorted_states[0][0]] = 'bull'
                    period_labels[sorted_states[-1][0]] = 'bear'
                    period_labels[sorted_states[1][0]] = 'neutral'
                elif len(sorted_states) == 2:
                    period_labels[sorted_states[0][0]] = 'bull'
                    period_labels[sorted_states[1][0]] = 'bear'
                else:
                    period_labels[sorted_states[0][0]] = 'neutral' if sorted_states else 'weak_bull'

                # Map numeric regimes to labels for this period
                labels_series.iloc[predict_start_pos:predict_end_pos] = period_regimes.map(period_labels)

            # Fill any remaining NaN values
            return labels_series.fillna('weak_bull')

        # Fallback: Use stored labels from HMM detector (these are based on training data only)
        if self.hmm_detector._state_labels:
            return regimes_numeric.map(self.hmm_detector._state_labels).fillna('weak_bull')

        # Second fallback: Use HMM model's state means (learned from initial training)
        if hasattr(self.hmm_detector.model, 'means_') and self.hmm_detector.model is not None:
            state_means = self.hmm_detector.model.means_.flatten()
            sorted_states = sorted(enumerate(state_means), key=lambda x: x[1], reverse=True)

            labels = {}
            if len(sorted_states) >= 4:
                labels[sorted_states[0][0]] = 'strong_bull'
                labels[sorted_states[1][0]] = 'weak_bull'
                labels[sorted_states[2][0]] = 'weak_bear'
                labels[sorted_states[3][0]] = 'strong_bear'
            elif len(sorted_states) == 3:
                labels[sorted_states[0][0]] = 'bull'
                labels[sorted_states[-1][0]] = 'bear'
                labels[sorted_states[1][0]] = 'neutral'
            elif len(sorted_states) == 2:
                labels[sorted_states[0][0]] = 'bull'
                labels[sorted_states[1][0]] = 'bear'
            else:
                labels[sorted_states[0][0]] = 'neutral' if sorted_states else 'weak_bull'

            return regimes_numeric.map(labels).fillna('weak_bull')

        # Last resort: simple labeling by state number
        labels = {state: f'state_{state}' for state in range(self.n_states)}
        return regimes_numeric.map(labels).fillna('weak_bull')

    def backtest(
        self,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame
    ) -> Dict:
        """Run backtest and return performance metrics."""
        if not self.fitted:
            self.fit(benchmark_data)

        signals = self.generate_signals(stock_data, benchmark_data)
        returns = signals['strategy_return']

        # Apply transaction costs if enabled
        if self.transaction_costs:
            # Use actual turnover from trade log
            # Create turnover series indexed by date
            turnover_dict = dict(self._turnover_log)
            turnover_series = pd.Series(0.0, index=returns.index)
            for date, turnover in turnover_dict.items():
                if date in turnover_series.index:
                    turnover_series.loc[date] = turnover

            # Apply costs: turnover * (spread + slippage + commission)
            # Include commission_pct in total cost model
            total_cost_pct = self.spread_pct + self.slippage_pct + self.commission_pct
            daily_costs = turnover_series.abs() * total_cost_pct
            returns_net = returns - daily_costs
            signals['strategy_return_net'] = returns_net
            signals['cumulative_return_net'] = (1 + returns_net).cumprod() - 1
            signals['turnover_series'] = turnover_series  # Include for analysis

            # Use net returns for metrics
            returns_for_metrics = returns_net
            total_return = signals['cumulative_return_net'].iloc[-1] if len(signals) > 0 else 0
        else:
            returns_for_metrics = returns
            total_return = signals['cumulative_return'].iloc[-1] if len(signals) > 0 else 0

        # Basic stats
        annualized_return = returns_for_metrics.mean() * 252
        volatility = returns_for_metrics.std() * np.sqrt(252)
        sharpe = annualized_return / volatility if volatility > 0 else 0

        # Sortino
        downside_returns = returns_for_metrics[returns_for_metrics < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = annualized_return / downside_std if downside_std > 0 else 0

        # Drawdown
        cumulative = (1 + returns_for_metrics).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_dd = drawdowns.min()

        # Win rate
        win_rate = (returns_for_metrics > 0).mean()

        # Profit factor
        gross_profit = returns_for_metrics[returns_for_metrics > 0].sum()
        gross_loss = abs(returns_for_metrics[returns_for_metrics < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Benchmark comparison
        benchmark_total = (1 + signals['benchmark_return']).cumprod().iloc[-1] - 1

        # Transaction cost summary
        if self.transaction_costs:
            total_costs = (returns - returns_for_metrics).sum()
            cost_drag_annual = total_costs / len(returns) * 252
            # Additional turnover statistics
            total_turnover = sum(t[1] for t in self._turnover_log)
            avg_turnover = total_turnover / len(self._turnover_log) if self._turnover_log else 0
            max_turnover = max(t[1] for t in self._turnover_log) if self._turnover_log else 0
        else:
            total_costs = 0
            cost_drag_annual = 0
            total_turnover = 0
            avg_turnover = 0
            max_turnover = 0

        # Trade statistics
        num_trades = len(self._trade_log)
        buy_trades = len([t for t in self._trade_log if t[2] == 'buy'])
        sell_trades = len([t for t in self._trade_log if t[2] == 'sell'])
        stop_loss_trades = len([t for t in self._trade_log if t[3] == 'stop_loss'])

        return {
            'signals': signals,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_days': len(returns),
            'benchmark_return': benchmark_total,
            'transaction_costs_total': total_costs,
            'cost_drag_annual': cost_drag_annual,
            'total_turnover': total_turnover,
            'avg_turnover': avg_turnover,
            'max_turnover': max_turnover,
            'num_trades': num_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'stop_loss_trades': stop_loss_trades,
            'trade_log': self._trade_log,
            'turnover_log': self._turnover_log,
            'rebalance_frequency_days': self.rebalance_frequency,
            'trading_frequency': 'daily',
            'uses_walkforward': self.use_walkforward,
            'transaction_costs_applied': self.transaction_costs
        }