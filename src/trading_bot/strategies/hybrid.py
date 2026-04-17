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
    5. Position Sizing: Equal weight allocation capped by position_size_pct.

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
        position_size_pct: float = _defaults.position_size_pct,
        regime_exposure: Optional[Dict[str, float]] = None,
        universe: Optional[List[str]] = None,
        universe_method: str = "static",
        universe_list: Optional[List[str]] = None,
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
        self.position_size_pct = position_size_pct  # Max allocation per position
        # Use provided universe, or resolve from universe_method, or use config's universe_list
        if universe:
            self.universe = universe
        elif universe_method == "survivorship_adjusted":
            self.universe = SURVIVORSHIP_ADJUSTED_UNIVERSE
        elif universe_method == "tech":
            self.universe = TECH_UNIVERSE
        else:
            # Default: use universe_list from config (set in StrategyConfig)
            # This allows full customization without modifying universes.py
            self.universe = universe_list or _defaults.universe_list
        self.use_walkforward = use_walkforward
        self.transaction_costs = transaction_costs
        self.spread_pct = spread_pct
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct

        self.regime_exposure = regime_exposure or dict(REGIME_EXPOSURE)

        self.hmm_detector = MarkovRegimeDetector(
            n_states=n_states,
            min_training_days=63,
            retrain_frequency=rebalance_frequency,  # Use config rebalance frequency
            use_volatility=True  # Enable multivariate HMM (returns + volatility)
        )
        self._stop_prices = {}
        self._entry_prices = {}
        self._entry_dates = {}  # Track entry dates for enriched trades
        self._entry_regimes = {}  # Track regime at entry for enriched trades
        self._entry_qtys = {}  # Track quantity at entry for enriched trades
        self._enriched_trades = []  # Enriched trade records for analytics dashboard
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
            'position_size_pct': config.position_size_pct,
            'regime_exposure': dict(config.regime_exposure),
            'spread_pct': config.spread_pct,
            'slippage_pct': config.slippage_pct,
            'commission_pct': config.commission_pct,
            'universe_method': config.universe_method,
            'universe_list': config.universe_list,
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

    def _get_momentum_scores(
        self,
        stock_data: Dict[str, pd.DataFrame],
        current_idx: int = None
    ) -> Dict[str, float]:
        """
        Calculate momentum scores for all stocks in universe.

        Args:
            stock_data: Dictionary of symbol -> DataFrame with price data
            current_idx: Index position for current date in backtest.

        Returns:
            Dictionary mapping symbol -> momentum score

        Momentum = short-term momentum + long-term momentum
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

        return momentum_scores

    def _get_momentum_ranking(
        self,
        stock_data: Dict[str, pd.DataFrame],
        current_idx: int = None
    ) -> List[str]:
        """
        Rank stocks by momentum and return top N symbols.

        Args:
            stock_data: Dictionary of symbol -> DataFrame with price data
            current_idx: Index position for current date in backtest.

        Returns:
            List of top N stock symbols by momentum score.
        """
        momentum_scores = self._get_momentum_scores(stock_data, current_idx)
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in ranked[:self.top_n_stocks]]

    def _calculate_momentum_weights(
        self,
        momentum_scores: Dict[str, float],
        exposure: float,
        top_n: int = None
    ) -> Dict[str, float]:
        """
        Calculate momentum-weighted position allocations.

        Higher momentum stocks get proportionally larger weights.
        Weights are scaled to sum to the target exposure level.

        Args:
            momentum_scores: Dictionary mapping symbol -> momentum score
            exposure: Target total exposure (0.0 to 1.0)
            top_n: Number of top stocks to select (uses self.top_n_stocks if None)

        Returns:
            Dictionary mapping symbol -> position weight

        Example:
            If NVDA has momentum score 0.15 and INTC has 0.05,
            NVDA gets 3x the weight of INTC.
        """
        top_n = top_n or self.top_n_stocks

        # Select top N stocks by momentum (may add more if needed to fill exposure)
        ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)

        if not ranked:
            return {}

        # Filter to only positive momentum stocks (avoid allocating to declining stocks)
        positive_momentum = [(sym, score) for sym, score in ranked if score > 0]

        if not positive_momentum:
            # If all momentum scores are negative, fall back to equal weight
            # This ensures we still participate when everything is weak
            positive_momentum = ranked[:top_n]

        if not positive_momentum:
            # If all momentum scores are negative, fall back to equal weight
            # This ensures we still participate when everything is weak
            positive_momentum = ranked

        # Calculate total momentum score
        total_momentum = sum(score for _, score in positive_momentum)

        if total_momentum <= 0:
            # Fallback to equal weight if total momentum is zero or negative
            weight_per_stock = exposure / len(positive_momentum)
            return {sym: weight_per_stock for sym, _ in positive_momentum}

        # Allocate weights proportional to momentum score
        # Each stock's weight = (its momentum / total momentum) * exposure
        # Capped at position_size_pct to avoid over-concentration
        weights = {}
        capped_symbols = set()

        # First pass: calculate raw weights and identify capped positions
        for symbol, score in positive_momentum:
            raw_weight = (score / total_momentum) * exposure
            if raw_weight >= self.position_size_pct:
                weights[symbol] = self.position_size_pct
                capped_symbols.add(symbol)

        # Check if we have remaining exposure to allocate
        current_total = sum(weights.values())
        remaining_exposure = exposure - current_total

        # Second pass: redistribute remaining exposure to uncapped symbols
        if remaining_exposure > 0.001:
            uncapped = [(sym, score) for sym, score in positive_momentum if sym not in capped_symbols]
            if uncapped:
                uncapped_total = sum(score for _, score in uncapped)
                for symbol, score in uncapped:
                    additional = (score / uncapped_total) * remaining_exposure
                    weights[symbol] = min(weights.get(symbol, 0) + additional, self.position_size_pct)

        # Final normalization: if still below target, scale up (some may be capped)
        current_total = sum(weights.values())
        if current_total > 0 and current_total < exposure:
            # Proportionally increase uncapped positions
            scale_factor = exposure / current_total
            final_weights = {}
            for sym, w in weights.items():
                scaled = w * scale_factor
                final_weights[sym] = min(scaled, self.position_size_pct)
            weights = final_weights

        # If still below target exposure, add more stocks from the ranked list
        current_total = sum(weights.values())
        if current_total < exposure - 0.01 and len(weights) < len(ranked):
            # Add additional stocks until we reach target or run out
            # Include stocks with any momentum score (even negative) to fill exposure
            for sym, score in ranked:
                if sym not in weights:
                    remaining = exposure - sum(weights.values())
                    weights[sym] = min(remaining, self.position_size_pct)
                    if sum(weights.values()) >= exposure - 0.01:
                        break

        return weights

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
                    # Create enriched trade record for stop loss
                    self._create_enriched_trade(
                        symbol=symbol,
                        exit_date=idx,
                        exit_price=fill_price,
                        reason='stop_loss',
                        regime_at_exit=regime,
                        price_data=price_data
                    )
                    # FIX: Only book TODAY'S return (yesterday close -> fill price)
                    # Daily returns for prior days were already booked via line 452-455.
                    # Booking the full entry-to-exit return would double-count.
                    prev_close = price_data[symbol].loc[
                        price_data[symbol].index[price_data[symbol].index < idx][-1]
                    ] if len(price_data[symbol].index[price_data[symbol].index < idx]) > 0 else entry_price
                    today_return = (fill_price - prev_close) / prev_close
                    daily_ret += today_return * position_weights.get(symbol, 1.0/len(active_positions))
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

            # Record the daily return (calculated BEFORE any rebalancing).
            # IMPORTANT: Returns are NOT scaled by exposure changes.
            # If you hold 100% invested and regime changes to 25%, you still
            # earned the full return on that day - you just sell 75% during
            # the rebalance. Exposure scaling affects FUTURE returns via
            # position_weights, not past returns.
            strategy_returns.append(daily_ret)

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
                # Get momentum scores for all stocks
                momentum_scores = self._get_momentum_scores(stock_data, current_idx=stock_pos)

                # Calculate momentum-weighted allocations
                position_weights_target = self._calculate_momentum_weights(
                    momentum_scores=momentum_scores,
                    exposure=exposure,
                    top_n=self.top_n_stocks
                )
                new_holdings = list(position_weights_target.keys())

                # Calculate turnover BEFORE making changes (for transaction costs)
                old_holdings = set(active_positions.keys())
                old_weights = dict(position_weights)
                new_holdings_set = set(new_holdings)

                # Close existing positions if needed
                if exposure <= 0:
                    # Exit all positions - turnover = sum of all old weights
                    for symbol in list(active_positions.keys()):
                        self._trade_log.append((idx, symbol, 'sell', 'regime_exit'))
                        # Create enriched trade record for regime exit
                        self._create_enriched_trade(
                            symbol=symbol,
                            exit_date=idx,
                            exit_price=price_data[symbol].loc[idx] if symbol in price_data else None,
                            reason='regime_exit',
                            regime_at_exit=regime,
                            price_data=price_data
                        )
                    turnover = sum(old_weights.values())
                    self._turnover_log.append((idx, turnover))
                    active_positions.clear()
                    position_weights.clear()
                else:
                    # Determine which positions to keep vs exit
                    positions_to_exit = old_holdings - new_holdings_set
                    positions_to_enter = new_holdings_set - old_holdings
                    positions_to_adjust = old_holdings & new_holdings_set

                    # Exit positions no longer in top N
                    for symbol in positions_to_exit:
                        if symbol in active_positions:
                            self._trade_log.append((idx, symbol, 'sell', 'rebalance'))
                            self._create_enriched_trade(
                                symbol=symbol,
                                exit_date=idx,
                                exit_price=price_data[symbol].loc[idx] if symbol in price_data else None,
                                reason='rebalance',
                                regime_at_exit=regime,
                                price_data=price_data
                            )
                            del active_positions[symbol]
                            del position_weights[symbol]

                    # Enter new positions
                    for symbol in positions_to_enter:
                        if symbol in price_data:
                            entry_price = price_data[symbol].loc[idx]
                            active_positions[symbol] = entry_price
                            self._stop_prices[symbol] = entry_price * (1 - self.stop_loss_pct)
                            position_weights[symbol] = position_weights_target.get(symbol, 0.0)
                            self._trade_log.append((idx, symbol, 'buy', 'rebalance'))
                            self._entry_prices[symbol] = entry_price
                            self._entry_dates[symbol] = idx
                            self._entry_regimes[symbol] = regime

                    # Adjust weights for existing positions (only if weight changed significantly)
                    for symbol in positions_to_adjust:
                        old_w = old_weights.get(symbol, 0)
                        new_w = position_weights_target.get(symbol, 0)
                        if abs(new_w - old_w) > 0.001:  # Only trade if change > 0.1%
                            position_weights[symbol] = new_w
                            # Log weight adjustment (not a full trade)
                            if new_w > old_w:
                                self._trade_log.append((idx, symbol, 'buy', 'rebalance'))
                            elif new_w < old_w:
                                self._trade_log.append((idx, symbol, 'sell', 'rebalance'))

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

    def _create_enriched_trade(
        self,
        symbol: str,
        exit_date,
        exit_price: float,
        reason: str,
        regime_at_exit: str,
        price_data: Dict[str, pd.Series],
    ):
        """
        Create an enriched trade record for the analytics dashboard.

        Args:
            symbol: Stock symbol
            exit_date: Exit date
            exit_price: Exit price
            reason: Reason for exit (stop_loss, rebalance, regime_exit)
            regime_at_exit: Market regime at exit
            price_data: Dictionary of symbol -> price series
        """
        # Get entry information
        entry_price = self._entry_prices.get(symbol)
        entry_date = self._entry_dates.get(symbol)
        entry_regime = self._entry_regimes.get(symbol)
        entry_stop_price = self._stop_prices.get(symbol, entry_price * (1 - self.stop_loss_pct) if entry_price else 0)

        if entry_price is None or entry_date is None:
            # Can't create enriched trade without entry info
            return

        # Calculate quantity from position value
        qty = 0.0
        if entry_price > 0:
            # Use position_size_pct from config for quantity calculation
            # Assuming $10k initial capital (used for analytics only)
            initial_capital = 10000
            position_value = initial_capital * self.position_size_pct
            qty = position_value / entry_price

        # Calculate P&L
        if exit_price is not None and entry_price > 0:
            realized_pnl = (exit_price - entry_price) * qty
            realized_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            realized_pnl = None
            realized_pnl_pct = None

        # Calculate holding period
        holding_days = (exit_date - entry_date).days if isinstance(entry_date, type(exit_date)) else None

        # Create enriched trade record
        from trading_bot.analytics.dashboard.models import EnrichedTrade

        trade = EnrichedTrade(
            entry_date=entry_date,
            exit_date=exit_date,
            symbol=symbol,
            side='buy',
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            entry_stop_price=entry_stop_price,
            realized_pnl=realized_pnl,
            realized_pnl_pct=realized_pnl_pct,
            unrealized_pnl=0.0,
            total_pnl=realized_pnl if realized_pnl else 0.0,
            holding_period_days=holding_days,
            regime_at_entry=entry_regime or 'unknown',
            regime_at_exit=regime_at_exit,
            reason=reason,
            momentum_score=None,
        )

        self._enriched_trades.append(trade)

        # Clean up entry tracking for this symbol
        self._entry_prices.pop(symbol, None)
        self._entry_dates.pop(symbol, None)
        self._entry_regimes.pop(symbol, None)

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

        # Basic stats - annualized return using geometric compounding
        n_days = len(returns_for_metrics)
        if n_days > 0 and total_return > -1:
            annualized_return = (1 + total_return) ** (252 / n_days) - 1
        else:
            annualized_return = 0.0
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
            'enriched_trades': self._enriched_trades,  # Enriched trade records for analytics dashboard
            'rebalance_frequency_days': self.rebalance_frequency,
            'trading_frequency': 'daily',
            'uses_walkforward': self.use_walkforward,
            'transaction_costs_applied': self.transaction_costs
        }