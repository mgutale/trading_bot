"""
Trade Enricher

Transforms raw trade_log tuples (date, symbol, side, reason) into EnrichedTrade objects
with entry/exit prices, P&L, holding period, and regime context.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

from trading_bot.analytics.dashboard.models import EnrichedTrade, create_enriched_trade

logger = logging.getLogger(__name__)


class TradeEnricher:
    """
    Reconstructs enriched trade records from strategy signals and raw trade log.

    The raw trade_log only contains (date, symbol, side, reason).
    This class enriches each trade with:
    - Entry/exit prices from price data
    - Realized P&L (computed from entry/exit)
    - Holding period
    - Regime at entry/exit
    - Attribution data

    Quantity is reconstructed from turnover_log, which tracks position weight changes.
    When a buy happens with no prior position, the weight change tells us the dollar
    amount invested, which divided by price gives quantity.
    """

    def __init__(
        self,
        raw_trade_log: List[Tuple],
        signals: pd.DataFrame,
        stock_data: Dict[str, pd.DataFrame],
        benchmark_data: pd.DataFrame,
        turnover_log: List[Tuple] = None,
        regime_exposure: Dict[str, float] = None,
        initial_capital: float = 10000.0,
        stop_loss_pct: float = 0.05,
    ):
        """
        Initialize the trade enricher.

        Args:
            raw_trade_log: List of (date, symbol, side, reason) tuples
            signals: DataFrame with regime, strategy_return, benchmark_return columns
            stock_data: Dict of symbol -> DataFrame with price data
            benchmark_data: DataFrame with benchmark prices
            turnover_log: List of (date, turnover_pct) tuples
            regime_exposure: Dict mapping regime names to exposure values
            initial_capital: Starting portfolio value
            stop_loss_pct: Stop loss percentage for stop price calculation
        """
        self.raw_trade_log = raw_trade_log or []
        self.signals = signals
        self.stock_data = stock_data
        self.benchmark_data = benchmark_data
        self.turnover_log = turnover_log or []
        self.regime_exposure = regime_exposure or {}
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct

        # Build lookup structures for efficient access
        self._price_cache = self._build_price_cache()
        self._regime_series = signals.get('regime', pd.Series(dtype=object)) if signals is not None else pd.Series(dtype=object)
        self._benchmark_series = signals.get('benchmark_cumulative', pd.Series(dtype=float)) if signals is not None else pd.Series(dtype=float)
        self._turnover_dict = dict(self.turnover_log) if self.turnover_log else {}

    def _build_price_cache(self) -> Dict[str, pd.Series]:
        """Build a cache of close prices for each symbol."""
        cache = {}
        for symbol, df in self.stock_data.items():
            if 'close' in df.columns:
                cache[symbol] = df['close']
        return cache

    def _get_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get the closing price for a symbol on a given date."""
        if symbol not in self._price_cache:
            return None

        prices = self._price_cache[symbol]

        # Find the closest price on or after the date
        try:
            # Try exact match first
            if date in prices.index:
                return float(prices.loc[date])
            # Find next available date
            future_dates = prices.index[prices.index >= date]
            if len(future_dates) > 0:
                return float(prices.loc[future_dates[0]])
        except (KeyError, IndexError, TypeError):
            pass

        return None

    def _get_regime_at(self, date: datetime) -> str:
        """Get the market regime at a given date."""
        try:
            if date in self._regime_series.index:
                return self._regime_series.loc[date]
            future_dates = self._regime_series.index[self._regime_series.index >= date]
            if len(future_dates) > 0:
                return self._regime_series.loc[future_dates[0]]
        except (KeyError, IndexError, TypeError):
            pass
        return 'unknown'

    def _get_benchmark_at(self, date: datetime) -> Optional[float]:
        """Get the benchmark cumulative return at a given date."""
        try:
            if date in self._benchmark_series.index:
                return self._benchmark_series.loc[date]
            future_dates = self._benchmark_series.index[self._benchmark_series.index >= date]
            if len(future_dates) > 0:
                return self._benchmark_series.loc[future_dates[0]]
        except (KeyError, IndexError, TypeError):
            pass
        return None

    def _reconstruct_qty(self, symbol: str, date: datetime, weight: float = 0.2) -> float:
        """
        Reconstruct the quantity traded from turnover and price data.

        The turnover_log tracks position weight changes. For a new position,
        the turnover change equals the weight * equity. Dividing by price gives qty.
        """
        price = self._get_price(symbol, date)
        if price is None or price <= 0:
            return 0.0

        # Get the turnover at this date to determine position size
        turnover = self._turnover_dict.get(date, weight)

        # For a new position, the turnover represents the weight added
        # Position value = turnover * initial_capital
        position_value = turnover * self.initial_capital

        # Quantity = position_value / price
        qty = position_value / price

        return qty

    def enrich(self) -> List[EnrichedTrade]:
        """
        Convert raw trade_log to enriched trades.

        Algorithm:
        1. Parse raw log chronologically
        2. Track open positions on 'buy' (store entry_price, entry_date, regime, qty)
        3. On 'sell', compute realized P&L using stored entry context
        4. Handle stop_loss/rebalance/regime_exit reasons with appropriate P&L
        5. For open positions at end, compute unrealized P&L with last known prices
        """
        enriched_trades = []
        open_positions: Dict[str, Dict] = {}  # symbol -> position details

        # Sort trades by date
        sorted_trades = sorted(self.raw_trade_log, key=lambda x: x[0])

        for trade_entry in sorted_trades:
            if len(trade_entry) < 4:
                continue

            date = trade_entry[0]
            symbol = trade_entry[1]
            side = trade_entry[2]
            reason = trade_entry[3]

            entry_price = self._get_price(symbol, date)

            if side == 'buy':
                # New or追加 position
                price = entry_price if entry_price else 0.0
                qty = self._reconstruct_qty(symbol, date)
                stop_price = price * (1 - self.stop_loss_pct) if price > 0 else 0.0
                regime = self._get_regime_at(date)
                benchmark = self._get_benchmark_at(date)

                # If adding to existing position, update average price
                if symbol in open_positions:
                    existing = open_positions[symbol]
                    total_qty = existing['qty'] + qty
                    avg_price = ((existing['entry_price'] * existing['qty']) + (price * qty)) / total_qty
                    existing['qty'] = total_qty
                    existing['entry_price'] = avg_price
                    existing['stop_price'] = avg_price * (1 - self.stop_loss_pct)
                    existing['last_update'] = date
                else:
                    open_positions[symbol] = {
                        'entry_date': date,
                        'entry_price': price,
                        'qty': qty,
                        'stop_price': stop_price,
                        'regime': regime,
                        'benchmark': benchmark,
                        'last_update': date,
                    }

            elif side == 'sell' and symbol in open_positions:
                pos = open_positions[symbol]
                exit_price = entry_price if entry_price else pos['entry_price']
                qty = pos['qty']

                # Calculate realized P&L
                realized_pnl = (exit_price - pos['entry_price']) * qty
                realized_pnl_pct = (exit_price - pos['entry_price']) / pos['entry_price'] if pos['entry_price'] > 0 else 0.0
                holding_days = (date - pos['entry_date']).days if isinstance(pos['entry_date'], datetime) else None
                regime_at_exit = self._get_regime_at(date)
                exit_benchmark = self._get_benchmark_at(date)

                enriched = create_enriched_trade(
                    entry_date=pos['entry_date'],
                    symbol=symbol,
                    side='buy',  # Original side
                    entry_price=pos['entry_price'],
                    exit_date=date,
                    exit_price=exit_price,
                    qty=qty,
                    entry_stop_price=pos['stop_price'],
                    regime_at_entry=pos['regime'],
                    reason=reason,
                    regime_at_exit=regime_at_exit,
                    momentum_score=None,
                )
                enriched_trades.append(enriched)
                del open_positions[symbol]

        # Handle remaining open positions with unrealized P&L
        for symbol, pos in open_positions.items():
            # Use naive datetime to match yfinance data (tz-naive)
            now = datetime.now().replace(tzinfo=None)
            current_price = self._get_price(symbol, now)
            if current_price is None:
                current_price = pos['entry_price']

            unrealized_pnl = (current_price - pos['entry_price']) * pos['qty']
            # Convert entry_date to tz-naive for comparison with naive now
            entry_date = pos['entry_date']
            if hasattr(entry_date, 'tz_localize'):
                entry_date = entry_date.tz_localize(None)
            holding_days = (now - entry_date).days if isinstance(entry_date, datetime) else None

            enriched = EnrichedTrade(
                entry_date=pos['entry_date'],
                exit_date=None,
                symbol=symbol,
                side='buy',
                entry_price=pos['entry_price'],
                exit_price=None,
                qty=pos['qty'],
                entry_stop_price=pos['stop_price'],
                realized_pnl=None,
                realized_pnl_pct=None,
                unrealized_pnl=unrealized_pnl,
                total_pnl=unrealized_pnl,
                holding_period_days=holding_days,
                regime_at_entry=pos['regime'],
                regime_at_exit=None,
                reason='open_position',
                momentum_score=None,
                benchmark_return=None,
                entry_benchmark=pos.get('benchmark'),
            )
            enriched_trades.append(enriched)

        return enriched_trades

    def enrich_from_backtest(
        self,
        active_positions: Dict[str, float],  # symbol -> entry_price at current date
        price_data: Dict[str, pd.Series],  # symbol -> price series
        current_date: datetime,
        regime: str,
    ) -> List[EnrichedTrade]:
        """
        Enrich trades during backtest simulation.

        This is called at each step to track currently held positions
        and compute unrealized P&L.
        """
        enriched = []

        for symbol, entry_price in active_positions.items():
            if symbol not in price_data:
                continue

            prices = price_data[symbol]
            try:
                current_price = prices.loc[current_date] if current_date in prices.index else prices.iloc[-1]
            except KeyError:
                current_price = entry_price

            qty = self._reconstruct_qty(symbol, current_date)
            unrealized_pnl = (current_price - entry_price) * qty if qty > 0 else 0.0
            holding_days = (current_date - current_date).days  # Placeholder

            enriched.append(EnrichedTrade(
                entry_date=current_date,
                exit_date=None,
                symbol=symbol,
                side='buy',
                entry_price=entry_price,
                exit_price=None,
                qty=qty,
                entry_stop_price=entry_price * (1 - self.stop_loss_pct),
                realized_pnl=None,
                realized_pnl_pct=None,
                unrealized_pnl=unrealized_pnl,
                total_pnl=unrealized_pnl,
                holding_period_days=holding_days,
                regime_at_entry=regime,
                regime_at_exit=None,
                reason='open_position',
            ))

        return enriched


def enrich_trade_log(
    raw_trade_log: List[Tuple],
    signals: pd.DataFrame,
    stock_data: Dict[str, pd.DataFrame],
    benchmark_data: pd.DataFrame,
    **kwargs
) -> List[EnrichedTrade]:
    """
    Convenience function to enrich a raw trade log.

    Args:
        raw_trade_log: List of (date, symbol, side, reason) tuples
        signals: DataFrame with regime and return columns
        stock_data: Dict of symbol -> DataFrame with price data
        benchmark_data: DataFrame with benchmark prices
        **kwargs: Additional arguments passed to TradeEnricher

    Returns:
        List of EnrichedTrade objects
    """
    enricher = TradeEnricher(
        raw_trade_log=raw_trade_log,
        signals=signals,
        stock_data=stock_data,
        benchmark_data=benchmark_data,
        **kwargs
    )
    return enricher.enrich()
