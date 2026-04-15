"""
Unified Strategy Engine

Position management for live trading via IBKR.
Used by daily_trade.py for order execution and stop loss management.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from trading_bot.config import StrategyConfig

_defaults = StrategyConfig()


@dataclass
class Position:
    """Represents a single trading position."""
    symbol: str
    qty: float
    entry_price: float
    current_price: float = 0.0
    stop_price: float = 0.0  # Trailing stop price
    take_profit_price: float = 0.0
    stop_loss_pct: float = _defaults.stop_loss_pct
    take_profit_pct: float = _defaults.take_profit_pct
    side: str = "long"
    entry_date: str = ""

    def __post_init__(self):
        # Initialize stop and take profit prices if not set
        if self.stop_price <= 0 and self.entry_price > 0:
            self.stop_price = self.entry_price * (1 - self.stop_loss_pct)
        if self.take_profit_price <= 0 and self.entry_price > 0:
            self.take_profit_price = self.entry_price * (1 + self.take_profit_pct)

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.qty <= 0 or self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.qty * self.current_price

    def update_price(self, price: float):
        """Update current price."""
        self.current_price = price

    def update_trailing_stop(self, stop_pct: float = None):
        """Update trailing stop - only moves up, never down."""
        if self.current_price <= 0:
            return
        pct = stop_pct if stop_pct is not None else self.stop_loss_pct
        new_stop = self.current_price * (1 - pct)
        self.stop_price = max(self.stop_price, new_stop)

    def is_stop_loss_hit(self) -> bool:
        """Check if stop loss is triggered."""
        if self.current_price <= 0 or self.stop_price <= 0:
            return False
        return self.current_price <= self.stop_price

    def is_take_profit_hit(self) -> bool:
        """Check if take profit is triggered."""
        if self.current_price <= 0 or self.take_profit_price <= 0:
            return False
        return self.current_price >= self.take_profit_price


class PositionManager:
    """
    Position manager for live trading via IBKR.

    Features:
    - Trailing stop loss (moves up only)
    - Take profit targets
    - Rebalancing logic
    - Equal weight position sizing

    All defaults from StrategyConfig (single source of truth).
    """

    def __init__(
        self,
        stop_loss_pct: float = _defaults.stop_loss_pct,
        take_profit_pct: float = _defaults.take_profit_pct,
        max_positions: int = _defaults.top_n_stocks,
        position_size_pct: float = _defaults.position_size_pct,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct

        self.positions: Dict[str, Position] = {}
        self.trade_log: List[Dict[str, Any]] = []
        self.cash = 0.0
        self.equity = 0.0

    def set_cash(self, cash: float):
        """Set available cash."""
        self.cash = cash
        self.equity = cash

    def set_equity(self, equity: float):
        """Set total equity (cash + positions)."""
        self.equity = equity

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all active positions."""
        return list(self.positions.values())

    def get_active_symbols(self) -> List[str]:
        """Get list of symbols with active positions."""
        return list(self.positions.keys())

    def position_count(self) -> int:
        """Get number of active positions."""
        return len(self.positions)

    def buy(
        self,
        symbol: str,
        qty: float,
        price: float,
        reason: str = "rebalance"
    ) -> Optional[Dict[str, Any]]:
        """Open or add to a position. Returns trade dict if successful."""
        if qty <= 0 or price <= 0:
            return None

        value = qty * price

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos.qty + qty
            avg_price = ((pos.qty * pos.entry_price) + value) / total_qty
            pos.qty = total_qty
            pos.entry_price = avg_price
            pos.current_price = price
            pos.update_trailing_stop(self.stop_loss_pct)
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                qty=qty,
                entry_price=price,
                current_price=price,
                stop_loss_pct=self.stop_loss_pct,
                take_profit_pct=self.take_profit_pct,
                stop_price=price * (1 - self.stop_loss_pct),
                take_profit_price=price * (1 + self.take_profit_pct),
            )

        trade = {
            "symbol": symbol, "action": "BUY", "qty": qty,
            "price": price, "value": value, "reason": reason,
        }
        self.trade_log.append(trade)
        return trade

    def sell(
        self,
        symbol: str,
        qty: Optional[float] = None,
        reason: str = "rebalance"
    ) -> Optional[Dict[str, Any]]:
        """Sell a position (fully or partially). Returns trade dict if successful."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        sell_qty = qty if qty is not None else pos.qty

        if sell_qty <= 0 or sell_qty > pos.qty:
            return None

        price = pos.current_price
        value = sell_qty * price

        if sell_qty >= pos.qty:
            del self.positions[symbol]
        else:
            pos.qty -= sell_qty

        trade = {
            "symbol": symbol, "action": "SELL", "qty": sell_qty,
            "price": price, "value": value, "reason": reason,
        }
        self.trade_log.append(trade)
        return trade

    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions."""
        for symbol, pos in self.positions.items():
            if symbol in prices:
                pos.update_price(prices[symbol])

    def update_trailing_stops(self):
        """Update trailing stops for all positions."""
        for pos in self.positions.values():
            pos.update_trailing_stop(self.stop_loss_pct)

    def check_stop_losses(self) -> List[Tuple[str, float]]:
        """Check which positions have hit stop loss. Returns list of (symbol, qty)."""
        return [(symbol, pos.qty) for symbol, pos in self.positions.items() if pos.is_stop_loss_hit()]

    def check_take_profits(self) -> List[Tuple[str, float]]:
        """Check which positions have hit take profit. Returns list of (symbol, qty)."""
        return [(symbol, pos.qty) for symbol, pos in self.positions.items() if pos.is_take_profit_hit()]

    def calculate_target_value(self, exposure: float = 1.0) -> float:
        """Calculate target value per position given exposure level."""
        return self.equity * exposure * self.position_size_pct

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        positions_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

        return {
            "equity": self.equity,
            "cash": self.cash,
            "positions_value": positions_value,
            "unrealized_pnl": positions_pnl,
            "position_count": len(self.positions),
            "positions": [
                {
                    "symbol": pos.symbol,
                    "qty": pos.qty,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "stop_price": pos.stop_price,
                    "take_profit_price": pos.take_profit_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                }
                for pos in self.positions.values()
            ],
            "trades_today": len(self.trade_log),
        }