"""
Interactive Brokers (IBKR) Trading Client

Handles all interactions with Interactive Brokers via TWS or IB Gateway.
Uses the ib_insync library for reliable, async-capable IBKR connectivity.
"""

import os
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv

load_dotenv()


class IBKRClient:
    """Client for interacting with Interactive Brokers via TWS/IB Gateway."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
    ):
        from ib_insync import IB

        self.host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("IBKR_PORT", "7497"))
        self.client_id = client_id or int(os.getenv("IBKR_CLIENT_ID", "1"))

        self.ib = IB()

        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to IBKR at {self.host}:{self.port}. "
                f"Ensure TWS or IB Gateway is running. Error: {e}"
            )

    def disconnect(self) -> None:
        """Disconnect from TWS/IB Gateway."""
        self.ib.disconnect()

    def get_account(self) -> Dict[str, Any]:
        """Get account summary information."""
        summary = self.ib.accountSummary()
        result = {}
        for item in summary:
            result[item.tag] = float(item.value) if item.value.replace(".", "").replace("-", "").isdigit() else item.value

        # Normalize to match expected interface
        return {
            "account_number": result.get("AccountCode", "N/A"),
            "equity": result.get("NetLiquidation", 0),
            "cash": result.get("TotalCashValue", 0),
            "buying_power": result.get("BuyingPower", 0),
            "available_funds": result.get("AvailableFunds", 0),
        }

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get current position for a symbol."""
        from ib_insync import Stock

        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        for pos in self.ib.positions():
            if pos.contract.symbol == symbol:
                return {
                    "symbol": pos.contract.symbol,
                    "qty": float(pos.position),
                    "avg_entry_price": float(pos.avgCost),
                    "market_value": float(pos.marketValue) if pos.marketValue else 0,
                    "unrealized_plpc": float(pos.unrealizedPNL / (pos.avgCost * pos.position)) if pos.avgCost and pos.position and pos.avgCost * pos.position != 0 else 0,
                }
        return {}

    def get_all_positions(self) -> List[Dict[str, Any]]:
        """Get all current positions."""
        positions = []
        for pos in self.ib.positions():
            positions.append({
                "symbol": pos.contract.symbol,
                "qty": float(pos.position),
                "avg_entry_price": float(pos.avgCost),
                "market_value": float(pos.marketValue) if pos.marketValue else 0,
                "unrealized_plpc": float(pos.unrealizedPNL / (pos.avgCost * pos.position)) if pos.avgCost and pos.position and pos.avgCost * pos.position != 0 else 0,
                "side": "long" if pos.position > 0 else "short",
                "current_price": 0,  # Filled below
            })

        # Fill current prices via tickers
        for pos in positions:
            ticker_data = self.get_current_price(pos["symbol"])
            if ticker_data:
                pos["current_price"] = ticker_data

        return positions

    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """
        Submit a new order.

        Args:
            symbol: Stock symbol (e.g., "AAPL")
            qty: Number of shares
            side: "buy" or "sell"
            order_type: "market" or "limit"
            limit_price: Required for limit orders
            time_in_force: "day" or "gtc"
        """
        from ib_insync import Stock, Order

        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        action = "BUY" if side.lower() == "buy" else "SELL"

        if order_type.lower() == "market":
            order = Order(action=action, totalQuantity=qty, orderType="MKT", tif=time_in_force.upper())
        elif order_type.lower() == "limit" and limit_price is not None:
            order = Order(action=action, totalQuantity=qty, orderType="LMT", lmtPrice=limit_price, tif=time_in_force.upper())
        else:
            raise ValueError(f"Invalid order type '{order_type}' or missing limit_price for limit order")

        trade = self.ib.placeOrder(contract, order)

        # Wait for order to be submitted (not necessarily filled)
        self.ib.sleep(1)

        return {
            "order_id": trade.order.orderId,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": trade.orderStatus.status,
            "filled": trade.orderStatus.filled,
            "avg_fill_price": trade.orderStatus.avgFillPrice,
        }

    def get_order(self, order_id: int) -> Dict[str, Any]:
        """Get order by ID."""
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                return {
                    "order_id": order_id,
                    "symbol": trade.contract.symbol,
                    "status": trade.orderStatus.status,
                    "filled": trade.orderStatus.filled,
                    "avg_fill_price": trade.orderStatus.avgFillPrice,
                }
        return {}

    def get_all_orders(self, status: str = "all") -> List[Dict[str, Any]]:
        """Get all orders."""
        trades = self.ib.openTrades() if status != "all" else self.ib.trades()
        return [
            {
                "order_id": t.order.orderId,
                "symbol": t.contract.symbol,
                "status": t.orderStatus.status,
                "filled": t.orderStatus.filled,
                "avg_fill_price": t.orderStatus.avgFillPrice,
            }
            for t in trades
        ]

    def cancel_order(self, order_id: int) -> None:
        """Cancel an order."""
        for trade in self.ib.openTrades():
            if trade.order.orderId == order_id:
                self.ib.cancelOrder(trade.order)
                return
        raise ValueError(f"Order {order_id} not found in open trades")

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        self.ib.reqGlobalCancel()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol."""
        from ib_insync import Stock

        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract, "", True, False)
        self.ib.sleep(2)
        price = ticker.marketPrice()
        self.ib.cancelMktData(contract)
        return price if price and price == price else None  # NaN check