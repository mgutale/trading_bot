"""
Telegram Notifier for Trading Bot

Sends daily trading reports and notifications via Telegram.
"""

import os
import logging
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send trading notifications via Telegram."""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(self, text: str, parse_mode: str = "Markdown", disable_notification: bool = False) -> bool:
        """Send a text message to Telegram."""
        if not self.bot_token:
            logger.warning("Telegram bot token not configured")
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": text
            }
            # Only add parse_mode if it's a valid value
            if parse_mode and parse_mode != "None":
                data["parse_mode"] = parse_mode
            if disable_notification:
                data["disable_notification"] = disable_notification

            response = requests.post(url, json=data, timeout=30)
            result = response.json()

            if result.get("ok"):
                logger.info(f"Telegram message sent to {self.chat_id}")
                return True
            else:
                logger.warning(f"Telegram error: {result}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_document(self, file_path: str, caption: str = "") -> bool:
        """Send a document (HTML report) to Telegram."""
        if not self.bot_token:
            logger.warning("Telegram bot token not configured")
            return False

        try:
            url = f"{self.base_url}/sendDocument"
            file_path_obj = Path(file_path)

            if not file_path_obj.exists():
                logger.warning(f"File not found: {file_path}")
                return False

            with open(file_path_obj, "rb") as f:
                files = {"document": f}
                data = {
                    "chat_id": self.chat_id,
                    "caption": caption,
                    "parse_mode": "Markdown"
                }
                response = requests.post(url, data=data, files=files, timeout=60)
                result = response.json()

            if result.get("ok"):
                logger.info(f"Telegram document sent: {file_path}")
                return True
            else:
                logger.warning(f"Telegram error: {result}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Telegram document: {e}")
            return False

    def send_trade_notification(self, action: str, symbol: str, qty: float,
                                 price: float, value: float) -> bool:
        """Send a trade execution notification."""
        emoji = "🟢" if action == "BUY" else "🔴"

        text = f"""
{emoji} *Trade Executed*

*Action:* {action}
*Symbol:* {symbol}
*Quantity:* {qty}
*Price:* ${price:,.2f}
*Value:* ${value:,.2f}

_Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET_
"""
        return self.send_message(text)

    def send_daily_report(self, account_data: Dict[str, Any],
                          regime: str, top_stocks: list,
                          trades_executed: int = 0) -> bool:
        """Send daily trading performance report."""
        equity = float(account_data.get('equity', 0))
        cash = float(account_data.get('cash', 0))
        buying_power = float(account_data.get('buying_power', 0))

        # Calculate daily P&L if we have previous equity
        prev_equity = float(account_data.get('last_equity', equity))
        daily_pnl = equity - prev_equity
        daily_pnl_pct = (daily_pnl / prev_equity * 100) if prev_equity > 0 else 0

        emoji = "📈" if daily_pnl >= 0 else "📉"

        # Format top stocks safely
        top_stocks_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(top_stocks[:5]))

        # Use HTML mode instead of Markdown to avoid parsing issues
        text = f"""
📊 Daily Trading Report
{datetime.now().strftime('%A, %B %d, %Y')}

{emoji} Account Summary
   Equity: ${equity:,.2f}
   Cash: ${cash:,.2f}
   Buying Power: ${buying_power:,.2f}
   Daily P&L: ${daily_pnl:+,.2f} ({daily_pnl_pct:+.2f}%)

🧠 Market Regime: {regime.upper()}

🏆 Top Momentum Stocks
{top_stocks_text}

💼 Trades Executed: {trades_executed}

Time: {datetime.now().strftime('%H:%M:%S')} ET
"""
        return self.send_message(text, parse_mode="")

    def send_error_notification(self, error_message: str) -> bool:
        """Send error notification."""
        text = f"""
⚠️ *Trading Bot Error*

{error_message}

_Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET_
"""
        return self.send_message(text)

    def send_startup_notification(self) -> bool:
        """Send notification when bot starts."""
        text = f"""
🚀 *Trading Bot Started*

Daily trading scheduled at 9:30 AM ET
Reports will be sent to this chat.

_Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET_
"""
        return self.send_message(text)


__all__ = ["TelegramNotifier"]
