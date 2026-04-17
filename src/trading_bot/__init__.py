"""
Trading Bot - Hybrid HMM + 5% Stop Loss Strategy

A production-grade algorithmic trading system featuring:
- Hidden Markov Model (HMM) for market regime detection
- Momentum-based stock selection
- Stop loss and take profit risk management
- Equal weight position sizing
"""

__version__ = "1.0.0"

from trading_bot.data.loader import DataLoader
from trading_bot.logging import setup_logging
from trading_bot.strategies.hybrid import HybridHMMStopLoss

__all__ = [
    "__version__",
    "HybridHMMStopLoss",
    "DataLoader",
    "setup_logging",
]
