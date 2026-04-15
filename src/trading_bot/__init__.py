"""
Trading Bot - Hybrid HMM + 5% Stop Loss Strategy

A production-grade algorithmic trading system featuring:
- Hidden Markov Model (HMM) for market regime detection (Bull/Neutral/Bear)
- Sector Momentum Rotation for stock selection
- 5% Stop Loss for risk management per position
- Equal weight position sizing (10% per stock)
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Team"

# Import main components for easy access
from trading_bot.ml.hybrid_with_stop import HybridHMMStopLoss
from trading_bot.data.loader import DataLoader
from trading_bot.logging import setup_logging

__all__ = [
    "__version__",
    "HybridHMMStopLoss",
    "DataLoader",
    "setup_logging",
]
