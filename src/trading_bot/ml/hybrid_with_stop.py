"""
DEPRECATED: This module is deprecated.

Import HybridHMMStopLoss from trading_bot.strategies.hybrid instead.

This shim exists for backwards compatibility only.
"""
import warnings
warnings.warn(
    "trading_bot.ml.hybrid_with_stop is deprecated. "
    "Import from trading_bot.strategies.hybrid instead.",
    DeprecationWarning,
    stacklevel=2,
)

from trading_bot.strategies.hybrid import HybridHMMStopLoss

__all__ = ["HybridHMMStopLoss"]