"""
Backwards compatibility re-export of stock universe constants.

Import directly from trading_bot.strategies.universes for new code.
"""
from trading_bot.strategies.universes import (
    TECH_UNIVERSE,
    SURVIVORSHIP_ADJUSTED_UNIVERSE,
    BENCHMARK_SYMBOL,
)

__all__ = ["TECH_UNIVERSE", "SURVIVORSHIP_ADJUSTED_UNIVERSE", "BENCHMARK_SYMBOL"]