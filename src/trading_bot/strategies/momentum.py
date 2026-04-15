"""
Momentum Ranking and Portfolio Construction

Extracted from HybridHMMStopLoss for reusability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def rank_by_momentum(
    prices: pd.DataFrame,
    short_window: int,
    long_window: int,
    top_n: int = 5,
    current_idx: Optional[int] = None
) -> List[str]:
    """
    Rank stocks by combined momentum and return top N.

    Args:
        prices: DataFrame of close prices indexed by date
        short_window: Short-term momentum lookback in days
        long_window: Long-term momentum lookback in days
        top_n: Number of top stocks to return
        current_idx: Index position for current date in backtest (avoids look-ahead).
                     If None, uses the last row (live trading).

    Returns:
        List of top N symbols sorted by momentum score descending
    """
    momentum_scores = {}

    for symbol in prices.columns:
        if len(prices) < long_window:
            continue

        if current_idx is not None and current_idx >= long_window and current_idx < len(prices):
            current_price = prices.iloc[current_idx][symbol]
            short_price = prices.iloc[current_idx - short_window][symbol]
            long_price = prices.iloc[current_idx - long_window][symbol]
        elif current_idx is None:
            current_price = prices.iloc[-1][symbol]
            short_price = prices.iloc[-short_window][symbol]
            long_price = prices.iloc[-long_window][symbol]
        else:
            continue

        mom_short = (current_price / short_price) - 1 if short_price > 0 else 0
        mom_long = (current_price / long_price) - 1 if long_price > 0 else 0
        momentum_scores[symbol] = mom_short + mom_long

    ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    return [s[0] for s in ranked[:top_n]]


def build_equal_weight_portfolio(
    symbols: List[str],
    exposure: float = 1.0
) -> Dict[str, float]:
    """
    Build equal-weight portfolio from list of symbols.

    Args:
        symbols: List of stock symbols
        exposure: Portfolio exposure (0.0 to 1.0)

    Returns:
        Dict mapping symbol -> weight
    """
    if not symbols:
        return {}
    weight_per_stock = exposure / len(symbols)
    return {symbol: weight_per_stock for symbol in symbols}