import pytest
from trading_bot.strategies.hybrid import HybridHMMStopLoss

def test_hybrid_class_exists():
    assert HybridHMMStopLoss is not None

def test_hybrid_inherits_base():
    from trading_bot.strategies.base import BaseStrategy
    assert issubclass(HybridHMMStopLoss, BaseStrategy)