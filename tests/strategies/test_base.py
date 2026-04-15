import pytest
from trading_bot.strategies.base import BaseStrategy

def test_base_strategy_is_abstract():
    """Cannot instantiate BaseStrategy directly."""
    with pytest.raises(TypeError, match="abstract"):
        strategy = BaseStrategy()