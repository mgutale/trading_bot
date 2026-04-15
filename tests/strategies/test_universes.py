import pytest
from trading_bot.strategies.universes import TECH_UNIVERSE, SURVIVORSHIP_ADJUSTED_UNIVERSE, BENCHMARK_SYMBOL

def test_tech_universe_is_list():
    assert isinstance(TECH_UNIVERSE, list)

def test_tech_universe_has_15_stocks():
    assert len(TECH_UNIVERSE) == 15

def test_tech_universe_contains_nvda():
    assert "NVDA" in TECH_UNIVERSE

def test_survivorship_adjusted_universe_has_15_stocks():
    assert len(SURVIVORSHIP_ADJUSTED_UNIVERSE) == 15

def test_benchmark_symbol_is_spy():
    assert BENCHMARK_SYMBOL == "SPY"