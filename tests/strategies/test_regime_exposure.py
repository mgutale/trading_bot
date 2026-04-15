import pytest
from trading_bot.strategies.regime_exposure import REGIME_EXPOSURE

def test_regime_exposure_has_all_regimes():
    assert set(REGIME_EXPOSURE.keys()) == {"strong_bull", "weak_bull", "weak_bear", "strong_bear"}

def test_regime_exposure_values_are_floats():
    for regime, exposure in REGIME_EXPOSURE.items():
        assert isinstance(exposure, float)
        assert 0.0 <= exposure <= 1.0

def test_strong_bull_is_full_exposure():
    assert REGIME_EXPOSURE["strong_bull"] == 1.0

def test_strong_bear_is_zero_exposure():
    assert REGIME_EXPOSURE["strong_bear"] == 0.0