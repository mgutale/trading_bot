"""
Regime Exposure Mapping

Single source of truth for market regime → portfolio exposure mapping.
Imported by config.py, strategies/hybrid.py, and optimization/optuna_optimizer.py.
"""

# Single source of truth for regime exposure mapping.
REGIME_EXPOSURE = {
    "strong_bull": 1.0,   # Full exposure in strong bull regime
    "weak_bull": 0.75,    # 75% exposure in weak bull
    "weak_bear": 0.25,    # 25% exposure in weak bear
    "strong_bear": 0.0,   # Cash in strong bear regime
}