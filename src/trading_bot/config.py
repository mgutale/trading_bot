"""
Configuration Management for Trading Bot

Loads and validates configuration from YAML files.
Supports loading optimized parameters from JSON file.
"""

import os
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any

import yaml

from trading_bot.strategies.regime_exposure import REGIME_EXPOSURE

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for Hybrid HMM + Momentum strategy.

    THIS IS THE SINGLE SOURCE OF TRUTH for all strategy parameters.
    All classes and scripts should read defaults from here, not hardcode values.
    Import as: from trading_bot.config import StrategyConfig, REGIME_EXPOSURE

    Parameters optimized on 2026-04-17 using 2 years of data (100 Optuna trials):
    - Optimized for Sharpe ratio maximization
    - Best Sharpe: 1.19 (vs 0.23 with previous params)
    """
    name: str = "hybrid_with_stop"
    timeframe: str = "1Day"           # Daily timeframe
    n_states: int = 4                 # 4-state HMM (Strong Bull/Weak Bull/Weak Bear/Strong Bear)
    momentum_short: int = 43          # Optimized: short momentum lookback (43 days)
    momentum_long: int = 176          # Optimized: long momentum lookback (176 days)
    top_n_stocks: int = 8             # Optimized: top 8 stocks by momentum
    rebalance_frequency: int = 19     # Optimized: rebalance every 19 days
    stop_loss_pct: float = 0.0722     # Optimized: 7.22% stop loss (wider to avoid whipsaw)
    take_profit_pct: float = 0.15     # 15% take profit
    position_size_pct: float = 0.20   # 20% per position (equal weight for 5 stocks)
    spread_pct: float = 0.001        # 0.1% bid-ask spread
    slippage_pct: float = 0.001      # 0.1% slippage + market impact
    commission_pct: float = 0.001     # 0.1% commission + PFOF cost + fees
    regime_exposure: Dict[str, float] = field(default_factory=lambda: dict(REGIME_EXPOSURE))
    universe_method: str = "static"  # "static" (TECH_UNIVERSE), "survivorship_adjusted", or "dynamic"
    # Stock universe - define which stocks to trade directly here
    # Common options: "tech" (TECH_UNIVERSE), "survivorship" (SURVIVORSHIP_ADJUSTED_UNIVERSE)
    # Or provide custom list like ["TSLA", "NVDA", "AAPL", "MSFT"]
    universe_list: list = field(default_factory=lambda: [
        # Semiconductors
        "NVDA", "AMD", "AVGO", "INTC", "TSM", "ASML",
        # Big Tech
        "MSFT", "GOOGL", "AMZN", "META", "AAPL", "ORCL",
        # Software
        "CRM", "ADBE", "NOW",
        # Electric Vehicles & Clean Energy
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI",
        "ENPH", "SEDG", "FSLR", "RUN",
        # Traditional Auto (EV transition)
        "F", "GM",
    ])

    def __post_init__(self):
        """Validate config parameters."""
        if self.momentum_short >= self.momentum_long:
            raise ValueError(
                f"momentum_short ({self.momentum_short}) must be less than "
                f"momentum_long ({self.momentum_long}). "
                "Short lookback must use fewer days than long lookback."
            )

    @classmethod
    def for_daily(cls) -> "StrategyConfig":
        """Create config for daily trading (optimized params)."""
        return cls()

    @classmethod
    def load_optimized(cls, params_path: Optional[str] = None) -> "StrategyConfig":
        """
        Load optimized parameters from JSON file.

        Args:
            params_path: Path to optimized params JSON file.
                        If None, uses default path data/optimized_params.json

        Returns:
            StrategyConfig with loaded parameters, or defaults if file not found.

        The JSON file should contain:
        {
            "momentum_short": 71,
            "momentum_long": 83,
            "stop_loss_pct": 0.053,
            "rebalance_frequency": 11,
            "top_n_stocks": 10,
            "take_profit_pct": 0.15,
            "position_size_pct": 0.20,
            "optimized_date": "2026-04-10",
            "sharpe_ratio": 4.87
        }
        """
        if params_path is None:
            params_path = "data/optimized_params.json"

        params_file = Path(params_path)
        if not params_file.exists():
            # Fall back to default optimized params
            return cls.for_daily()

        try:
            with open(params_file, "r") as f:
                data = json.load(f)

            return cls(
                n_states=data.get("n_states", 4),
                momentum_short=data.get("momentum_short", 71),
                momentum_long=data.get("momentum_long", 83),
                top_n_stocks=data.get("top_n_stocks", 5),
                rebalance_frequency=data.get("rebalance_frequency", 11),
                stop_loss_pct=data.get("stop_loss_pct", 0.053),
                take_profit_pct=data.get("take_profit_pct", 0.15),
                position_size_pct=data.get("position_size_pct", 0.20),
                spread_pct=data.get("spread_pct", 0.001),
                slippage_pct=data.get("slippage_pct", 0.001),
                commission_pct=data.get("commission_pct", 0.001),
                regime_exposure=data.get("regime_exposure", dict(REGIME_EXPOSURE)),
            )
        except Exception as e:
            # Fall back to defaults on error
            logger.warning(f"Failed to load optimized params: {e}")
            return cls.for_daily()

    def save_optimized(self, params_path: str, sharpe_ratio: float = 0.0):
        """
        Save current parameters to JSON file.

        Args:
            params_path: Path to save JSON file
            sharpe_ratio: Optional Sharpe ratio achieved with these params
        """
        params_file = Path(params_path)
        params_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "n_states": self.n_states,
            "momentum_short": self.momentum_short,
            "momentum_long": self.momentum_long,
            "top_n_stocks": self.top_n_stocks,
            "rebalance_frequency": self.rebalance_frequency,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "position_size_pct": self.position_size_pct,
            "spread_pct": self.spread_pct,
            "slippage_pct": self.slippage_pct,
            "commission_pct": self.commission_pct,
            "regime_exposure": self.regime_exposure,
            "optimized_date": datetime.now().strftime("%Y-%m-%d"),
            "sharpe_ratio": sharpe_ratio,
        }

        with open(params_file, "w") as f:
            json.dump(data, f, indent=2)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    cache_dir: str = "data"
    default_timeframe: str = "1Day"
    default_start_days: int = 365 * 10
    use_cache: bool = True


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    initial_capital: float = 5000.0
    max_position_pct: float = 0.10
    max_portfolio_drawdown: float = 0.20
    var_confidence: float = 0.95


@dataclass
class TradingBotConfig:
    """Main configuration container."""
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str) -> "TradingBotConfig":
        """Load configuration from YAML file."""
        yaml_path = Path(path)
        if not yaml_path.exists():
            return cls()  # Return defaults

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Parse strategy config
        strategy_data = data.get("strategy", {})
        strategy = StrategyConfig(
            name=strategy_data.get("name", "hybrid"),
            n_states=strategy_data.get("n_states", 4),
            momentum_short=strategy_data.get("momentum_short", 71),
            momentum_long=strategy_data.get("momentum_long", 83),
            top_n_stocks=strategy_data.get("top_n_stocks", 5),
            rebalance_frequency=strategy_data.get("rebalance_frequency", 11),
            stop_loss_pct=strategy_data.get("stop_loss_pct", 0.053),
            take_profit_pct=strategy_data.get("take_profit_pct", 0.15),
            position_size_pct=strategy_data.get("position_size_pct", 0.20),
            spread_pct=strategy_data.get("spread_pct", 0.001),
            slippage_pct=strategy_data.get("slippage_pct", 0.001),
            commission_pct=strategy_data.get("commission_pct", 0.001),
            regime_exposure=strategy_data.get("regime_exposure", dict(REGIME_EXPOSURE)),
        )

        # Parse data config
        data_data = data.get("data", {})
        data_cfg = DataConfig(
            cache_dir=data_data.get("cache_dir", "data"),
            default_timeframe=data_data.get("default_timeframe", "1Day"),
            default_start_days=data_data.get("default_start_days", 365 * 10),
            use_cache=data_data.get("use_cache", True),
        )

        # Parse risk config
        risk_data = data.get("risk", {})
        risk = RiskConfig(
            initial_capital=risk_data.get("initial_capital", 5000.0),
            max_position_pct=risk_data.get("max_position_pct", 0.10),
            max_portfolio_drawdown=risk_data.get("max_portfolio_drawdown", 0.20),
            var_confidence=risk_data.get("var_confidence", 0.95),
        )

        return cls(
            strategy=strategy,
            data=data_cfg,
            risk=risk,
            log_level=data.get("log_level", "INFO"),
            log_file=data.get("log_file"),
        )

    @classmethod
    def from_env(cls) -> "TradingBotConfig":
        """Load configuration from environment variables."""
        config_path = os.getenv("TRADING_BOT_CONFIG", "config/config.yaml")
        return cls.from_yaml(config_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "strategy": {
                "name": self.strategy.name,
                "n_states": self.strategy.n_states,
                "momentum_short": self.strategy.momentum_short,
                "momentum_long": self.strategy.momentum_long,
                "top_n_stocks": self.strategy.top_n_stocks,
                "rebalance_frequency": self.strategy.rebalance_frequency,
                "stop_loss_pct": self.strategy.stop_loss_pct,
                "take_profit_pct": self.strategy.take_profit_pct,
                "position_size_pct": self.strategy.position_size_pct,
                "spread_pct": self.strategy.spread_pct,
                "slippage_pct": self.strategy.slippage_pct,
                "commission_pct": self.strategy.commission_pct,
                "regime_exposure": self.strategy.regime_exposure,
            },
            "data": {
                "cache_dir": self.data.cache_dir,
                "default_timeframe": self.data.default_timeframe,
                "default_start_days": self.data.default_start_days,
                "use_cache": self.data.use_cache,
            },
            "risk": {
                "initial_capital": self.risk.initial_capital,
                "max_position_pct": self.risk.max_position_pct,
                "max_portfolio_drawdown": self.risk.max_portfolio_drawdown,
                "var_confidence": self.risk.var_confidence,
            },
            "log_level": self.log_level,
            "log_file": self.log_file,
        }


# Default configuration instance
DEFAULT_CONFIG = TradingBotConfig()
