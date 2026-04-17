# HMM-Based Momentum Trading Bot

A **modular algorithmic trading system** featuring Hidden Markov Model (HMM) regime detection, momentum-based stock selection, and automated risk management.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Performance (1-Year Backtest)

| Metric | Value |
|--------|-------|
| **Total Return** | 56.3% |
| **Annualized Return** | 56.8% |
| **Sharpe Ratio** | 4.23 |
| **Sortino Ratio** | 6.95 |
| **Max Drawdown** | -6.7% |
| **Win Rate** | 57.8% |
| **Profit Factor** | 1.80 |

**Portfolio:** $5,000 → $7,810 (+$2,810)  
**Benchmark (SPY):** 34.8%  
**Outperformance:** +21.4%

*Period: 2025-04-17 to 2026-04-17 | 249 days | 192 trades*

---

## Strategy Overview

### Core Components

1. **Multivariate Hidden Markov Model (HMM) Regime Detection**
   - 4-state market regime classification (Strong Bull, Weak Bull, Weak Bear, Strong Bear)
   - **Multivariate features**: Uses both returns AND volatility for better regime separation
   - Walk-forward training to prevent look-ahead bias
   - Forward-only filtering for real-time regime assignment (no Viterbi look-ahead)
   - BIC-based model selection for deterministic results

2. **Momentum-Based Stock Selection**
   - Ranks technology stocks by combined short/long-term momentum
   - Selects top N stocks with equal-weight position sizing
   - Configurable universe (default: 27 tech stocks across semiconductors, big tech, EV, software)

3. **Risk Management**
   - Trailing stop loss (7.2% per position, moves up only)
   - Regime-based exposure control (100%/75%/25%/0% based on market state)
   - Transaction cost modeling (0.3% total: spread + slippage + commission)
   - Position size cap prevents over-concentration

### Regime Exposure

| Regime | Exposure | Description |
|--------|----------|-------------|
| Strong Bull | 100% | Full investment |
| Weak Bull | 75% | Mostly invested |
| Weak Bear | 25% | Defensive position |
| Strong Bear | 0% | All cash |

### Multivariate HMM

The strategy uses a **multivariate Gaussian HMM** that considers both:
- **Returns**: Daily percentage price changes
- **Volatility**: 21-day rolling volatility (annualized)

This creates meaningful differentiation between regimes:

| Regime | Days | Mean Return | Volatility (Ann.) |
|--------|------|-------------|-------------------|
| Strong Bull | ~54% | Highest | Low (~12%) |
| Weak Bull | ~19% | Moderate | High (~15%) |
| Weak Bear | ~22% | Low | Moderate |
| Strong Bear | ~2% | Negative | Highest |

---

## Project Structure

```
trading_bot/
├── src/trading_bot/
│   ├── strategies/              # All trading strategy logic
│   │   ├── base.py             # Abstract BaseStrategy interface
│   │   ├── hybrid.py           # HybridHMMStopLoss (main strategy)
│   │   ├── momentum.py         # Momentum ranking utilities
│   │   ├── regime_exposure.py  # REGIME_EXPOSURE constant
│   │   └── universes.py        # TECH_UNIVERSE, SURVIVORSHIP_ADJUSTED_UNIVERSE
│   ├── ml/                      # Raw ML models
│   │   └── markov_regime.py    # MarkovRegimeDetector (HMM)
│   ├── data/                    # Data fetching
│   │   └── loader.py           # DataLoader (Yahoo Finance)
│   ├── optimization/            # Hyperparameter optimization
│   │   └── optuna_optimizer.py # Optuna-based parameter search
│   ├── analytics/               # Dashboard & reporting
│   │   └── dashboard/          # Live analytics dashboard
│   ├── config.py               # StrategyConfig (single source of truth)
│   ├── logging.py              # Logging setup
│   └── main.py                 # CLI runner
├── tests/                       # Unit tests
├── data/                        # Cached market data (parquet files)
├── results/                     # Backtest reports and charts
├── pyproject.toml               # Package configuration
└── requirements.txt             # Pip dependencies
```

---

## Configuration

All strategy parameters are defined in **`StrategyConfig`** (`src/trading_bot/config.py`):

```python
@dataclass
class StrategyConfig:
    n_states: int = 4                 # 4-state HMM
    momentum_short: int = 43          # Short momentum lookback (days)
    momentum_long: int = 176          # Long momentum lookback (days)
    top_n_stocks: int = 8             # Number of stocks to hold
    rebalance_frequency: int = 19     # Rebalance every 19 days
    stop_loss_pct: float = 0.0722     # 7.22% trailing stop
    position_size_pct: float = 0.20   # Max 20% per position
    spread_pct: float = 0.001         # 0.1% bid-ask spread
    slippage_pct: float = 0.001       # 0.1% slippage
    commission_pct: float = 0.001     # 0.1% commission
```

**Stock Universe** (`src/trading_bot/strategies/universes.py`):
```python
TECH_UNIVERSE = [
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
]
BENCHMARK_SYMBOL = "SPY"  # For HMM regime detection
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .

# Or install from requirements
pip install -r requirements.txt

# With optimization support
pip install -e ".[optimization]"
```


---

## Quick Start

### Backtest

```bash
# 1-year backtest (default)
python -m trading_bot.main backtest --years 1

# With custom capital
python -m trading_bot.main backtest --years 1 --capital 10000

# 3-year backtest
python -m trading_bot.main backtest --years 3

# Generate HTML report
python -m trading_bot.main backtest --years 1 --report
```

### Optimize Parameters

```bash
# Optimize with walk-forward validation (recommended)
python -m trading_bot.main optimize --trials 100 --years 2 --walk-forward

# Quick optimization
python -m trading_bot.main optimize --trials 50 --years 1
```

---

## Python API

```python
from trading_bot.strategies import HybridHMMStopLoss
from trading_bot.data.loader import DataLoader
from trading_bot.config import StrategyConfig

# Load data
loader = DataLoader()
stock_data = loader.get_multiple_symbols(
    symbols=['NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    start='2024-01-01',
    end='2026-04-17'
)
benchmark = loader.get_historical_data('SPY')

# Run backtest
config = StrategyConfig.load_optimized()
strategy = HybridHMMStopLoss(
    n_states=config.n_states,
    momentum_short=config.momentum_short,
    momentum_long=config.momentum_long,
    top_n_stocks=config.top_n_stocks,
    rebalance_frequency=config.rebalance_frequency,
    stop_loss_pct=config.stop_loss_pct,
    position_size_pct=config.position_size_pct,
    spread_pct=config.spread_pct,
    slippage_pct=config.slippage_pct,
    commission_pct=config.commission_pct,
    regime_exposure=config.regime_exposure,
)
results = strategy.backtest(stock_data, benchmark)

print(f"Return: {results['total_return']:.2%}")
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
print(f"Max DD: {results['max_drawdown']:.2%}")
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src/trading_bot
```

---

## Key Features

### Verified
- **No look-ahead bias** — Walk-forward HMM training, forward-only filtering
- **Multivariate HMM** — Uses returns + volatility for regime detection
- **Deterministic results** — BIC-based model selection, fixed random seeds
- **Realistic transaction costs** — Applied based on actual turnover

### Risk Controls
- **Per-position stop loss** — 7.2% trailing stop limits individual losses
- **Regime-based exposure** — Reduces exposure in bear markets
- **Position size cap** — Maximum 20% per stock prevents concentration
- **Diversification** — Holds up to 8 stocks across tech sector

### Known Limitations
- **Survivorship bias** — Uses stocks that survived to 2026
- **Tech-only universe** — Results may not generalize to other sectors
- **Backtest assumptions** — Live results may vary due to execution slippage

---

## Reproducibility

To get identical results:

1. Use the same Python version (3.12+)
2. Install exact dependencies: `pip install -r requirements.txt`
3. Clear data cache: `rm -rf data/*.parquet`
4. Run: `python -m trading_bot.main backtest --years 1`

The HMM uses fixed random seeds and BIC-based model selection for deterministic results.
Minor variations (<0.1%) may occur due to floating-point precision in EM algorithm.

---

## Disclaimer

This software is for **educational and research purposes only**. Not financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results.
