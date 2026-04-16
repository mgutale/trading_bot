# HMM-Based Momentum Trading Bot

A **modular algorithmic trading system** featuring Hidden Markov Model (HMM) regime detection, momentum-based stock selection, and automated risk management.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Performance (3-Year Backtest)

| Metric | Value |
|--------|-------|
| **Total Return** | 503.1% |
| **Annualized Return** | 61.6% |
| **Sharpe Ratio** | 3.90 |
| **Sortino Ratio** | 6.26 |
| **Max Drawdown** | -7.2% |
| **Win Rate** | 50.5% |
| **Profit Factor** | 2.51 |

**Portfolio:** $5,000 → $30,157 (+$25,157)  
**Benchmark (SPY):** 75.7%  
**Outperformance:** 427.4%

*Period: 2023-04-17 to 2026-04-16 | 751 days | 2,566 trades*

---

## Strategy Overview

### Core Components

1. **Hidden Markov Model (HMM) Regime Detection**
   - 4-state market regime classification (Strong Bull, Weak Bull, Weak Bear, Strong Bear)
   - Walk-forward training to prevent look-ahead bias
   - Adaptive state transition probabilities based on market conditions

2. **Momentum-Based Stock Selection**
   - Ranks 15 technology stocks by combined short/long-term momentum
   - Selects top N stocks with **momentum-weighted position sizing**
   - Higher momentum stocks receive larger allocations (capped at 20% per position)

3. **Risk Management**
   - Trailing stop loss (2.5% per position, moves up only)
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

### Position Sizing

The strategy uses **momentum-weighted allocation**:
- Stocks are ranked by momentum score (short + long term)
- Higher momentum = larger position weight
- Individual positions capped at 20% of portfolio
- Negative momentum stocks only included to fill remaining exposure

Example allocation (100% exposure, 5 stocks):

| Stock | Momentum | Weight |
|-------|----------|--------|
| NVDA | 0.45 (highest) | 20.0% |
| AMD | 0.25 | 20.0% |
| AVGO | 0.15 | 20.0% |
| MSFT | 0.10 | 20.0% |
| GOOGL | 0.05 | 11.1% |
| INTC | -0.10 (negative) | 8.9% |

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
│   │   └── universes.py        # TECH_UNIVERSE, BENCHMARK_SYMBOL
│   ├── ml/                      # Raw ML models
│   │   └── markov_regime.py    # MarkovRegimeDetector (HMM)
│   ├── data/                    # Data fetching
│   │   └── loader.py           # DataLoader (Yahoo Finance + IBKR)
│   ├── optimization/            # Hyperparameter optimization
│   │   └── optuna_optimizer.py # Optuna-based parameter search
│   ├── core/                    # IBKR execution client
│   ├── analytics/               # Dashboard & reporting
│   │   └── dashboard/          # Live analytics dashboard
│   ├── config.py               # StrategyConfig (single source of truth)
│   ├── strategy_engine.py      # Position manager for live trading
│   ├── logging.py              # Logging setup
│   ├── telegram_notifier.py    # Telegram notifications
│   └── main.py                 # CLI runner
├── tests/
│   ├── strategies/             # Unit tests for strategy modules
│   ├── test_data_loader.py
│   ├── test_look_ahead_bias.py
│   └── test_paper_trading.py
├── data/                       # Cached market data (parquet files)
└── results/                    # Backtest reports and charts
```

---

## Configuration

All strategy parameters are defined in **`StrategyConfig`** (`src/trading_bot/config.py`):

```python
@dataclass
class StrategyConfig:
    n_states: int = 4                 # 4-state HMM
    momentum_short: int = 110         # Short momentum lookback (days)
    momentum_long: int = 228          # Long momentum lookback (days)
    top_n_stocks: int = 5             # Number of stocks to hold
    rebalance_frequency: int = 5      # Rebalance every 5 days
    stop_loss_pct: float = 0.025      # 2.5% trailing stop
    position_size_pct: float = 0.20   # Max 20% per position
    spread_pct: float = 0.001         # 0.1% bid-ask spread
    slippage_pct: float = 0.001       # 0.1% slippage
    commission_pct: float = 0.001     # 0.1% commission
```

**Stock Universe** (`src/trading_bot/strategies/universes.py`):
```python
TECH_UNIVERSE = [
    "NVDA", "AMD", "AVGO", "INTC", "TSM", "ASML",
    "MSFT", "GOOGL", "AMZN", "META", "AAPL", "ORCL",
    "CRM", "ADBE", "NOW",
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

# With optimization support
pip install -e ".[optimization]"
```

### Environment Variables (Optional)

For live trading with IBKR and Telegram notifications:
```bash
# .env file
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## Quick Start

### Backtest

```bash
# 1-year backtest
python3 -m trading_bot.main backtest --years 1 --capital 10000

# 3-year backtest
python3 -m trading_bot.main backtest --years 3

# 10-year backtest (full tech universe history)
python3 -m trading_bot.main backtest --years 10

# With live dashboard
python3 -m trading_bot.main backtest --years 1 --dashboard
```

### Live Trading

```bash
# Paper trading (dry run)
python3 -m trading_bot.main live --dry-run

# Live trading (requires TWS or IB Gateway running)
python3 -m trading_bot.main live

# Daily scheduled runs
python3 -m trading_bot.main live --daily
```

### Optimize Parameters

```bash
# Optimize with walk-forward validation (recommended)
python3 -m trading_bot.main optimize --trials 100 --years 3 --walk-forward

# Quick optimization
python3 -m trading_bot.main optimize --trials 50 --years 2
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
    start='2021-01-01',
    end='2026-04-16'
)
benchmark = loader.get_historical_data('SPY')

# Run backtest
config = StrategyConfig()
strategy = HybridHMMStopLoss.from_config(config)
results = strategy.backtest(stock_data, benchmark)

print(f"Return: {results['total_return']:.2%}")
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
print(f"Max DD: {results['max_drawdown']:.2%}")
```

---

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ --cov=src/trading_bot
```

---

## Key Features

### Verified
- **No look-ahead bias** — Walk-forward HMM training, proper index alignment
- **Momentum-weighted sizing** — Higher momentum stocks get larger allocations
- **Complete trade lifecycle** — Opens and closes positions properly
- **Realistic transaction costs** — Applied based on actual turnover

### Risk Controls
- **Per-position stop loss** — 2.5% trailing stop limits individual losses
- **Regime-based exposure** — Reduces exposure in bear markets
- **Position size cap** — Maximum 20% per stock prevents concentration
- **Diversification** — Holds up to 5 stocks across tech sector

### Known Limitations
- **Survivorship bias** — Uses stocks that survived to 2026
- **Tech-only universe** — Results may not generalize to other sectors
- **Backtest assumptions** — Live results may vary due to execution slippage

---

## Disclaimer

This software is for **educational and research purposes only**. Not financial advice. Trading involves substantial risk of loss. Past performance does not guarantee future results.
