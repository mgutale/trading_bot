"""
Stock Universe Constants

Single source of truth for stock universes used by strategies and CLI.
"""

# Tech stock universe for backtesting
TECH_UNIVERSE = [
    "NVDA", "AMD", "AVGO", "INTC", "TSM", "ASML",
    "MSFT", "GOOGL", "AMZN", "META", "AAPL", "ORCL",
    "CRM", "ADBE", "NOW",
]

# Survivorship-adjusted universe: replaces top 3 outperformers (NVDA, AVGO, TSM)
# with large-cap tech stocks available in 2021 to reduce survivorship bias.
SURVIVORSHIP_ADJUSTED_UNIVERSE = [
    "CSCO", "AMD", "QCOM", "INTC", "IBM", "ASML",
    "MSFT", "GOOGL", "AMZN", "META", "AAPL", "ORCL",
    "CRM", "ADBE", "NOW",
]

# Benchmark symbol for HMM regime detection
BENCHMARK_SYMBOL = "SPY"