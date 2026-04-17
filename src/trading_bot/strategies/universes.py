"""
Stock Universe Constants

Single source of truth for stock universes used by strategies and CLI.
"""

# Tech stock universe for backtesting
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

# Survivorship-adjusted universe: replaces top 3 outperformers (NVDA, AVGO, TSM)
# with large-cap tech stocks available in 2021 to reduce survivorship bias.
SURVIVORSHIP_ADJUSTED_UNIVERSE = [
    "CSCO", "AMD", "QCOM", "INTC", "IBM", "ASML",
    "MSFT", "GOOGL", "AMZN", "META", "AAPL", "ORCL",
    "CRM", "ADBE", "NOW",
]

# Benchmark symbol for HMM regime detection
BENCHMARK_SYMBOL = "SPY"