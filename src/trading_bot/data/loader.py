"""
Data Loader Module

Production wrapper for loading historical stock data.
Uses Yahoo Finance as primary source with optional IBKR data source for live trading.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import yfinance as yf

logger = logging.getLogger(__name__)


class DataLoader:
    """
    DataLoader with production logging.

    Handles fetching and caching historical stock market data.
    Uses Yahoo Finance by default. When IBKR is configured (TWS/IB Gateway running),
    it can also fetch data from Interactive Brokers.
    """

    def __init__(self, cache_dir: str = "data", use_ibkr: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_ibkr = use_ibkr
        self._ibkr_client = None

        if use_ibkr:
            self._init_ibkr()

    def _init_ibkr(self) -> None:
        """Initialize IBKR data connection if available."""
        try:
            from trading_bot.core.ibkr_client import IBKRClient
            self._ibkr_client = IBKRClient()
            logger.info("IBKR data source connected")
        except Exception as e:
            logger.warning(f"IBKR not available, using Yahoo Finance only: {e}")
            self._ibkr_client = None
            self.use_ibkr = False

    def _get_cache_path(self, symbol: str, timeframe: str, start: str, end: str) -> Path:
        """Generate cache file path"""
        return self.cache_dir / f"{symbol}_{timeframe}_{start}_{end}.parquet"

    def _load_from_cache(self, symbol: str, timeframe: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """Try to load data from cache"""
        cache_path = self._get_cache_path(symbol, timeframe, start, end)
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None

    def _save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str, start: str, end: str) -> None:
        """Save data to cache"""
        cache_path = self._get_cache_path(symbol, timeframe, start, end)
        df.to_parquet(cache_path)

    def get_historical_data(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a stock symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL", "MSFT", "NVDA")
            timeframe: "1Day" for daily data
            start: Start date in YYYY-MM-DD format
            end: End date in YYYY-MM-DD format
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        logger.debug(f"Fetching data for {symbol}: {start} to {end}")
        result = self._get_historical_data_impl(symbol, timeframe, start, end, use_cache)
        if result.empty:
            logger.warning(f"No data returned for {symbol}")
        else:
            logger.debug(f"Loaded {len(result)} rows for {symbol}")
        return result

    def _get_historical_data_impl(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Internal implementation of get_historical_data."""
        # Set defaults
        if end is None:
            end = datetime.now().strftime("%Y-%m-%d")
        if start is None:
            start = (datetime.now() - timedelta(days=365*3)).strftime("%Y-%m-%d")

        # Try cache first
        if use_cache:
            cached = self._load_from_cache(symbol, timeframe, start, end)
            if cached is not None:
                logger.info(f"Loaded {symbol} from cache")
                return cached

        # Try IBKR first if available
        if self.use_ibkr and self._ibkr_client:
            logger.info(f"Fetching {symbol} from IBKR ({timeframe})...")
            try:
                df = self._fetch_from_ibkr(symbol, timeframe, start, end)
                if not df.empty:
                    if use_cache:
                        self._save_to_cache(df, symbol, timeframe, start, end)
                        logger.info(f"Cached {symbol} to {self.cache_dir}")
                    return df
            except Exception as e:
                logger.warning(f"IBKR failed for {symbol}, falling back to Yahoo Finance: {e}")

        # Yahoo Finance (primary for backtest, fallback for live)
        logger.info(f"Fetching {symbol} from Yahoo Finance ({timeframe})...")
        try:
            ticker = yf.Ticker(symbol)
            interval_map = {
                "1Hour": "1h",
                "1Min": "1m",
                "15Min": "15m",
                "1Day": "1d"
            }
            yahoo_interval = interval_map.get(timeframe, "1d")
            df = ticker.history(start=start, end=end, interval=yahoo_interval)

            if df.empty:
                logger.warning(f"No data from Yahoo Finance for {symbol}")
                return pd.DataFrame()

            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume'
            })
            df = df[['open', 'high', 'low', 'close', 'volume']]

            if use_cache:
                self._save_to_cache(df, symbol, timeframe, start, end)
                logger.info(f"Cached {symbol} to {self.cache_dir}")

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_from_ibkr(self, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
        """Fetch data from Interactive Brokers."""
        from ib_insync import Stock, util

        contract = Stock(symbol, "SMART", "USD")
        self._ibkr_client.ib.qualifyContracts(contract)

        duration_map = {"1Day": "1 D", "1Hour": "3600 S", "15Min": "900 S"}
        bar_size_map = {"1Day": "1 day", "1Hour": "1 hour", "15Min": "15 mins"}

        duration = duration_map.get(timeframe, "1 D")
        bar_size = bar_size_map.get(timeframe, "1 day")

        # Calculate duration string (IBKR needs this format)
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        days = (end_dt - start_dt).days + 1
        duration_str = f"{days} D"

        bars = self._ibkr_client.ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=duration_str,
            barSizeSetting=bar_size,
            whatToShow="ADJUSTED_LAST",
            useRTH=True,
        )

        if not bars:
            return pd.DataFrame()

        df = util.df(bars)
        df = df.rename(columns={
            'open': 'open', 'high': 'high', 'low': 'low',
            'close': 'close', 'volume': 'volume'
        })

        if 'date' in df.columns:
            df = df.set_index('date')

        return df[['open', 'high', 'low', 'close', 'volume']]

    def get_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.

        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        logger.info(f"Loading data for {len(symbols)} symbols")
        result = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, timeframe, start, end, use_cache)
            if not df.empty:
                result[symbol] = df
        logger.info(f"Successfully loaded {len(result)} of {len(symbols)} symbols")
        return result

    def clear_cache(self) -> None:
        """Clear all cached data"""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        logger.info(f"Cleared cache at {self.cache_dir}")


# Backwards compatibility — prefer importing from strategies/universes.py
from trading_bot.strategies.universes import TECH_UNIVERSE, SURVIVORSHIP_ADJUSTED_UNIVERSE, BENCHMARK_SYMBOL

__all__ = ["DataLoader", "TECH_UNIVERSE", "SURVIVORSHIP_ADJUSTED_UNIVERSE", "BENCHMARK_SYMBOL"]