"""
Data Loader Module

Fetches historical stock data from Yahoo Finance with local caching.
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
    DataLoader for fetching and caching historical stock market data.
    Uses Yahoo Finance as the data source.
    """

    def __init__(self, cache_dir: str = "data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

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

        # Fetch from Yahoo Finance
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


__all__ = ["DataLoader"]
