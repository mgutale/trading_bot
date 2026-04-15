"""
Tests for Data Loader Module

Tests caching, API fallback, and data validation.
"""

import pandas as pd
import pytest
from pathlib import Path

from trading_bot.data.loader import DataLoader, TECH_UNIVERSE, BENCHMARK_SYMBOL


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_initialization(self, temp_cache_dir):
        """Test DataLoader initializes correctly."""
        loader = DataLoader(cache_dir=temp_cache_dir)
        assert loader.cache_dir == Path(temp_cache_dir)
        assert loader.use_ibkr is False

    def test_cache_path_generation(self, temp_cache_dir):
        """Test cache path is generated correctly."""
        loader = DataLoader(cache_dir=temp_cache_dir)
        path = loader._get_cache_path("AAPL", "1Day", "2024-01-01", "2024-12-31")

        assert "AAPL" in str(path)
        assert "1Day" in str(path)
        assert path.suffix == ".parquet"

    def test_cache_save_and_load(self, sample_prices, temp_cache_dir):
        """Test saving and loading from cache."""
        loader = DataLoader(cache_dir=temp_cache_dir)

        # Save to cache
        loader._save_to_cache(
            sample_prices, "TEST", "1Day",
            "2024-01-01", "2024-12-31"
        )

        # Load from cache
        loaded = loader._load_from_cache(
            "TEST", "1Day",
            "2024-01-01", "2024-12-31"
        )

        assert loaded is not None
        assert len(loaded) == len(sample_prices)
        assert list(loaded.columns) == list(sample_prices.columns)

    def test_cache_miss_returns_none(self, temp_cache_dir):
        """Test cache miss returns None."""
        loader = DataLoader(cache_dir=temp_cache_dir)
        result = loader._load_from_cache(
            "NONEXISTENT", "1Day",
            "2024-01-01", "2024-12-31"
        )
        assert result is None

    def test_tech_universe_not_empty(self):
        """Test TECH_UNIVERSE has symbols."""
        assert len(TECH_UNIVERSE) > 0
        assert "AAPL" in TECH_UNIVERSE
        assert "MSFT" in TECH_UNIVERSE

    def test_benchmark_symbol_not_empty(self):
        """Test BENCHMARK_SYMBOL is set."""
        assert BENCHMARK_SYMBOL == "SPY"


class TestDataValidation:
    """Tests for data validation."""

    def test_empty_dataframe_handling(self, temp_cache_dir):
        """Test handling of empty data."""
        loader = DataLoader(cache_dir=temp_cache_dir)

        # Save empty dataframe
        empty_df = pd.DataFrame()
        loader._save_to_cache(empty_df, "EMPTY", "1Day", "2024-01-01", "2024-12-31")

        loaded = loader._load_from_cache("EMPTY", "1Day", "2024-01-01", "2024-12-31")
        assert loaded is not None
        assert len(loaded) == 0

    def test_cache_clear(self, sample_prices, temp_cache_dir):
        """Test clearing cache."""
        loader = DataLoader(cache_dir=temp_cache_dir)

        # Save some data
        loader._save_to_cache(sample_prices, "TEST1", "1Day", "2024-01-01", "2024-12-31")
        loader._save_to_cache(sample_prices, "TEST2", "1Day", "2024-01-01", "2024-12-31")

        # Clear cache
        loader.clear_cache()

        # Verify cache is empty
        cache_files = list(Path(temp_cache_dir).glob("*.parquet"))
        assert len(cache_files) == 0
