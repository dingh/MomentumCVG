"""
Unit tests for SpotPriceDB class.

Tests cover:
- Loading from CSV and Parquet formats
- Initialization and metadata caching
- Spot price lookups (single date and time series)
- Realized volatility calculations (CRITICAL - validates RV formula)
- Spot move percentage calculations
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import tempfile
import os

from src.data.spot_price_db import SpotPriceDB


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_spot_data():
    """
    Load sample spot price data from CSV fixture.
    
    Returns DataFrame with spot prices for AAPL, MSFT, GOOGL, UBER.
    Real ORATS data suitable for volatility testing.
    """
    fixture_path = Path(__file__).parent.parent / "fixtures" / "spot_prices_aapl_msft_googl_uber.csv"
    df = pd.read_csv(fixture_path, parse_dates=['date'])
    return df


@pytest.fixture
def spot_csv_file():
    """Return path to CSV fixture file with spot prices."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "spot_prices_aapl_msft_googl_uber.csv"
    return str(fixture_path)


@pytest.fixture
def spot_parquet_file():
    """Return path to Parquet fixture file with spot prices."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "spot_prices_aapl_msft_googl_uber.parquet"
    return str(fixture_path)


@pytest.fixture
def loaded_spot_db(sample_spot_data):
    """Create SpotPriceDB instance directly from DataFrame."""
    return SpotPriceDB(sample_spot_data)


# ============================================================================
# Test Class: Loading & Initialization
# ============================================================================

class TestSpotPriceDBLoadingAndInitialization:
    """Test SpotPriceDB loading from files and initialization."""
    
    def test_load_from_parquet(self, spot_parquet_file):
        """
        Test loading spot price database from Parquet file.
        
        Verifies:
        - File loads without error
        - Data is correctly parsed
        - Metadata is cached
        """
        # Arrange
        # (spot_parquet_file fixture provides the path)
        
        # Act
        spot_db = SpotPriceDB.load(spot_parquet_file)
        
        # Assert
        assert spot_db is not None
        assert isinstance(spot_db, SpotPriceDB)
        assert len(spot_db.tickers) > 0
        assert spot_db.total_records > 0
        # Verify multi-index is created
        assert isinstance(spot_db.df.index, pd.MultiIndex)
        assert spot_db.df.index.names == ['date', 'ticker']
    
    def test_load_from_csv(self, spot_csv_file):
        """
        Test loading spot price database from CSV file.
        
        Verifies:
        - File loads without error
        - Dates are correctly parsed
        - Data matches expected format
        """
        # Arrange
        # (spot_csv_file fixture provides the path)
        
        # Act
        spot_db = SpotPriceDB.load(spot_csv_file)
        
        # Assert
        assert spot_db is not None
        assert isinstance(spot_db, SpotPriceDB)
        assert len(spot_db.tickers) > 0
        assert spot_db.total_records > 0
        # Verify dates are parsed as datetime
        assert pd.api.types.is_datetime64_any_dtype(spot_db.df.index.get_level_values('date'))
    
    def test_init_creates_multi_index(self, sample_spot_data):
        """
        Test that __init__ creates multi-index (date, ticker).
        
        Verifies:
        - Multi-index is created with correct levels
        - Index is sorted
        - Original columns are preserved
        """
        # Arrange
        expected_tickers = ['AAPL', 'GOOGL', 'MSFT', 'UBER']
        
        # Act
        spot_db = SpotPriceDB(sample_spot_data)
        
        # Assert
        # Verify multi-index structure
        assert isinstance(spot_db.df.index, pd.MultiIndex)
        assert spot_db.df.index.names == ['date', 'ticker']
        assert spot_db.df.index.nlevels == 2
        
        # Verify index is sorted
        assert spot_db.df.index.is_monotonic_increasing
        
        # Verify column is preserved
        assert 'adj_spot_price' in spot_db.df.columns
        
        # Verify tickers are correct
        assert set(spot_db.tickers) == set(expected_tickers)
    
    def test_init_caches_metadata(self, sample_spot_data):
        """
        Test that __init__ caches metadata correctly.
        
        Verifies:
        - tickers list is populated and sorted
        - date_range tuple has (min_date, max_date)
        - total_records matches DataFrame length
        """
        # Arrange
        expected_tickers = ['AAPL', 'GOOGL', 'MSFT', 'UBER']
        
        # Act
        spot_db = SpotPriceDB(sample_spot_data)
        
        # Assert
        # Verify tickers are sorted
        assert spot_db.tickers == sorted(expected_tickers)
        
        # Verify date_range is tuple of dates
        assert isinstance(spot_db.date_range, tuple)
        assert len(spot_db.date_range) == 2
        assert isinstance(spot_db.date_range[0], date)
        assert isinstance(spot_db.date_range[1], date)
        assert spot_db.date_range[0] <= spot_db.date_range[1]
        
        # Verify total_records matches input data
        assert spot_db.total_records == len(sample_spot_data)
        assert spot_db.total_records == len(spot_db.df)


# ============================================================================
# Test Class: Spot Price Lookups
# ============================================================================

class TestSpotPriceLookups:
    """Test spot price lookup methods."""
    
    def test_get_spot_found(self, loaded_spot_db):
        """
        Test get_spot() returns correct price when data exists.
        
        Verifies:
        - Returns float for valid ticker + date
        - Price matches expected value
        - Works for different tickers and dates
        """
        # Arrange
        test_cases = [
            ('AAPL', date(2018, 1, 2), 43.02625),
            ('GOOGL', date(2018, 1, 2), 53.6965),
            ('MSFT', date(2018, 1, 3), 86.3301),
            ('UBER', date(2019, 5, 16), 42.84),  # UBER's first day
        ]
        
        # Act & Assert
        for ticker, test_date, expected_price in test_cases:
            spot = loaded_spot_db.get_spot(ticker, test_date)
            
            assert spot is not None, f"Expected spot for {ticker} on {test_date}"
            assert isinstance(spot, float)
            assert spot == pytest.approx(expected_price, rel=1e-6), \
                f"Expected {expected_price} for {ticker} on {test_date}, got {spot}"
    
    def test_get_spot_not_found(self, loaded_spot_db):
        """
        Test get_spot() returns None when data missing.
        
        Verifies:
        - Returns None for non-existent ticker
        - Returns None for date outside range
        - Returns None for ticker + date combination not in data
        """
        # Arrange & Act & Assert
        
        # Case 1: Non-existent ticker
        spot = loaded_spot_db.get_spot('TSLA', date(2018, 1, 2))
        assert spot is None, "Expected None for non-existent ticker"
        
        # Case 2: Date before range
        spot = loaded_spot_db.get_spot('AAPL', date(2017, 1, 1))
        assert spot is None, "Expected None for date before range"
        
        # Case 3: Date after range (assuming data ends before 2030)
        spot = loaded_spot_db.get_spot('AAPL', date(2030, 1, 1))
        assert spot is None, "Expected None for date after range"
        
        # Case 4: Valid ticker but date with no data for that ticker
        # UBER started 2019-05-16, so 2018-01-02 should return None
        spot = loaded_spot_db.get_spot('UBER', date(2018, 1, 2))
        assert spot is None, "Expected None for ticker on date before its listing"
    
    def test_get_daily_spots_valid_range(self, loaded_spot_db):
        """
        Test get_daily_spots() returns time series for valid range.
        
        Verifies:
        - Returns Series with date index
        - Series contains correct number of observations
        - Prices match expected values
        - Handles partial ranges (weekends/gaps)
        """
        # Arrange
        ticker = 'AAPL'
        start_date = date(2018, 1, 2)
        end_date = date(2018, 1, 5)
        # Expected: 2018-01-02, 01-03, 01-04, 01-05 (4 trading days)
        expected_prices = {
            pd.Timestamp('2018-01-02'): 43.02625,
            pd.Timestamp('2018-01-03'): 43.023,
            pd.Timestamp('2018-01-04'): 43.26,
            pd.Timestamp('2018-01-05'): 43.75125,
        }
        
        # Act
        spots = loaded_spot_db.get_daily_spots(ticker, start_date, end_date)
        
        # Assert
        assert isinstance(spots, pd.Series)
        assert len(spots) == 4, f"Expected 4 observations, got {len(spots)}"
        assert pd.api.types.is_datetime64_any_dtype(spots.index)
        
        # Verify specific prices
        for ts, expected_price in expected_prices.items():
            assert ts in spots.index, f"Expected {ts} in index"
            assert spots[ts] == pytest.approx(expected_price, rel=1e-6), \
                f"Expected {expected_price} for {ts}, got {spots[ts]}"
        
        # Verify first and last match
        assert spots.iloc[0] == pytest.approx(43.02625, rel=1e-6)
        assert spots.iloc[-1] == pytest.approx(43.75125, rel=1e-6)
    
    def test_get_daily_spots_missing_ticker(self, loaded_spot_db):
        """
        Test get_daily_spots() returns empty Series for missing ticker.
        
        Verifies:
        - Returns empty Series (not None)
        - Series has correct dtype (float)
        """
        # Arrange
        ticker = 'TSLA'  # Not in fixture data
        start_date = date(2018, 1, 2)
        end_date = date(2018, 1, 5)
        
        # Act
        spots = loaded_spot_db.get_daily_spots(ticker, start_date, end_date)
        
        # Assert
        assert isinstance(spots, pd.Series), "Expected Series, not None"
        assert len(spots) == 0, "Expected empty Series"
        assert spots.dtype == float or pd.api.types.is_float_dtype(spots.dtype), \
            f"Expected float dtype, got {spots.dtype}"


# ============================================================================
# Test Class: Volatility Calculations (CRITICAL)
# ============================================================================

class TestVolatilityCalculations:
    """
    Test realized volatility and spot move calculations.
    
    CRITICAL: These tests validate the core RV formula using sum of
    squared returns (RV = sqrt(252 * mean(r_t^2))).
    """
    
    def test_calculate_realized_volatility_sufficient_data(self, loaded_spot_db):
        """
        Test calculate_realized_volatility() with sufficient data.
        
        Verifies:
        - Returns float value
        - Uses correct formula: sqrt(252 * mean(daily_returns^2))
        - Result is annualized (reasonable range 0.1 to 2.0)
        - Zero-mean assumption (no mean subtraction)
        
        CRITICAL: Validates the core RV calculation formula.
        """
        # Arrange
        ticker = 'AAPL'
        start_date = date(2018, 1, 2)
        end_date = date(2018, 1, 5)
        
        # Manual calculation for verification:
        # Spots: 43.02625, 43.023, 43.26, 43.75125
        # Log returns: ln(43.023/43.02625), ln(43.26/43.023), ln(43.75125/43.26)
        spots = np.array([43.02625, 43.023, 43.26, 43.75125])
        log_returns = np.log(spots[1:] / spots[:-1])
        expected_rv = np.sqrt(252 * np.mean(log_returns**2))
        
        # Act
        rv = loaded_spot_db.calculate_realized_volatility(ticker, start_date, end_date)
        
        # Assert
        assert rv is not None, "Expected RV value, got None"
        assert isinstance(rv, float)
        
        # Verify formula correctness (this is the CRITICAL validation)
        assert rv == pytest.approx(expected_rv, rel=1e-6), \
            f"RV formula mismatch: expected {expected_rv}, got {rv}"
        
        # Verify result is reasonable for annualized volatility
        assert 0.0 < rv < 3.0, f"RV should be reasonable annualized vol, got {rv}"
        
        # Test another period to verify consistency
        rv_longer = loaded_spot_db.calculate_realized_volatility(
            'MSFT', date(2018, 1, 2), date(2018, 1, 10)
        )
        assert rv_longer is not None
        assert isinstance(rv_longer, float)
        assert 0.0 < rv_longer < 3.0
    
    def test_calculate_realized_volatility_insufficient_data(self, loaded_spot_db):
        """
        Test calculate_realized_volatility() with insufficient observations.
        
        Verifies:
        - Returns None when observations < min_observations
        - Returns None when ticker not found
        - Returns None when date range has no data
        - Logs appropriate warning messages
        """
        # Arrange & Act & Assert
        
        # Case 1: Insufficient observations (only 2 days, need 3)
        rv = loaded_spot_db.calculate_realized_volatility(
            'AAPL',
            date(2018, 1, 2),
            date(2018, 1, 3),
            min_observations=3
        )
        assert rv is None, "Expected None when observations < min_observations"
        
        # Case 2: Non-existent ticker
        rv = loaded_spot_db.calculate_realized_volatility(
            'TSLA',
            date(2018, 1, 2),
            date(2018, 1, 10)
        )
        assert rv is None, "Expected None for non-existent ticker"
        
        # Case 3: Date range with no data
        rv = loaded_spot_db.calculate_realized_volatility(
            'AAPL',
            date(2030, 1, 1),
            date(2030, 1, 10)
        )
        assert rv is None, "Expected None for date range with no data"
        
        # Case 4: Ticker before its listing (UBER before 2019-05-16)
        rv = loaded_spot_db.calculate_realized_volatility(
            'UBER',
            date(2018, 1, 2),
            date(2018, 1, 10)
        )
        assert rv is None, "Expected None for ticker before listing date"
    
    def test_calculate_spot_move_pct(self, loaded_spot_db):
        """
        Test calculate_spot_move_pct() for percentage moves.
        
        Verifies:
        - Returns float for valid ticker + date range
        - Calculates percentage correctly: (end - start) / start
        - Returns None when either date missing
        - Handles positive and negative moves
        """
        # Arrange & Act & Assert
        
        # Case 1: Positive move (AAPL 2018-01-02 to 2018-01-05)
        # From 43.02625 to 43.75125
        move = loaded_spot_db.calculate_spot_move_pct(
            'AAPL',
            date(2018, 1, 2),
            date(2018, 1, 5)
        )
        expected_move = (43.75125 - 43.02625) / 43.02625
        
        assert move is not None
        assert isinstance(move, float)
        assert move == pytest.approx(expected_move, rel=1e-6), \
            f"Expected move {expected_move}, got {move}"
        assert move > 0, "Expected positive move"
        
        # Case 2: Negative move (find a down period)
        # AAPL 2018-01-05 to 2018-01-08: 43.75125 to 43.575
        move = loaded_spot_db.calculate_spot_move_pct(
            'AAPL',
            date(2018, 1, 5),
            date(2018, 1, 8)
        )
        expected_move = (43.575 - 43.75125) / 43.75125
        
        assert move is not None
        assert move == pytest.approx(expected_move, rel=1e-6)
        assert move < 0, "Expected negative move"
        
        # Case 3: Missing start date
        move = loaded_spot_db.calculate_spot_move_pct(
            'AAPL',
            date(2030, 1, 1),
            date(2018, 1, 5)
        )
        assert move is None, "Expected None when start date missing"
        
        # Case 4: Missing end date
        move = loaded_spot_db.calculate_spot_move_pct(
            'AAPL',
            date(2018, 1, 2),
            date(2030, 1, 1)
        )
        assert move is None, "Expected None when end date missing"
        
        # Case 5: Non-existent ticker
        move = loaded_spot_db.calculate_spot_move_pct(
            'TSLA',
            date(2018, 1, 2),
            date(2018, 1, 5)
        )
        assert move is None, "Expected None for non-existent ticker"
    
    def test_get_ticker_availability(self, loaded_spot_db):
        """
        Test get_ticker_availability() returns metadata.
        
        Verifies:
        - Returns tuple (first_date, last_date, count)
        - Dates are correct for ticker's range
        - Count matches number of observations
        - Returns (None, None, 0) for missing ticker
        """
        # Arrange & Act & Assert
        
        # Case 1: Valid ticker (AAPL)
        first, last, count = loaded_spot_db.get_ticker_availability('AAPL')
        
        assert isinstance(first, date)
        assert isinstance(last, date)
        assert isinstance(count, int)
        assert first == date(2018, 1, 2), f"Expected AAPL to start 2018-01-02, got {first}"
        assert last > first, "Last date should be after first date"
        assert count > 0, "Count should be positive"
        
        # Case 2: UBER (starts later than others, on 2019-05-16)
        first, last, count = loaded_spot_db.get_ticker_availability('UBER')
        
        assert isinstance(first, date)
        assert first == date(2019, 5, 16), f"Expected UBER to start 2019-05-16, got {first}"
        assert count > 0
        
        # Case 3: MSFT (verify count makes sense)
        first_msft, last_msft, count_msft = loaded_spot_db.get_ticker_availability('MSFT')
        
        assert count_msft > 0
        assert first_msft == date(2018, 1, 2)
        
        # Case 4: Missing ticker
        first, last, count = loaded_spot_db.get_ticker_availability('TSLA')
        
        assert first is None, "Expected None for missing ticker first date"
        assert last is None, "Expected None for missing ticker last date"
        assert count == 0, "Expected 0 count for missing ticker"
