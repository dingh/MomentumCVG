"""
Unit tests for MomentumCalculator.

Tests cover:
- Basic window calculation (mean, sum, count, std)
- NaN handling in return data
- Insufficient data scenarios (count < min_periods)
- Boundary conditions (early dates, collapsed windows)
- Consistency between calculate() and calculate_bulk()
- Multiple windows
- Sparse data handling
"""

import pytest
from pathlib import Path
from datetime import datetime, date, timedelta
from decimal import Decimal
import pandas as pd
import numpy as np
from unittest.mock import Mock

from src.features.momentum_calculator import MomentumCalculator
from src.features.base import FeatureDataContext


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_straddle_history():
    """
    Create sample straddle history with dense date structure (no gaps).
    
    Loaded from tests/fixtures/sample_straddle_history.csv - real 2019 data.
    
    Returns DataFrame with columns:
        - ticker: Stock ticker (AAPL, TSLA, UBER, ADP)
        - entry_date: Trade entry dates (weekly Fridays, ALL dates present)
        - return_pct: Realized return percentage (NaN for weeks without trades)
        
    Data characteristics:
        - ALL tickers have rows for ALL ~52 weeks (dense structure)
        - AAPL: Valid returns for all weeks
        - TSLA: Valid returns for all weeks
        - ADP: Valid returns with some NaN scattered
        - UBER: return_pct = NaN before 2019-05-31 (pre-IPO), valid after
    """
    fixtures_dir = Path(__file__).parent.parent / 'fixtures'
    df = pd.read_csv(fixtures_dir / 'sample_straddle_history.csv')
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    return df


@pytest.fixture
def sample_straddle_history_all_nan():
    """
    Create straddle history with ticker that has all NaN returns.
    
    Returns DataFrame for edge case testing where a ticker exists in the
    dataset but has no valid trades (all return_pct = NaN).
    
    Data characteristics:
        - XYZ: Has rows for all 52 weeks, but ALL return_pct = NaN
        
    Use case: Test handling of tickers with no valid data to calculate momentum.
    """
    fixtures_dir = Path(__file__).parent.parent / 'fixtures'
    df = pd.read_csv(fixtures_dir / 'sample_straddle_history.csv')
    df['entry_date'] = pd.to_datetime(df['entry_date'])
    # Take all unique dates from AAPL and create XYZ rows with all NaN returns
    dates = df[df['ticker'] == 'AAPL']['entry_date'].values
    xyz_rows = pd.DataFrame({'ticker': 'XYZ', 'entry_date': dates, 'return_pct': np.nan})
    return pd.concat([df, xyz_rows], ignore_index=True)


@pytest.fixture
def feature_context(sample_straddle_history):
    """
    Create FeatureDataContext with straddle history.
    
    Args:
        sample_straddle_history: DataFrame from fixture
        
    Returns:
        FeatureDataContext instance with 'straddle_history' data source
    """
    return FeatureDataContext({'straddle_history': sample_straddle_history})


@pytest.fixture
def momentum_calculator():
    """
    Create MomentumCalculator with standard configuration.
    
    Configuration:
        - windows: [(12, 2)] - Single 11-week window from t-12 to t-2
        - min_periods: 3 - Require at least 3 observations
    """
    return MomentumCalculator(windows=[(12, 2)], min_periods=3)


@pytest.fixture
def momentum_calculator_multi_window():
    """
    Create MomentumCalculator with multiple windows.
    
    Configuration:
        - windows: [(12, 2), (8, 1), (20, 4)] - Three different windows
        - min_periods: 3
    """
    return MomentumCalculator(windows=[(12, 2), (8, 1), (20, 4)], min_periods=3)


@pytest.fixture
def momentum_calculator_no_min():
    """
    Create MomentumCalculator with no minimum periods requirement.
    
    Configuration:
        - windows: [(12, 2)]
        - min_periods: 1 - Allow single observation
    """
    return MomentumCalculator(windows=[(12, 2)], min_periods=1)


# ============================================================================
# Test Class: Initialization & Configuration
# ============================================================================

class TestMomentumCalculatorInit:
    """Test MomentumCalculator initialization and configuration."""
    
    def test_init_default_parameters(self):
        """
        Test initialization with default parameters.
        
        Verifies:
        - Default window is [(12, 2)]
        - Default min_periods is 1
        - Feature names generated correctly
        """
        # Arrange & Act
        calc = MomentumCalculator()
        
        # Assert
        assert calc.windows == [(12, 2)]
        assert calc.min_periods == 1
        assert calc.feature_names == ['mom_12_2_mean', 'mom_12_2_sum', 'mom_12_2_count', 'mom_12_2_std']
    
    def test_init_custom_windows(self):
        """
        Test initialization with custom windows.
        
        Verifies:
        - Custom windows stored correctly
        - Feature names generated for all windows (4 stats × 3 windows = 12)
        """
        # Arrange & Act
        calc = MomentumCalculator(windows=[(8, 1), (12, 2), (20, 4)], min_periods=3)
        
        # Assert windows stored correctly
        assert calc.windows == [(8, 1), (12, 2), (20, 4)]
        
        # Assert 12 feature names (4 stats × 3 windows)
        assert len(calc.feature_names) == 12
        
        # Assert all window prefixes present
        names = calc.feature_names
        assert 'mom_8_1_mean' in names
        assert 'mom_12_2_mean' in names
        assert 'mom_20_4_mean' in names
        
        # Assert all stats present for each window
        for prefix in ['mom_8_1', 'mom_12_2', 'mom_20_4']:
            for stat in ['mean', 'sum', 'count', 'std']:
                assert f'{prefix}_{stat}' in names
    
    def test_init_invalid_window_max_equal_to_min(self):
        """
        Test validation rejects window where max_lag == min_lag.
        
        Verifies ValueError raised (window would have zero width).
        """
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="max_lag"):
            MomentumCalculator(windows=[(2, 2)])
    
    def test_init_invalid_window_max_less_than_min(self):
        """
        Test validation rejects window where max_lag < min_lag.
        
        Verifies ValueError raised (inverted window).
        """
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="max_lag"):
            MomentumCalculator(windows=[(2, 5)])
    
    def test_init_invalid_window_negative_min_lag(self):
        """
        Test validation rejects negative min_lag.
        
        Verifies ValueError raised (would introduce look-ahead bias).
        """
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="min_lag"):
            MomentumCalculator(windows=[(12, -1)])
    
    def test_feature_names_order_consistent(self):
        """
        Test feature_names property returns names in consistent order.
        
        Verifies:
        - Stats grouped by window (not interleaved)
        - Order within each window: mean, sum, count, std
        """
        # Arrange & Act
        calc = MomentumCalculator(windows=[(12, 2), (8, 1)])
        names = calc.feature_names
        
        # Assert order: all mom_12_2 stats first, then all mom_8_1 stats
        assert names.index('mom_12_2_mean') < names.index('mom_12_2_std')
        assert names.index('mom_12_2_std') < names.index('mom_8_1_mean')
        assert names.index('mom_8_1_mean') < names.index('mom_8_1_std')
    
    def test_required_data_sources(self):
        """
        Test required_data_sources property returns correct list.
        
        Verifies:
        - Returns ['straddle_history']
        """
        # Arrange & Act
        calc = MomentumCalculator()
        
        # Assert
        assert calc.required_data_sources == ['straddle_history']


# ============================================================================
# Test Class: Window Feature Calculation (_calculate_window_features)
# ============================================================================

class TestWindowFeatureCalculation:
    """Test _calculate_window_features method (internal helper)."""
    
    def test_calculate_window_features_basic(self, momentum_calculator):
        """
        Test basic window feature calculation with clean data.
        
        Given: Window with 5 returns [10, 20, 30, 40, 50]
        Expected:
        - mean = 30.0
        - sum = 150.0
        - count = 5
        - std ≈ 15.81 (sample std with ddof=1)
        """
        # TODO: Create DataFrame with return_pct = [10, 20, 30, 40, 50]
        # TODO: Call _calculate_window_features(df, 'mom_12_2')
        # TODO: Assert mean ≈ 30.0
        # TODO: Assert sum ≈ 150.0
        # TODO: Assert count == 5
        # TODO: Assert std ≈ 15.81 (use pytest.approx)
        pass
    
    def test_calculate_window_features_with_nan(self, momentum_calculator):
        """
        Test window calculation excludes NaN values.
        
        Given: Returns [10, NaN, 20, NaN, 30]
        Expected:
        - Only non-NaN values used: [10, 20, 30]
        - mean = 20.0
        - count = 3 (not 5)
        - sum = 60.0
        """
        # TODO: Create DataFrame with mixed NaN values
        # TODO: Call _calculate_window_features
        # TODO: Assert statistics computed only on non-NaN values
        pass
    
    def test_calculate_window_features_insufficient_data(self, momentum_calculator):
        """
        Test window calculation returns NaN when count < min_periods.
        
        Given: 
        - Calculator with min_periods=3
        - Window with only 2 returns [10, 20]
        Expected:
        - All features NaN except count=2
        """
        # TODO: Create DataFrame with only 2 returns
        # TODO: Call _calculate_window_features
        # TODO: Assert mean, sum, std are NaN
        # TODO: Assert count == 2
        pass
    
    def test_calculate_window_features_single_observation(self, momentum_calculator_no_min):
        """
        Test window calculation with single observation (when allowed).
        
        Given:
        - Calculator with min_periods=1
        - Window with single return [25.5]
        Expected:
        - mean = 25.5
        - sum = 25.5
        - count = 1
        - std = 0.0 (only 1 value)
        """
        # TODO: Use calculator with min_periods=1
        # TODO: Create DataFrame with single return
        # TODO: Assert mean == sum == 25.5
        # TODO: Assert std == 0.0
        pass
    
    def test_calculate_window_features_negative_returns(self, momentum_calculator):
        """
        Test window calculation with negative returns.
        
        Given: Returns [-10, -20, -30, 5, 10]
        Expected:
        - mean = -9.0
        - sum = -45.0
        - std calculated correctly (handles negatives)
        """
        # TODO: Create DataFrame with negative returns
        # TODO: Call _calculate_window_features
        # TODO: Assert mean < 0
        # TODO: Assert sum < 0
        # TODO: Assert std > 0
        pass
    
    def test_calculate_window_features_all_nan(self, momentum_calculator):
        """
        Test window calculation with all NaN returns.
        
        Given: Returns [NaN, NaN, NaN]
        Expected:
        - All features NaN except count=0
        """
        # TODO: Create DataFrame with all NaN
        # TODO: Assert count == 0
        # TODO: Assert all other features are NaN
        pass


# ============================================================================
# Test Class: Single Date Calculation (calculate method)
# ============================================================================

class TestCalculateSingleDate:
    """Test calculate() method for single date feature generation."""
    
    def test_calculate_basic_single_ticker(self, momentum_calculator, feature_context):
        """
        Test basic momentum calculation for single ticker at one date.
        
        Given:
        - AAPL with 30 weeks of history
        - Calculate at week 20 (position 20)
        - Window (12, 2): uses rows 8-18 (11 rows)
        
        Verifies:
        - Returns DataFrame with 1 row
        - All expected columns present
        - Feature values are numeric (not NaN)
        """
        # TODO: Get date at position 20 from sample data
        # TODO: Call calculator.calculate(context, date, ['AAPL'])
        # TODO: Assert result has 1 row
        # TODO: Assert columns: ticker, date, mom_12_2_mean, mom_12_2_sum, mom_12_2_count, mom_12_2_std
        # TODO: Assert count == 11 (or close, depending on NaN handling)
        pass
    
    def test_calculate_multiple_tickers(self, momentum_calculator, feature_context):
        """
        Test momentum calculation for multiple tickers at same date.
        
        Given:
        - Calculate for ['AAPL', 'TSLA', 'UBER'] at week 40
        
        Verifies:
        - Returns 3 rows (one per ticker)
        - Each ticker has independent features
        - UBER may have fewer observations (IPO scenario)
        """
        # TODO: Get date at position 40
        # TODO: Call calculate with ['AAPL', 'TSLA', 'UBER']
        # TODO: Assert result has 3 rows
        # TODO: Assert UBER count < AAPL count (partial history)
        pass
    
    def test_calculate_ticker_not_in_history(self, momentum_calculator, feature_context):
        """
        Test calculation for ticker with no history.
        
        Given: Ticker 'XYZ' not in straddle_history
        Expected:
        - Returns 1 row with all features NaN
        - No error raised
        """
        # TODO: Call calculate with ['XYZ']
        # TODO: Assert result has 1 row
        # TODO: Assert all features are NaN
        pass
    
    def test_calculate_date_not_in_history(self, momentum_calculator, feature_context):
        """
        Test calculation for date not in ticker's history.
        
        Given:
        - AAPL has data for weeks 1-52
        - Request calculation at week 25 (but week 25 missing for AAPL)
        
        Expected:
        - Returns 1 row with all features NaN
        """
        # TODO: Use sample_straddle_history_with_gaps fixture
        # TODO: Call calculate for AAPL at missing date
        # TODO: Assert all features NaN
        pass
    
    def test_calculate_boundary_early_position(self, momentum_calculator, feature_context):
        """
        Test calculation when target position < max_lag.
        
        Given:
        - Calculate at position 5 (week 5)
        - Window (12, 2): start_idx = 5-12 = -7 → clamped to 0
        - Uses rows [0, 1, 2, 3] (partial window, only 4 rows)
        
        Verifies:
        - Features calculated on available data
        - count = 4 (or fewer if NaN returns)
        - mean/sum/std calculated correctly on partial window
        """
        # TODO: Get date at position 5
        # TODO: Call calculate
        # TODO: Assert count <= 4 (partial window)
        # TODO: Assert features calculated (not NaN) if count >= min_periods
        pass
    
    def test_calculate_boundary_collapsed_window(self, momentum_calculator, feature_context):
        """
        Test calculation when window collapses (end_idx <= start_idx).
        
        Given:
        - Calculate at position 1 (week 1)
        - Window (12, 2): start=0, end=1-2=-1
        - Window is invalid (end < start)
        
        Expected:
        - count = 0
        - All features NaN
        """
        # TODO: Get date at position 1
        # TODO: Call calculate
        # TODO: Assert count == 0
        # TODO: Assert all features NaN
        pass
    
    def test_calculate_empty_history(self, momentum_calculator):
        """
        Test calculation with empty straddle history.
        
        Given: Context with empty DataFrame
        Expected:
        - Returns rows for all requested tickers
        - All features NaN
        """
        # TODO: Create empty DataFrame
        # TODO: Create context with empty history
        # TODO: Call calculate with ['AAPL', 'TSLA']
        # TODO: Assert 2 rows returned
        # TODO: Assert all features NaN
        pass
    
    def test_calculate_uppercase_ticker_conversion(self, momentum_calculator, feature_context):
        """
        Test that ticker symbols are converted to uppercase.
        
        Given: Call calculate with lowercase tickers ['aapl', 'tsla']
        Expected: Correctly matches uppercase tickers in history
        """
        # TODO: Call calculate with ['aapl', 'tsla'] (lowercase)
        # TODO: Assert results returned (not empty)
        # TODO: Assert ticker column contains uppercase 'AAPL', 'TSLA'
        pass
    
    def test_calculate_with_nan_returns_excluded(self, momentum_calculator, feature_context):
        """
        Test that NaN returns are excluded from statistics.
        
        Given:
        - TSLA has some NaN returns in window
        - Window should have 11 rows, but only 7 non-NaN
        
        Expected:
        - count = 7 (not 11)
        - mean/sum/std calculated on 7 values only
        """
        # TODO: Use TSLA (has NaN returns in fixture)
        # TODO: Call calculate
        # TODO: Verify count reflects only non-NaN values
        pass


# ============================================================================
# Test Class: Bulk Calculation (calculate_bulk method)
# ============================================================================

class TestCalculateBulk:
    """Test calculate_bulk() method for efficient date range processing."""
    
    def test_calculate_bulk_single_ticker(self, momentum_calculator, feature_context):
        """
        Test bulk calculation for single ticker over date range.
        
        Given:
        - AAPL history from week 1-52
        - Calculate bulk for weeks 20-30 (11 dates)
        
        Verifies:
        - Returns 11 rows (one per date)
        - Features calculated correctly for each date
        - Output columns match expected schema
        """
        # TODO: Get start_date (week 20) and end_date (week 30)
        # TODO: Call calculate_bulk(context, start_date, end_date, ['AAPL'])
        # TODO: Assert result has 11 rows
        # TODO: Assert ticker column all 'AAPL'
        # TODO: Assert date range matches request
        pass
    
    def test_calculate_bulk_multiple_tickers(self, momentum_calculator, feature_context):
        """
        Test bulk calculation for multiple tickers.
        
        Given:
        - Calculate for ['AAPL', 'TSLA'] over weeks 20-30
        
        Verifies:
        - Returns 22 rows (11 dates × 2 tickers)
        - Each ticker has independent features
        """
        # TODO: Call calculate_bulk with ['AAPL', 'TSLA']
        # TODO: Assert result has 22 rows
        # TODO: Assert 2 unique tickers
        pass
    
    def test_calculate_bulk_all_tickers(self, momentum_calculator, feature_context):
        """
        Test bulk calculation without ticker filter (all tickers).
        
        Given:
        - sample_straddle_history has 3 tickers (AAPL, TSLA, UBER)
        - Calculate for all tickers over weeks 40-45
        
        Verifies:
        - UBER may have fewer rows (started later)
        - All tickers processed correctly
        """
        # TODO: Call calculate_bulk with tickers=None
        # TODO: Assert 3 unique tickers in result
        # TODO: Assert UBER has fewer rows than AAPL (IPO scenario)
        pass
    
    def test_calculate_bulk_date_filtering(self, momentum_calculator, feature_context):
        """
        Test that date range filtering works correctly.
        
        Given:
        - Full history weeks 1-52
        - Request weeks 10-20
        
        Verifies:
        - Only dates in [10, 20] returned
        - No dates outside range
        """
        # TODO: Call calculate_bulk with specific date range
        # TODO: Assert min(result.date) == start_date
        # TODO: Assert max(result.date) == end_date
        # TODO: Assert all dates in range
        pass
    
    def test_calculate_bulk_empty_date_range(self, momentum_calculator, feature_context):
        """
        Test bulk calculation with no dates in range.
        
        Given:
        - History has data for 2023
        - Request date range in 2025 (future, no data)
        
        Expected:
        - Returns empty DataFrame
        - Correct column schema preserved
        """
        # TODO: Call calculate_bulk with future dates
        # TODO: Assert result is empty (len == 0)
        # TODO: Assert columns present
        pass
    
    def test_calculate_bulk_sparse_data(self, momentum_calculator, feature_context):
        """
        Test bulk calculation with sparse data (missing weeks).
        
        Given:
        - sample_straddle_history_with_gaps (AAPL missing weeks 10, 20, 30)
        
        Verifies:
        - Only actual dates in history returned
        - Missing dates not included in result
        - Features calculated correctly around gaps
        """
        # TODO: Use sample_straddle_history_with_gaps fixture
        # TODO: Call calculate_bulk
        # TODO: Assert missing weeks not in result
        # TODO: Assert row count matches actual data points
        pass
    
    def test_calculate_bulk_output_schema(self, momentum_calculator, feature_context):
        """
        Test that bulk calculation returns correct schema.
        
        Verifies:
        - Columns: ['ticker', 'date', 'mom_12_2_mean', 'mom_12_2_sum', 'mom_12_2_count', 'mom_12_2_std']
        - entry_date renamed to 'date'
        - No extra columns
        """
        # TODO: Call calculate_bulk
        # TODO: Assert column names match expected
        # TODO: Assert 'entry_date' not in columns (renamed to 'date')
        pass
    
    def test_calculate_bulk_ticker_uppercase_conversion(self, momentum_calculator, feature_context):
        """
        Test ticker uppercase conversion in bulk mode.
        
        Given: tickers=['aapl', 'tsla'] (lowercase)
        Expected: Results returned (matched uppercase in history)
        """
        # TODO: Call calculate_bulk with lowercase tickers
        # TODO: Assert results returned (not empty)
        pass


# ============================================================================
# Test Class: Multiple Windows
# ============================================================================

class TestMultipleWindows:
    """Test calculator with multiple momentum windows."""
    
    def test_multiple_windows_feature_count(self, momentum_calculator_multi_window):
        """
        Test that multiple windows generate correct number of features.
        
        Given: windows=[(12, 2), (8, 1), (20, 4)]
        Expected: 12 features (4 stats × 3 windows)
        """
        # TODO: Assert len(feature_names) == 12
        # TODO: Assert all window prefixes present: mom_12_2, mom_8_1, mom_20_4
        pass
    
    def test_multiple_windows_calculate(self, momentum_calculator_multi_window, feature_context):
        """
        Test calculate() with multiple windows.
        
        Verifies:
        - All windows calculated independently
        - Features for each window present
        - Values differ between windows (different lookback periods)
        """
        # TODO: Call calculate with multi-window calculator
        # TODO: Assert all 12 feature columns present
        # TODO: Assert mom_12_2_mean != mom_8_1_mean (different windows)
        pass
    
    def test_multiple_windows_calculate_bulk(self, momentum_calculator_multi_window, feature_context):
        """
        Test calculate_bulk() with multiple windows.
        
        Verifies:
        - Bulk calculation works with multiple windows
        - Performance reasonable (vectorization still effective)
        """
        # TODO: Call calculate_bulk with multi-window calculator
        # TODO: Assert all 12 feature columns present in result
        pass
    
    def test_multiple_windows_different_min_periods(self, feature_context):
        """
        Test that windows can have different effective min_periods.
        
        Given:
        - Window (8, 1): 8-week lookback → may have 7 observations
        - Window (20, 4): 17-week lookback → needs more data
        - min_periods=3 applies to all
        
        Verifies:
        - Shorter window (8,1) valid at earlier positions
        - Longer window (20,4) NaN until more history available
        """
        # TODO: Create calculator with windows=[(8, 1), (20, 4)], min_periods=3
        # TODO: Calculate at early position (e.g., week 10)
        # TODO: Assert mom_8_1 features valid (enough data)
        # TODO: Assert mom_20_4 features NaN (insufficient history)
        pass


# ============================================================================
# Test Class: Consistency Between Methods
# ============================================================================

class TestConsistency:
    """Test consistency between calculate() and calculate_bulk()."""
    
    def test_calculate_vs_bulk_single_date_single_ticker(self, momentum_calculator, feature_context):
        """
        Test that calculate() matches calculate_bulk() for single date+ticker.
        
        Given:
        - Same context, same date, same ticker
        - Call both calculate() and calculate_bulk()
        
        Expected:
        - Feature values match exactly (within floating point tolerance)
        """
        # TODO: Select a date and ticker
        # TODO: result_single = calculate(context, date, ['AAPL'])
        # TODO: result_bulk = calculate_bulk(context, date, date, ['AAPL'])
        # TODO: Assert feature values match (use pytest.approx for floats)
        pass
    
    def test_calculate_vs_bulk_multiple_dates(self, momentum_calculator, feature_context):
        """
        Test that calculate() matches calculate_bulk() across date range.
        
        Given:
        - Calculate features for AAPL at 10 different dates using calculate()
        - Calculate same using calculate_bulk()
        
        Expected:
        - All feature values match
        """
        # TODO: Select 10 dates
        # TODO: Loop: call calculate() for each date, collect results
        # TODO: Call calculate_bulk() for date range
        # TODO: Compare row-by-row
        pass
    
    def test_bulk_with_ticker_filter_matches_all_tickers(self, momentum_calculator, feature_context):
        """
        Test that bulk with ticker filter matches bulk without filter.
        
        Given:
        - calculate_bulk(tickers=['AAPL', 'TSLA'])
        - calculate_bulk(tickers=None) filtered to AAPL+TSLA
        
        Expected:
        - Results identical
        """
        # TODO: result1 = calculate_bulk(..., tickers=['AAPL', 'TSLA'])
        # TODO: result2 = calculate_bulk(..., tickers=None)
        # TODO: result2_filtered = result2[result2['ticker'].isin(['AAPL', 'TSLA'])]
        # TODO: Assert results match
        pass


# ============================================================================
# Test Class: Edge Cases & Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_ticker_list(self, momentum_calculator, feature_context):
        """
        Test calculation with empty ticker list.
        
        Expected: Returns empty DataFrame with correct schema
        """
        # TODO: Call calculate with tickers=[]
        # TODO: Assert result is empty
        # TODO: Assert columns present
        pass
    
    def test_date_before_all_history(self, momentum_calculator, feature_context):
        """
        Test calculation at date before any history exists.
        
        Given:
        - History starts at 2023-01-06
        - Request calculation at 2022-01-01
        
        Expected:
        - Empty history after filtering
        - Returns rows with all NaN features
        """
        # TODO: Call calculate with very early date
        # TODO: Assert all features NaN
        pass
    
    def test_single_return_in_window(self, momentum_calculator_no_min, feature_context):
        """
        Test window with only 1 return (when min_periods=1).
        
        Expected:
        - mean = sum = that single return
        - std = 0.0 (only 1 observation)
        - count = 1
        """
        # TODO: Use early position where window has only 1 return
        # TODO: Assert mean == sum == single return value
        # TODO: Assert std == 0.0
        # TODO: Assert count == 1
        pass
    
    def test_all_returns_identical(self, momentum_calculator):
        """
        Test window with all identical returns.
        
        Given: Returns [10, 10, 10, 10, 10]
        Expected:
        - mean = 10
        - std = 0.0 (no variance)
        """
        # TODO: Create fixture with identical returns
        # TODO: Call _calculate_window_features
        # TODO: Assert mean == 10
        # TODO: Assert std == 0.0
        pass
    
    def test_very_large_returns(self, momentum_calculator):
        """
        Test calculation with very large return values.
        
        Verifies numerical stability with large numbers.
        """
        # TODO: Create window with returns like [10000, 20000, 30000]
        # TODO: Call _calculate_window_features
        # TODO: Assert calculations correct (no overflow)
        pass
    
    def test_date_type_datetime_vs_date(self, momentum_calculator, feature_context):
        """
        Test that both datetime and date objects work for date parameter.
        
        Given:
        - Call calculate with datetime object
        - Call calculate with date object
        
        Expected: Both work correctly (pandas handles conversion)
        """
        # TODO: Test with datetime.datetime(2023, 6, 1)
        # TODO: Test with datetime.date(2023, 6, 1)
        # TODO: Assert both return results
        pass


# ============================================================================
# Test Class: Performance & Optimization
# ============================================================================

class TestPerformance:
    """Test performance characteristics (not strict benchmarks)."""
    
    def test_bulk_faster_than_loop(self, momentum_calculator, feature_context):
        """
        Test that calculate_bulk() is significantly faster than looping calculate().
        
        This is a sanity check, not a strict benchmark.
        
        Expected: bulk is at least 5x faster for 50+ dates
        """
        # TODO: Time calculate_bulk for 50 dates
        # TODO: Time loop of calculate() for same 50 dates
        # TODO: Assert bulk_time < loop_time / 5
        pass
    
    def test_bulk_memory_efficient(self, momentum_calculator, feature_context):
        """
        Test that calculate_bulk() doesn't create excessive intermediate data.
        
        Verifies output DataFrame size is reasonable.
        """
        # TODO: Call calculate_bulk for large date range
        # TODO: Check result.memory_usage()
        # TODO: Assert memory < threshold (e.g., 10MB for 10k rows)
        pass


# ============================================================================
# Notes for Implementation
# ============================================================================

"""
Fixtures to implement:
1. sample_straddle_history: 52 weeks, 3 tickers (AAPL full, TSLA with NaN, UBER partial)
2. sample_straddle_history_with_gaps: Sparse data with intentional gaps
3. feature_context: Wrapper for straddle_history in FeatureDataContext
4. momentum_calculator: Standard config (window=[(12,2)], min_periods=3)
5. momentum_calculator_multi_window: Multiple windows config
6. momentum_calculator_no_min: Permissive config (min_periods=1)

Test priorities:
- HIGH: Basic calculation, NaN handling, boundary conditions, consistency
- MEDIUM: Multiple windows, edge cases, sparse data
- LOW: Performance tests (sanity checks only)

Total tests: ~35-40 tests across 8 test classes
Estimated implementation time: 6-8 hours
"""
