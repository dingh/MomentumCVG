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
    return FeatureDataContext(straddle_history=sample_straddle_history)


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
        # Arrange
        window_data = pd.DataFrame({'return_pct': [10.0, 20.0, 30.0, 40.0, 50.0]})
        
        # Act
        result = momentum_calculator._calculate_window_features(window_data, 'mom_12_2')
        
        # Assert
        assert result['mom_12_2_mean'] == pytest.approx(30.0)
        assert result['mom_12_2_sum'] == pytest.approx(150.0)
        assert result['mom_12_2_count'] == 5
        assert result['mom_12_2_std'] == pytest.approx(15.811, rel=1e-3)
    
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
        # Arrange
        window_data = pd.DataFrame({'return_pct': [10.0, np.nan, 20.0, np.nan, 30.0]})
        
        # Act
        result = momentum_calculator._calculate_window_features(window_data, 'mom_12_2')
        
        # Assert
        assert result['mom_12_2_count'] == 3          # NaN rows excluded
        assert result['mom_12_2_mean'] == pytest.approx(20.0)
        assert result['mom_12_2_sum'] == pytest.approx(60.0)
        assert result['mom_12_2_std'] == pytest.approx(10.0)
    
    def test_calculate_window_features_insufficient_data(self, momentum_calculator):
        """
        Test window calculation returns NaN when count < min_periods.
        
        Given: 
        - Calculator with min_periods=3
        - Window with only 2 returns [10, 20]
        Expected:
        - All features NaN except count=2
        """
        # Arrange
        window_data = pd.DataFrame({'return_pct': [10.0, 20.0]})
        
        # Act
        result = momentum_calculator._calculate_window_features(window_data, 'mom_12_2')
        
        # Assert - count reported, but stats are NaN due to insufficient data
        assert result['mom_12_2_count'] == 2
        assert np.isnan(result['mom_12_2_mean'])
        assert np.isnan(result['mom_12_2_sum'])
        assert np.isnan(result['mom_12_2_std'])
    
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
        # Arrange
        window_data = pd.DataFrame({'return_pct': [25.5]})
        
        # Act
        result = momentum_calculator_no_min._calculate_window_features(window_data, 'mom_12_2')
        
        # Assert
        assert result['mom_12_2_mean'] == pytest.approx(25.5)
        assert result['mom_12_2_sum'] == pytest.approx(25.5)
        assert result['mom_12_2_count'] == 1
        assert result['mom_12_2_std'] == pytest.approx(0.0)
    
    def test_calculate_window_features_negative_returns(self, momentum_calculator):
        """
        Test window calculation with negative returns.
        
        Given: Returns [-10, -20, -30, 5, 10]
        Expected:
        - mean = -9.0
        - sum = -45.0
        - std calculated correctly (handles negatives)
        """
        # Arrange
        window_data = pd.DataFrame({'return_pct': [-10.0, -20.0, -30.0, 5.0, 10.0]})
        
        # Act
        result = momentum_calculator._calculate_window_features(window_data, 'mom_12_2')
        
        # Assert
        assert result['mom_12_2_mean'] == pytest.approx(-9.0)
        assert result['mom_12_2_sum'] == pytest.approx(-45.0)
        assert result['mom_12_2_count'] == 5
        assert result['mom_12_2_std'] > 0
    
    def test_calculate_window_features_all_nan(self, momentum_calculator):
        """
        Test window calculation with all NaN returns.
        
        Given: Returns [NaN, NaN, NaN]
        Expected:
        - All features NaN except count=0
        """
        # Arrange
        window_data = pd.DataFrame({'return_pct': [np.nan, np.nan, np.nan]})
        
        # Act
        result = momentum_calculator._calculate_window_features(window_data, 'mom_12_2')
        
        # Assert
        assert result['mom_12_2_count'] == 0
        assert np.isnan(result['mom_12_2_mean'])
        assert np.isnan(result['mom_12_2_sum'])
        assert np.isnan(result['mom_12_2_std'])


# ============================================================================
# Test Class: Single Date Calculation (calculate method)
# ============================================================================

class TestCalculateSingleDate:
    """Test calculate() method for single date feature generation."""
    
    def test_calculate_basic_single_ticker(self, momentum_calculator, feature_context):
        """
        Test basic momentum calculation for single ticker at one date.
        
        Given:
        - AAPL at position 19 (2019-05-17), all returns valid
        - Window (12, 2): rows 7-17 inclusive = 11 rows, all AAPL non-NaN
        
        Verifies:
        - Returns DataFrame with 1 row
        - All expected columns present
        - count == 11, mean/sum/std all non-NaN
        """
        # Arrange: position 19 in the fixture (0-indexed from 2019-01-04)
        calc_date = pd.Timestamp('2019-05-17')
        expected_cols = {'ticker', 'date', 'mom_12_2_mean', 'mom_12_2_sum',
                         'mom_12_2_count', 'mom_12_2_std'}
        
        # Act
        result = momentum_calculator.calculate(feature_context, calc_date, ['AAPL'])
        
        # Assert
        assert len(result) == 1
        assert set(result.columns) == expected_cols
        assert result.iloc[0]['mom_12_2_count'] == 11
        assert result.iloc[0]['mom_12_2_mean'] == pytest.approx(22.001535, rel=1e-5)
        assert result.iloc[0]['mom_12_2_sum']  == pytest.approx(242.016881, rel=1e-5)
        assert result.iloc[0]['mom_12_2_std']  == pytest.approx(117.347678, rel=1e-5)
    
    def test_calculate_multiple_tickers(self, momentum_calculator, feature_context):
        """
        Test momentum calculation for multiple tickers at same date.
        
        Given:
        - Calculate for ['AAPL', 'TSLA', 'UBER'] at position 29 (2019-07-26)
        - UBER IPO at position 21 (2019-05-31); window rows 17-27 include
          4 pre-IPO NaN rows → UBER count=7, AAPL count=11
        
        Verifies:
        - Returns 3 rows (one per ticker)
        - UBER count < AAPL count (partial valid history in window)
        """
        # Arrange: UBER first valid return at 2019-05-31 (position 21)
        calc_date = pd.Timestamp('2019-07-26')
        
        # Act
        result = momentum_calculator.calculate(feature_context, calc_date, ['AAPL', 'TSLA', 'UBER'])
        
        # Assert
        assert len(result) == 3
        assert set(result['ticker'].tolist()) == {'AAPL', 'TSLA', 'UBER'}
        
        uber_count = result.loc[result['ticker'] == 'UBER', 'mom_12_2_count'].iloc[0]
        aapl_count = result.loc[result['ticker'] == 'AAPL', 'mom_12_2_count'].iloc[0]
        assert uber_count < aapl_count
    
    def test_calculate_ticker_not_in_history(self, momentum_calculator, feature_context):
        """
        Test calculation for ticker with no history.
        
        Given: Ticker 'XYZ' not in straddle_history
        Expected:
        - Returns 1 row (no error raised)
        - All features NaN (including count, since no ticker data at all)
        """
        # Arrange
        calc_date = pd.Timestamp('2019-05-17')
        
        # Act
        result = momentum_calculator.calculate(feature_context, calc_date, ['XYZ'])
        
        # Assert
        assert len(result) == 1
        assert result.iloc[0]['ticker'] == 'XYZ'
        for col in momentum_calculator.feature_names:
            assert pd.isna(result.iloc[0][col])
    
    def test_calculate_date_not_in_history(self, momentum_calculator, feature_context):
        """
        Test calculation for date not present in ticker's dense history.
        
        Given:
        - 2019-03-06 is a Wednesday, not in any ticker's weekly Friday history
        
        Expected:
        - target_rows lookup returns empty → row with all features NaN
        """
        # Arrange: a Wednesday, so not in weekly-Friday fixture
        missing_date = pd.Timestamp('2019-03-06')
        
        # Act
        result = momentum_calculator.calculate(feature_context, missing_date, ['AAPL'])
        
        # Assert
        assert len(result) == 1
        for col in momentum_calculator.feature_names:
            assert pd.isna(result.iloc[0][col])
    
    def test_calculate_boundary_early_position(self, momentum_calculator, feature_context):
        """
        Test calculation when target position is less than max_lag.
        
        Given:
        - AAPL at position 5 (2019-02-08)
        - Window (12, 2): start = 5-12 = -7 → clamped to 0; end = 5-2 = 3
        - Covers rows [0, 1, 2, 3] = 4 rows, all AAPL returns valid
        
        Verifies:
        - count == 4 (partial window, clamped at start)
        - count(4) >= min_periods(3) → mean/std are not NaN
        """
        # Arrange: position 5 = 2019-02-08
        calc_date = pd.Timestamp('2019-02-08')
        
        # Act
        result = momentum_calculator.calculate(feature_context, calc_date, ['AAPL'])
        
        # Assert
        assert len(result) == 1
        assert result.iloc[0]['mom_12_2_count'] == 4
        assert not pd.isna(result.iloc[0]['mom_12_2_mean'])
        assert not pd.isna(result.iloc[0]['mom_12_2_std'])
    
    def test_calculate_boundary_collapsed_window(self, momentum_calculator, feature_context):
        """
        Test calculation when min_lag pushes end_idx before start_idx.
        
        Given:
        - AAPL at position 1 (2019-01-11)
        - Window (12, 2): start = 1-12 = -11 → clamped to 0; end = 1-2 = -1
        - end(-1) <= start(0) → window collapses
        
        Expected:
        - count == 0 (explicitly set on collapse)
        - mean, sum, std all NaN
        """
        # Arrange: position 1 = 2019-01-11
        calc_date = pd.Timestamp('2019-01-11')
        
        # Act
        result = momentum_calculator.calculate(feature_context, calc_date, ['AAPL'])
        
        # Assert
        assert len(result) == 1
        assert result.iloc[0]['mom_12_2_count'] == 0
        assert pd.isna(result.iloc[0]['mom_12_2_mean'])
        assert pd.isna(result.iloc[0]['mom_12_2_sum'])
        assert pd.isna(result.iloc[0]['mom_12_2_std'])
    
    def test_calculate_empty_history(self, momentum_calculator):
        """
        Test calculation with empty straddle history.
        
        Given: Context with empty DataFrame (no rows)
        Expected:
        - Returns 2 rows (one per requested ticker)
        - All feature columns NaN (including count)
        """
        # Arrange
        empty_history = pd.DataFrame(columns=['ticker', 'entry_date', 'return_pct'])
        empty_context = FeatureDataContext(straddle_history=empty_history)
        calc_date = pd.Timestamp('2019-05-17')
        
        # Act
        result = momentum_calculator.calculate(empty_context, calc_date, ['AAPL', 'TSLA'])
        
        # Assert
        assert len(result) == 2
        for col in momentum_calculator.feature_names:
            assert result[col].isna().all()
    
    def test_calculate_uppercase_ticker_conversion(self, momentum_calculator, feature_context):
        """
        Test that ticker symbols are converted to uppercase before lookup.
        
        Given: Lowercase tickers ['aapl', 'tsla']
        Expected:
        - Both matched against uppercase 'AAPL' / 'TSLA' in history
        - Result ticker column contains uppercase values
        - Features are valid (not NaN)
        """
        # Arrange
        calc_date = pd.Timestamp('2019-05-17')
        
        # Act
        result = momentum_calculator.calculate(feature_context, calc_date, ['aapl', 'tsla'])
        
        # Assert
        assert len(result) == 2
        assert set(result['ticker'].tolist()) == {'AAPL', 'TSLA'}
        assert not result['mom_12_2_mean'].isna().any()
    
    def test_calculate_with_nan_returns_excluded(self, momentum_calculator, feature_context):
        """
        Test that NaN returns in the window are excluded from statistics.
        
        Given:
        - ADP at position 19 (2019-05-17)
        - Window rows [7..17]: positions 7 (2019-02-22) and 14 (2019-04-12)
          are NaN for ADP → 11 rows total, 9 valid
        
        Expected:
        - count == 9 (not 11)
        - mean/sum/std computed on 9 values only
        """
        # Arrange: ADP has scattered NaN; window rows 7-17 have 2 NaN at pos 7 and 14
        calc_date = pd.Timestamp('2019-05-17')
        
        # Act
        result = momentum_calculator.calculate(feature_context, calc_date, ['ADP'])
        
        # Assert
        assert len(result) == 1
        assert result.iloc[0]['mom_12_2_count'] == 9
        assert not pd.isna(result.iloc[0]['mom_12_2_mean'])
        assert not pd.isna(result.iloc[0]['mom_12_2_sum'])


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
        - Calculate bulk for weeks 20-30 (2019-05-17 to 2019-07-26 = 11 dates)
        
        Verifies:
        - Returns 11 rows (one per date)
        - ticker column is all 'AAPL'
        - date range matches request (min == start, max == end)
        - AAPL has valid (non-NaN) mean features at this point in the year
        """
        # Arrange: positions 19-29 in the fixture (0-indexed from 2019-01-04)
        start_date = pd.Timestamp('2019-05-17')
        end_date   = pd.Timestamp('2019-07-26')
        
        # Act
        result = momentum_calculator.calculate_bulk(feature_context, start_date, end_date, ['AAPL'])
        
        # Assert
        assert len(result) == 11
        assert (result['ticker'] == 'AAPL').all()
        assert result['date'].min() == start_date
        assert result['date'].max() == end_date
        # AAPL has dense valid returns by week 20 — at least some means are non-NaN
        assert not result['mom_12_2_mean'].isna().all()
    
    def test_calculate_bulk_multiple_tickers(self, momentum_calculator, feature_context):
        """
        Test bulk calculation for multiple tickers.
        
        Given:
        - Calculate for ['AAPL', 'TSLA'] over weeks 20-30
          (2019-05-17 to 2019-07-26 = 11 dates, dense fixture)
        
        Verifies:
        - Returns 22 rows (11 dates × 2 tickers)
        - Both tickers present, each with exactly 11 rows
        """
        # Arrange
        start_date = pd.Timestamp('2019-05-17')
        end_date   = pd.Timestamp('2019-07-26')
        
        # Act
        result = momentum_calculator.calculate_bulk(feature_context, start_date, end_date, ['AAPL', 'TSLA'])
        
        # Assert
        assert len(result) == 22
        assert set(result['ticker'].unique()) == {'AAPL', 'TSLA'}
        # Each ticker should have exactly 11 rows (dense fixture)
        per_ticker = result.groupby('ticker').size()
        assert (per_ticker == 11).all()
    
    def test_calculate_bulk_all_tickers(self, momentum_calculator, feature_context):
        """
        Test bulk calculation without ticker filter (all tickers).
        
        Given:
        - sample_straddle_history has 4 tickers (AAPL, TSLA, ADP, UBER),
          all present at every date in the dense fixture
        - tickers=None → uses all tickers
        
        Verifies:
        - All 4 unique tickers appear in result
        - Each ticker has 11 rows for the 11-date range
        """
        # Arrange
        start_date = pd.Timestamp('2019-05-17')
        end_date   = pd.Timestamp('2019-07-26')
        
        # Act: no tickers argument → all tickers
        result = momentum_calculator.calculate_bulk(feature_context, start_date, end_date)
        
        # Assert
        assert set(result['ticker'].unique()) == {'AAPL', 'TSLA', 'ADP', 'UBER'}
        # Dense fixture: every ticker has a row at every date
        per_ticker = result.groupby('ticker').size()
        assert (per_ticker == 11).all()
    
    def test_calculate_bulk_date_filtering(self, momentum_calculator, feature_context):
        """
        Test that date range filtering works correctly.
        
        Given:
        - Full 52-week history
        - Request weeks 10-20 (2019-03-08 to 2019-05-17 = 11 dates)
        
        Verifies:
        - Only dates in [start_date, end_date] are returned
        - min(date) == start_date, max(date) == end_date
        - No dates outside the requested range
        """
        # Arrange: positions 9-19 in the fixture
        start_date = pd.Timestamp('2019-03-08')
        end_date   = pd.Timestamp('2019-05-17')
        
        # Act
        result = momentum_calculator.calculate_bulk(feature_context, start_date, end_date, ['AAPL'])
        
        # Assert
        assert len(result) == 11
        assert result['date'].min() == start_date
        assert result['date'].max() == end_date
        assert (result['date'] >= start_date).all()
        assert (result['date'] <= end_date).all()
    
    def test_calculate_bulk_empty_date_range(self, momentum_calculator, feature_context):
        """
        Test bulk calculation with no dates in range.
        
        Given:
        - History has data only for 2019
        - Request date range in 2025 (future, no data)
        
        Expected:
        - Returns empty DataFrame
        - Correct column schema still present
        """
        # Arrange: future dates not in fixture
        start_date = pd.Timestamp('2025-01-01')
        end_date   = pd.Timestamp('2025-12-31')
        
        # Act
        result = momentum_calculator.calculate_bulk(feature_context, start_date, end_date, ['AAPL'])
        
        # Assert
        assert len(result) == 0
        expected_cols = ['ticker', 'date', 'mom_12_2_mean', 'mom_12_2_sum',
                         'mom_12_2_count', 'mom_12_2_std']
        assert list(result.columns) == expected_cols
    
    def test_calculate_bulk_nan_returns_excluded_from_count(self, momentum_calculator, feature_context):
        """
        Test that NaN return_pct values are excluded from window counts in bulk mode.

        In this system every trade date always has a row for every ticker, but a
        ticker can have NaN return_pct (UBER pre-IPO, ADP data gaps).  The rolling
        window must count only non-NaN returns — this is the only form of "sparse"
        data that actually occurs.

        Given:
        - At 2019-07-26 (position 29), window (12, 2) covers rows [17..27]
        - AAPL: all 11 returns valid → count == 11
        - UBER: IPO at position 21 (2019-05-31); rows 17-20 are NaN → count == 7

        Verifies:
        - UBER count < AAPL count at the same date
        - UBER count > 0 (has valid returns after IPO inside the window)
        - AAPL count == 11 (all valid)
        """
        # Arrange
        start_date = pd.Timestamp('2019-07-26')
        end_date   = pd.Timestamp('2019-07-26')

        # Act
        result = momentum_calculator.calculate_bulk(
            feature_context, start_date, end_date, ['AAPL', 'UBER']
        )

        # Assert
        assert len(result) == 2
        aapl_count = result.loc[result['ticker'] == 'AAPL', 'mom_12_2_count'].iloc[0]
        uber_count = result.loc[result['ticker'] == 'UBER', 'mom_12_2_count'].iloc[0]

        assert aapl_count == 11          # all 11 window rows valid for AAPL
        assert 0 < uber_count < aapl_count  # UBER has fewer valid returns (pre-IPO NaNs)
    
    def test_calculate_bulk_output_schema(self, momentum_calculator, feature_context):
        """
        Test that bulk calculation returns the correct column schema.
        
        Verifies:
        - Columns: ['ticker', 'date', 'mom_12_2_mean', 'mom_12_2_sum',
                    'mom_12_2_count', 'mom_12_2_std']
        - 'entry_date' renamed to 'date' (no raw 'entry_date' column)
        - No extra columns beyond expected schema
        """
        # Arrange
        start_date = pd.Timestamp('2019-05-17')
        end_date   = pd.Timestamp('2019-07-26')
        
        # Act
        result = momentum_calculator.calculate_bulk(feature_context, start_date, end_date, ['AAPL'])
        
        # Assert
        expected_cols = ['ticker', 'date', 'mom_12_2_mean', 'mom_12_2_sum',
                         'mom_12_2_count', 'mom_12_2_std']
        assert list(result.columns) == expected_cols
        assert 'entry_date' not in result.columns
    
    def test_calculate_bulk_ticker_uppercase_conversion(self, momentum_calculator, feature_context):
        """
        Test ticker uppercase conversion in bulk mode.
        
        Given: tickers=['aapl', 'tsla'] (lowercase)
        Expected:
        - Matched against uppercase 'AAPL' / 'TSLA' in history
        - Result contains rows (not empty)
        - Result ticker column values are uppercase
        """
        # Arrange
        start_date = pd.Timestamp('2019-05-17')
        end_date   = pd.Timestamp('2019-07-26')
        
        # Act
        result = momentum_calculator.calculate_bulk(
            feature_context, start_date, end_date, ['aapl', 'tsla']
        )
        
        # Assert
        assert len(result) > 0
        assert set(result['ticker'].unique()) == {'AAPL', 'TSLA'}


# ============================================================================
# Test Class: Multiple Windows
# ============================================================================

class TestMultipleWindows:
    """Test calculator with multiple momentum windows."""
    
    def test_multiple_windows_feature_count(self, momentum_calculator_multi_window):
        """
        Test that multiple windows generate correct number of features.
        
        Given: windows=[(12, 2), (8, 1), (20, 4)]
        Expected: 12 features (4 stats × 3 windows), one group per window
        """
        # Arrange & Act
        names = momentum_calculator_multi_window.feature_names
        
        # Assert total count
        assert len(names) == 12

        # Assert all 4 stats present for each of the 3 window prefixes
        for prefix in ['mom_12_2', 'mom_8_1', 'mom_20_4']:
            for stat in ['mean', 'sum', 'count', 'std']:
                assert f'{prefix}_{stat}' in names
    
    def test_multiple_windows_calculate(self, momentum_calculator_multi_window, feature_context):
        """
        Test calculate() produces independent features for every window.
        
        Given:
        - AAPL at position 19 (2019-05-17), plenty of history for all three windows
          - (12,2): rows [7..17] = 11 rows
          - (8,1):  rows [11..18] = 8 rows
          - (20,4): rows [0..15]  = 16 rows (clamped start)
        
        Verifies:
        - All 12 feature columns present in result
        - All three window means are non-NaN (sufficient data)
        - mom_12_2_mean != mom_8_1_mean (different lookback periods → different values)
        """
        # Arrange
        calc_date = pd.Timestamp('2019-05-17')
        
        # Act
        result = momentum_calculator_multi_window.calculate(feature_context, calc_date, ['AAPL'])
        
        # Assert schema: ticker + date + 12 feature columns
        assert len(result) == 1
        expected_cols = {'ticker', 'date'} | set(momentum_calculator_multi_window.feature_names)
        assert set(result.columns) == expected_cols
        
        # Assert all three windows produced valid (non-NaN) means
        row = result.iloc[0]
        assert not pd.isna(row['mom_12_2_mean'])
        assert not pd.isna(row['mom_8_1_mean'])
        assert not pd.isna(row['mom_20_4_mean'])
        
        # Assert windows differ (different lookback periods cover different returns)
        assert row['mom_12_2_mean'] != pytest.approx(row['mom_8_1_mean'])
    
    def test_multiple_windows_calculate_bulk(self, momentum_calculator_multi_window, feature_context):
        """
        Test calculate_bulk() produces all window columns across a date range.
        
        Given:
        - AAPL over weeks 20-30 (2019-05-17 to 2019-07-26 = 11 dates)
        - All three windows have sufficient history by week 20
        
        Verifies:
        - 11 rows returned
        - Column list is exactly ['ticker', 'date'] + all 12 feature names
        - No window column is entirely NaN for AAPL in this range
        """
        # Arrange
        start_date = pd.Timestamp('2019-05-17')
        end_date   = pd.Timestamp('2019-07-26')
        
        # Act
        result = momentum_calculator_multi_window.calculate_bulk(
            feature_context, start_date, end_date, ['AAPL']
        )
        
        # Assert row count and schema
        assert len(result) == 11
        expected_cols = ['ticker', 'date'] + momentum_calculator_multi_window.feature_names
        assert list(result.columns) == expected_cols
        
        # Assert each window has at least some valid values for AAPL
        assert not result['mom_12_2_mean'].isna().all()
        assert not result['mom_8_1_mean'].isna().all()
        assert not result['mom_20_4_mean'].isna().all()
    
    def test_multiple_windows_different_min_periods(self, feature_context):
        """
        Test that a shorter window can be valid while a longer window is still NaN.
        
        Given:
        - Calculator with windows=[(8, 1), (20, 4)], min_periods=3
        - AAPL at position 5 (2019-02-08)
          - Window (8,1):  start=max(0,5-8)=0, end=5-1=4 → rows [0..4] = 5 rows ≥ 3 → valid
          - Window (20,4): start=max(0,5-20)=0, end=5-4=1 → rows [0..1] = 2 rows < 3 → NaN
        
        Verifies:
        - mom_8_1_mean is not NaN and mom_8_1_count == 5
        - mom_20_4_mean is NaN and mom_20_4_count == 2 (reported even when below threshold)
        """
        # Arrange
        calc = MomentumCalculator(windows=[(8, 1), (20, 4)], min_periods=3)
        calc_date = pd.Timestamp('2019-02-08')  # position 5 in the fixture
        
        # Act
        result = calc.calculate(feature_context, calc_date, ['AAPL'])
        
        # Assert
        assert len(result) == 1
        row = result.iloc[0]
        
        # Shorter window (8,1): 5 valid rows → sufficient
        assert not pd.isna(row['mom_8_1_mean'])
        assert row['mom_8_1_count'] == 5
        
        # Longer window (20,4): only 2 rows available → below min_periods=3
        assert pd.isna(row['mom_20_4_mean'])
        assert row['mom_20_4_count'] == 2


# ============================================================================
# Test Class: Consistency Between Methods
# ============================================================================

class TestConsistency:
    """Test consistency between calculate() and calculate_bulk()."""

    @staticmethod
    def _assert_features_match(single_row, bulk_row, feature_names, label):
        """
        Compare one row from calculate() against one row from calculate_bulk().

        Known divergence on *_count for collapsed windows:
          calculate()      → count = 0   (explicit, no valid window)
          calculate_bulk() → count = NaN (rolling+shift puts NaN before data starts)
        Both correctly mean "no usable observations", so we treat 0 ≡ NaN for count.
        mean/sum/std must agree exactly (rel=1e-6).
        """
        signal_cols = [f for f in feature_names if not f.endswith('_count')]
        count_cols  = [f for f in feature_names if f.endswith('_count')]

        for col in signal_cols:
            s_val = single_row[col]
            b_val = bulk_row[col]
            if pd.isna(s_val):
                assert pd.isna(b_val), f"{label} {col}: single=NaN but bulk={b_val}"
            else:
                assert b_val == pytest.approx(s_val, rel=1e-6), (
                    f"{label} {col}: single={s_val:.6f}, bulk={b_val:.6f}"
                )

        for col in count_cols:
            s_val = single_row[col]
            b_val = bulk_row[col]
            # Treat (bulk=NaN, single=0) as equivalent — both mean "collapsed window"
            s_zero_or_nan = pd.isna(s_val) or s_val == 0
            b_zero_or_nan = pd.isna(b_val) or b_val == 0
            if b_zero_or_nan:
                assert s_zero_or_nan, (
                    f"{label} {col}: bulk={b_val} (no window) but single={s_val}"
                )
            elif pd.isna(s_val):
                assert pd.isna(b_val), f"{label} {col}: single=NaN but bulk={b_val}"
            else:
                assert b_val == pytest.approx(s_val, rel=1e-6), (
                    f"{label} {col}: single={s_val}, bulk={b_val}"
                )

    def test_calculate_vs_bulk_single_date_all_tickers(
        self, momentum_calculator, feature_context, sample_straddle_history
    ):
        """
        Cross-validate calculate() vs calculate_bulk() for all 4 tickers at one date.

        Uses 2019-07-26 (position 29) — well after the UBER IPO so all tickers
        have at least some valid returns in their window.

        Verifies that both methods agree on every (ticker, feature) pair
        at this single date.
        """
        # Arrange
        calc_date   = pd.Timestamp('2019-07-26')
        all_tickers = sorted(sample_straddle_history['ticker'].unique().tolist())
        feature_names = momentum_calculator.feature_names

        # Act
        single = (
            momentum_calculator
            .calculate(feature_context, calc_date, all_tickers)
            .sort_values('ticker')
            .reset_index(drop=True)
        )
        bulk = (
            momentum_calculator
            .calculate_bulk(feature_context, calc_date, calc_date, all_tickers)
            .sort_values('ticker')
            .reset_index(drop=True)
        )

        # Assert same row count (one per ticker)
        assert len(single) == len(all_tickers)
        assert len(bulk)   == len(all_tickers)

        for i in range(len(all_tickers)):
            ticker = single.iloc[i]['ticker']
            self._assert_features_match(
                single.iloc[i], bulk.iloc[i], feature_names,
                label=f"ticker={ticker}"
            )

    def test_calculate_vs_bulk_full_fixture(
        self, momentum_calculator, feature_context, sample_straddle_history
    ):
        """
        Exhaustive cross-validation: calculate() vs calculate_bulk() for
        every date × every ticker in the entire fixture (~52 dates × 4 tickers).

        This is the strongest consistency check — it exercises:
          - AAPL / TSLA (all valid returns throughout the year)
          - ADP (scattered NaN returns in many windows)
          - UBER (all-NaN pre-IPO windows, valid returns from week 22 onward)
          - Early dates with collapsed / partial windows
          - Full windows in the middle and late year

        mean, sum, std must match to rel=1e-6.
        count: calculate()=0 and bulk()=NaN are treated as equivalent
               (both signal "no valid window"; see _assert_features_match).
        """
        # Arrange: pull every unique date and ticker directly from the fixture
        all_dates   = sorted(sample_straddle_history['entry_date'].unique())
        all_tickers = sorted(sample_straddle_history['ticker'].unique().tolist())
        start_date  = pd.Timestamp(all_dates[0])
        end_date    = pd.Timestamp(all_dates[-1])
        feature_names = momentum_calculator.feature_names

        # Build ground-truth row-by-row via calculate()
        single_rows = []
        for d in all_dates:
            rows = momentum_calculator.calculate(
                feature_context, pd.Timestamp(d), all_tickers
            )
            single_rows.append(rows)
        single_df = (
            pd.concat(single_rows, ignore_index=True)
            .sort_values(['ticker', 'date'])
            .reset_index(drop=True)
        )

        # One vectorised call covering the entire fixture range
        bulk_df = (
            momentum_calculator
            .calculate_bulk(feature_context, start_date, end_date, all_tickers)
            .sort_values(['ticker', 'date'])
            .reset_index(drop=True)
        )

        # Assert same row count
        assert len(bulk_df) == len(single_df), (
            f"Row count mismatch: bulk={len(bulk_df)}, single={len(single_df)}"
        )

        # Assert every (ticker, date, feature) triple agrees
        for i in range(len(bulk_df)):
            ticker = bulk_df.iloc[i]['ticker']
            date   = bulk_df.iloc[i]['date']
            self._assert_features_match(
                single_df.iloc[i], bulk_df.iloc[i], feature_names,
                label=f"ticker={ticker} date={date.date()}"
            )

    def test_bulk_with_ticker_filter_matches_all_tickers(
        self, momentum_calculator, feature_context, sample_straddle_history
    ):
        """
        Verify that an explicit ticker filter in calculate_bulk() returns
        the same values as running without a filter and subsetting afterward.

        This confirms the per-ticker groupby rolling is not affected by which
        other tickers are present in the DataFrame.

        Given:
        - calculate_bulk(tickers=['AAPL', 'TSLA']) over weeks 20-30
        - calculate_bulk(tickers=None) over same range, then filtered to AAPL+TSLA

        Expected: identical DataFrames after sorting.
        """
        # Arrange
        start_date    = pd.Timestamp('2019-05-17')
        end_date      = pd.Timestamp('2019-07-26')
        feature_names = momentum_calculator.feature_names
        subset        = ['AAPL', 'TSLA']

        # Act
        filtered = (
            momentum_calculator
            .calculate_bulk(feature_context, start_date, end_date, subset)
            .sort_values(['ticker', 'date'])
            .reset_index(drop=True)
        )
        unfiltered_subset = (
            momentum_calculator
            .calculate_bulk(feature_context, start_date, end_date)
            [lambda df: df['ticker'].isin(subset)]
            .sort_values(['ticker', 'date'])
            .reset_index(drop=True)
        )

        # Assert same shape
        assert filtered.shape == unfiltered_subset.shape

        # Assert every cell matches
        for i in range(len(filtered)):
            ticker = filtered.iloc[i]['ticker']
            date   = filtered.iloc[i]['date']
            self._assert_features_match(
                filtered.iloc[i], unfiltered_subset.iloc[i], feature_names,
                label=f"ticker={ticker} date={date.date()}"
            )


# ============================================================================
# Test Class: Edge Cases & Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_ticker_list(self, momentum_calculator, feature_context):
        """
        Test calculation with empty ticker list.

        Expected: Returns empty DataFrame with no rows.
        """
        # Act
        result = momentum_calculator.calculate(
            feature_context, pd.Timestamp('2019-05-17'), []
        )

        # Assert
        assert len(result) == 0

    def test_date_before_all_history(self, momentum_calculator, feature_context):
        """
        Test calculation at date before any history exists.

        Given:
        - Fixture history starts 2019-01-04
        - Request calculation at 2018-01-01

        Expected:
        - History filtered to date <= 2018-01-01 is empty
        - Returns 1 row per requested ticker, all features NaN
        """
        # Arrange
        calc_date = pd.Timestamp('2018-01-01')

        # Act
        result = momentum_calculator.calculate(feature_context, calc_date, ['AAPL'])

        # Assert
        assert len(result) == 1
        assert result.iloc[0]['ticker'] == 'AAPL'
        for col in momentum_calculator.feature_names:
            assert pd.isna(result.iloc[0][col])

    def test_single_return_in_window(self, momentum_calculator_no_min, feature_context):
        """
        Test window with exactly 1 valid return (when min_periods=1).

        Given:
        - AAPL at position 2 (2019-01-18) with min_periods=1
        - Window (12, 2): start=max(0, 2-12)=0, end=2-2=0
          → end(0) < start(0) is False → iloc[0:1] = exactly 1 row
        - AAPL row 0 (2019-01-04) return_pct = -18.96869244935543

        Expected:
        - count == 1
        - mean == sum == that single return value
        - std == 0.0 (undefined for 1 value; implementation returns 0.0)
        """
        # Arrange: position 2 in AAPL history; window collapses to exactly 1 prior row
        calc_date = pd.Timestamp('2019-01-18')

        # Act
        result = momentum_calculator_no_min.calculate(feature_context, calc_date, ['AAPL'])

        # Assert
        assert len(result) == 1
        row = result.iloc[0]
        assert row['mom_12_2_count'] == 1
        assert row['mom_12_2_mean'] == pytest.approx(-18.96869244935543, rel=1e-5)
        assert row['mom_12_2_sum']  == pytest.approx(-18.96869244935543, rel=1e-5)
        assert row['mom_12_2_std']  == pytest.approx(0.0)

    def test_all_returns_identical(self, momentum_calculator):
        """
        Test window with all identical returns.

        Given: Returns [10, 10, 10, 10, 10]
        Expected:
        - mean == 10.0, sum == 50.0, count == 5
        - std == 0.0 (zero variance when all values are equal)
        """
        # Arrange
        window_data = pd.DataFrame({'return_pct': [10.0, 10.0, 10.0, 10.0, 10.0]})

        # Act
        result = momentum_calculator._calculate_window_features(window_data, 'mom_12_2')

        # Assert
        assert result['mom_12_2_mean']  == pytest.approx(10.0)
        assert result['mom_12_2_sum']   == pytest.approx(50.0)
        assert result['mom_12_2_count'] == 5
        assert result['mom_12_2_std']   == pytest.approx(0.0)

    def test_very_large_returns(self, momentum_calculator):
        """
        Test calculation with very large return values.

        Verifies numerical stability — no overflow or unexpected NaN.

        Given: Returns [10000, 20000, 30000, 40000, 50000]
        Expected: mean=30000, sum=150000, std non-NaN
        """
        # Arrange
        window_data = pd.DataFrame(
            {'return_pct': [10000.0, 20000.0, 30000.0, 40000.0, 50000.0]}
        )

        # Act
        result = momentum_calculator._calculate_window_features(window_data, 'mom_12_2')

        # Assert
        assert result['mom_12_2_mean']  == pytest.approx(30000.0)
        assert result['mom_12_2_sum']   == pytest.approx(150000.0)
        assert result['mom_12_2_count'] == 5
        assert not np.isnan(result['mom_12_2_std'])


