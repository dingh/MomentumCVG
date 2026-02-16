"""
Unit tests for StraddleHistoryBuilder.

Tests cover:
- Liquidity metrics recording
- Expiry selection logic (weekly and monthly)
- Process single straddle workflow (integration tests)
"""

import pytest
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import Mock, MagicMock, patch

from src.features.straddle_analyzer import StraddleHistoryBuilder
from src.core.models import OptionQuote, OptionStrategy, OptionLeg, Position
from src.data.spot_price_db import SpotPriceDB


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_call_quote():
    """Create sample call option quote."""
    return OptionQuote(
        ticker='AAPL',
        trade_date=date(2024, 1, 5),
        expiry_date=date(2024, 1, 12),
        strike=Decimal('150.0'),
        option_type='call',
        bid=Decimal('3.80'),
        ask=Decimal('4.20'),
        mid=Decimal('4.00'),
        iv=0.25,
        delta=0.52,
        gamma=0.03,
        vega=0.15,
        theta=-0.05,
        volume=500,
        open_interest=1000
    )


@pytest.fixture
def sample_put_quote():
    """Create sample put option quote."""
    return OptionQuote(
        ticker='AAPL',
        trade_date=date(2024, 1, 5),
        expiry_date=date(2024, 1, 12),
        strike=Decimal('150.0'),
        option_type='put',
        bid=Decimal('3.60'),
        ask=Decimal('4.00'),
        mid=Decimal('3.80'),
        iv=0.26,
        delta=-0.48,
        gamma=0.03,
        vega=0.15,
        theta=-0.05,
        volume=450,
        open_interest=900
    )


@pytest.fixture
def mock_spot_db():
    """Create mock SpotPriceDB."""
    mock_db = Mock(spec=SpotPriceDB)
    mock_db.calculate_spot_move_pct.return_value = 0.0235  # 2.35%
    mock_db.calculate_realized_volatility.return_value = 0.18  # 18%
    return mock_db


@pytest.fixture
def straddle_builder(mock_spot_db):
    """Create StraddleHistoryBuilder instance."""
    return StraddleHistoryBuilder(
        data_root='c:/ORATS/data/ORATS_Adjusted',
        spot_db=mock_spot_db,
        dte_target=7,
        max_spread_pct=0.50,
        min_volume=10,
        min_oi=0
    )


# ============================================================================
# Test Class: Liquidity Metrics
# ============================================================================

class TestLiquidityMetrics:
    """Test liquidity metrics recording."""
    
    def test_record_liquidity_metrics_calculation(self, straddle_builder, sample_call_quote, sample_put_quote):
        """
        Test that liquidity metrics are correctly calculated.
        
        Verifies:
        - Spread percentages calculated correctly
        - Volume and OI recorded
        - Average spread calculated
        """
        # Arrange
        # Call: bid=3.80, ask=4.20, mid=4.00, spread=0.40 (10%)
        # Put: bid=3.60, ask=4.00, mid=3.80, spread=0.40 (10.5%)
        
        # Act
        metrics = straddle_builder.record_liquidity_metrics(
            sample_call_quote, sample_put_quote
        )
        
        # Assert
        assert 'call_spread_pct' in metrics
        assert 'put_spread_pct' in metrics
        assert 'avg_spread_pct' in metrics
        assert 'call_volume' in metrics
        assert 'put_volume' in metrics
        assert 'call_open_interest' in metrics
        assert 'put_open_interest' in metrics
        
        # Verify spread calculations
        expected_call_spread = (4.20 - 3.80) / 4.00  # 10%
        expected_put_spread = (4.00 - 3.60) / 3.80   # 10.5%
        
        assert metrics['call_spread_pct'] == pytest.approx(expected_call_spread, rel=1e-4)
        assert metrics['put_spread_pct'] == pytest.approx(expected_put_spread, rel=1e-4)
        assert metrics['avg_spread_pct'] == pytest.approx((expected_call_spread + expected_put_spread) / 2, rel=1e-4)
        
        # Verify volume and OI
        assert metrics['call_volume'] == 500
        assert metrics['put_volume'] == 450
        assert metrics['call_open_interest'] == 1000
        assert metrics['put_open_interest'] == 900
    
    def test_record_liquidity_metrics_zero_premium(self, straddle_builder, sample_call_quote):
        """
        Test handling of zero premium (edge case).
        
        Verifies spread_pct = 999.0 when mid = 0.
        """
        # Arrange
        zero_premium_put = OptionQuote(
            ticker='AAPL',
            trade_date=date(2024, 1, 5),
            expiry_date=date(2024, 1, 12),
            strike=Decimal('150.0'),
            option_type='put',
            bid=Decimal('0.0'),
            ask=Decimal('0.0'),
            mid=Decimal('0.0'),
            iv=0.0,
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            volume=0,
            open_interest=0
        )
        
        # Act
        metrics = straddle_builder.record_liquidity_metrics(
            sample_call_quote, zero_premium_put
        )
        
        # Assert
        assert metrics['put_spread_pct'] == 999.0


# ============================================================================
# Test Class: Expiry Selection - Weekly
# ============================================================================

class TestExpirySelectionWeekly:
    """Test _find_best_expiry() for weekly strategies (7 DTE)."""
    
    def test_find_best_expiry_exact_match_friday(self, straddle_builder):
        """
        Test finding expiry when exact 7 DTE Friday exists.
        
        Verifies correct expiry selected when perfect match available.
        """
        # Arrange
        trade_date = date(2024, 1, 5)  # Friday
        # 7 days later = 2024-01-12 (also Friday)
        available_expiries = [
            date(2024, 1, 12),  # 7 DTE - Friday (exact match)
            date(2024, 1, 19),  # 14 DTE
            date(2024, 1, 26),  # 21 DTE
        ]
        
        straddle_builder.provider = Mock()
        straddle_builder.provider.get_available_expiries.return_value = available_expiries
        
        # Act
        result = straddle_builder._find_best_expiry('AAPL', trade_date, target_dte=7)
        
        # Assert
        assert result == date(2024, 1, 12)
        assert result.weekday() == 4  # Friday
    
    def test_find_best_expiry_closest_friday(self, straddle_builder):
        """
        Test finding closest Friday when no exact match.
        
        Verifies preference for Friday within tolerance.
        """
        # Arrange
        trade_date = date(2024, 1, 5)  # Friday
        available_expiries = [
            date(2024, 1, 11),  # 6 DTE - Thursday
            date(2024, 1, 12),  # 7 DTE - Friday
            date(2024, 1, 13),  # 8 DTE - Saturday (won't be selected)
        ]
        
        straddle_builder.provider = Mock()
        straddle_builder.provider.get_available_expiries.return_value = available_expiries
        
        # Act
        result = straddle_builder._find_best_expiry('AAPL', trade_date, target_dte=7)
        
        # Assert
        assert result == date(2024, 1, 12)  # Prefers Friday over Thursday
        assert result.weekday() == 4
    
    def test_find_best_expiry_thursday_fallback(self, straddle_builder):
        """
        Test fallback to Thursday when no Friday available.
        
        Verifies Thursday is acceptable alternative.
        """
        # Arrange
        trade_date = date(2024, 1, 5)
        available_expiries = [
            date(2024, 1, 11),  # 6 DTE - Thursday
            date(2024, 1, 18),  # 13 DTE - Thursday (too far)
        ]
        
        straddle_builder.provider = Mock()
        straddle_builder.provider.get_available_expiries.return_value = available_expiries
        
        # Act
        result = straddle_builder._find_best_expiry('AAPL', trade_date, target_dte=7)
        
        # Assert
        assert result == date(2024, 1, 11)  # Thursday acceptable
        assert result.weekday() == 3
    
    def test_find_best_expiry_no_valid_expiry(self, straddle_builder):
        """
        Test returns None when no expiry within tolerance.
        
        Verifies None returned when all expiries too far.
        """
        # Arrange
        trade_date = date(2024, 1, 5)
        available_expiries = [
            date(2024, 1, 26),  # 21 DTE (too far, >7+4=11)
            date(2024, 2, 2),   # 28 DTE
        ]
        
        straddle_builder.provider = Mock()
        straddle_builder.provider.get_available_expiries.return_value = available_expiries
        
        # Act
        result = straddle_builder._find_best_expiry('AAPL', trade_date, target_dte=7, tolerance_days=4)
        
        # Assert
        assert result is None
    
    def test_find_best_expiry_empty_list(self, straddle_builder):
        """
        Test returns None when no expiries available.
        
        Verifies graceful handling of missing data.
        """
        # Arrange
        straddle_builder.provider = Mock()
        straddle_builder.provider.get_available_expiries.return_value = []
        
        # Act
        result = straddle_builder._find_best_expiry('AAPL', date(2024, 1, 5), target_dte=7)
        
        # Assert
        assert result is None


# ============================================================================
# Test Class: Process Single Straddle (Integration Tests)
# ============================================================================

class TestProcessSingleStraddle:
    """Integration tests for process_single_straddle() workflow."""
    
    def test_process_straddle_no_spot_price_at_entry(self, straddle_builder):
        """
        Test handling when spot price missing at entry.
        
        Verifies failure_reason set correctly.
        """
        # Arrange
        straddle_builder.provider = Mock()
        straddle_builder.provider.get_spot_price.return_value = None
        
        # Act
        result = straddle_builder.process_single_straddle('AAPL', date(2024, 1, 5))
        
        # Assert
        assert result['failure_reason'] == 'no_spot_price'
        assert result['entry_spot'] is None
        assert 'processing_time' in result
    
    def test_process_straddle_no_expiry_found(self, straddle_builder):
        """
        Test handling when no valid expiry found.
        
        Verifies failure_reason set correctly.
        """
        # Arrange
        straddle_builder.provider = Mock()
        straddle_builder.provider.get_spot_price.return_value = Decimal('150.0')
        straddle_builder.provider.get_available_expiries.return_value = []
        
        # Act
        result = straddle_builder.process_single_straddle('AAPL', date(2024, 1, 5))
        
        # Assert
        assert result['failure_reason'] == 'no_expiry_found'
        assert result['entry_spot'] == 150.0
        assert result['expiry_date'] is None
    
    def test_process_straddle_no_options_at_entry(self, straddle_builder):
        """
        Test handling when option chain is empty.
        
        Verifies failure_reason set correctly.
        """
        # Arrange
        straddle_builder.provider = Mock()
        straddle_builder.provider.get_spot_price.return_value = Decimal('150.0')
        straddle_builder.provider.get_available_expiries.return_value = [date(2024, 1, 12)]
        straddle_builder.provider.get_option_chain.return_value = []
        
        # Act
        result = straddle_builder.process_single_straddle('AAPL', date(2024, 1, 5))
        
        # Assert
        assert result['failure_reason'] == 'no_options_at_entry'
    
    def test_process_straddle_build_strategy_failure(self, straddle_builder, sample_call_quote):
        """
        Test handling when StraddleBuilder raises ValueError.
        
        Verifies failure_reason contains error message.
        """
        # Arrange
        straddle_builder.provider = Mock()
        straddle_builder.provider.get_spot_price.return_value = Decimal('150.0')
        straddle_builder.provider.get_available_expiries.return_value = [date(2024, 1, 12)]
        straddle_builder.provider.get_option_chain.return_value = [sample_call_quote]
        
        straddle_builder.builder = Mock()
        straddle_builder.builder.build_strategy.side_effect = ValueError("No call at strike")
        
        # Act
        result = straddle_builder.process_single_straddle('AAPL', date(2024, 1, 5))
        
        # Assert
        assert 'build_failed' in result['failure_reason']
        assert 'No call at strike' in result['failure_reason']
    
    def test_process_straddle_no_spot_price_at_expiry(self, straddle_builder, sample_call_quote, sample_put_quote):
        """
        Test handling when spot price missing at expiry.
        
        Verifies processing continues until exit spot lookup.
        """
        # Arrange
        call_leg = OptionLeg(option=sample_call_quote, quantity=1)
        put_leg = OptionLeg(option=sample_put_quote, quantity=1)
        
        from src.core.models import StrategyType
        mock_strategy = OptionStrategy(
            ticker='AAPL',
            strategy_type=StrategyType.STRADDLE,
            legs=(call_leg, put_leg),
            trade_date=date(2024, 1, 5)
        )
        
        straddle_builder.provider = Mock()
        straddle_builder.provider.get_spot_price.side_effect = [
            Decimal('150.0'),  # Entry spot (success)
            None               # Exit spot (failure)
        ]
        straddle_builder.provider.get_available_expiries.return_value = [date(2024, 1, 12)]
        straddle_builder.provider.get_option_chain.return_value = [sample_call_quote, sample_put_quote]
        
        straddle_builder.builder = Mock()
        straddle_builder.builder.build_strategy.return_value = mock_strategy
        
        # Act
        result = straddle_builder.process_single_straddle('AAPL', date(2024, 1, 5))
        
        # Assert
        assert result['failure_reason'] == 'no_spot_price_at_expiry'
        assert result['entry_spot'] == 150.0
        assert result['entry_cost'] is not None  # Entry processed
        assert result['exit_spot'] is None
        assert result['pnl'] is None
