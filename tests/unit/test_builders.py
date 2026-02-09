"""
Unit tests for strategy builders.

Test Coverage:
- StraddleBuilder: ATM straddle construction from market data
- StraddleAnalyzer: Historical straddle analysis (TODO)

Testing Philosophy:
- Builders are PURE FUNCTIONS: deterministic, no side effects
- Test both happy paths and all error conditions
- Validate strike selection logic thoroughly (business-critical)
- Use real-world-like fixtures (spreads, strikes, greeks)
"""

import pytest
from datetime import date
from decimal import Decimal
from typing import List

from src.core.models import OptionQuote, OptionStrategy, OptionLeg, StrategyType
from src.strategy.builders import StraddleBuilder


# =============================================================================
# StraddleBuilder Tests
# =============================================================================

class TestStraddleBuilderHappyPath:
    """Test successful straddle construction with valid inputs."""
    
    def test_build_strategy_happy_path(
        self, 
        sample_option_chain_atm, 
        trade_date, 
        expiry_date, 
        ticker
    ):
        """
        Test that builder constructs a valid unit straddle from a clean option chain.
        
        Given:
        - Valid option chain with strikes 43.5/44.0/44.5/45.0 (both calls and puts)
        - Spot price = 44.50 (closest to strike 44.5)
        - Single expiry date matching trade date
        - All options have positive premiums, reasonable greeks
        
        When:
        - build_strategy() is called
        
        Then:
        - Returns OptionStrategy with:
            - strategy_type = StrategyType.STRADDLE
            - exactly 2 legs (call + put)
            - both legs at strike 44.5 (ATM)
            - both legs have quantity = 1 (unit strategy)
            - one leg is 'call', one is 'put'
            - ticker and trade_date match inputs
            - net_premium > 0 (sum of call and put premiums)
        
        This is the most important test - validates core builder functionality.
        """
        # Arrange
        builder = StraddleBuilder()
        spot_price = Decimal('44.50')  # Exactly at strike 44.5
        
        # Act
        strategy = builder.build_strategy(
            ticker=ticker,
            trade_date=trade_date,
            expiry_date=expiry_date,
            option_chain=sample_option_chain_atm,
            spot_price=spot_price
        )
        
        # Assert - Strategy metadata
        assert strategy.strategy_type == StrategyType.STRADDLE
        assert strategy.ticker == ticker
        assert strategy.trade_date == trade_date
        
        # Assert - Structure
        assert len(strategy.legs) == 2
        
        # Assert - Both legs at ATM strike
        atm_strike = Decimal('44.5')
        assert all(leg.option.strike == atm_strike for leg in strategy.legs)
        
        # Assert - Unit strategy (quantity = 1)
        assert all(leg.quantity == 1 for leg in strategy.legs)
        
        # Assert - One call, one put
        leg_types = {leg.option.option_type for leg in strategy.legs}
        assert leg_types == {'call', 'put'}
        
        # Assert - Positive net premium (paying for straddle)
        assert strategy.net_premium > 0
        
        # Assert - Net premium is sum of call and put
        call_leg = next(leg for leg in strategy.legs if leg.option.option_type == 'call')
        put_leg = next(leg for leg in strategy.legs if leg.option.option_type == 'put')
        expected_premium = call_leg.option.mid + put_leg.option.mid
        assert strategy.net_premium == expected_premium


class TestStraddleBuilderATMStrikeSelection:
    """Test _find_atm_strike() logic in isolation."""
    
    def test_find_atm_strike_exact_match(self, sample_option_chain_atm):
        """
        Test ATM strike selection when spot exactly equals a strike.
        
        Given:
        - Option chain with strikes [43.5, 44.0, 44.5, 45.0]
        - Spot price = Decimal('44.00') (exact match)
        
        When:
        - _find_atm_strike() is called
        
        Then:
        - Returns Decimal('44.00')
        
        Edge case: Ensures exact matches are preferred (distance = 0).
        """
        # Arrange
        builder = StraddleBuilder()
        spot_price = Decimal('44.00')
        
        # Act
        atm_strike = builder._find_atm_strike(sample_option_chain_atm, spot_price)
        
        # Assert
        assert atm_strike == Decimal('44.0')
    
    def test_find_atm_strike_closest(self, sample_option_chain_atm):
        """
        Test ATM strike selection when spot is between strikes.
        
        Given:
        - Option chain with strikes [43.5, 44.0, 44.5, 45.0]
        - Spot price = Decimal('44.25') (closer to 44.0 than 44.5)
        
        When:
        - _find_atm_strike() is called
        
        Then:
        - Returns Decimal('44.0') (closest strike)
        
        Validates: Euclidean distance minimization logic.
        """
        # Arrange
        builder = StraddleBuilder()
        spot_price = Decimal('44.25')  # 0.25 from 44.0, 0.25 from 44.5 (tie, lower wins)
        
        # Act
        atm_strike = builder._find_atm_strike(sample_option_chain_atm, spot_price)
        
        # Assert
        assert atm_strike == Decimal('44.0')  # Lower strike wins on tie
    
    def test_find_atm_strike_tie_lower_wins(self, sample_option_chain_atm):
        """
        Test tie-breaking rule: lower strike wins when equidistant.
        
        Given:
        - Option chain with strikes [43.5, 44.0, 44.5, 45.0]
        - Spot price = Decimal('44.75') (exactly mid-way between 44.5 and 45.0)
        
        When:
        - _find_atm_strike() is called
        
        Then:
        - Returns Decimal('44.5') (lower strike, not 45.0)
        
        Critical: Ensures deterministic behavior via min() key tuple (distance, strike).
        Standard market convention: round down on ties.
        """
        # Arrange
        builder = StraddleBuilder()
        spot_price = Decimal('44.75')  # Exactly midway: 0.25 from both 44.5 and 45.0
        
        # Act
        atm_strike = builder._find_atm_strike(sample_option_chain_atm, spot_price)
        
        # Assert
        assert atm_strike == Decimal('44.5')  # Lower strike wins
    
    def test_find_atm_strike_unsorted_input(self, sample_option_chain_unsorted):
        """
        Test that strike selection works with unsorted option chain.
        
        Given:
        - Option chain with strikes in random order: [44.0, 44.5, 43.5, 45.0, ...]
        - Spot price = Decimal('44.60')
        
        When:
        - _find_atm_strike() is called
        
        Then:
        - Returns Decimal('44.5') (correct ATM strike)
        
        Validates: Internal sorting logic handles arbitrary input order.
        Real-world data providers may not guarantee sorted output.
        """
        # Arrange
        builder = StraddleBuilder()
        spot_price = Decimal('44.60')  # Closest to 44.5
        
        # Act
        atm_strike = builder._find_atm_strike(sample_option_chain_unsorted, spot_price)
        
        # Assert
        assert atm_strike == Decimal('44.5')
    
    def test_find_atm_strike_empty_chain(self):
        """
        Test error handling when option chain contains no strikes.
        
        Given:
        - Empty option chain (no strikes available)
        
        When:
        - _find_atm_strike() is called
        
        Then:
        - Raises ValueError with message "No strikes found in option chain"
        
        Edge case: Protects against malformed data provider output.
        """
        # Arrange
        builder = StraddleBuilder()
        empty_chain = []
        spot_price = Decimal('44.00')
        
        # Act & Assert
        with pytest.raises(ValueError, match="No strikes found in option chain"):
            builder._find_atm_strike(empty_chain, spot_price)


class TestStraddleBuilderOptionLookup:
    """Test _get_option_at_strike() helper method."""
    
    def test_get_option_at_strike_found(self, sample_option_chain_atm):
        """
        Test that option lookup returns correct option when present.
        
        Given:
        - Option chain containing call and put at strike 44.5
        
        When:
        - _get_option_at_strike(chain, Decimal('44.5'), 'call') is called
        - _get_option_at_strike(chain, Decimal('44.5'), 'put') is called
        
        Then:
        - Returns OptionQuote with correct strike and option_type
        - Call and put are different objects
        
        Validates: Lookup logic correctly filters by strike AND option_type.
        """
        # Arrange
        builder = StraddleBuilder()
        target_strike = Decimal('44.5')
        
        # Act
        call_option = builder._get_option_at_strike(
            sample_option_chain_atm, 
            target_strike, 
            'call'
        )
        put_option = builder._get_option_at_strike(
            sample_option_chain_atm, 
            target_strike, 
            'put'
        )
        
        # Assert - Both found
        assert call_option is not None
        assert put_option is not None
        
        # Assert - Correct attributes
        assert call_option.strike == target_strike
        assert call_option.option_type == 'call'
        assert put_option.strike == target_strike
        assert put_option.option_type == 'put'
        
        # Assert - Different objects
        assert call_option is not put_option
    
    def test_get_option_at_strike_not_found(self, sample_option_chain_atm):
        """
        Test that option lookup returns None when option is missing.
        
        Given:
        - Option chain with strikes [43.5, 44.0, 44.5, 45.0] (missing 50.0)
        
        When:
        - _get_option_at_strike(chain, Decimal('50.0'), 'call') is called
        
        Then:
        - Returns None
        
        Edge case: Allows caller (build_strategy) to raise descriptive error.
        """
        # Arrange
        builder = StraddleBuilder()
        missing_strike = Decimal('50.0')
        
        # Act
        result = builder._get_option_at_strike(
            sample_option_chain_atm,
            missing_strike,
            'call'
        )
        
        # Assert
        assert result is None


class TestStraddleBuilderErrorHandling:
    """Test build_strategy() error cases with invalid inputs."""
    
    def test_build_strategy_empty_chain(self, trade_date, expiry_date, ticker):
        """
        Test error when option chain is empty.
        
        Given:
        - Empty option_chain list
        
        When:
        - build_strategy() is called
        
        Then:
        - Raises ValueError with message "Empty option chain for {ticker} on {trade_date}"
        
        Critical: Prevents silent failures on data feed outages.
        """
        # Arrange
        builder = StraddleBuilder()
        empty_chain = []
        spot_price = Decimal('44.00')
        
        # Act & Assert
        with pytest.raises(ValueError, match=f"Empty option chain for {ticker}"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=empty_chain,
                spot_price=spot_price
            )
    
    def test_build_strategy_multiple_expiries(
        self, 
        sample_option_chain_multiple_expiries,
        trade_date, 
        expiry_date, 
        ticker
    ):
        """
        Test error when option chain contains mixed expiry dates.
        
        Given:
        - Option chain with options expiring on 2024-12-06 AND 2024-12-13
        
        When:
        - build_strategy() is called with expiry_date=2024-12-06
        
        Then:
        - Raises ValueError with message "Option chain contains multiple expiries: ..."
        
        Design constraint: Builder expects pre-filtered chain for single expiry.
        Data provider should handle filtering.
        """
        # Arrange
        builder = StraddleBuilder()
        spot_price = Decimal('44.50')
        
        # Act & Assert
        with pytest.raises(ValueError, match="Option chain contains multiple expiries"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=sample_option_chain_multiple_expiries,
                spot_price=spot_price
            )
    
    def test_build_strategy_expiry_mismatch(
        self,
        sample_option_chain_atm,
        trade_date,
        ticker
    ):
        """
        Test error when option chain expiry doesn't match expected expiry.
        
        Given:
        - Option chain with all options expiring 2024-12-06
        - expiry_date argument = 2024-12-13 (mismatch)
        
        When:
        - build_strategy() is called
        
        Then:
        - Raises ValueError with message "Option chain expiry mismatch. Expected ..., got ..."
        
        Critical: Catches data provider logic errors (wrong expiry returned).
        """
        # Arrange
        builder = StraddleBuilder()
        spot_price = Decimal('44.50')
        wrong_expiry = date(2024, 12, 13)  # Chain has 2024-12-06
        
        # Act & Assert
        with pytest.raises(ValueError, match="Option chain expiry mismatch"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=wrong_expiry,
                option_chain=sample_option_chain_atm,
                spot_price=spot_price
            )
    
    def test_build_strategy_missing_call(
        self,
        sample_option_chain_missing_call,
        trade_date,
        expiry_date,
        ticker
    ):
        """
        Test error when call option is missing at ATM strike.
        
        Given:
        - Option chain has puts at strikes 43.5/44.0/44.5/45.0
        - Option chain is MISSING call at strike 44.5 (ATM)
        - Spot price = 44.50 (ATM strike = 44.5)
        
        When:
        - build_strategy() is called
        
        Then:
        - Raises ValueError with message "No call option found at ATM strike ..."
        
        Real-world cause: Illiquid options filtered out by data quality checks.
        """
        # Arrange
        builder = StraddleBuilder()
        spot_price = Decimal('44.50')  # ATM = 44.5, but call is missing
        
        # Act & Assert
        with pytest.raises(ValueError, match="No call option found at ATM strike"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=sample_option_chain_missing_call,
                spot_price=spot_price
            )
    
    def test_build_strategy_missing_put(
        self,
        sample_option_chain_missing_put,
        trade_date,
        expiry_date,
        ticker
    ):
        """
        Test error when put option is missing at ATM strike.
        
        Given:
        - Option chain has calls at strikes 43.5/44.0/44.5/45.0
        - Option chain is MISSING put at strike 44.5 (ATM)
        - Spot price = 44.50 (ATM strike = 44.5)
        
        When:
        - build_strategy() is called
        
        Then:
        - Raises ValueError with message "No put option found at ATM strike ..."
        
        Symmetric to test_build_strategy_missing_call.
        """
        # Arrange
        builder = StraddleBuilder()
        spot_price = Decimal('44.50')  # ATM = 44.5, but put is missing
        
        # Act & Assert
        with pytest.raises(ValueError, match="No put option found at ATM strike"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=sample_option_chain_missing_put,
                spot_price=spot_price
            )
    
    def test_build_strategy_invalid_call_premium(
        self,
        sample_option_chain_invalid_call_premium,
        trade_date,
        expiry_date,
        ticker
    ):
        """
        Test error when call option has non-positive premium (mid <= 0).
        
        Given:
        - Option chain has ATM call at 44.5 with mid = Decimal('0.00')
        - Option chain has valid ATM put with mid = Decimal('0.44')
        - Spot price = 44.50 selects strike 44.5 as ATM
        
        When:
        - build_strategy() is called
        
        Then:
        - Raises ValueError with message "Invalid call premium ... Check data quality filters."
        
        Real-world cause: Bad data (missing quotes, stale prices).
        Critical: Prevents trading with invalid prices.
        """
        # Arrange
        builder = StraddleBuilder()
        spot_price = Decimal('44.50')  # ATM = 44.5, call has mid=0
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid call premium.*Check data quality"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=sample_option_chain_invalid_call_premium,
                spot_price=spot_price
            )
    
    def test_build_strategy_invalid_put_premium(
        self,
        sample_option_chain_invalid_put_premium,
        trade_date,
        expiry_date,
        ticker
    ):
        """
        Test error when put option has non-positive premium (mid <= 0).
        
        Given:
        - Option chain has valid ATM call at 44.5 with mid = Decimal('0.40')
        - Option chain has ATM put at 44.5 with mid = Decimal('0.00') (invalid)
        - Spot price = 44.50 selects strike 44.5 as ATM
        
        When:
        - build_strategy() is called
        
        Then:
        - Raises ValueError with message "Invalid put premium ... Check data quality filters."
        
        Symmetric to test_build_strategy_invalid_call_premium.
        Edge case: Zero mid can occur with missing/stale data.
        """
        # Arrange
        builder = StraddleBuilder()
        spot_price = Decimal('44.50')  # ATM = 44.5, put has mid=0
        
        # Act & Assert
        with pytest.raises(ValueError, match="Invalid put premium.*Check data quality"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=sample_option_chain_invalid_put_premium,
                spot_price=spot_price
            )


# =============================================================================
# StraddleAnalyzer Tests (TODO)
# =============================================================================

class TestStraddleAnalyzer:
    """
    Tests for StraddleAnalyzer (historical straddle analysis).
    
    TODO: Implement after StraddleBuilder tests are complete.
    """
    pass
