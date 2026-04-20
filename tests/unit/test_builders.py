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
from src.strategy.builders import StraddleBuilder, IronButterflyBuilder, IronButterflyCandidate


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


# =============================================================================
# IronButterflyBuilder Tests
# =============================================================================


class TestIronButterflyBuilderInit:
    """
    Test __init__() parameter validation.

    IronButterflyBuilder accepts three tunable thresholds. Each has a valid
    range; values outside that range must raise ValueError immediately so
    mis-configured builders are caught at construction time rather than
    silently producing bad strategies at runtime.
    """

    def test_init_stores_valid_defaults(self):
        """
        When no arguments are passed the builder stores the documented defaults.

        Purpose: Confirms that wing_delta=0.15, max_spread_cost_ratio=0.25, and
        min_yield_on_capital=0.05 are the shipped defaults and that they are
        accessible as instance attributes.
        """
        builder = IronButterflyBuilder()

        assert builder.wing_delta == 0.15
        assert builder.max_spread_cost_ratio == 0.25
        assert builder.min_yield_on_capital == 0.05

    def test_init_stores_custom_params(self):
        """
        Custom valid arguments are stored on the instance without modification.

        Purpose: Ensures the builder does not clamp, round, or transform
        valid user-supplied values.
        """
        builder = IronButterflyBuilder(wing_delta=0.20, max_spread_cost_ratio=0.10, min_yield_on_capital=0.08)

        assert builder.wing_delta == 0.20
        assert builder.max_spread_cost_ratio == 0.10
        assert builder.min_yield_on_capital == 0.08

    def test_init_wing_delta_zero_raises(self):
        """
        wing_delta=0 is rejected because a zero-delta 'wing' is ATM, not OTM.

        Purpose: Guards against a degenerate configuration that would place
        the long wing at the body strike, collapsing the strategy.
        Expected: ValueError matching 'wing_delta'.
        """
        with pytest.raises(ValueError, match="wing_delta"):
            IronButterflyBuilder(wing_delta=0.0)

    def test_init_wing_delta_at_or_above_half_raises(self):
        """
        wing_delta >= 0.5 is rejected because the long wing would be ITM or
        at-the-money, which inverts the wing/body relationship.

        Purpose: Guards against configurations where the 'wing' delta is as
        large or larger than the ATM delta.
        Expected: ValueError matching 'wing_delta' for both 0.5 and 0.6.
        """
        with pytest.raises(ValueError, match="wing_delta"):
            IronButterflyBuilder(wing_delta=0.5)

        with pytest.raises(ValueError, match="wing_delta"):
            IronButterflyBuilder(wing_delta=0.6)

    def test_init_wing_delta_negative_raises(self):
        """
        Negative wing_delta has no financial meaning in this context.

        Purpose: Input sanitisation — the caller should always pass an
        absolute (positive) delta target.
        Expected: ValueError matching 'wing_delta'.
        """
        with pytest.raises(ValueError, match="wing_delta"):
            IronButterflyBuilder(wing_delta=-0.10)

    def test_init_max_spread_cost_ratio_zero_raises(self):
        """
        max_spread_cost_ratio=0 means every option would fail the spread filter,
        making the builder permanently unusable.

        Purpose: Prevents a configuration that can never produce a valid
        strategy.
        Expected: ValueError matching 'max_spread_cost_ratio'.
        """
        with pytest.raises(ValueError, match="max_spread_cost_ratio"):
            IronButterflyBuilder(max_spread_cost_ratio=0.0)

    def test_init_min_yield_on_capital_negative_raises(self):
        """
        A negative yield threshold is financially meaningless — any credit
        strategy would trivially pass.

        Purpose: Input sanitisation; min_yield_on_capital must be >= 0.
        Expected: ValueError matching 'min_yield_on_capital'.
        """
        with pytest.raises(ValueError, match="min_yield_on_capital"):
            IronButterflyBuilder(min_yield_on_capital=-0.01)


class TestIronButterflyBuilderHappyPath:
    """
    Test successful iron butterfly construction with a well-formed chain.

    All tests use sample_ibf_chain_atm (AAPL, 2026-02-13, body=255.0,
    single symmetric wing pair at ±10.0, tight spreads, positive net credit).
    Use ibf_ticker / ibf_trade_date / ibf_expiry_date / ibf_spot_price fixtures.
    """

    def test_build_strategy_returns_iron_butterfly_type(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date, ibf_ticker, ibf_spot_price
    ):
        """
        The returned OptionStrategy has strategy_type == IRON_BUTTERFLY.

        Purpose: Verifies the builder stamps the correct StrategyType enum
        value, which is used downstream to dispatch expiry/P&L logic.
        """
        # Arrange
        builder = IronButterflyBuilder(wing_delta=0.17, max_spread_cost_ratio=0.25, min_yield_on_capital=0.05)

        # Act
        strategy = builder.build_strategy(
            ticker=ibf_ticker,
            trade_date=ibf_trade_date,
            expiry_date=ibf_expiry_date,
            option_chain=sample_ibf_chain_atm,
            spot_price=ibf_spot_price,
        )

        # Assert
        assert strategy.strategy_type == StrategyType.IRON_BUTTERFLY

    def test_build_strategy_has_four_legs(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date, ibf_ticker, ibf_spot_price
    ):
        """
        The strategy always contains exactly 4 OptionLeg objects.

        Purpose: An iron butterfly is a 4-leg structure by definition.
        Any other count signals a construction bug.
        """
        # Arrange
        builder = IronButterflyBuilder(wing_delta=0.17, max_spread_cost_ratio=0.25, min_yield_on_capital=0.05)

        # Act
        strategy = builder.build_strategy(
            ticker=ibf_ticker,
            trade_date=ibf_trade_date,
            expiry_date=ibf_expiry_date,
            option_chain=sample_ibf_chain_atm,
            spot_price=ibf_spot_price,
        )

        # Assert
        assert len(strategy.legs) == 4

    def test_build_strategy_leg_order_and_quantities(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date, ibf_ticker, ibf_spot_price
    ):
        """
        Legs are ordered [long put wing, short put body, short call body,
        long call wing] with quantities [+1, -1, -1, +1].

        Purpose: Downstream P&L and greek aggregation code relies on a
        stable leg order.  quantity sign drives credit/debit accounting in
        OptionLeg.net_premium.
        """
        # Arrange
        builder = IronButterflyBuilder(wing_delta=0.17, max_spread_cost_ratio=0.25, min_yield_on_capital=0.05)

        # Act
        strategy = builder.build_strategy(
            ticker=ibf_ticker,
            trade_date=ibf_trade_date,
            expiry_date=ibf_expiry_date,
            option_chain=sample_ibf_chain_atm,
            spot_price=ibf_spot_price,
        )

        # Assert - quantities
        assert [leg.quantity for leg in strategy.legs] == [1, -1, -1, 1]

        # Assert - option types match expected leg roles
        assert strategy.legs[0].option.option_type == 'put'   # long put wing
        assert strategy.legs[1].option.option_type == 'put'   # short put body
        assert strategy.legs[2].option.option_type == 'call'  # short call body
        assert strategy.legs[3].option.option_type == 'call'  # long call wing

    def test_build_strategy_body_at_atm_strike(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date, ibf_ticker, ibf_spot_price
    ):
        """
        Both short (body) legs are at the ATM strike (closest to spot_price).

        Purpose: The 'body' of the iron butterfly must sit at-the-money.
        If the body drifts away from ATM the max-profit point moves away
        from the current price.
        """
        # Arrange
        builder = IronButterflyBuilder(wing_delta=0.17, max_spread_cost_ratio=0.25, min_yield_on_capital=0.05)
        expected_body_strike = Decimal("255.0")

        # Act
        strategy = builder.build_strategy(
            ticker=ibf_ticker,
            trade_date=ibf_trade_date,
            expiry_date=ibf_expiry_date,
            option_chain=sample_ibf_chain_atm,
            spot_price=ibf_spot_price,
        )

        # Assert - both short legs sit at ATM
        assert strategy.legs[1].option.strike == expected_body_strike  # short put body
        assert strategy.legs[2].option.strike == expected_body_strike  # short call body

    def test_build_strategy_wings_equidistant(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date, ibf_ticker, ibf_spot_price
    ):
        """
        |body_strike - call_wing_strike| == |body_strike - put_wing_strike|.

        Purpose: The iron butterfly definition requires symmetric wings.
        Asymmetric wings would produce a skewed risk profile and incorrect
        capital / yield calculations downstream.
        """
        # Arrange
        builder = IronButterflyBuilder(wing_delta=0.17, max_spread_cost_ratio=0.25, min_yield_on_capital=0.05)

        # Act
        strategy = builder.build_strategy(
            ticker=ibf_ticker,
            trade_date=ibf_trade_date,
            expiry_date=ibf_expiry_date,
            option_chain=sample_ibf_chain_atm,
            spot_price=ibf_spot_price,
        )

        # Assert
        body_strike = strategy.legs[1].option.strike
        put_wing_strike = strategy.legs[0].option.strike
        call_wing_strike = strategy.legs[3].option.strike
        assert abs(body_strike - call_wing_strike) == abs(body_strike - put_wing_strike)

    def test_build_strategy_call_wing_above_body(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date, ibf_ticker, ibf_spot_price
    ):
        """
        Long call wing strike > body strike (OTM call).

        Purpose: Verifies spatial correctness of the structure — the call
        wing must cap the upside loss, so it must be above the body.
        """
        # Arrange
        builder = IronButterflyBuilder(wing_delta=0.17, max_spread_cost_ratio=0.25, min_yield_on_capital=0.05)

        # Act
        strategy = builder.build_strategy(
            ticker=ibf_ticker,
            trade_date=ibf_trade_date,
            expiry_date=ibf_expiry_date,
            option_chain=sample_ibf_chain_atm,
            spot_price=ibf_spot_price,
        )

        # Assert
        body_strike = strategy.legs[2].option.strike      # short call body
        call_wing_strike = strategy.legs[3].option.strike  # long call wing
        assert call_wing_strike > body_strike

    def test_build_strategy_put_wing_below_body(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date, ibf_ticker, ibf_spot_price
    ):
        """
        Long put wing strike < body strike (OTM put).

        Purpose: Symmetric to the call wing check; the put wing must cap
        the downside loss so it must be below the body.
        """
        # Arrange
        builder = IronButterflyBuilder(wing_delta=0.17, max_spread_cost_ratio=0.25, min_yield_on_capital=0.05)

        # Act
        strategy = builder.build_strategy(
            ticker=ibf_ticker,
            trade_date=ibf_trade_date,
            expiry_date=ibf_expiry_date,
            option_chain=sample_ibf_chain_atm,
            spot_price=ibf_spot_price,
        )

        # Assert
        body_strike = strategy.legs[1].option.strike     # short put body
        put_wing_strike = strategy.legs[0].option.strike  # long put wing
        assert put_wing_strike < body_strike

    def test_build_strategy_net_premium_is_negative(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date, ibf_ticker, ibf_spot_price
    ):
        """
        strategy.net_premium < 0, indicating a net credit is received.

        Purpose: An iron butterfly is always entered for a credit (the
        short body premiums exceed the long wing premiums).  A positive
        net_premium would mean the strategy costs money — a silent data
        or logic error.
        """
        # Arrange
        builder = IronButterflyBuilder(wing_delta=0.17, max_spread_cost_ratio=0.25, min_yield_on_capital=0.05)

        # Act
        strategy = builder.build_strategy(
            ticker=ibf_ticker,
            trade_date=ibf_trade_date,
            expiry_date=ibf_expiry_date,
            option_chain=sample_ibf_chain_atm,
            spot_price=ibf_spot_price,
        )

        # Assert
        assert strategy.net_premium < 0

    def test_build_strategy_net_premium_matches_arithmetic(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date, ibf_ticker, ibf_spot_price
    ):
        """
        strategy.net_premium == -(body_call_mid + body_put_mid - wing_call_mid - wing_put_mid).

        From sample_ibf_chain_atm.csv (AAPL, body=255.0, wings at 245/265):
          body_call_mid = 4.65  (255 call)
          body_put_mid  = 3.65  (255 put)
          wing_call_mid = 0.87  (265 call)
          wing_put_mid  = 1.19  (245 put)
          net_credit    = (4.65 + 3.65) - (0.87 + 1.19) = 6.24
          strategy.net_premium = Decimal('-6.24')

        Purpose: Regression anchor for the Decimal arithmetic path through
        build_strategy(). Hard-coded expected value catches any rounding or
        sign-flip regressions after refactors.
        """
        # Arrange
        builder = IronButterflyBuilder(wing_delta=0.17, max_spread_cost_ratio=0.25, min_yield_on_capital=0.05)

        # Act
        strategy = builder.build_strategy(
            ticker=ibf_ticker,
            trade_date=ibf_trade_date,
            expiry_date=ibf_expiry_date,
            option_chain=sample_ibf_chain_atm,
            spot_price=ibf_spot_price,
        )

        # Assert
        assert strategy.net_premium == Decimal('-6.24')

    def test_build_strategy_propagates_ticker_and_trade_date(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date, ibf_ticker, ibf_spot_price
    ):
        """
        strategy.ticker and strategy.trade_date match the inputs passed to
        build_strategy().

        Purpose: Metadata must survive construction unchanged so that
        downstream position tracking can identify which instrument and date
        the strategy was priced on.
        """
        # Arrange
        builder = IronButterflyBuilder(wing_delta=0.17, max_spread_cost_ratio=0.25, min_yield_on_capital=0.05)

        # Act
        strategy = builder.build_strategy(
            ticker=ibf_ticker,
            trade_date=ibf_trade_date,
            expiry_date=ibf_expiry_date,
            option_chain=sample_ibf_chain_atm,
            spot_price=ibf_spot_price,
        )

        # Assert
        assert strategy.ticker == ibf_ticker
        assert strategy.trade_date == ibf_trade_date

    def test_max_spread_cost_ratio_filters_out_illiquid_candidates(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date, ibf_ticker, ibf_spot_price
    ):
        """
        Builder with max_spread_cost_ratio=0.001 rejects every candidate on
        sample_ibf_chain_atm, raising ValueError.  Builder with
        max_spread_cost_ratio=10.0 accepts the same chain.

        Purpose: Verifies that max_spread_cost_ratio is wired end-to-end
        from IronButterflyBuilder.__init__() through build_strategy() to
        enumerate_candidates().  Any non-trivial bid/ask spread will exceed
        0.001, so the tight builder exercises the filter path cleanly.
        """
        # Tight limit — no candidate survives the spread_cost_ratio filter
        builder_tight = IronButterflyBuilder(
            wing_delta=0.17,
            max_spread_cost_ratio=0.001,
            min_yield_on_capital=0.0,
        )
        with pytest.raises(ValueError, match="No valid symmetric wing pairs"):
            builder_tight.build_strategy(
                ticker=ibf_ticker,
                trade_date=ibf_trade_date,
                expiry_date=ibf_expiry_date,
                option_chain=sample_ibf_chain_atm,
                spot_price=ibf_spot_price,
            )

        # Permissive limit — same chain builds successfully
        builder_loose = IronButterflyBuilder(
            wing_delta=0.17,
            max_spread_cost_ratio=10.0,
            min_yield_on_capital=0.0,
        )
        strategy = builder_loose.build_strategy(
            ticker=ibf_ticker,
            trade_date=ibf_trade_date,
            expiry_date=ibf_expiry_date,
            option_chain=sample_ibf_chain_atm,
            spot_price=ibf_spot_price,
        )
        assert strategy.strategy_type == StrategyType.IRON_BUTTERFLY


# =============================================================================
# TestEnumerateCandidates
# =============================================================================


class TestEnumerateCandidates:
    """
    Test enumerate_candidates() — the two-phase bucketing algorithm that
    returns at most one symmetric wing candidate per delta-target level.

    Phase 1: build full pool (spread filter, yield filter, mirror check).
    Phase 2: nearest-neighbour bucketing — each candidate claims the closest
             target; strict < tie-breaking means insertion order (ascending
             wing_width = narrower wing) wins on equal distance.
    """

    # ── Shared dates / ticker for all inline chains ────────────────────────
    TRADE  = date(2026, 2, 13)
    EXPIRY = date(2026, 2, 20)
    TKTR   = "AAPL"

    def _opt(
        self,
        strike: float,
        option_type: str,
        bid: float,
        ask: float,
        mid: float,
        delta: float,
        gamma: float = 0.02,
        vega: float  = 0.10,
        theta: float = -0.20,
    ) -> OptionQuote:
        """Minimal OptionQuote factory for inline chain construction."""
        return OptionQuote(
            ticker=self.TKTR,
            trade_date=self.TRADE,
            expiry_date=self.EXPIRY,
            strike=Decimal(str(strike)),
            option_type=option_type,
            bid=Decimal(str(bid)),
            ask=Decimal(str(ask)),
            mid=Decimal(str(mid)),
            iv=0.30,
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            volume=100,
            open_interest=100,
        )

    def _body_pair(self, strike: float = 100.0):
        """Return (short_call, short_put) body pair at the given ATM strike."""
        sc = self._opt(strike, "call", bid=5.00, ask=5.10, mid=5.05, delta=0.55,
                       gamma=0.04, vega=0.15, theta=-0.35)
        sp = self._opt(strike, "put",  bid=4.90, ask=5.00, mid=4.95, delta=-0.45,
                       gamma=0.04, vega=0.15, theta=-0.35)
        return sc, sp

    # ── Phase 1 filter tests ───────────────────────────────────────────────

    def test_returns_empty_list_when_no_otm_strikes(self):
        """
        Pool is empty when chain contains only the body (no OTM strikes).
        No OTM calls → no candidates → empty list returned.
        """
        sc, sp = self._body_pair()
        chain = [sc, sp]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)

        result = builder.enumerate_candidates(
            chain, Decimal("100"), sc, sp
        )

        assert result == []

    def test_returns_empty_list_when_no_mirror_put(self, sample_ibf_chain_no_mirror,
                                                    ibf_trade_date, ibf_expiry_date):
        """
        OTM call strikes exist but no symmetric put strike present.
        enumerate_candidates() silently skips unmatched calls — empty list.
        Uses sample_ibf_chain_no_mirror (body call+put at 255.0, OTM calls at
        245 and 265 but puts at those strikes removed).
        """
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)
        body_strike = Decimal("255.0")
        sc = next(q for q in sample_ibf_chain_no_mirror
                  if q.strike == body_strike and q.option_type == "call")
        sp = next(q for q in sample_ibf_chain_no_mirror
                  if q.strike == body_strike and q.option_type == "put")

        result = builder.enumerate_candidates(
            sample_ibf_chain_no_mirror, body_strike, sc, sp
        )

        assert result == []

    def test_returns_empty_list_when_net_credit_zero(self):
        """
        Wing cost equals body premium → net_credit == 0 → filtered out.
        The filter is strict (net_credit <= 0), so exactly zero is excluded.
        """
        sc, sp = self._body_pair()   # body total mid = 5.05 + 4.95 = 10.00
        # Wing mids sum to exactly body mids sum → net_credit = 0
        lc = self._opt(110.0, "call", bid=4.99, ask=5.01, mid=5.00, delta=0.15)
        lp = self._opt( 90.0, "put",  bid=4.99, ask=5.01, mid=5.00, delta=-0.15)
        chain = [sc, sp, lc, lp]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)

        result = builder.enumerate_candidates(
            chain, Decimal("100"), sc, sp
        )

        assert result == []

    def test_returns_empty_list_when_spread_too_wide(self):
        """
        One wing leg has a very wide bid/ask spread → total spread_cost_ratio > max_spread_cost_ratio.
        builder.max_spread_cost_ratio=0.25; total_spread / net_credit ≈ 0.80 (>> 0.25).
        """
        sc, sp = self._body_pair()
        # Wide spread: mid=2.505, spread=4.99 → spread_pct ≈ 1.99
        lc = self._opt(110.0, "call", bid=0.01, ask=5.00, mid=2.505, delta=0.15)
        lp = self._opt( 90.0, "put",  bid=0.95, ask=0.97, mid=0.960, delta=-0.15)
        chain = [sc, sp, lc, lp]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.25, min_yield_on_capital=0.0)

        result = builder.enumerate_candidates(
            chain, Decimal("100"), sc, sp
        )

        assert result == []

    def test_returns_empty_list_when_yield_below_threshold(self):
        """
        net_credit / wing_width < min_yield_on_capital → silently filtered.
        wing_width=10, net_credit=0.20, yield=0.02 < min_yield=0.05.
        """
        # body total mid = 3.0 + 2.0 = 5.0
        sc = self._opt(100.0, "call", bid=2.95, ask=3.05, mid=3.00, delta=0.55)
        sp = self._opt(100.0, "put",  bid=1.95, ask=2.05, mid=2.00, delta=-0.45)
        # wing total mid = 2.40 + 2.40 = 4.80 → net_credit = 0.20 → yield = 0.02
        lc = self._opt(110.0, "call", bid=2.35, ask=2.45, mid=2.40, delta=0.15)
        lp = self._opt( 90.0, "put",  bid=2.35, ask=2.45, mid=2.40, delta=-0.15)
        chain = [sc, sp, lc, lp]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.05)

        result = builder.enumerate_candidates(
            chain, Decimal("100"), sc, sp
        )

        assert result == []

    # ── Happy path / structure tests ───────────────────────────────────────

    def test_single_valid_pair_returns_one_candidate(
        self, sample_ibf_chain_atm, ibf_trade_date, ibf_expiry_date
    ):
        """
        Chain with exactly one symmetric pair → list of exactly 1 candidate.
        sample_ibf_chain_atm has body=255.0 and one wing pair ±10.
        """
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)
        body_strike = Decimal("255.0")
        sc = next(q for q in sample_ibf_chain_atm
                  if q.strike == body_strike and q.option_type == "call")
        sp = next(q for q in sample_ibf_chain_atm
                  if q.strike == body_strike and q.option_type == "put")

        result = builder.enumerate_candidates(sample_ibf_chain_atm, body_strike, sc, sp)

        assert len(result) == 1
        assert isinstance(result[0], IronButterflyCandidate)

    def test_candidate_fields_correct(self):
        """
        All 14 fields of the returned IronButterflyCandidate are correct.
        Values are computed manually from a single inline pair.

        Body  @100: call mid=5.05 (bid=5.00, ask=5.10), delta=+0.55, gamma=0.04, vega=0.15, theta=-0.35
                    put  mid=4.95 (bid=4.90, ask=5.00), delta=-0.45, gamma=0.04, vega=0.15, theta=-0.35
        Wings @110/90: call mid=1.01 (bid=1.00, ask=1.02), delta=+0.15, gamma=0.02, vega=0.10, theta=-0.20
                       put  mid=0.96 (bid=0.95, ask=0.97), delta=-0.15, gamma=0.02, vega=0.10, theta=-0.20

        net_credit       = (5.05+4.95) - (1.01+0.96) = 8.03
        credit_to_width  = 8.03 / 10 = 0.803
        avg_wing_delta   = (0.15 + 0.15) / 2 = 0.15
        net_delta        = (-0.15) + -(-0.45) + -(0.55) + 0.15 = -0.10
        net_gamma        = 0.02+0.02 - 0.04-0.04 = -0.04
        net_vega         = 0.10+0.10 - 0.15-0.15 = -0.10
        net_theta        = (-0.20)+(-0.20) - (-0.35)-(-0.35) = 0.30
        """
        sc, sp = self._body_pair()   # strike=100, call mid=5.05, put mid=4.95
        lc = self._opt(110.0, "call", bid=1.00, ask=1.02, mid=1.01, delta=0.15,
                       gamma=0.02, vega=0.10, theta=-0.20)
        lp = self._opt( 90.0, "put",  bid=0.95, ask=0.97, mid=0.96, delta=-0.15,
                       gamma=0.02, vega=0.10, theta=-0.20)
        chain = [sc, sp, lc, lp]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)

        result = builder.enumerate_candidates(chain, Decimal("100"), sc, sp)

        assert len(result) == 1
        c = result[0]

        assert c.body_strike       == Decimal("100")
        assert c.wing_width        == Decimal("10")
        assert c.call_wing_strike  == Decimal("110.0")
        assert c.put_wing_strike   == Decimal("90.0")
        assert c.net_credit        == Decimal("8.03")
        assert abs(c.credit_to_width  - 0.803)  < 1e-9
        assert abs(c.avg_wing_delta   - 0.15)   < 1e-9
        assert c.short_call is sc
        assert c.short_put  is sp
        assert c.long_call  is lc
        assert c.long_put   is lp

    # ── Net greeks test ────────────────────────────────────────────────────

    def test_net_greeks_computed_correctly(self):
        """
        Net greeks across all 4 legs use signed quantities (+1 long, -1 short).

        net_delta = lp.delta*+1 + sp.delta*-1 + sc.delta*-1 + lc.delta*+1
        net_gamma = lp.gamma + lc.gamma - sp.gamma - sc.gamma
        net_vega  = lp.vega  + lc.vega  - sp.vega  - sc.vega
        net_theta = lp.theta + lc.theta - sp.theta - sc.theta
        """
        sc, sp = self._body_pair()   # delta ±0.55/−0.45, gamma=0.04, vega=0.15, theta=-0.35
        lc = self._opt(110.0, "call", bid=1.00, ask=1.02, mid=1.01, delta=0.15,
                       gamma=0.02, vega=0.10, theta=-0.20)
        lp = self._opt( 90.0, "put",  bid=0.95, ask=0.97, mid=0.96, delta=-0.15,
                       gamma=0.02, vega=0.10, theta=-0.20)
        chain = [sc, sp, lc, lp]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)

        c = builder.enumerate_candidates(chain, Decimal("100"), sc, sp)[0]

        expected_net_delta = (-0.15) + -(-0.45) + -(0.55) + 0.15        # = -0.10
        expected_net_gamma = 0.02 + 0.02 - 0.04 - 0.04                  # = -0.04
        expected_net_vega  = 0.10 + 0.10 - 0.15 - 0.15                  # = -0.10
        expected_net_theta = (-0.20) + (-0.20) - (-0.35) - (-0.35)      # = +0.30

        assert abs(c.net_delta - expected_net_delta) < 1e-9
        assert abs(c.net_gamma - expected_net_gamma) < 1e-9
        assert abs(c.net_vega  - expected_net_vega)  < 1e-9
        assert abs(c.net_theta - expected_net_theta) < 1e-9

    # ── Bucketing algorithm tests ──────────────────────────────────────────

    def test_bucketing_two_candidates_same_bucket_closer_wins(self):
        """
        Two candidates that both claim the 0.15 target; the closer one wins.

        Pair A (width=10): avg_delta=0.134  → dist to 0.15 = 0.016
        Pair B (width= 8): avg_delta=0.152  → dist to 0.15 = 0.002  ← wins

        Pool insertion order is ascending wing_width (Pair A first), so
        Pair A does NOT win the tie even if considered first — Pair B is
        strictly closer and beats it.
        Result: [Pair B] only.
        """
        sc, sp = self._body_pair()
        # Pair A  (width=10)
        lc_a = self._opt(110.0, "call", bid=0.95, ask=0.97, mid=0.96, delta= 0.134)
        lp_a = self._opt( 90.0, "put",  bid=0.90, ask=0.92, mid=0.91, delta=-0.134)
        # Pair B  (width=8)
        lc_b = self._opt(108.0, "call", bid=1.00, ask=1.02, mid=1.01, delta= 0.152)
        lp_b = self._opt( 92.0, "put",  bid=0.95, ask=0.97, mid=0.96, delta=-0.152)
        chain = [sc, sp, lc_a, lp_a, lc_b, lp_b]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)

        result = builder.enumerate_candidates(chain, Decimal("100"), sc, sp)

        assert len(result) == 1
        assert abs(result[0].avg_wing_delta - 0.152) < 1e-9
        assert result[0].call_wing_strike == Decimal("108.0")

    def test_bucketing_two_candidates_different_buckets(self):
        """
        Two candidates that claim different target buckets → both returned.

        Pair A (width=8 ): avg_delta=0.12 → nearest 0.10 (dist=0.02)
        Pair B (width=10): avg_delta=0.22 → nearest 0.20 (dist=0.02)

        Each wins its own bucket; result has 2 candidates sorted by width.
        """
        sc, sp = self._body_pair()
        # Pair A — claims 0.10 bucket
        lc_a = self._opt(108.0, "call", bid=0.95, ask=0.97, mid=0.96, delta= 0.12)
        lp_a = self._opt( 92.0, "put",  bid=0.90, ask=0.92, mid=0.91, delta=-0.12)
        # Pair B — claims 0.20 bucket
        lc_b = self._opt(110.0, "call", bid=0.85, ask=0.87, mid=0.86, delta= 0.22)
        lp_b = self._opt( 90.0, "put",  bid=0.80, ask=0.82, mid=0.81, delta=-0.22)
        chain = [sc, sp, lc_a, lp_a, lc_b, lp_b]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)

        result = builder.enumerate_candidates(chain, Decimal("100"), sc, sp)

        assert len(result) == 2
        widths = [c.wing_width for c in result]
        assert widths[0] <= widths[1],  "results must be sorted ascending by wing_width"
        deltas = {round(c.avg_wing_delta, 2) for c in result}
        assert deltas == {0.12, 0.22}

    def test_bucketing_returns_at_most_one_per_target(self):
        """
        Five candidates all nearest to the 0.15 target; only the closest wins.

        Deltas: 0.140, 0.145, 0.150, 0.155, 0.160 — all claim 0.15.
        0.150 is equidistant to 0.15 and... it IS 0.15, so dist=0.000 wins.
        Result: exactly 1 candidate with avg_wing_delta ≈ 0.150.
        """
        sc, sp = self._body_pair()
        pairs = [
            (106.0, 94.0, 0.140),
            (107.0, 93.0, 0.145),
            (108.0, 92.0, 0.150),
            (109.0, 91.0, 0.155),
            (110.0, 90.0, 0.160),
        ]
        chain = [sc, sp]
        for call_k, put_k, delta in pairs:
            chain.append(self._opt(call_k, "call", bid=0.95, ask=0.97, mid=0.96, delta= delta))
            chain.append(self._opt(put_k,  "put",  bid=0.90, ask=0.92, mid=0.91, delta=-delta))
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)

        result = builder.enumerate_candidates(chain, Decimal("100"), sc, sp)

        assert len(result) == 1
        assert abs(result[0].avg_wing_delta - 0.150) < 1e-9

    def test_custom_wing_delta_targets_respected(self):
        """
        Passing wing_delta_targets changes which candidates survive bucketing.

        Chain has two pairs:
          Pair A (width=8):  avg_delta=0.12 — with default targets claims 0.10
          Pair B (width=10): avg_delta=0.28 — with default targets claims 0.30

        Default [0.10, 0.15, 0.20, 0.30] → 2 results (each wins its bucket).
        Custom  [0.15]                    → 1 result (both claim 0.15; Pair A
                                             wins at dist=0.03 vs Pair B at 0.13).
        """
        sc, sp = self._body_pair()
        lc_a = self._opt(108.0, "call", bid=0.95, ask=0.97, mid=0.96, delta= 0.12)
        lp_a = self._opt( 92.0, "put",  bid=0.90, ask=0.92, mid=0.91, delta=-0.12)
        lc_b = self._opt(110.0, "call", bid=0.85, ask=0.87, mid=0.86, delta= 0.28)
        lp_b = self._opt( 90.0, "put",  bid=0.80, ask=0.82, mid=0.81, delta=-0.28)
        chain = [sc, sp, lc_a, lp_a, lc_b, lp_b]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)

        result_default = builder.enumerate_candidates(chain, Decimal("100"), sc, sp)
        result_custom  = builder.enumerate_candidates(
            chain, Decimal("100"), sc, sp, wing_delta_targets=[0.15]
        )

        assert len(result_default) == 2
        assert len(result_custom)  == 1
        assert abs(result_custom[0].avg_wing_delta - 0.12) < 1e-9

    def test_sorted_ascending_by_wing_width(self):
        """
        Returned list is sorted ascending by wing_width regardless of pool
        construction order.

        Three pairs at widths 8, 10, 12 with deltas claiming three different
        target buckets — all three survive bucketing and must come back
        ordered 8 < 10 < 12.
        """
        sc, sp = self._body_pair()
        # width 8  → delta 0.12 → claims 0.10
        lc_a = self._opt(108.0, "call", bid=0.95, ask=0.97, mid=0.96, delta= 0.12)
        lp_a = self._opt( 92.0, "put",  bid=0.90, ask=0.92, mid=0.91, delta=-0.12)
        # width 10 → delta 0.20 → claims 0.20
        lc_b = self._opt(110.0, "call", bid=0.85, ask=0.87, mid=0.86, delta= 0.20)
        lp_b = self._opt( 90.0, "put",  bid=0.80, ask=0.82, mid=0.81, delta=-0.20)
        # width 12 → delta 0.29 → claims 0.30
        lc_c = self._opt(112.0, "call", bid=0.75, ask=0.77, mid=0.76, delta= 0.29)
        lp_c = self._opt( 88.0, "put",  bid=0.70, ask=0.72, mid=0.71, delta=-0.29)
        chain = [sc, sp, lc_a, lp_a, lc_b, lp_b, lc_c, lp_c]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)

        result = builder.enumerate_candidates(chain, Decimal("100"), sc, sp)

        assert len(result) == 3
        widths = [c.wing_width for c in result]
        assert widths == sorted(widths), f"Expected ascending widths, got {widths}"

    def test_default_targets_are_05_10_15_20_30(
        self, sample_ibf_chain_multi_width, ibf_trade_date, ibf_expiry_date
    ):
        """
        Calling enumerate_candidates() without wing_delta_targets uses the
        default [0.05, 0.10, 0.15, 0.20, 0.30], so at most 5 candidates are returned.
        Uses sample_ibf_chain_multi_width (many available symmetric pairs).
        """
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)
        body_strike = Decimal("255.0")
        sc = next(q for q in sample_ibf_chain_multi_width
                  if q.strike == body_strike and q.option_type == "call")
        sp = next(q for q in sample_ibf_chain_multi_width
                  if q.strike == body_strike and q.option_type == "put")

        result = builder.enumerate_candidates(sample_ibf_chain_multi_width, body_strike, sc, sp)

        assert len(result) <= 5, (
            f"Default targets have 5 buckets, got {len(result)} candidates"
        )
        # All results must be valid IronButterflyCandidate instances
        assert all(isinstance(c, IronButterflyCandidate) for c in result)
        # Widths must be ascending
        widths = [c.wing_width for c in result]
        assert widths == sorted(widths)


class TestIronButterflyBuilderChainValidation:
    """
    Test build_strategy() input guard rails that mirror StraddleBuilder.

    These tests use inline minimal chains (constructed directly in the test
    body) rather than fixtures, because each case requires a subtly broken
    chain that does not merit a standalone CSV file.
    """

    def test_empty_chain_raises(self, trade_date, expiry_date, ticker):
        """
        An empty option_chain list is immediately rejected.

        Purpose: Prevents silent failures when a data feed returns nothing
        (e.g., market holiday, bad ticker symbol).
        Expected: ValueError matching 'Empty option chain'.
        """
        builder = IronButterflyBuilder()
        with pytest.raises(ValueError, match="Empty option chain"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=[],
                spot_price=Decimal("100.0"),
            )

    def test_multiple_expiries_raises(
        self, sample_option_chain_multiple_expiries, trade_date, expiry_date, ticker
    ):
        """
        A chain containing options from more than one expiry date is rejected.

        Purpose: The builder expects the caller to pre-filter to a single
        expiry.  Mixed expiries mean the legs would have different DTEs,
        breaking the strategy's risk profile.
        Expected: ValueError matching 'multiple expiries'.
        """
        builder = IronButterflyBuilder()
        with pytest.raises(ValueError, match="multiple expiries"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=sample_option_chain_multiple_expiries,
                spot_price=Decimal("44.50"),
            )

    def test_expiry_mismatch_raises(self, trade_date, expiry_date, ticker):
        """
        Chain expiry is 2024-12-06 but expiry_date argument is 2024-12-13.

        Purpose: Catches data provider bugs where the returned chain is for
        the wrong expiry.  If unchecked, the strategy would be priced on
        the wrong date with incorrect DTE.
        Expected: ValueError matching 'expiry mismatch'.
        """
        # Chain has expiry_date (2024-12-06); pass a later date to trigger mismatch
        chain = [
            OptionQuote(
                ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                strike=Decimal("100"), option_type="call",
                bid=Decimal("5.00"), ask=Decimal("5.10"), mid=Decimal("5.05"),
                iv=0.30, delta=0.55, gamma=0.04, vega=0.15, theta=-0.35,
                volume=100, open_interest=100,
            )
        ]
        builder = IronButterflyBuilder()
        with pytest.raises(ValueError, match="expiry mismatch"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=date(2024, 12, 13),
                option_chain=chain,
                spot_price=Decimal("100.0"),
            )

    def test_missing_body_call_raises(self, trade_date, expiry_date, ticker):
        """
        Chain has a put at the ATM body strike but no call at that strike.

        Purpose: The short call body is required.  Missing it means the
        strategy cannot be formed.  This can happen when deep-ITM calls are
        filtered out by liquidity screens before the chain reaches the builder.
        Expected: ValueError matching 'No call option found at body strike'.
        """
        # Chain has put@100 but no call@100; ATM snaps to 100, call lookup returns None
        chain = [
            OptionQuote(
                ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                strike=Decimal("100"), option_type="put",
                bid=Decimal("4.90"), ask=Decimal("5.00"), mid=Decimal("4.95"),
                iv=0.30, delta=-0.45, gamma=0.04, vega=0.15, theta=-0.35,
                volume=100, open_interest=100,
            ),
            OptionQuote(
                ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                strike=Decimal("105"), option_type="call",
                bid=Decimal("2.00"), ask=Decimal("2.10"), mid=Decimal("2.05"),
                iv=0.30, delta=0.35, gamma=0.03, vega=0.12, theta=-0.25,
                volume=100, open_interest=100,
            ),
        ]
        builder = IronButterflyBuilder()
        with pytest.raises(ValueError, match="No call option found at body strike"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=chain,
                spot_price=Decimal("100.0"),
            )

    def test_missing_body_put_raises(self, trade_date, expiry_date, ticker):
        """
        Chain has a call at the ATM body strike but no put at that strike.

        Purpose: Symmetric to test_missing_body_call_raises.  The short put
        body is equally required.
        Expected: ValueError matching 'No put option found at body strike'.
        """
        # Chain has call@100 but no put@100; ATM snaps to 100, put lookup returns None
        chain = [
            OptionQuote(
                ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                strike=Decimal("100"), option_type="call",
                bid=Decimal("5.00"), ask=Decimal("5.10"), mid=Decimal("5.05"),
                iv=0.30, delta=0.55, gamma=0.04, vega=0.15, theta=-0.35,
                volume=100, open_interest=100,
            ),
            OptionQuote(
                ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                strike=Decimal("95"), option_type="put",
                bid=Decimal("1.50"), ask=Decimal("1.60"), mid=Decimal("1.55"),
                iv=0.30, delta=-0.25, gamma=0.03, vega=0.12, theta=-0.25,
                volume=100, open_interest=100,
            ),
        ]
        builder = IronButterflyBuilder()
        with pytest.raises(ValueError, match="No put option found at body strike"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=chain,
                spot_price=Decimal("100.0"),
            )

    def test_zero_mid_body_call_raises(self, trade_date, expiry_date, ticker):
        """
        Short call body has mid=0.00 (stale or missing quote).

        Purpose: Trading against a zero-mid option would produce nonsensical
        P&L.  This guard catches bad data before a strategy object is created.
        Expected: ValueError matching 'Invalid short call mid'.
        """
        chain = [
            OptionQuote(
                ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                strike=Decimal("100"), option_type="call",
                bid=Decimal("0.00"), ask=Decimal("0.00"), mid=Decimal("0.00"),
                iv=0.30, delta=0.55, gamma=0.04, vega=0.15, theta=-0.35,
                volume=0, open_interest=0,
            ),
            OptionQuote(
                ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                strike=Decimal("100"), option_type="put",
                bid=Decimal("4.90"), ask=Decimal("5.00"), mid=Decimal("4.95"),
                iv=0.30, delta=-0.45, gamma=0.04, vega=0.15, theta=-0.35,
                volume=100, open_interest=100,
            ),
        ]
        builder = IronButterflyBuilder()
        with pytest.raises(ValueError, match="Invalid short call mid"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=chain,
                spot_price=Decimal("100.0"),
            )

    def test_zero_mid_body_put_raises(self, trade_date, expiry_date, ticker):
        """
        Short put body has mid=0.00 (stale or missing quote).

        Purpose: Symmetric to test_zero_mid_body_call_raises.
        Expected: ValueError matching 'Invalid short put mid'.
        """
        chain = [
            OptionQuote(
                ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                strike=Decimal("100"), option_type="call",
                bid=Decimal("5.00"), ask=Decimal("5.10"), mid=Decimal("5.05"),
                iv=0.30, delta=0.55, gamma=0.04, vega=0.15, theta=-0.35,
                volume=100, open_interest=100,
            ),
            OptionQuote(
                ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                strike=Decimal("100"), option_type="put",
                bid=Decimal("0.00"), ask=Decimal("0.00"), mid=Decimal("0.00"),
                iv=0.30, delta=-0.45, gamma=0.04, vega=0.15, theta=-0.35,
                volume=0, open_interest=0,
            ),
        ]
        builder = IronButterflyBuilder()
        with pytest.raises(ValueError, match="Invalid short put mid"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=chain,
                spot_price=Decimal("100.0"),
            )


class TestIronButterflyBuilderWingValidation:
    """
    Test the three post-wing-selection guards:
      1. Spread quality on all 4 legs (max_spread_cost_ratio)
      2. Net credit must be positive
      3. Yield-on-capital must meet the minimum threshold

    All tests use inline chains to isolate exactly the failing condition.
    """

    def test_no_symmetric_wing_pair_raises(
        self, sample_ibf_chain_no_mirror, trade_date, expiry_date, ticker
    ):
        """
        Chain has OTM call candidates but no matching mirrored put strikes.

        Purpose: _select_wing_pair() must raise rather than return None or
        pick an asymmetric pair.  Uses sample_ibf_chain_no_mirror which
        contains a body but only OTM calls (no OTM puts on the other side).
        Expected: ValueError matching 'No valid symmetric wing pairs found'.
        """
        # Extract dates from the chain itself (IBF/AAPL dates, not VZ fixtures)
        chain_expiry = sample_ibf_chain_no_mirror[0].expiry_date
        chain_trade_date = sample_ibf_chain_no_mirror[0].trade_date
        chain_ticker = sample_ibf_chain_no_mirror[0].ticker
        builder = IronButterflyBuilder()
        with pytest.raises(ValueError, match="No valid symmetric wing pairs found"):
            builder.build_strategy(
                ticker=chain_ticker,
                trade_date=chain_trade_date,
                expiry_date=chain_expiry,
                option_chain=sample_ibf_chain_no_mirror,
                spot_price=Decimal("255.81"),
            )

    def test_wide_spread_on_long_call_wing_raises(self, trade_date, expiry_date, ticker):
        """
        Long call wing has an extremely wide bid-ask spread (bid=0.01, ask=0.50),
        giving spread_cost_ratio >> max_spread_cost_ratio=0.25.

        Purpose: Illiquid wing options make entry/exit expensive.  The spread
        filter enforces a minimum liquidity standard.  Must flag the specific
        leg name in the error message.
        Expected: ValueError matching 'long call wing'.
        """
        # lc@110 has extremely wide spread → spread_cost_ratio >> 0.25 → filtered → no candidates
        chain = [
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="call",
                        bid=Decimal("5.00"), ask=Decimal("5.10"), mid=Decimal("5.05"),
                        iv=0.30, delta=0.55, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="put",
                        bid=Decimal("4.90"), ask=Decimal("5.00"), mid=Decimal("4.95"),
                        iv=0.30, delta=-0.45, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("110"), option_type="call",
                        bid=Decimal("0.01"), ask=Decimal("5.00"), mid=Decimal("2.505"),
                        iv=0.30, delta=0.15, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=1, open_interest=1),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("90"), option_type="put",
                        bid=Decimal("0.95"), ask=Decimal("0.97"), mid=Decimal("0.96"),
                        iv=0.30, delta=-0.15, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=100, open_interest=100),
        ]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.25, min_yield_on_capital=0.0)
        with pytest.raises(ValueError, match="No valid symmetric wing pairs found"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=chain,
                spot_price=Decimal("100.0"),
            )

    def test_wide_spread_on_long_put_wing_raises(self, trade_date, expiry_date, ticker):
        """
        Long put wing has an extremely wide bid-ask spread.

        Purpose: Same motivation as the call wing test; validates that the
        spread check is applied to ALL four legs, not just the calls.
        Expected: ValueError matching 'long put wing'.
        """
        # lp@90 has extremely wide spread → spread_cost_ratio >> 0.25 → filtered → no candidates
        chain = [
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="call",
                        bid=Decimal("5.00"), ask=Decimal("5.10"), mid=Decimal("5.05"),
                        iv=0.30, delta=0.55, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="put",
                        bid=Decimal("4.90"), ask=Decimal("5.00"), mid=Decimal("4.95"),
                        iv=0.30, delta=-0.45, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("110"), option_type="call",
                        bid=Decimal("1.00"), ask=Decimal("1.02"), mid=Decimal("1.01"),
                        iv=0.30, delta=0.15, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("90"), option_type="put",
                        bid=Decimal("0.01"), ask=Decimal("5.00"), mid=Decimal("2.505"),
                        iv=0.30, delta=-0.15, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=1, open_interest=1),
        ]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.25, min_yield_on_capital=0.0)
        with pytest.raises(ValueError, match="No valid symmetric wing pairs found"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=chain,
                spot_price=Decimal("100.0"),
            )

    def test_wide_spread_on_short_call_body_raises(self, trade_date, expiry_date, ticker):
        """
        Short call body has an extremely wide bid-ask spread.

        Purpose: The body legs can also be illiquid.  Ensures the spread
        check covers the body, not just the wings.
        Expected: ValueError matching 'short call body'.
        """
        # sc@100 has extremely wide spread → spread_cost_ratio >> 0.25 → filtered → no candidates
        chain = [
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="call",
                        bid=Decimal("0.01"), ask=Decimal("5.00"), mid=Decimal("2.505"),
                        iv=0.30, delta=0.55, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=1, open_interest=1),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="put",
                        bid=Decimal("4.90"), ask=Decimal("5.00"), mid=Decimal("4.95"),
                        iv=0.30, delta=-0.45, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("110"), option_type="call",
                        bid=Decimal("1.00"), ask=Decimal("1.02"), mid=Decimal("1.01"),
                        iv=0.30, delta=0.15, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("90"), option_type="put",
                        bid=Decimal("0.95"), ask=Decimal("0.97"), mid=Decimal("0.96"),
                        iv=0.30, delta=-0.15, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=100, open_interest=100),
        ]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.25, min_yield_on_capital=0.0)
        with pytest.raises(ValueError, match="No valid symmetric wing pairs found"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=chain,
                spot_price=Decimal("100.0"),
            )

    def test_wide_spread_on_short_put_body_raises(self, trade_date, expiry_date, ticker):
        """
        Short put body has an extremely wide bid-ask spread.

        Purpose: Symmetric to test_wide_spread_on_short_call_body_raises.
        Expected: ValueError matching 'short put body'.
        """
        # sp@100 has extremely wide spread → spread_cost_ratio >> 0.25 → filtered → no candidates
        chain = [
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="call",
                        bid=Decimal("5.00"), ask=Decimal("5.10"), mid=Decimal("5.05"),
                        iv=0.30, delta=0.55, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="put",
                        bid=Decimal("0.01"), ask=Decimal("5.00"), mid=Decimal("2.505"),
                        iv=0.30, delta=-0.45, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=1, open_interest=1),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("110"), option_type="call",
                        bid=Decimal("1.00"), ask=Decimal("1.02"), mid=Decimal("1.01"),
                        iv=0.30, delta=0.15, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("90"), option_type="put",
                        bid=Decimal("0.95"), ask=Decimal("0.97"), mid=Decimal("0.96"),
                        iv=0.30, delta=-0.15, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=100, open_interest=100),
        ]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.25, min_yield_on_capital=0.0)
        with pytest.raises(ValueError, match="No valid symmetric wing pairs found"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=chain,
                spot_price=Decimal("100.0"),
            )

    def test_negative_net_credit_raises(self, trade_date, expiry_date, ticker):
        """
        Wings are priced more expensively than the body, yielding a net debit.

        This can happen in inverted vol surfaces or extremely wide skews.

        Purpose: A debit iron butterfly is a data anomaly — the strategy
        should be rejected rather than recorded as a credit strategy with a
        negative premium.
        Expected: ValueError matching 'non-positive net credit'.
        """
        # Wing mids sum (5.50) > body mids sum (5.00) → net_credit = -0.50 ≤ 0 → filtered
        chain = [
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="call",
                        bid=Decimal("2.95"), ask=Decimal("3.05"), mid=Decimal("3.00"),
                        iv=0.30, delta=0.55, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="put",
                        bid=Decimal("1.95"), ask=Decimal("2.05"), mid=Decimal("2.00"),
                        iv=0.30, delta=-0.45, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("110"), option_type="call",
                        bid=Decimal("3.45"), ask=Decimal("3.55"), mid=Decimal("3.50"),
                        iv=0.30, delta=0.30, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("90"), option_type="put",
                        bid=Decimal("1.95"), ask=Decimal("2.05"), mid=Decimal("2.00"),
                        iv=0.30, delta=-0.30, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=100, open_interest=100),
        ]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.0)
        with pytest.raises(ValueError, match="No valid symmetric wing pairs found"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=chain,
                spot_price=Decimal("100.0"),
            )

    def test_yield_below_threshold_raises(self, trade_date, expiry_date, ticker):
        """
        Net credit is positive but tiny relative to wing width,
        so yield_on_capital < min_yield_on_capital.

        Example: body_mid_total=1.00, wing_mid_total=0.96, wing_width=10.0
        net_credit=0.04, yield=0.004 which is below the default 0.05.

        Purpose: Screens out mathematically valid but economically
        uninteresting butterflies where the limited credit doesn't
        justify the capital tied up as collateral.
        Expected: ValueError matching 'Yield-on-capital'.
        """
        # net_credit=0.20, wing_width=10 → yield=0.02 < min_yield=0.05 → filtered
        chain = [
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="call",
                        bid=Decimal("2.95"), ask=Decimal("3.05"), mid=Decimal("3.00"),
                        iv=0.30, delta=0.55, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("100"), option_type="put",
                        bid=Decimal("1.95"), ask=Decimal("2.05"), mid=Decimal("2.00"),
                        iv=0.30, delta=-0.45, gamma=0.04, vega=0.15, theta=-0.35,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("110"), option_type="call",
                        bid=Decimal("2.35"), ask=Decimal("2.45"), mid=Decimal("2.40"),
                        iv=0.30, delta=0.15, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=100, open_interest=100),
            OptionQuote(ticker=ticker, trade_date=trade_date, expiry_date=expiry_date,
                        strike=Decimal("90"), option_type="put",
                        bid=Decimal("2.35"), ask=Decimal("2.45"), mid=Decimal("2.40"),
                        iv=0.30, delta=-0.15, gamma=0.02, vega=0.10, theta=-0.20,
                        volume=100, open_interest=100),
        ]
        builder = IronButterflyBuilder(max_spread_cost_ratio=0.99, min_yield_on_capital=0.05)
        with pytest.raises(ValueError, match="No valid symmetric wing pairs found"):
            builder.build_strategy(
                ticker=ticker,
                trade_date=trade_date,
                expiry_date=expiry_date,
                option_chain=chain,
                spot_price=Decimal("100.0"),
            )


class TestIronButterflyBuilderHelpers:
    """
    Test the two shared private helpers (_find_atm_strike, _get_option_at_strike)
    on IronButterflyBuilder and the IBF-specific _compute_yield_on_capital.

    These helpers are tested in isolation so that failures here give precise
    diagnostic information independent of the full build_strategy() flow.
    """

    def test_compute_yield_on_capital_correct_value(self):
        """
        _compute_yield_on_capital(net_credit, wing_width) returns net_credit / wing_width
        as a plain float.

        Example: 4.20 / 10.00 == 0.42.

        Purpose: Locks in the arithmetic and confirms the return type is
        float (not Decimal), consistent with the greek fields on OptionQuote.
        """
        builder = IronButterflyBuilder()
        result = builder._compute_yield_on_capital(Decimal("4.20"), Decimal("10.00"))
        assert isinstance(result, float)
        assert abs(result - 0.42) < 1e-9

    def test_compute_yield_on_capital_zero_width_raises(self):
        """
        wing_width=0 would cause division by zero.

        Purpose: Defensive guard; a zero-width wing means the long and short
        legs are at the same strike — not a valid iron butterfly.
        Expected: ValueError matching 'wing_width must be positive'.
        """
        builder = IronButterflyBuilder()
        with pytest.raises(ValueError, match="wing_width must be positive"):
            builder._compute_yield_on_capital(Decimal("4.20"), Decimal("0"))

    def test_find_atm_strike_exact_match(self, sample_ibf_chain_atm, ibf_spot_price):
        """
        Spot price exactly on a strike returns that strike.

        Purpose: Validates the common case where spot lands exactly on a
        listed strike.  The result should be deterministic and not depend
        on floating-point ordering.
        """
        builder = IronButterflyBuilder()
        result = builder._find_atm_strike(sample_ibf_chain_atm, Decimal("255.0"))
        assert result == Decimal("255.0")

    def test_find_atm_strike_between_strikes(self, sample_ibf_chain_atm, ibf_spot_price):
        """
        Spot between two strikes returns the closer one.

        Purpose: The distance minimisation (abs(strike - spot)) must select
        the numerically closer strike, not simply the first or last.
        """
        builder = IronButterflyBuilder()
        # ibf_spot_price=255.81: |255.81-255.0|=0.81 < |255.81-265.0|=9.19 → 255.0 is nearest
        result = builder._find_atm_strike(sample_ibf_chain_atm, ibf_spot_price)
        assert result == Decimal("255.0")

    def test_get_option_at_strike_found(self, sample_ibf_chain_atm, ibf_spot_price):
        """
        Returns the matching OptionQuote when strike and option_type are present.

        Purpose: Confirms the lookup correctly filters by both strike AND
        option_type (not just strike), so call and put at the same strike
        are distinguishable.
        """
        builder = IronButterflyBuilder()
        result = builder._get_option_at_strike(sample_ibf_chain_atm, Decimal("255.0"), "call")
        assert result is not None
        assert result.strike == Decimal("255.0")
        assert result.option_type == "call"

    def test_get_option_at_strike_not_found(self, sample_ibf_chain_atm, ibf_spot_price):
        """
        Returns None when the requested strike/type combination is absent.

        Purpose: The caller (build_strategy) relies on a None return to
        produce a descriptive error message.  A KeyError or exception here
        would surface a confusing internal traceback instead.
        """
        builder = IronButterflyBuilder()
        result = builder._get_option_at_strike(sample_ibf_chain_atm, Decimal("999.0"), "call")
        assert result is None
