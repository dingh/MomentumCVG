"""
Unit tests for core models - Layer 1
File: tests/unit/test_models.py
Target: src/core/models.py
"""

import pytest
from datetime import date
from decimal import Decimal
from dataclasses import FrozenInstanceError

from src.core.models import (
    OptionQuote,
    OptionLeg,
    OptionStrategy,
    Signal,
    Position,
    StrategyType,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def vz_put():
    """VZ put option - real data from backtest"""
    return OptionQuote(
        ticker='VZ',
        trade_date=date(2024, 11, 29),
        expiry_date=date(2024, 12, 6),
        strike=Decimal('44.5'),
        option_type='put',
        bid=Decimal('0.43'),
        ask=Decimal('0.45'),
        mid=Decimal('0.44'),
        iv=0.160773,
        delta=-0.5130319999999999,
        gamma=0.40047643,
        vega=0.02481564,
        theta=-0.02910814,
        volume=491,
        open_interest=785
    )


@pytest.fixture
def vz_call():
    """VZ call option - real data from backtest"""
    return OptionQuote(
        ticker='VZ',
        trade_date=date(2024, 11, 29),
        expiry_date=date(2024, 12, 6),
        strike=Decimal('44.5'),
        option_type='call',
        bid=Decimal('0.39'),
        ask=Decimal('0.41'),
        mid=Decimal('0.40'),
        iv=0.160773,
        delta=0.486968,
        gamma=0.40047643,
        vega=0.02481564,
        theta=-0.02910814,
        volume=1029,
        open_interest=1248
    )


# ============================================================================
# LAYER 1.1: OptionQuote Tests
# ============================================================================

class TestOptionQuote:
    """Test OptionQuote dataclass properties and calculations"""
    
    def test_dte_calculation(self):
        """Test days to expiration calculation
        
        Should calculate correct number of days between trade_date and expiry_date.
        Using real VZ option from 2024-11-29 expiring 2024-12-06 (7 days).
        """
        # ARRANGE: Create real option from your backtest
        quote = OptionQuote(
            ticker='VZ',
            trade_date=date(2024, 11, 29),
            expiry_date=date(2024, 12, 6),
            strike=Decimal('44.5'),
            option_type='put',
            bid=Decimal('0.43'),
            ask=Decimal('0.45'),
            mid=Decimal('0.44'),
            iv=0.160773,
            delta=-0.5130319999999999,
            gamma=0.40047643,
            vega=0.02481564,
            theta=-0.02910814,
            volume=491,
            open_interest=785
        )
        
        # ACT: Calculate DTE
        actual_dte = quote.dte
        
        # ASSERT: Should be 7 days
        assert actual_dte == 7, f"Expected 7 days, got {actual_dte}"
    
    def test_spread_calculation(self):
        """Test bid-ask spread calculation
        
        Should calculate spread as ask - bid.
        Using real VZ option: ask=0.45, bid=0.43 → spread=0.02
        """
        # ARRANGE
        quote = OptionQuote(
            ticker='VZ',
            trade_date=date(2024, 11, 29),
            expiry_date=date(2024, 12, 6),
            strike=Decimal('44.5'),
            option_type='put',
            bid=Decimal('0.43'),
            ask=Decimal('0.45'),
            mid=Decimal('0.44'),
            iv=0.160773,
            delta=-0.5130319999999999,
            gamma=0.40047643,
            vega=0.02481564,
            theta=-0.02910814,
            volume=491,
            open_interest=785
        )
        
        # ACT
        actual_spread = quote.spread
        
        # ASSERT: 0.45 - 0.43 = 0.02
        expected_spread = Decimal('0.02')
        assert actual_spread == expected_spread, f"Expected {expected_spread}, got {actual_spread}"
    
    def test_spread_pct_normal(self):
        """Test spread percentage calculation
        
        Should calculate spread / mid as percentage.
        Real VZ option: spread=0.02, mid=0.44 → ~4.55%
        """
        # ARRANGE
        quote = OptionQuote(
            ticker='VZ',
            trade_date=date(2024, 11, 29),
            expiry_date=date(2024, 12, 6),
            strike=Decimal('44.5'),
            option_type='put',
            bid=Decimal('0.43'),
            ask=Decimal('0.45'),
            mid=Decimal('0.44'),
            iv=0.160773,
            delta=-0.5130319999999999,
            gamma=0.40047643,
            vega=0.02481564,
            theta=-0.02910814,
            volume=491,
            open_interest=785
        )
        
        # ACT
        actual_spread_pct = quote.spread_pct
        
        # ASSERT: 0.02 / 0.44 = 0.04545... (4.55%)
        expected_spread_pct = 0.02 / 0.44
        assert abs(actual_spread_pct - expected_spread_pct) < 0.0001, \
            f"Expected ~{expected_spread_pct:.4f}, got {actual_spread_pct:.4f}"
    
    def test_spread_pct_zero_mid(self):
        """Test spread_pct handles zero mid price gracefully
        
        Edge case: When mid=0, should return 0.0 (not raise division error).
        This tests defensive programming for zero division protection.
        """
        # ARRANGE: Option with zero mid price (pathological case)
        quote = OptionQuote(
            ticker='TEST',
            trade_date=date(2024, 1, 5),
            expiry_date=date(2024, 2, 2),
            strike=Decimal('100.0'),
            option_type='call',
            bid=Decimal('0.00'),
            ask=Decimal('0.01'),
            mid=Decimal('0.00'),  # ← Zero mid!
            iv=0.20,
            delta=0.10,
            gamma=0.01,
            vega=0.05,
            theta=-0.01,
            volume=0,
            open_interest=0
        )
        
        # ACT: Should not raise ZeroDivisionError
        actual_spread_pct = quote.spread_pct
        
        # ASSERT: Should return 0.0
        assert actual_spread_pct == 0.0, \
            f"Expected 0.0 for zero mid, got {actual_spread_pct}"
    
    def test_immutability(self):
        """Test OptionQuote is frozen and cannot be modified
        
        Should raise FrozenInstanceError when attempting to modify any attribute.
        Example: quote.ticker = 'TSLA' should raise error
        """
        pass

        quote = OptionQuote(
            ticker='VZ',
            trade_date=date(2024, 11, 29),
            expiry_date=date(2024, 12, 6),
            strike=Decimal('44.5'),
            option_type='put',
            bid=Decimal('0.43'),
            ask=Decimal('0.45'),
            mid=Decimal('0.44'),
            iv=0.160773,
            delta=-0.5130319999999999,
            gamma=0.40047643,
            vega=0.02481564,
            theta=-0.02910814,
            volume=491,
            open_interest=785
        )

        try:
            quote.ticker = 'TSLA'
            assert False, "Expected FrozenInstanceError when modifying attribute"
        except FrozenInstanceError:
            pass


# ============================================================================
# LAYER 1.2: OptionLeg Tests
# ============================================================================

class TestOptionLeg:
    """Test OptionLeg calculations (premium, intrinsic value, greeks)"""
    
    def test_long_leg_direction(self, vz_put):
        """Test long leg identified when quantity > 0"""
        # ACT: Create long leg (positive quantity)
        long_leg = OptionLeg(option=vz_put, quantity=10)
        
        # ASSERT
        assert long_leg.is_long is True
        assert long_leg.is_short is False
    
    def test_short_leg_direction(self, vz_put):
        """Test short leg identified when quantity < 0"""
        # ACT: Create short leg (negative quantity)
        short_leg = OptionLeg(option=vz_put, quantity=-10)
        
        # ASSERT
        assert short_leg.is_long is False
        assert short_leg.is_short is True
    
    def test_long_leg_premium(self, vz_put):
        """Test long leg pays premium (positive net_premium)"""
        # ACT
        long_leg = OptionLeg(option=vz_put, quantity=10)
        
        # ASSERT: 0.44 * 10 = 4.40 (positive = we pay)
        expected_premium = Decimal('0.44') * 10
        assert long_leg.net_premium == expected_premium
        assert long_leg.net_premium > 0, "Long leg should have positive premium (debit)"
    
    def test_short_leg_premium(self, vz_put):
        """Test short leg receives premium (negative net_premium)"""
        # ACT
        short_leg = OptionLeg(option=vz_put, quantity=-10)
        
        # ASSERT: 0.44 * 10 * (-1) = -4.40 (negative = we receive)
        expected_premium = Decimal('0.44') * 10 * -1
        assert short_leg.net_premium == expected_premium
        assert short_leg.net_premium < 0, "Short leg should have negative premium (credit)"
    
    def test_call_intrinsic_itm(self, vz_call):
        """Test call intrinsic when ITM: spot > strike"""
        # ACT
        leg = OptionLeg(option=vz_call, quantity=1)
        intrinsic = leg.calculate_intrinsic_value(Decimal('45.50'))
        
        # ASSERT: max(45.50 - 44.50, 0) = 1.00
        assert intrinsic == Decimal('1.00')
    
    def test_call_intrinsic_atm(self, vz_call):
        """Test call intrinsic at ATM: spot = strike"""
        # ACT
        leg = OptionLeg(option=vz_call, quantity=1)
        intrinsic = leg.calculate_intrinsic_value(Decimal('44.50'))
        
        # ASSERT: max(44.50 - 44.50, 0) = 0.00
        assert intrinsic == Decimal('0.00')
    
    def test_call_intrinsic_otm(self, vz_call):
        """Test call intrinsic when OTM: spot < strike"""
        # ACT
        leg = OptionLeg(option=vz_call, quantity=1)
        intrinsic = leg.calculate_intrinsic_value(Decimal('44.00'))
        
        # ASSERT: max(44.00 - 44.50, 0) = 0.00
        assert intrinsic == Decimal('0.00')
    
    def test_put_intrinsic_itm(self, vz_put):
        """Test put intrinsic when ITM: spot < strike"""
        # ACT
        leg = OptionLeg(option=vz_put, quantity=1)
        intrinsic = leg.calculate_intrinsic_value(Decimal('44.00'))
        
        # ASSERT: max(44.50 - 44.00, 0) = 0.50
        assert intrinsic == Decimal('0.50')
    
    def test_put_intrinsic_atm(self, vz_put):
        """Test put intrinsic at ATM: spot = strike"""
        # ACT
        leg = OptionLeg(option=vz_put, quantity=1)
        intrinsic = leg.calculate_intrinsic_value(Decimal('44.50'))
        
        # ASSERT: max(44.50 - 44.50, 0) = 0.00
        assert intrinsic == Decimal('0.00')
    
    def test_put_intrinsic_otm(self, vz_put):
        """Test put intrinsic when OTM: spot > strike"""
        # ACT
        leg = OptionLeg(option=vz_put, quantity=1)
        intrinsic = leg.calculate_intrinsic_value(Decimal('45.00'))
        
        # ASSERT: max(44.50 - 45.00, 0) = 0.00
        assert intrinsic == Decimal('0.00')
    
    def test_greek_exposures(self, vz_put):
        """Test greeks scaled by signed quantity"""
        # ACT: Long leg
        long_leg = OptionLeg(option=vz_put, quantity=10)
        
        # ASSERT: Greeks scaled by quantity
        expected_delta = vz_put.delta * 10
        expected_gamma = vz_put.gamma * 10
        expected_vega = vz_put.vega * 10
        expected_theta = vz_put.theta * 10
        
        assert abs(long_leg.delta_exposure - expected_delta) < 0.0001
        assert abs(long_leg.gamma_exposure - expected_gamma) < 0.0001
        assert abs(long_leg.vega_exposure - expected_vega) < 0.0001
        assert abs(long_leg.theta_exposure - expected_theta) < 0.0001
        
        # Test short leg (negative quantity flips sign)
        short_leg = OptionLeg(option=vz_put, quantity=-10)
        expected_delta_short = vz_put.delta * -10
        
        assert abs(short_leg.delta_exposure - expected_delta_short) < 0.0001
    
    def test_intrinsic_value_unsigned_regardless_of_quantity(self, vz_put):
        """Test intrinsic value is unsigned for both long and short legs
        
        Key insight: calculate_intrinsic_value() returns unsigned value.
        Direction is applied elsewhere (in calculate_payoff via leg.quantity).
        """
        # ACT
        long_leg = OptionLeg(option=vz_put, quantity=10)
        short_leg = OptionLeg(option=vz_put, quantity=-10)
        
        # ASSERT: Both should return same unsigned intrinsic
        spot = Decimal('44.00')
        long_intrinsic = long_leg.calculate_intrinsic_value(spot)
        short_intrinsic = short_leg.calculate_intrinsic_value(spot)
        
        # Both return 0.50 (unsigned)
        assert long_intrinsic == Decimal('0.50')
        assert short_intrinsic == Decimal('0.50')
        assert long_intrinsic == short_intrinsic


# ============================================================================
# LAYER 1.3: OptionStrategy Tests
# ============================================================================

@pytest.fixture
def atm_straddle(vz_call, vz_put):
    """ATM straddle constructed from VZ call + put"""
    call_leg = OptionLeg(option=vz_call, quantity=1)
    put_leg = OptionLeg(option=vz_put, quantity=1)
    
    return OptionStrategy(
        ticker='VZ',
        strategy_type=StrategyType.STRADDLE,
        legs=(call_leg, put_leg),
        trade_date=date(2024, 11, 29)
    )


@pytest.fixture
def synthetic_long(vz_call, vz_put):
    """Synthetic long: long call + short put at same strike
    
    Replicates long stock position with options:
    - Long call (quantity=1): unlimited upside
    - Short put (quantity=-1): unlimited downside
    
    Net premium: call.mid - put.mid = 0.40 - 0.44 = -0.04 (credit)
    Net delta: call.delta + put.delta = 0.487 + (-0.513) = -0.026 ≈ 0 (should be ~1.0 for true synthetic)
    
    Note: Real synthetic long has delta ~1.0. Our ATM example has delta ≈ 0 because
    call delta (0.487) and put delta (-0.513) nearly cancel. For proper synthetic long,
    use ITM options where call delta → 1.0 and put delta → 0.
    """
    call_leg = OptionLeg(option=vz_call, quantity=1)   # Long call
    put_leg = OptionLeg(option=vz_put, quantity=-1)    # Short put (negative quantity)
    
    return OptionStrategy(
        ticker='VZ',
        strategy_type=StrategyType.CUSTOM,
        legs=(call_leg, put_leg),
        trade_date=date(2024, 11, 29)
    )


class TestOptionStrategy:
    """Test OptionStrategy aggregation and payoff calculations"""
    
    def test_straddle_net_premium(self, atm_straddle, vz_call, vz_put):
        """Test net premium is sum of leg premiums
        
        Long straddle: pay premium for both call and put (debit).
        Expected: call.mid + put.mid = 0.40 + 0.44 = 0.84
        """
        # ACT
        net_premium = atm_straddle.net_premium
        
        # ASSERT: Sum of individual leg premiums
        expected = vz_call.mid + vz_put.mid  # 0.40 + 0.44 = 0.84
        assert net_premium == expected, f"Expected {expected}, got {net_premium}"
    
    def test_straddle_is_debit_spread(self, atm_straddle):
        """Test long straddle is a debit spread (net_premium > 0)
        
        Long straddle: we pay to enter, so positive premium.
        """
        # ACT & ASSERT
        assert atm_straddle.is_debit_spread is True
        assert atm_straddle.is_credit_spread is False
        assert atm_straddle.net_premium > 0, "Long straddle should have positive premium"
    
    def test_straddle_net_delta_near_zero(self, atm_straddle):
        """Test ATM straddle has near-zero delta
        
        ATM straddle: call delta ~+0.5, put delta ~-0.5 → net ~0.
        Tests delta neutrality of ATM straddle.
        """
        # ACT
        net_delta = atm_straddle.net_delta
        
        # ASSERT: call.delta (0.487) + put.delta (-0.513) ≈ -0.026
        assert abs(net_delta) < 0.1, f"ATM straddle should have near-zero delta, got {net_delta}"
    
    def test_straddle_payoff_up_move(self, atm_straddle):
        """Test straddle payoff when spot moves up (call ITM)
        
        Entry: strike=44.50, premium=0.84
        Exit: spot=46.00
        - Call intrinsic: 46.00 - 44.50 = 1.50
        - Put intrinsic: 0.00 (OTM)
        - Total payoff: 1.50
        """
        # ACT: Spot moves up to 46.00
        spot_at_expiry = Decimal('46.00')
        expiry_date = atm_straddle.max_expiry
        payoff = atm_straddle.calculate_payoff({expiry_date: spot_at_expiry})
        
        # ASSERT: Call intrinsic = 1.50, put = 0
        expected = Decimal('1.50')
        assert payoff == expected, f"Expected {expected}, got {payoff}"
    
    def test_straddle_payoff_down_move(self, atm_straddle):
        """Test straddle payoff when spot moves down (put ITM)
        
        Entry: strike=44.50, premium=0.84
        Exit: spot=43.00
        - Call intrinsic: 0.00 (OTM)
        - Put intrinsic: 44.50 - 43.00 = 1.50
        - Total payoff: 1.50
        """
        # ACT: Spot moves down to 43.00
        spot_at_expiry = Decimal('43.00')
        expiry_date = atm_straddle.max_expiry
        payoff = atm_straddle.calculate_payoff({expiry_date: spot_at_expiry})
        
        # ASSERT: Put intrinsic = 1.50, call = 0
        expected = Decimal('1.50')
        assert payoff == expected, f"Expected {expected}, got {payoff}"
    
    def test_straddle_payoff_no_move(self, atm_straddle):
        """Test straddle payoff at ATM expiry (both legs expire worthless)
        
        Exit: spot=44.50 (same as strike)
        - Call intrinsic: 0.00 (ATM)
        - Put intrinsic: 0.00 (ATM)
        - Total payoff: 0.00 (max loss = premium paid)
        """
        # ACT: Spot stays at strike (44.50)
        spot_at_expiry = Decimal('44.50')
        expiry_date = atm_straddle.max_expiry
        payoff = atm_straddle.calculate_payoff({expiry_date: spot_at_expiry})
        
        # ASSERT: Both legs worthless
        expected = Decimal('0.00')
        assert payoff == expected, f"Expected {expected}, got {payoff}"
    
    def test_calendar_spread_multiple_expiries(self, vz_call):
        """Test strategy with multiple expiration dates
        
        Calendar spread: different expiries for different legs.
        Tests expiry_dates, min_expiry, max_expiry properties.
        """
        # ARRANGE: Create calendar spread with two expiries
        near_expiry = date(2024, 12, 6)
        far_expiry = date(2024, 12, 20)
        
        # Near-term short call
        near_call = OptionQuote(
            ticker='VZ', trade_date=date(2024, 11, 29), expiry_date=near_expiry,
            strike=Decimal('45.0'), option_type='call',
            bid=Decimal('0.30'), ask=Decimal('0.32'), mid=Decimal('0.31'),
            iv=0.15, delta=0.45, gamma=0.35, vega=0.02, theta=-0.025,
            volume=100, open_interest=200
        )
        
        # Far-term long call
        far_call = OptionQuote(
            ticker='VZ', trade_date=date(2024, 11, 29), expiry_date=far_expiry,
            strike=Decimal('45.0'), option_type='call',
            bid=Decimal('0.50'), ask=Decimal('0.52'), mid=Decimal('0.51'),
            iv=0.15, delta=0.50, gamma=0.30, vega=0.03, theta=-0.020,
            volume=150, open_interest=300
        )
        
        calendar = OptionStrategy(
            ticker='VZ',
            strategy_type=StrategyType.CALENDAR_SPREAD,
            legs=(
                OptionLeg(option=near_call, quantity=-1),  # Short near
                OptionLeg(option=far_call, quantity=1),     # Long far
            ),
            trade_date=date(2024, 11, 29)
        )
        
        # ACT & ASSERT
        assert len(calendar.expiry_dates) == 2, "Should have 2 unique expiries"
        assert calendar.min_expiry == near_expiry
        assert calendar.max_expiry == far_expiry
        assert near_expiry in calendar.expiry_dates
        assert far_expiry in calendar.expiry_dates
    
    def test_greek_aggregation(self, atm_straddle, vz_call, vz_put):
        """Test net greeks are sum of leg exposures
        
        Straddle greeks = call greeks + put greeks
        Tests: net_delta, net_vega, net_gamma, net_theta
        """
        # ACT
        net_delta = atm_straddle.net_delta
        net_vega = atm_straddle.net_vega
        net_gamma = atm_straddle.net_gamma
        net_theta = atm_straddle.net_theta
        
        # ASSERT: Should be sum of individual leg exposures
        # Each leg has quantity=1, so exposure = greek * 1
        expected_delta = vz_call.delta + vz_put.delta
        expected_vega = vz_call.vega + vz_put.vega
        expected_gamma = vz_call.gamma + vz_put.gamma
        expected_theta = vz_call.theta + vz_put.theta
        
        assert abs(net_delta - expected_delta) < 0.0001, f"Delta: expected {expected_delta}, got {net_delta}"
        assert abs(net_vega - expected_vega) < 0.0001, f"Vega: expected {expected_vega}, got {net_vega}"
        assert abs(net_gamma - expected_gamma) < 0.0001, f"Gamma: expected {expected_gamma}, got {net_gamma}"
        assert abs(net_theta - expected_theta) < 0.0001, f"Theta: expected {expected_theta}, got {net_theta}"
    
    # ========================================================================
    # Synthetic Long Tests - Tests mixed long/short legs
    # ========================================================================
    
    def test_synthetic_long_net_premium(self, synthetic_long, vz_call, vz_put):
        """Test synthetic long net premium (credit spread)
        
        Synthetic long: long call + short put
        Net premium: call.mid - put.mid = 0.40 - 0.44 = -0.04 (credit received)
        """
        # ACT
        net_premium = synthetic_long.net_premium
        
        # ASSERT: Long call pays 0.40, short put receives 0.44 → net -0.04
        expected = vz_call.mid - vz_put.mid  # 0.40 - 0.44 = -0.04
        assert net_premium == expected, f"Expected {expected}, got {net_premium}"
    
    def test_synthetic_long_is_credit_spread(self, synthetic_long):
        """Test synthetic long is a credit spread (net_premium < 0)
        
        Synthetic long: receive more premium from short put than paid for long call.
        """
        # ACT & ASSERT
        assert synthetic_long.is_credit_spread is True
        assert synthetic_long.is_debit_spread is False
        assert synthetic_long.net_premium < 0, "Synthetic long should have negative premium (credit)"
    
    def test_synthetic_long_payoff_up_move(self, synthetic_long):
        """Test synthetic long payoff when spot moves up
        
        Spot moves from 44.50 → 46.00 (+1.50)
        - Long call: intrinsic = 1.50 (quantity=1)
        - Short put: intrinsic = 0.00 (quantity=-1, expires OTM)
        - Total payoff: +1.50 (linear profit like long stock)
        """
        # ACT: Spot moves up to 46.00
        spot_at_expiry = Decimal('46.00')
        expiry_date = synthetic_long.max_expiry
        payoff = synthetic_long.calculate_payoff({expiry_date: spot_at_expiry})
        
        # ASSERT: Call gains 1.50, put expires worthless
        expected = Decimal('1.50')
        assert payoff == expected, f"Expected {expected}, got {payoff}"
    
    def test_synthetic_long_payoff_down_move(self, synthetic_long):
        """Test synthetic long payoff when spot moves down
        
        Spot moves from 44.50 → 43.00 (-1.50)
        - Long call: intrinsic = 0.00 (expires OTM)
        - Short put: intrinsic = 1.50, but we're short (quantity=-1) → -1.50
        - Total payoff: -1.50 (linear loss like long stock)
        """
        # ACT: Spot moves down to 43.00
        spot_at_expiry = Decimal('43.00')
        expiry_date = synthetic_long.max_expiry
        payoff = synthetic_long.calculate_payoff({expiry_date: spot_at_expiry})
        
        # ASSERT: Call expires worthless, put obligation = -1.50
        expected = Decimal('-1.50')
        assert payoff == expected, f"Expected {expected}, got {payoff}"
    
    def test_synthetic_long_payoff_no_move(self, synthetic_long):
        """Test synthetic long payoff at ATM expiry
        
        Spot stays at 44.50 (strike)
        - Long call: intrinsic = 0.00 (ATM)
        - Short put: intrinsic = 0.00 (ATM)
        - Total payoff: 0.00
        """
        # ACT: Spot stays at strike
        spot_at_expiry = Decimal('44.50')
        expiry_date = synthetic_long.max_expiry
        payoff = synthetic_long.calculate_payoff({expiry_date: spot_at_expiry})
        
        # ASSERT: Both legs ATM, zero payoff
        expected = Decimal('0.00')
        assert payoff == expected, f"Expected {expected}, got {payoff}"
    
    def test_synthetic_long_greek_aggregation(self, synthetic_long, vz_call, vz_put):
        """Test synthetic long greeks with mixed long/short legs
        
        Synthetic long: long call (qty=1) + short put (qty=-1)
        Net greeks = call greeks + put greeks * (-1)
        """
        # ACT
        net_delta = synthetic_long.net_delta
        net_vega = synthetic_long.net_vega
        net_gamma = synthetic_long.net_gamma
        net_theta = synthetic_long.net_theta
        
        # ASSERT: Long call + short put
        # Short put exposure = put.greek * (-1)
        expected_delta = vz_call.delta + (vz_put.delta * -1)
        expected_vega = vz_call.vega + (vz_put.vega * -1)
        expected_gamma = vz_call.gamma + (vz_put.gamma * -1)
        expected_theta = vz_call.theta + (vz_put.theta * -1)
        
        assert abs(net_delta - expected_delta) < 0.0001, f"Delta: expected {expected_delta}, got {net_delta}"
        assert abs(net_vega - expected_vega) < 0.0001, f"Vega: expected {expected_vega}, got {net_vega}"
        assert abs(net_gamma - expected_gamma) < 0.0001, f"Gamma: expected {expected_gamma}, got {net_gamma}"
        assert abs(net_theta - expected_theta) < 0.0001, f"Theta: expected {expected_theta}, got {net_theta}"


# ============================================================================
# LAYER 1.4: Signal Tests
# ============================================================================

class TestSignal:
    """Test Signal validation and immutability"""
    
    def test_signal_validation_valid(self):
        """Test Signal accepts valid conviction values (0.0 to 1.0)
        
        Conviction must be in range [0.0, 1.0]. Test boundary and mid values.
        """
        # ARRANGE & ACT: Create signals with valid conviction values
        signal_min = Signal(
            ticker='AAPL',
            signal_date=date(2024, 12, 1),
            strategy_type=StrategyType.STRADDLE,
            direction='long',
            conviction=0.0,  # Minimum valid
            features={'mom_60_8_mean': 0.05, 'cvg_60_8': 0.3},
            metadata={'reason': 'low_conviction'}
        )
        
        signal_mid = Signal(
            ticker='MSFT',
            signal_date=date(2024, 12, 1),
            strategy_type=StrategyType.STRADDLE,
            direction='short',
            conviction=0.5,  # Mid-range
            features={'mom_60_8_mean': 0.10, 'cvg_60_8': 0.5},
            metadata={'reason': 'medium_conviction'}
        )
        
        signal_max = Signal(
            ticker='GOOG',
            signal_date=date(2024, 12, 1),
            strategy_type=StrategyType.STRADDLE,
            direction='long',
            conviction=1.0,  # Maximum valid
            features={'mom_60_8_mean': 0.25, 'cvg_60_8': 0.8},
            metadata={'reason': 'high_conviction'}
        )
        
        # ASSERT: All should be created successfully
        assert signal_min.conviction == 0.0
        assert signal_mid.conviction == 0.5
        assert signal_max.conviction == 1.0
    
    def test_signal_validation_invalid(self):
        """Test Signal rejects invalid conviction values
        
        Conviction < 0.0 or > 1.0 should raise ValueError.
        """
        # ARRANGE & ACT & ASSERT: Negative conviction
        with pytest.raises(ValueError, match="Conviction must be between 0 and 1"):
            Signal(
                ticker='AAPL',
                signal_date=date(2024, 12, 1),
                strategy_type=StrategyType.STRADDLE,
                direction='long',
                conviction=-0.1,  # Invalid: below 0
                features={},
                metadata={}
            )
        
        # ARRANGE & ACT & ASSERT: Conviction > 1.0
        with pytest.raises(ValueError, match="Conviction must be between 0 and 1"):
            Signal(
                ticker='AAPL',
                signal_date=date(2024, 12, 1),
                strategy_type=StrategyType.STRADDLE,
                direction='long',
                conviction=1.5,  # Invalid: above 1
                features={},
                metadata={}
            )
    
    def test_signal_immutability(self):
        """Test Signal is frozen and cannot be modified
        
        Should raise FrozenInstanceError when attempting to modify any attribute.
        """
        # ARRANGE
        signal = Signal(
            ticker='AAPL',
            signal_date=date(2024, 12, 1),
            strategy_type=StrategyType.STRADDLE,
            direction='long',
            conviction=0.75,
            features={'mom': 0.15},
            metadata={'strike': 150}
        )
        
        # ACT & ASSERT: Attempt to modify should raise error
        with pytest.raises(FrozenInstanceError):
            signal.ticker = 'MSFT'
        
        with pytest.raises(FrozenInstanceError):
            signal.conviction = 0.5


# ============================================================================
# LAYER 1.5: Position Tests
# ============================================================================

class TestPosition:
    """Test Position P&L calculations, state management, and greek exposures"""
    
    def test_open_position_state(self, atm_straddle):
        """Test open position has correct state flags
        
        Open position: exit_date=None, exit_value=None
        - is_open = True
        - is_closed = False
        - pnl = None (can't calculate until closed)
        - pnl_pct = None
        - holding_period = None
        """
        # ARRANGE & ACT: Create open position
        position = Position(
            ticker='VZ',
            entry_date=date(2024, 11, 29),
            strategy=atm_straddle,
            quantity=5.0,
            entry_cost=Decimal('420.00'),  # 5 straddles @ $0.84 each
            exit_date=None,
            exit_value=None,
            metadata={}
        )
        
        # ASSERT: Open state
        assert position.is_open is True
        assert position.is_closed is False
        assert position.pnl is None
        assert position.pnl_pct is None
        assert position.holding_period is None
    
    def test_closed_position_state(self, atm_straddle):
        """Test closed position has correct state flags
        
        Closed position: exit_date set, exit_value set
        - is_open = False
        - is_closed = True
        - pnl calculated
        - pnl_pct calculated
        - holding_period calculated
        """
        # ARRANGE & ACT: Create closed position
        position = Position(
            ticker='VZ',
            entry_date=date(2024, 11, 29),
            strategy=atm_straddle,
            quantity=5.0,
            entry_cost=Decimal('420.00'),
            exit_date=date(2024, 12, 6),  # 7 days later
            exit_value=Decimal('750.00'),
            metadata={}
        )
        
        # ASSERT: Closed state
        assert position.is_open is False
        assert position.is_closed is True
        assert position.pnl is not None
        assert position.pnl_pct is not None
        assert position.holding_period is not None
    
    def test_winning_debit_position_pnl(self, atm_straddle):
        """Test P&L calculation for profitable long position
        
        Long straddle (debit spread):
        - Entry: pay $420 (positive entry_cost)
        - Exit: receive $750 (positive exit_value)
        - P&L = $750 - $420 = +$330 (profit)
        """
        # ARRANGE: Long position that wins
        position = Position(
            ticker='VZ',
            entry_date=date(2024, 11, 29),
            strategy=atm_straddle,
            quantity=5.0,  # Long 5 straddles
            entry_cost=Decimal('420.00'),  # We paid $420
            exit_date=date(2024, 12, 6),
            exit_value=Decimal('750.00'),  # Closed for $750
            metadata={}
        )
        
        # ACT & ASSERT
        expected_pnl = Decimal('750.00') - Decimal('420.00')
        assert position.pnl == expected_pnl
        assert position.pnl == Decimal('330.00')
        assert position.pnl > 0, "Should be profitable"
    
    def test_losing_debit_position_pnl(self, atm_straddle):
        """Test P&L calculation for losing long position
        
        Long straddle (debit spread):
        - Entry: pay $420 (positive entry_cost)
        - Exit: receive $150 (positive exit_value, but less than entry)
        - P&L = $150 - $420 = -$270 (loss)
        """
        # ARRANGE: Long position that loses
        position = Position(
            ticker='VZ',
            entry_date=date(2024, 11, 29),
            strategy=atm_straddle,
            quantity=5.0,
            entry_cost=Decimal('420.00'),  # We paid $420
            exit_date=date(2024, 12, 6),
            exit_value=Decimal('150.00'),  # Closed for only $150
            metadata={}
        )
        
        # ACT & ASSERT
        expected_pnl = Decimal('150.00') - Decimal('420.00')
        assert position.pnl == expected_pnl
        assert position.pnl == Decimal('-270.00')
        assert position.pnl < 0, "Should be a loss"
    
    def test_winning_credit_position_pnl(self, atm_straddle):
        """Test P&L calculation for profitable short position
        
        Short straddle (credit spread):
        - Entry: receive $420 (negative entry_cost)
        - Exit: pay $150 (negative exit_value)
        - P&L = -$150 - (-$420) = +$270 (profit from keeping premium)
        """
        # ARRANGE: Short position that wins (options expire worthless)
        position = Position(
            ticker='VZ',
            entry_date=date(2024, 11, 29),
            strategy=atm_straddle,
            quantity=-5.0,  # Short 5 straddles
            entry_cost=Decimal('-420.00'),  # We received $420 credit
            exit_date=date(2024, 12, 6),
            exit_value=Decimal('-150.00'),  # We paid $150 to close
            metadata={}
        )
        
        # ACT & ASSERT
        expected_pnl = Decimal('-150.00') - Decimal('-420.00')
        assert position.pnl == expected_pnl
        assert position.pnl == Decimal('270.00')
        assert position.pnl > 0, "Should be profitable"
    
    def test_losing_credit_position_pnl(self, atm_straddle):
        """Test P&L calculation for losing short position
        
        Short straddle (credit spread):
        - Entry: receive $420 (negative entry_cost)
        - Exit: pay $750 (negative exit_value, more than we received)
        - P&L = -$750 - (-$420) = -$330 (loss)
        """
        # ARRANGE: Short position that loses (options finish ITM)
        position = Position(
            ticker='VZ',
            entry_date=date(2024, 11, 29),
            strategy=atm_straddle,
            quantity=-5.0,
            entry_cost=Decimal('-420.00'),  # We received $420 credit
            exit_date=date(2024, 12, 6),
            exit_value=Decimal('-750.00'),  # We paid $750 to close (ouch!)
            metadata={}
        )
        
        # ACT & ASSERT
        expected_pnl = Decimal('-750.00') - Decimal('-420.00')
        assert position.pnl == expected_pnl
        assert position.pnl == Decimal('-330.00')
        assert position.pnl < 0, "Should be a loss"
    
    def test_pnl_pct_calculation(self, atm_straddle):
        """Test percentage return calculation
        
        P&L % = P&L / |entry_cost|
        Example: $330 profit on $420 risk = 78.57%
        """
        # ARRANGE: Winning long position
        position = Position(
            ticker='VZ',
            entry_date=date(2024, 11, 29),
            strategy=atm_straddle,
            quantity=5.0,
            entry_cost=Decimal('420.00'),
            exit_date=date(2024, 12, 6),
            exit_value=Decimal('750.00'),
            metadata={}
        )
        
        # ACT
        pnl_pct = position.pnl_pct
        
        # ASSERT: (750 - 420) / 420 = 330 / 420 = 0.7857... (78.57%)
        expected_pct = float(Decimal('330.00') / Decimal('420.00'))
        assert pnl_pct is not None
        assert abs(pnl_pct - expected_pct) < 0.0001
        assert abs(pnl_pct - 0.7857) < 0.01
    
    def test_pnl_pct_zero_entry_cost(self, atm_straddle):
        """Test pnl_pct handles zero entry_cost edge case
        
        Edge case: If entry_cost = 0, should return None (avoid division by zero).
        This is a pathological case but tests defensive programming.
        """
        # ARRANGE: Position with zero entry cost (shouldn't happen in practice)
        position = Position(
            ticker='VZ',
            entry_date=date(2024, 11, 29),
            strategy=atm_straddle,
            quantity=5.0,
            entry_cost=Decimal('0.00'),  # Edge case
            exit_date=date(2024, 12, 6),
            exit_value=Decimal('100.00'),
            metadata={}
        )
        
        # ACT & ASSERT: Should return None, not raise ZeroDivisionError
        assert position.pnl_pct is None
    
    def test_holding_period_calculation(self, atm_straddle):
        """Test holding period calculation
        
        Holding period = exit_date - entry_date (in days)
        Example: 2024-11-29 to 2024-12-06 = 7 days
        """
        # ARRANGE
        position = Position(
            ticker='VZ',
            entry_date=date(2024, 11, 29),
            strategy=atm_straddle,
            quantity=5.0,
            entry_cost=Decimal('420.00'),
            exit_date=date(2024, 12, 6),
            exit_value=Decimal('750.00'),
            metadata={}
        )
        
        # ACT & ASSERT
        assert position.holding_period == 7
    
    def test_greek_exposures_with_quantity(self, atm_straddle, vz_call, vz_put):
        """Test position greeks are strategy greeks scaled by quantity
        
        Position greeks = strategy greeks × quantity
        Tests: net_delta, net_vega, net_gamma, net_theta
        
        Strategy net_delta = call.delta + put.delta ≈ -0.026
        Position (qty=5): net_delta = -0.026 × 5 = -0.13
        """
        # ARRANGE: Position with quantity = 5
        position = Position(
            ticker='VZ',
            entry_date=date(2024, 11, 29),
            strategy=atm_straddle,
            quantity=5.0,
            entry_cost=Decimal('420.00'),
            exit_date=None,
            exit_value=None,
            metadata={}
        )
        
        # ACT: Get position greeks
        pos_delta = position.net_delta
        pos_vega = position.net_vega
        pos_gamma = position.net_gamma
        pos_theta = position.net_theta
        
        # ASSERT: Position greeks = strategy greeks × 5
        strategy_delta = atm_straddle.net_delta
        strategy_vega = atm_straddle.net_vega
        strategy_gamma = atm_straddle.net_gamma
        strategy_theta = atm_straddle.net_theta
        
        assert abs(pos_delta - strategy_delta * 5.0) < 0.0001
        assert abs(pos_vega - strategy_vega * 5.0) < 0.0001
        assert abs(pos_gamma - strategy_gamma * 5.0) < 0.0001
        assert abs(pos_theta - strategy_theta * 5.0) < 0.0001
    
    def test_strategy_type_property(self, atm_straddle):
        """Test strategy_type convenience property
        
        position.strategy_type should return strategy.strategy_type
        """
        # ARRANGE
        position = Position(
            ticker='VZ',
            entry_date=date(2024, 11, 29),
            strategy=atm_straddle,
            quantity=5.0,
            entry_cost=Decimal('420.00'),
            exit_date=None,
            exit_value=None,
            metadata={}
        )
        
        # ACT & ASSERT
        assert position.strategy_type == StrategyType.STRADDLE
        assert position.strategy_type == atm_straddle.strategy_type
