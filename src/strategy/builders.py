"""
Strategy builders: construct option strategies from signals.

This module provides builders that convert trading signals into concrete
OptionStrategy objects using current market data.

Key Design Principle:
- Builders are PURE FUNCTIONS: signal + market data → strategy
- NO knowledge of backtest vs live trading
- Same code used in historical analysis AND live execution
"""

from typing import Protocol, List, Optional
from datetime import date
from decimal import Decimal

from src.core.models import Signal, OptionQuote, OptionStrategy, OptionLeg, StrategyType


class IStrategyBuilder(Protocol):
    """
    Protocol for strategy builders.
    
    Builders are pure functions that construct option strategies from
    market data. They have NO knowledge of:
    - Trade direction (long/short)
    - Position sizing
    - Signal logic
    
    They only define strategy STRUCTURE (which options, which strikes).
    """
    
    def build_strategy(
        self,
        ticker: str,
        trade_date: date,
        expiry_date: date,
        option_chain: List[OptionQuote],
        spot_price: Decimal
    ) -> OptionStrategy:
        """
        Build option strategy from market data.
        
        Returns unit strategy (quantity=1) with structure only.
        Optimizer handles scaling and direction.
        """
        ...


class StraddleBuilder:
    """
    Build ATM straddle strategies from market data.
    
    A straddle consists of:
    - 1 ATM call (quantity = +1)
    - 1 ATM put (quantity = +1)
    - Same strike (closest to spot price)
    - Same expiry
    
    Output is a "unit strategy" (quantity=1 for both legs). The optimizer
    will later scale quantity and apply direction (long/short).
    
    Example:
        >>> builder = StraddleBuilder()
        >>> strategy = builder.build_strategy(
        ...     ticker='AAPL',
        ...     trade_date=date(2023, 1, 6),
        ...     expiry_date=date(2023, 2, 3),
        ...     option_chain=chain,  # From data provider
        ...     spot_price=Decimal('150.00')
        ... )
        >>> strategy.legs[0].quantity  # Call
        1
        >>> strategy.legs[1].quantity  # Put
        1
        >>> strategy.net_premium  # Cost of 1 unit
        Decimal('7.50')
    """
    
    def __init__(self):
        """Initialize straddle builder."""
        pass
    
    def build_strategy(
        self,
        ticker: str,
        trade_date: date,
        expiry_date: date,
        option_chain: List[OptionQuote],
        spot_price: Decimal
    ) -> OptionStrategy:
        """
        Build ATM straddle from market data.
        
        Creates a unit straddle (1 call + 1 put at ATM strike). Direction
        and position sizing are handled downstream by the optimizer.
        
        Steps:
        1. Find ATM strike (closest to spot price)
        2. Get call and put at that strike
        3. Create legs with quantity=1 (unit strategy)
        4. Return OptionStrategy
        
        Args:
            ticker: Underlying ticker symbol
            trade_date: Date strategy is being constructed
            expiry_date: Target expiration date for options
            option_chain: List of available options at expiry_date
                         (pre-filtered by data provider)
            spot_price: Current price of underlying
            
        Returns:
            OptionStrategy with 2 legs (call + put), both quantity=1
            
        Raises:
            ValueError: If ATM strike not found, options missing, or invalid premiums
            
        Example:
            >>> # AAPL at $150, ATM options available
            >>> strategy = builder.build_straddle(
            ...     ticker='AAPL',
            ...     trade_date=date(2023, 1, 6),
            ...     expiry_date=date(2023, 2, 3),
            ...     option_chain=chain,
            ...     spot_price=Decimal('150.00')
            ... )
            >>> 
            >>> # Result: ATM straddle at $150 strike
            >>> strategy.strategy_type
            <StrategyType.STRADDLE: 'straddle'>
            >>> len(strategy.legs)
            2
            >>> strategy.legs[0].option.strike
            Decimal('150.00')
            >>> strategy.net_premium  # Call $5.30 + Put $2.20
            Decimal('7.50')
        """
        # Validate inputs
        if not option_chain:
            raise ValueError(f"Empty option chain for {ticker} on {trade_date}")
        
        # Validate expiry consistency
        expiries = {opt.expiry_date for opt in option_chain}
        
        if len(expiries) == 0:
            # Should never happen due to check above, but be defensive
            raise ValueError(
                f"No expiry dates found in option chain for {ticker} on {trade_date}"
            )
        
        if len(expiries) > 1:
            raise ValueError(
                f"Option chain contains multiple expiries: {sorted(expiries)}. "
                f"Expected single expiry: {expiry_date}"
            )
        
        # Now we know there's exactly 1 expiry
        actual_expiry = next(iter(expiries))
        if actual_expiry != expiry_date:
            raise ValueError(
                f"Option chain expiry mismatch. Expected {expiry_date}, "
                f"got {actual_expiry}"
            )
        
        # Find ATM strike
        atm_strike = self._find_atm_strike(option_chain, spot_price)
        
        # Get call and put at ATM strike
        atm_call = self._get_option_at_strike(option_chain, atm_strike, 'call')
        atm_put = self._get_option_at_strike(option_chain, atm_strike, 'put')
        
        # Validate options exist
        if atm_call is None:
            raise ValueError(
                f"No call option found at ATM strike ${atm_strike} "
                f"for {ticker} expiring {expiry_date}"
            )
        if atm_put is None:
            raise ValueError(
                f"No put option found at ATM strike ${atm_strike} "
                f"for {ticker} expiring {expiry_date}"
            )
        
        # Validate premiums are positive
        if atm_call.mid <= 0:
            raise ValueError(
                f"Invalid call premium ${atm_call.mid} at strike ${atm_strike}. "
                f"Check data quality filters."
            )
        if atm_put.mid <= 0:
            raise ValueError(
                f"Invalid put premium ${atm_put.mid} at strike ${atm_strike}. "
                f"Check data quality filters."
            )
        
        # Create unit strategy (quantity = 1 for both legs)
        # Optimizer will later scale and apply direction
        call_leg = OptionLeg(option=atm_call, quantity=1)
        put_leg = OptionLeg(option=atm_put, quantity=1)
        
        # Build strategy
        strategy = OptionStrategy(
            ticker=ticker,
            strategy_type=StrategyType.STRADDLE,
            legs=(call_leg, put_leg),
            trade_date=trade_date
        )
        
        return strategy
    
    def _find_atm_strike(
        self,
        option_chain: List[OptionQuote],
        spot_price: Decimal
    ) -> Decimal:
        """
        Find ATM (at-the-money) strike closest to spot price.
        
        Uses Euclidean distance to find closest strike. If two strikes
        are equidistant, returns the lower one (standard convention).
        
        Args:
            option_chain: List of option quotes
            spot_price: Current underlying price
            
        Returns:
            Strike price closest to spot
            
        Raises:
            ValueError: If no strikes available in chain
            
        Example:
            >>> # Spot = $150.25, strikes = [$145, $150, $155]
            >>> atm_strike = self._find_atm_strike(chain, Decimal('150.25'))
            >>> atm_strike
            Decimal('150.00')  # Closest to spot
        """
        # Get unique strikes from chain
        strikes = sorted({opt.strike for opt in option_chain})
        
        if not strikes:
            raise ValueError("No strikes found in option chain")
        
        # Find strike with minimum distance to spot
        # If tie (e.g., spot=150.5, strikes=[150, 151]), returns lower strike
        atm_strike = min(strikes, key=lambda s: (abs(s - spot_price), s))
        
        return atm_strike
    
    def _get_option_at_strike(
        self,
        option_chain: List[OptionQuote],
        strike: Decimal,
        option_type: str
    ) -> Optional[OptionQuote]:
        """
        Get specific option from chain by strike and type.
        
        Args:
            option_chain: List of option quotes
            strike: Target strike price
            option_type: 'call' or 'put'
            
        Returns:
            OptionQuote if found, None otherwise
            
        Example:
            >>> call = self._get_option_at_strike(chain, Decimal('150'), 'call')
            >>> call.strike
            Decimal('150.00')
            >>> call.option_type
            'call'
        """
        for opt in option_chain:
            if opt.strike == strike and opt.option_type == option_type:
                return opt
        
        return None


class IronCondorBuilder:
    """
    Build iron condor strategies using delta-targeted strike selection.

    A standard iron condor consists of four legs (same expiry):
    - Short put  (nearer-to-money put)
    - Long put   (farther OTM put, lower strike)
    - Short call (nearer-to-money call)
    - Long call  (farther OTM call, higher strike)

    Strike selection is delta-based:
    - Short legs target +/- short_delta
    - Long legs target +/- long_delta
    """

    def __init__(self, short_delta: float = 0.30, long_delta: float = 0.15):
        if not (0 < long_delta < short_delta < 1):
            raise ValueError(
                "Expected 0 < long_delta < short_delta < 1 for iron condor builder"
            )

        self.short_delta = short_delta
        self.long_delta = long_delta

    def build_strategy(
        self,
        ticker: str,
        trade_date: date,
        expiry_date: date,
        option_chain: List[OptionQuote],
        spot_price: Decimal,
    ) -> OptionStrategy:
        """Build a unit iron condor from market data."""
        del spot_price  # Delta-based builder does not currently use spot.

        if not option_chain:
            raise ValueError(f"Empty option chain for {ticker} on {trade_date}")

        expiries = {opt.expiry_date for opt in option_chain}
        if len(expiries) > 1:
            raise ValueError(
                f"Option chain contains multiple expiries: {sorted(expiries)}. "
                f"Expected single expiry: {expiry_date}"
            )

        actual_expiry = next(iter(expiries))
        if actual_expiry != expiry_date:
            raise ValueError(
                f"Option chain expiry mismatch. Expected {expiry_date}, got {actual_expiry}"
            )

        puts = [opt for opt in option_chain if opt.option_type == 'put']
        calls = [opt for opt in option_chain if opt.option_type == 'call']

        short_put_candidates = [
            opt for opt in puts if any(other.strike < opt.strike for other in puts)
        ]
        short_call_candidates = [
            opt for opt in calls if any(other.strike > opt.strike for other in calls)
        ]

        short_put = self._find_by_target_delta(
            short_put_candidates, -self.short_delta, "short put"
        )
        short_call = self._find_by_target_delta(
            short_call_candidates, self.short_delta, "short call"
        )

        long_put_candidates = [opt for opt in puts if opt.strike < short_put.strike]
        long_call_candidates = [opt for opt in calls if opt.strike > short_call.strike]

        long_put = self._find_by_target_delta(
            long_put_candidates, -self.long_delta, "long put"
        )
        long_call = self._find_by_target_delta(
            long_call_candidates, self.long_delta, "long call"
        )

        for leg_name, option in {
            "short put": short_put,
            "long put": long_put,
            "short call": short_call,
            "long call": long_call,
        }.items():
            if option.mid <= 0:
                raise ValueError(
                    f"Invalid {leg_name} premium ${option.mid} at strike ${option.strike}. "
                    "Check data quality filters."
                )

        return OptionStrategy(
            ticker=ticker,
            strategy_type=StrategyType.IRON_CONDOR,
            legs=(
                OptionLeg(option=short_put, quantity=-1),
                OptionLeg(option=long_put, quantity=1),
                OptionLeg(option=short_call, quantity=-1),
                OptionLeg(option=long_call, quantity=1),
            ),
            trade_date=trade_date,
        )

    def _find_by_target_delta(
        self,
        options: List[OptionQuote],
        target_delta: float,
        leg_name: str,
    ) -> OptionQuote:
        if not options:
            raise ValueError(f"No candidates available for {leg_name} leg")

        return min(options, key=lambda opt: abs(opt.delta - target_delta))
