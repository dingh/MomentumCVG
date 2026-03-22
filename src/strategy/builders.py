"""
Strategy builders: construct option strategies from signals.

This module provides builders that convert trading signals into concrete
OptionStrategy objects using current market data.

Key Design Principle:
- Builders are PURE FUNCTIONS: signal + market data → strategy
- NO knowledge of backtest vs live trading
- Same code used in historical analysis AND live execution
"""

from dataclasses import dataclass
from typing import Protocol, List, Optional, Tuple
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


@dataclass(frozen=True)
class IronButterflyCandidate:
    """
    Immutable snapshot of one valid symmetric wing candidate.

    Computed once by ``enumerate_candidates()`` and returned to callers.
    Enables post-hoc analysis scripts to persist the full wing surface for
    a fixed body/expiry without rebuilding the chain repeatedly.

    All Decimal fields are per-share (matching OptionQuote.mid convention).
    Greeks are net across all 4 legs (signed by qty: short=-1, long=+1).
    Note: put deltas on OptionQuote are already negative (e.g. -0.15),
    so net_delta arithmetic works directly with raw delta values.
    """
    # Wing geometry
    body_strike: Decimal
    wing_width: Decimal          # call_wing_strike - body_strike (symmetric)
    call_wing_strike: Decimal
    put_wing_strike: Decimal

    # Economics
    net_credit: Decimal          # short body premium - long wing cost  (> 0)
    credit_to_width: float       # net_credit / wing_width  (= return on max-loss capital)
    return_on_max_loss: float    # alias for credit_to_width (for clarity in analysis)
    total_spread: Decimal        # sum of (ask - bid) across all 4 legs (entry slippage budget)
    spread_cost_ratio: float     # total_spread / net_credit  (e.g. 0.10 = spread costs 10% of credit)

    # Wing delta score
    avg_wing_delta: float        # (|long_call.delta| + |long_put.delta|) / 2

    # Net greeks across all 4 legs
    net_delta: float
    net_gamma: float
    net_vega: float
    net_theta: float

    # Underlying option quotes
    short_call: OptionQuote
    short_put: OptionQuote
    long_call: OptionQuote
    long_put: OptionQuote


class IronButterflyBuilder:
    """
    Build ATM iron butterfly strategies from market data.

    An iron butterfly consists of 4 legs on the same expiry:
        - Short call  at the body strike (ATM, qty=-1)
        - Short put   at the body strike (ATM, qty=-1)
        - Long  call  above the body    (OTM wing,  qty=+1)
        - Long  put   below the body    (OTM wing,  qty=+1)

    The two wings are **equidistant** from the body (symmetric width).
    Wing placement is driven by ``wing_delta``: all OTM call strikes above
    the body are enumerated as candidates; only those that have a perfectly
    mirrored put strike at ``body - (candidate - body)`` are kept; the
    candidate pair whose average long-wing |delta| is closest to
    ``wing_delta`` is selected.

    This is a **credit strategy** — the body premium exceeds the wing cost.
    Validation ensures minimum spread quality and yield-on-capital.

    Args:
        wing_delta:          Target absolute delta of the long OTM wings.
                             E.g. 0.15 selects ~15-delta wings. Default 0.15.
        max_spread_pct:      Maximum bid-ask spread as fraction of mid for
                             any individual leg. Default 0.25 (25%).
        min_yield_on_capital: Minimum net credit / wing_width (both per-share).
                             Screens out low-value butterflies. Default 0.05.

    Output is a **unit strategy** (qty magnitudes are 1). Direction and
    position sizing are handled downstream by the optimizer.

    Example::

        >>> builder = IronButterflyBuilder(wing_delta=0.15)
        >>> strategy = builder.build_strategy(
        ...     ticker='AAPL',
        ...     trade_date=date(2023, 1, 6),
        ...     expiry_date=date(2023, 2, 3),
        ...     option_chain=chain,
        ...     spot_price=Decimal('150.00')
        ... )
        >>> # Body at $150, wings at $140 (put) and $160 (call)
        >>> strategy.strategy_type
        <StrategyType.IRON_BUTTERFLY: 'iron_butterfly'>
        >>> len(strategy.legs)
        4
        >>> strategy.net_premium          # Negative = credit received
        Decimal('-4.20')
        >>> # wing_width = 10.00, net_credit = 4.20
        >>> # yield_on_capital = 4.20 / 10.00 = 0.42 (42%)
    """

    def __init__(
        self,
        wing_delta: float = 0.15,
        max_spread_pct: float = 0.25,
        min_yield_on_capital: float = 0.05,
        selection_mode: str = "closest_delta",
    ):
        """
        Initialize iron butterfly builder.

        Args:
            wing_delta:           Target |delta| for OTM long wings (default 0.15).
            max_spread_pct:       Max bid-ask spread / mid per leg (default 0.25).
            min_yield_on_capital: Min net_credit / wing_width (default 0.05).
            selection_mode:       How to pick among valid candidates:
                                  ``'closest_delta'``  – wing whose avg |delta| is
                                      nearest to ``wing_delta`` (default).
                                  ``'max_credit_to_width'`` – highest net_credit /
                                      wing_width ratio.
                                  ``'max_credit'`` – highest absolute net credit.
        """
        if not 0 < wing_delta < 0.5:
            raise ValueError(
                f"wing_delta must be between 0 and 0.5, got {wing_delta}. "
                "Long wings should be OTM (delta < 0.5)."
            )
        if not 0 < max_spread_pct <= 1:
            raise ValueError(
                f"max_spread_pct must be between 0 and 1, got {max_spread_pct}."
            )
        if min_yield_on_capital < 0:
            raise ValueError(
                f"min_yield_on_capital must be non-negative, got {min_yield_on_capital}."
            )
        valid_modes = {"closest_delta", "max_credit_to_width", "max_credit"}
        if selection_mode not in valid_modes:
            raise ValueError(
                f"selection_mode must be one of {valid_modes}, got {selection_mode!r}."
            )
        self.wing_delta = wing_delta
        self.max_spread_pct = max_spread_pct
        self.min_yield_on_capital = min_yield_on_capital
        self.selection_mode = selection_mode

    # ------------------------------------------------------------------
    # IStrategyBuilder interface
    # ------------------------------------------------------------------

    def build_strategy(
        self,
        ticker: str,
        trade_date: date,
        expiry_date: date,
        option_chain: List[OptionQuote],
        spot_price: Decimal
    ) -> OptionStrategy:
        """
        Build ATM iron butterfly from market data.

        Algorithm:
            1. Validate chain (non-empty, single expiry matching ``expiry_date``).
            2. Find body strike (ATM, closest to ``spot_price``).
            3. Fetch short call and short put at body; validate mid > 0.
            4. Enumerate all OTM call strikes above body that have an exact
               mirror put strike at ``body - (call_strike - body)``
               (symmetric wing candidates).
            5. For each candidate pair, confirm both long options exist with
               mid > 0.  Score by ``|avg_|delta| - wing_delta|``;  select
               the closest match.
            6. Validate all 4 legs pass ``max_spread_pct``.
            7. Compute net credit and yield-on-capital; raise if below
               ``min_yield_on_capital``.
            8. Assemble and return the OptionStrategy.

        Args:
            ticker:        Underlying ticker symbol.
            trade_date:    Date the strategy is being constructed.
            expiry_date:   Target expiration date (all legs share this expiry).
            option_chain:  List of OptionQuote objects pre-filtered to one expiry.
            spot_price:    Current underlying price.

        Returns:
            OptionStrategy with 4 legs ordered
            [long put wing, short put body, short call body, long call wing].
            Legs signed: short legs qty=-1, long wing legs qty=+1.
            ``strategy.net_premium`` will be negative (credit received).

        Raises:
            ValueError: Empty chain, multiple expiries, expiry mismatch,
                        missing body options, no valid symmetric wing pair,
                        spread too wide, or yield-on-capital below threshold.

        Example::

            >>> # AAPL at $150; 30-DTE chain
            >>> strategy = builder.build_strategy(
            ...     ticker='AAPL',
            ...     trade_date=date(2023, 1, 6),
            ...     expiry_date=date(2023, 2, 3),
            ...     option_chain=chain,
            ...     spot_price=Decimal('150.00')
            ... )
            >>> strategy.net_premium          # credit
            Decimal('-4.20')
            >>> [leg.option.strike for leg in strategy.legs]
            [Decimal('140'), Decimal('150'), Decimal('150'), Decimal('160')]
            >>> [leg.quantity for leg in strategy.legs]
            [1, -1, -1, 1]
        """
        # Validate then delegate — keeps IStrategyBuilder contract intact
        self._validate_chain(ticker, trade_date, expiry_date, option_chain)
        body_strike = self._find_atm_strike(option_chain, spot_price)
        return self.build_strategy_at_body(
            ticker=ticker,
            trade_date=trade_date,
            expiry_date=expiry_date,
            option_chain=option_chain,
            body_strike=body_strike,
        )

    def build_strategy_at_body(
        self,
        ticker: str,
        trade_date: date,
        expiry_date: date,
        option_chain: List[OptionQuote],
        body_strike: Decimal,
    ) -> OptionStrategy:
        """
        Build iron butterfly around an explicitly supplied body strike.

        Useful when comparing straddle vs iron butterfly on the same
        opportunity: both builders can be forced to use the identical body
        strike instead of independently snapping to ATM.

        Chain validation is the caller's responsibility (call
        ``_validate_chain()`` first, or use ``build_strategy()`` which
        does both steps automatically).

        Args:
            ticker:       Underlying ticker symbol.
            trade_date:   Date the strategy is being constructed.
            expiry_date:  Target expiration date.
            option_chain: Pre-validated option chain (single expiry).
            body_strike:  Strike to use as the short body (ATM centre).

        Returns:
            OptionStrategy with 4 legs ordered
            [long put wing, short put body, short call body, long call wing].

        Raises:
            ValueError: Missing/invalid body options or no valid wing pair
                        after spread and yield filters.
        """
        short_call = self._get_option_at_strike(option_chain, body_strike, 'call')
        short_put  = self._get_option_at_strike(option_chain, body_strike, 'put')

        if short_call is None:
            raise ValueError(
                f"No call option found at body strike ${body_strike} "
                f"for {ticker} expiring {expiry_date}"
            )
        if short_put is None:
            raise ValueError(
                f"No put option found at body strike ${body_strike} "
                f"for {ticker} expiring {expiry_date}"
            )
        if short_call.mid <= 0:
            raise ValueError(
                f"Invalid short call mid ${short_call.mid} at body strike ${body_strike}."
            )
        if short_put.mid <= 0:
            raise ValueError(
                f"Invalid short put mid ${short_put.mid} at body strike ${body_strike}."
            )

        candidates = self.enumerate_candidates(
            option_chain=option_chain,
            body_strike=body_strike,
            short_call=short_call,
            short_put=short_put,
        )

        if not candidates:
            raise ValueError(
                f"No valid symmetric wing pairs found for {ticker} "
                f"expiring {expiry_date} around body strike ${body_strike}. "
                "Check that the chain has OTM strikes on both sides with "
                "positive mid prices, acceptable spreads, and sufficient yield."
            )

        # Select best candidate based on selection_mode
        if self.selection_mode == "closest_delta":
            best = min(candidates, key=lambda c: abs(c.avg_wing_delta - self.wing_delta))
        elif self.selection_mode == "max_credit_to_width":
            best = max(candidates, key=lambda c: c.credit_to_width)
        elif self.selection_mode == "max_credit":
            best = max(candidates, key=lambda c: c.net_credit)
        else:
            raise ValueError(f"Unknown selection_mode: {self.selection_mode!r}")

        legs = (
            OptionLeg(option=best.long_put,   quantity=1),
            OptionLeg(option=best.short_put,   quantity=-1),
            OptionLeg(option=best.short_call,  quantity=-1),
            OptionLeg(option=best.long_call,   quantity=1),
        )

        return OptionStrategy(
            ticker=ticker,
            strategy_type=StrategyType.IRON_BUTTERFLY,
            legs=legs,
            trade_date=trade_date,
        )

    def enumerate_candidates(
        self,
        option_chain: List[OptionQuote],
        body_strike: Decimal,
        short_call: OptionQuote,
        short_put: OptionQuote,
    ) -> List[IronButterflyCandidate]:
        """
        Return all valid symmetric wing candidates around ``body_strike``.

        Candidates that fail spread quality or yield-on-capital filters are
        silently excluded — the caller sees only structurally buildable wings.
        This lets analysis scripts iterate over the full candidate surface
        without the builder raising exceptions.

        Args:
            option_chain: Full pre-validated option chain.
            body_strike:  ATM body strike (shared by short_call and short_put).
            short_call:   Short call body leg (already fetched and validated).
            short_put:    Short put body leg (already fetched and validated).

        Returns:
            List of ``IronButterflyCandidate`` objects, one per valid symmetric
            wing pair, sorted ascending by wing_width.
        """
        chain_lookup: dict = {
            (opt.strike, opt.option_type): opt for opt in option_chain
        }
        otm_call_strikes = sorted(
            {opt.strike for opt in option_chain if opt.strike > body_strike}
        )

        candidates: List[IronButterflyCandidate] = []
        for call_strike in otm_call_strikes:
            wing_width = call_strike - body_strike
            put_strike = body_strike - wing_width  # mirror strike

            long_call = chain_lookup.get((call_strike, 'call'))
            long_put  = chain_lookup.get((put_strike,  'put'))

            # Both symmetric legs must exist with tradeable premiums
            if long_call is None or long_put is None:
                continue
            if long_call.mid <= 0 or long_put.mid <= 0:
                continue

            # Spread filter — silently skip illiquid candidates
            all_legs = [short_call, short_put, long_call, long_put]
            if any(opt.spread_pct > self.max_spread_pct for opt in all_legs):
                continue

            # Yield filter — silently skip uneconomical candidates
            net_credit = (
                (short_call.mid + short_put.mid)
                - (long_call.mid + long_put.mid)
            )
            if net_credit <= 0:
                continue
            credit_to_width = float(net_credit / wing_width)
            if credit_to_width < self.min_yield_on_capital:
                continue

            avg_wing_delta = (abs(long_call.delta) + abs(long_put.delta)) / 2.0

            total_spread = (
                (short_call.ask - short_call.bid)
                + (short_put.ask  - short_put.bid)
                + (long_call.ask  - long_call.bid)
                + (long_put.ask   - long_put.bid)
            )
            spread_cost_ratio = float(total_spread / net_credit)

            # Net greeks across all 4 legs (signed by qty).
            # Put deltas on OptionQuote are already negative (e.g. -0.15)
            # so arithmetic works directly with raw delta values.
            net_delta = (
                long_put.delta  * 1
                + short_put.delta  * -1
                + short_call.delta * -1
                + long_call.delta  * 1
            )
            net_gamma = (
                long_put.gamma + long_call.gamma
                - short_put.gamma - short_call.gamma
            )
            net_vega = (
                long_put.vega + long_call.vega
                - short_put.vega - short_call.vega
            )
            net_theta = (
                long_put.theta + long_call.theta
                - short_put.theta - short_call.theta
            )

            candidates.append(IronButterflyCandidate(
                body_strike=body_strike,
                wing_width=wing_width,
                call_wing_strike=call_strike,
                put_wing_strike=put_strike,
                net_credit=net_credit,
                credit_to_width=credit_to_width,
                return_on_max_loss=credit_to_width,
                total_spread=total_spread,
                spread_cost_ratio=spread_cost_ratio,
                avg_wing_delta=avg_wing_delta,
                net_delta=net_delta,
                net_gamma=net_gamma,
                net_vega=net_vega,
                net_theta=net_theta,
                short_call=short_call,
                short_put=short_put,
                long_call=long_call,
                long_put=long_put,
            ))

        return candidates

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_chain(
        self,
        ticker: str,
        trade_date: date,
        expiry_date: date,
        option_chain: List[OptionQuote],
    ) -> None:
        """
        Validate that the option chain satisfies the builder's preconditions.

        Extracted so ``build_strategy()`` and any future public entry points
        share a single validation path without duplication.

        
        Raises:
            ValueError: Empty chain, multiple expiries, or expiry mismatch.
        """
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
                f"Option chain expiry mismatch. Expected {expiry_date}, "
                f"got {actual_expiry}"
            )

    def _find_atm_strike(
        self,
        option_chain: List[OptionQuote],
        spot_price: Decimal
    ) -> Decimal:
        """
        Find ATM strike closest to spot price.

        If two strikes are equidistant, returns the lower one.

        Args:
            option_chain: List of option quotes.
            spot_price:   Current underlying price.

        Returns:
            Strike price closest to spot.

        Raises:
            ValueError: If no strikes available in chain.
        """
        strikes = sorted({opt.strike for opt in option_chain})
        if not strikes:
            raise ValueError("No strikes found in option chain")
        return min(strikes, key=lambda s: (abs(s - spot_price), s))

    def _get_option_at_strike(
        self,
        option_chain: List[OptionQuote],
        strike: Decimal,
        option_type: str
    ) -> Optional[OptionQuote]:
        """
        Fetch a specific option from the chain by strike and type.

        Args:
            option_chain: List of option quotes.
            strike:       Target strike price.
            option_type:  ``'call'`` or ``'put'``.

        Returns:
            Matching OptionQuote, or None if not found.
        """
        for opt in option_chain:
            if opt.strike == strike and opt.option_type == option_type:
                return opt
        return None

    def _select_wing_pair(
        self,
        option_chain: List[OptionQuote],
        body_strike: Decimal,
        ticker: str,
        expiry_date: date,
    ) -> tuple:
        """
        Select the best symmetric wing pair closest to ``self.wing_delta``.

        Iterates over every OTM call strike above ``body_strike``.  A
        candidate call wing is valid when:
            * The mirror put strike ``body - (call_strike - body)`` exists
              in the chain (guarantees symmetry).
            * Both long options have ``mid > 0``.

        Among all valid candidates the one whose average long-wing |delta|
        is nearest to ``self.wing_delta`` is returned.

        Args:
            option_chain: Full option chain for the expiry.
            body_strike:  ATM body strike.
            ticker:       Ticker name (used in error messages).
            expiry_date:  Expiry date (used in error messages).

        Returns:
            ``(long_call_wing, long_put_wing)`` OptionQuote tuple.

        Raises:
            ValueError: If no valid symmetric candidate pairs are found.
        """
        # Build lookup: (strike, option_type) -> OptionQuote for fast access
        chain_lookup = {
            (opt.strike, opt.option_type): opt
            for opt in option_chain
        }

        # All distinct OTM call strikes above the body
        otm_call_strikes = sorted(
            {opt.strike for opt in option_chain if opt.strike > body_strike}
        )

        candidates = []
        for call_strike in otm_call_strikes:
            wing_width  = call_strike - body_strike
            put_strike  = body_strike - wing_width  # mirror strike

            long_call = chain_lookup.get((call_strike, 'call'))
            long_put  = chain_lookup.get((put_strike, 'put'))

            # Both legs must exist and have tradeable premiums
            if long_call is None or long_put is None:
                continue
            if long_call.mid <= 0 or long_put.mid <= 0:
                continue

            # Score by proximity of average |delta| to wing_delta target
            avg_abs_delta = (abs(long_call.delta) + abs(long_put.delta)) / 2.0
            score = abs(avg_abs_delta - self.wing_delta)
            candidates.append((score, long_call, long_put))

        if not candidates:
            raise ValueError(
                f"No valid symmetric wing pairs found for {ticker} "
                f"expiring {expiry_date} around body strike ${body_strike}. "
                "Check that the chain has OTM strikes on both sides with "
                "positive mid prices."
            )

        # Pick candidate with score closest to zero
        candidates.sort(key=lambda x: x[0])
        _, long_call_wing, long_put_wing = candidates[0]
        return long_call_wing, long_put_wing

    def _compute_yield_on_capital(
        self,
        net_credit: Decimal,
        wing_width: Decimal,
    ) -> float:
        """
        Compute yield-on-capital for the iron butterfly.

        Both inputs are per-share values.  The wing_width is the max loss
        per share on either side (capital at risk per share for the strategy).

        Args:
            net_credit: Premium collected (short body) minus premium paid
                        (long wings), per share. Should be positive.
            wing_width: Distance from body to either wing strike, per share.

        Returns:
            Decimal fraction (e.g. 0.42 = 42%).

        Example::

            >>> builder._compute_yield_on_capital(
            ...     Decimal('4.20'), Decimal('10.00')
            ... )
            0.42
        """
        if wing_width <= 0:
            raise ValueError(f"wing_width must be positive, got {wing_width}")
        return float(net_credit / wing_width)

