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
    target_bucket_delta: float   # nearest target in wing_delta_targets this candidate claimed

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
        max_spread_cost_ratio: Maximum total bid-ask spread as fraction of net
                             credit. Default 0.25 (25% of collected credit).
        min_yield_on_capital: Minimum net credit / wing_width (both per-share).
                             Screens out low-value butterflies. Default 0.05.

    Wing selection always uses the ``closest_delta`` rule: the candidate whose
    ``avg_wing_delta`` is nearest to ``wing_delta`` is chosen.

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
        max_spread_cost_ratio: float = 0.25,
        min_yield_on_capital: float = 0.05,
    ):
        """
        Initialize iron butterfly builder.

        Args:
            wing_delta:             Target |delta| for OTM long wings (default 0.15).
            max_spread_cost_ratio:  Max total_spread / net_credit (default 0.25).
                                    Filters candidates where total bid-ask slippage
                                    exceeds this fraction of collected credit.
            min_yield_on_capital:   Min net_credit / wing_width (default 0.05).
        """
        if not 0 < wing_delta < 0.5:
            raise ValueError(
                f"wing_delta must be between 0 and 0.5, got {wing_delta}. "
                "Long wings should be OTM (delta < 0.5)."
            )
        if max_spread_cost_ratio <= 0:
            raise ValueError(
                f"max_spread_cost_ratio must be positive, got {max_spread_cost_ratio}."
            )
        if min_yield_on_capital < 0:
            raise ValueError(
                f"min_yield_on_capital must be non-negative, got {min_yield_on_capital}."
            )
        self.wing_delta = wing_delta
        self.max_spread_cost_ratio = max_spread_cost_ratio
        self.min_yield_on_capital = min_yield_on_capital

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
            4. Enumerate symmetric wing candidates via ``enumerate_candidates()``:
               - For each OTM call strike, require a mirrored put at
                 ``body - (call_strike - body)``.
               - Filter: net credit > 0, total_spread / net_credit <=
                 ``max_spread_cost_ratio``, credit / wing_width >=
                 ``min_yield_on_capital``.
               - Bucket survivors by nearest target in ``[0.05, 0.10, 0.15, 0.20, 0.30]``;
                 keep at most one per bucket.
            5. Select best candidate according to ``selection_mode``
               (default ``'closest_delta'``: nearest ``avg_wing_delta`` to
               ``wing_delta``).
            6. Assemble and return the OptionStrategy.

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

        # Always include self.wing_delta as an explicit bucket target so the
        # nearest-neighbour bucketing in enumerate_candidates() never silently
        # displaces the candidate that best matches the configured wing_delta.
        # e.g. wing_delta=0.17 sits between 0.15 and 0.20 targets — without
        # this injection a candidate at exactly 0.17 could lose its bucket to
        # a neighbour and be dropped before selection_mode ever sees it.
        _default_targets = [0.05, 0.10, 0.15, 0.20, 0.30]
        _targets = sorted(set(_default_targets) | {self.wing_delta})
        candidates = self.enumerate_candidates(
            option_chain=option_chain,
            body_strike=body_strike,
            short_call=short_call,
            short_put=short_put,
            wing_delta_targets=_targets,
        )

        if not candidates:
            raise ValueError(
                f"No valid symmetric wing pairs found for {ticker} "
                f"expiring {expiry_date} around body strike ${body_strike}. "
                "Check that the chain has OTM strikes on both sides with "
                "positive mid prices, acceptable spreads, and sufficient yield."
            )

        # Select the candidate whose avg_wing_delta is nearest to self.wing_delta
        best = min(candidates, key=lambda c: abs(c.avg_wing_delta - self.wing_delta))

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
        wing_delta_targets: Optional[List[float]] = None,
    ) -> List[IronButterflyCandidate]:
        """
        Return up to one candidate per target delta level in ``WING_DELTA_TARGETS``.

        **Selection algorithm (nearest-neighbor bucketing):**

        1. Build the full pool of structurally valid, spread-filtered,
           yield-filtered symmetric wing pairs (same logic as before).
        2. For each candidate, determine its *nearest target delta*
           (the element of ``WING_DELTA_TARGETS`` closest to
           ``candidate.avg_wing_delta``).  This is the candidate's "claim".
        3. Group candidates by the target they claimed.  For each target
           group keep only the single candidate with the smallest
           ``|avg_wing_delta - target|``.
        4. Candidates that lost to a closer rival in the same bucket are
           dropped entirely — they are NOT reassigned to a second-choice
           target.

        **Example:** available deltas are 13.4 and 15.2; targets are
        ``[0.10, 0.15, 0.20, 0.30]``.

        * 13.4 → nearest target 0.15 (|13.4-15|=1.6 < |13.4-10|=3.4)
        * 15.2 → nearest target 0.15 (|15.2-15|=0.2)

        Both contest 0.15.  15.2 wins (closer).  13.4 is unassigned — the
        0.10 bucket stays empty.  Result: ``[cand@15.2]``.

        Candidates that fail spread quality or yield-on-capital filters are
        silently excluded before bucketing.

        Args:
            option_chain:       Full pre-validated option chain.
            body_strike:        ATM body strike (shared by short_call and short_put).
            short_call:         Short call body leg (already fetched and validated).
            short_put:          Short put body leg (already fetched and validated).
            wing_delta_targets: Ordered list of target |delta| levels used for
                                nearest-neighbor bucketing.  At most one candidate
                                is returned per level.  Defaults to
                                ``[0.05, 0.10, 0.15, 0.20, 0.30]``.

        Returns:
            List of ``IronButterflyCandidate`` objects — at most one per
            element of ``wing_delta_targets`` — sorted ascending by
            wing_width.
        """
        if wing_delta_targets is None:
            wing_delta_targets = [0.05, 0.10, 0.15, 0.20, 0.30]
        chain_lookup: dict = {
            (opt.strike, opt.option_type): opt for opt in option_chain
        }
        otm_call_strikes = sorted(
            {opt.strike for opt in option_chain if opt.strike > body_strike}
        )

        # ── Step 1: Build the full candidate pool ──────────────────────────
        pool: List[dict] = []
        for call_strike in otm_call_strikes:
            wing_width = call_strike - body_strike
            put_strike = body_strike - wing_width  # mirror strike

            long_call = chain_lookup.get((call_strike, 'call'))
            long_put  = chain_lookup.get((put_strike,  'put'))

            # Both symmetric legs must exist with tradeable premiums
            if long_call is None or long_put is None:
                continue
            if long_call.bid <= 0 or long_call.ask <= 0 or long_call.mid <= 0 \
                    or long_put.bid <= 0 or long_put.ask <= 0 or long_put.mid <= 0:
                continue

            # Economics — net credit must be positive before computing ratios
            net_credit = (
                (short_call.mid + short_put.mid)
                - (long_call.mid + long_put.mid)
            )
            if net_credit <= 0:
                continue

            # Spread cost filter — total bid-ask slippage as fraction of net credit
            total_spread = (
                (short_call.ask - short_call.bid)
                + (short_put.ask  - short_put.bid)
                + (long_call.ask  - long_call.bid)
                + (long_put.ask   - long_put.bid)
            )
            spread_cost_ratio = float(total_spread / net_credit)
            if spread_cost_ratio > self.max_spread_cost_ratio:
                continue

            # Yield filter — silently skip uneconomical candidates
            credit_to_width = float(net_credit / wing_width)
            if credit_to_width < self.min_yield_on_capital:
                continue

            avg_wing_delta = (abs(long_call.delta) + abs(long_put.delta)) / 2.0

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

            pool.append(dict(
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

        if not pool:
            return []

        # ── Step 2: Nearest-neighbour bucketing by target delta ────────────
        # Each candidate claims the target it is closest to.  Within each
        # target bucket only the single best (smallest distance) wins.
        # Losers are dropped — they are NOT reassigned to a secondary target.
        buckets: dict = {}   # target_delta -> (distance, candidate_kwargs)
        for cand_kwargs in pool:
            nearest = min(
                wing_delta_targets,
                key=lambda t: abs(cand_kwargs['avg_wing_delta'] - t),
            )
            dist = abs(cand_kwargs['avg_wing_delta'] - nearest)
            if nearest not in buckets or dist < buckets[nearest][0]:
                buckets[nearest] = (dist, cand_kwargs)
            # Note: strict < means ties are broken by pool insertion order
            # (ascending wing_width), so the narrower wing wins on equal distance.

        # Construct IronButterflyCandidate objects with target_bucket_delta assigned
        candidates = [
            IronButterflyCandidate(**cand_kwargs, target_bucket_delta=nearest)
            for nearest, (_dist, cand_kwargs) in buckets.items()
        ]
        candidates.sort(key=lambda c: c.wing_width)
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


# ======================================================================
# Iron Condor
# ======================================================================

@dataclass(frozen=True)
class IronCondorCandidate:
    """
    Immutable snapshot of one valid iron condor candidate.

    An iron condor has 4 legs with DIFFERENT body strikes (strangle body):
        - Short call at body_delta (OTM)
        - Short put  at body_delta (OTM, symmetric |delta|)
        - Long  call further OTM   (wing)
        - Long  put  further OTM   (wing)

    Unlike the iron butterfly (shared ATM body strike), the condor has
    asymmetric widths on each side because strikes are placed by delta.
    """
    # Body geometry (strangle)
    short_call_strike: Decimal
    short_put_strike: Decimal
    body_delta_target: float          # target |delta| used for body selection
    avg_body_delta: float             # (|sc.delta| + |sp.delta|) / 2

    # Wing geometry
    long_call_strike: Decimal
    long_put_strike: Decimal
    wing_delta_target: float          # target |delta| used for wing selection
    avg_wing_delta: float             # (|lc.delta| + |lp.delta|) / 2

    # Widths
    call_spread_width: Decimal        # long_call_strike - short_call_strike
    put_spread_width: Decimal         # short_put_strike - long_put_strike
    max_loss_width: Decimal           # max(call_spread_width, put_spread_width)

    # Economics
    net_credit: Decimal               # short premium - long premium (> 0)
    credit_to_width: float            # net_credit / max_loss_width
    total_spread: Decimal             # sum of (ask - bid) across all 4 legs
    spread_cost_ratio: float          # total_spread / net_credit

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


class IronCondorBuilder:
    """
    Build iron condor strategies from market data using delta-based placement.

    An iron condor consists of 4 legs on the same expiry:
        - Short call at ``body_delta`` (OTM, qty=-1)
        - Short put  at ``body_delta`` (OTM, symmetric |delta|, qty=-1)
        - Long  call at ``wing_delta`` (further OTM, qty=+1)
        - Long  put  at ``wing_delta`` (further OTM, qty=+1)

    Body and wing strikes are selected by finding the option whose |delta|
    is closest to the target.  Each side (call/put) is placed independently,
    so spread widths may differ.

    Args:
        body_delta_targets: List of target |delta| for the short body legs.
                            E.g. ``[0.30, 0.40]``. One candidate per target.
        wing_delta:         Target |delta| for the long OTM wings (default 0.10).
        max_spread_cost_ratio: Maximum total bid-ask spread as fraction of net
                            credit (default 0.25).
        min_yield_on_capital: Minimum net_credit / max_loss_width (default 0.0).
    """

    def __init__(
        self,
        body_delta_targets: Optional[List[float]] = None,
        wing_delta: float = 0.10,
        max_spread_cost_ratio: float = 0.25,
        min_yield_on_capital: float = 0.0,
    ):
        if body_delta_targets is None:
            body_delta_targets = [0.30, 0.40]
        for bd in body_delta_targets:
            if not 0 < bd < 0.5:
                raise ValueError(
                    f"body_delta_targets must be between 0 and 0.5, got {bd}"
                )
        if not 0 < wing_delta < 0.5:
            raise ValueError(
                f"wing_delta must be between 0 and 0.5, got {wing_delta}"
            )
        self.body_delta_targets = sorted(body_delta_targets)
        self.wing_delta = wing_delta
        self.max_spread_cost_ratio = max_spread_cost_ratio
        self.min_yield_on_capital = min_yield_on_capital

    def enumerate_candidates(
        self,
        option_chain: List[OptionQuote],
    ) -> List[IronCondorCandidate]:
        """
        Enumerate iron condor candidates, one per ``body_delta_target``.

        For each body delta target:
            1. Find the OTM call whose |delta| is closest to target.
            2. Find the OTM put whose |delta| is closest to target.
            3. Find the further-OTM long call whose |delta| is closest
               to ``wing_delta`` AND strike > short call strike.
            4. Find the further-OTM long put whose |delta| is closest
               to ``wing_delta`` AND strike < short put strike.
            5. Validate: all 4 mids > 0, spread filter, net credit > 0,
               yield filter.
            6. Build ``IronCondorCandidate``.

        Args:
            option_chain: Pre-validated option chain (single expiry).

        Returns:
            List of ``IronCondorCandidate`` — at most one per body delta target,
            sorted ascending by body_delta_target.
        """
        chain_lookup: dict = {
            (opt.strike, opt.option_type): opt for opt in option_chain
        }

        # Separate calls and puts
        calls = [opt for opt in option_chain if opt.option_type == 'call']
        puts = [opt for opt in option_chain if opt.option_type == 'put']

        candidates: List[IronCondorCandidate] = []

        for body_delta_tgt in self.body_delta_targets:
            cand = self._build_candidate_for_body_delta(
                calls, puts, chain_lookup, body_delta_tgt,
            )
            if cand is not None:
                candidates.append(cand)

        candidates.sort(key=lambda c: c.body_delta_target)
        return candidates

    def _build_candidate_for_body_delta(
        self,
        calls: List[OptionQuote],
        puts: List[OptionQuote],
        chain_lookup: dict,
        body_delta_target: float,
    ) -> Optional[IronCondorCandidate]:
        """Build a single condor candidate for a given body delta target."""

        # ── Find short call: OTM call closest to body_delta_target ────────
        # OTM calls have delta > 0 and < 0.5 (approximately)
        otm_calls = [
            opt for opt in calls
            if opt.bid > 0 and opt.ask > 0 and opt.mid > 0 and 0 < opt.delta < 0.50
        ]
        if not otm_calls:
            return None
        short_call = min(otm_calls, key=lambda o: abs(o.delta - body_delta_target))
        if abs(short_call.delta - body_delta_target) > 0.05:
            return None

        # ── Find short put: OTM put closest to body_delta_target ──────────
        # OTM puts have delta < 0; we compare |delta| to body_delta_target
        otm_puts = [
            opt for opt in puts
            if opt.bid > 0 and opt.ask > 0 and opt.mid > 0 and -0.50 < opt.delta < 0
        ]
        if not otm_puts:
            return None
        short_put = min(otm_puts, key=lambda o: abs(abs(o.delta) - body_delta_target))
        if abs(abs(short_put.delta) - body_delta_target) > 0.05:
            return None

        # sanity: short put strike must be below short call strike
        if short_put.strike >= short_call.strike:
            return None

        # ── Find long call wing: further OTM than short call, closest to wing_delta
        wing_calls = [
            opt for opt in calls
            if opt.strike > short_call.strike
            and opt.bid > 0 and opt.ask > 0 and opt.mid > 0
            and 0 < opt.delta < short_call.delta  # further OTM = smaller delta
        ]
        if not wing_calls:
            return None
        long_call = min(wing_calls, key=lambda o: abs(o.delta - self.wing_delta))
        if abs(long_call.delta - self.wing_delta) > 0.05:
            return None

        # ── Find long put wing: further OTM than short put, closest to wing_delta
        wing_puts = [
            opt for opt in puts
            if opt.strike < short_put.strike
            and opt.bid > 0 and opt.ask > 0 and opt.mid > 0
            and opt.delta > short_put.delta  # further OTM = delta closer to 0
        ]
        if not wing_puts:
            return None
        long_put = min(wing_puts, key=lambda o: abs(abs(o.delta) - self.wing_delta))
        if abs(abs(long_put.delta) - self.wing_delta) > 0.05:
            return None

        # ── Economics ─────────────────────────────────────────────────────
        net_credit = (
            (short_call.mid + short_put.mid)
            - (long_call.mid + long_put.mid)
        )
        if net_credit <= 0:
            return None

        # Spread cost filter — total bid-ask slippage as fraction of net credit
        total_spread = (
            (short_call.ask - short_call.bid)
            + (short_put.ask - short_put.bid)
            + (long_call.ask - long_call.bid)
            + (long_put.ask - long_put.bid)
        )
        spread_cost_ratio = float(total_spread / net_credit)
        if spread_cost_ratio > self.max_spread_cost_ratio:
            return None

        call_spread_width = long_call.strike - short_call.strike
        put_spread_width = short_put.strike - long_put.strike
        max_loss_width = max(call_spread_width, put_spread_width)

        if max_loss_width <= 0:
            return None

        credit_to_width = float(net_credit / max_loss_width)
        if credit_to_width < self.min_yield_on_capital:
            return None

        avg_body_delta = (abs(short_call.delta) + abs(short_put.delta)) / 2.0
        avg_wing_delta = (abs(long_call.delta) + abs(long_put.delta)) / 2.0

        # ── Net greeks (signed by qty) ────────────────────────────────────
        net_delta = (
            long_put.delta * 1
            + short_put.delta * -1
            + short_call.delta * -1
            + long_call.delta * 1
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

        return IronCondorCandidate(
            short_call_strike=short_call.strike,
            short_put_strike=short_put.strike,
            body_delta_target=body_delta_target,
            avg_body_delta=avg_body_delta,
            long_call_strike=long_call.strike,
            long_put_strike=long_put.strike,
            wing_delta_target=self.wing_delta,
            avg_wing_delta=avg_wing_delta,
            call_spread_width=call_spread_width,
            put_spread_width=put_spread_width,
            max_loss_width=max_loss_width,
            net_credit=net_credit,
            credit_to_width=credit_to_width,
            total_spread=total_spread,
            spread_cost_ratio=spread_cost_ratio,
            net_delta=net_delta,
            net_gamma=net_gamma,
            net_vega=net_vega,
            net_theta=net_theta,
            short_call=short_call,
            short_put=short_put,
            long_call=long_call,
            long_put=long_put,
        )

