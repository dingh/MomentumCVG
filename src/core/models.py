"""
Core data models for option trading system.

This module defines immutable data structures for:
- OptionQuote: Single option contract pricing snapshot
- OptionLeg: Single leg within a multi-leg strategy
- OptionStrategy: Generic multi-leg option strategy (straddle, condor, etc.)
- Signal: Trading recommendation from strategy
- Position: Actual opened/closed trade with P&L tracking
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Literal
from enum import Enum


class StrategyType(Enum):
    """Known option strategy types"""
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    BUTTERFLY = "butterfly"
    IRON_CONDOR = "iron_condor"
    VERTICAL_SPREAD = "vertical_spread"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    CUSTOM = "custom"


@dataclass(frozen=True)
class OptionQuote:
    """
    Immutable snapshot of option pricing at a point in time.
    
    Represents a single option contract (call or put) with market data
    and greeks. All quotes are frozen to ensure data integrity during
    backtesting and strategy calculations.
    """
    ticker: str                          # Underlying ticker symbol
    trade_date: date                     # Date of this quote
    expiry_date: date                    # Option expiration date
    strike: Decimal                      # Strike price
    option_type: Literal['call', 'put']  # Call or put
    bid: Decimal                         # Bid price
    ask: Decimal                         # Ask price
    mid: Decimal                         # Mid price (for entry/exit)
    iv: float                            # Implied volatility
    delta: float                         # Delta greek
    gamma: float                         # Gamma greek
    vega: float                          # Vega greek
    theta: float                         # Theta greek
    volume: int                          # Daily volume
    open_interest: int                   # Open interest
    
    @property
    def dte(self) -> int:
        """Days to expiration"""
        return (self.expiry_date - self.trade_date).days
    
    @property
    def spread(self) -> Decimal:
        """Bid-ask spread"""
        return self.ask - self.bid
    
    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as percentage of mid price"""
        if self.mid > 0:
            return float(self.spread / self.mid)
        return 0.0


@dataclass(frozen=True)
class OptionLeg:
    """
    A single leg within a multi-leg option strategy.

    Represents one option contract with a specified quantity within a strategy
    definition. The quantity's sign indicates whether this leg is long (+) or 
    short (-) within the strategy unit.

    Two-Level System:
    1. Strategy Unit (this level): Defines the strategy's composition
        - Quantity determines if leg is long/short within the unit
        - Example: Straddle = call (qty=1) + put (qty=1)
        - Example: Ratio spread = call (qty=1) + call (qty=-2)
    
    2. Position Level: Scales the entire strategy
        - Position.quantity scales all legs together
        - Positive position = long the strategy as defined
        - Negative position = short the strategy (inverts all legs)

    Current Usage: StraddleBuilder creates symmetric strategies (all qty=+1).
    Future: Will support complex strategies with mixed long/short legs.
    """
    option: OptionQuote  # The option contract
    quantity: int        # Positive for long, negative for short
    
    @property
    def is_long(self) -> bool:
        """True if this is a long position (buying)"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """True if this is a short position (selling)"""
        return self.quantity < 0
    
    @property
    def net_premium(self) -> Decimal:
        """
        Premium received (negative) or paid (positive) at mid price.
        
        Phase 1: Uses mid price for simplicity (optimistic assumption)
        Phase 2: Will add configurable slippage/execution models in executor
        
        For long positions: pay premium (positive cost)
        For short positions: receive premium (negative cost)
        """
        # Use mid price for both long and short
        # Slippage modeling will be handled by the executor in Phase 2
        price = self.option.mid
        
        # Return signed premium (long = positive cost, short = negative cost)
        return price * abs(self.quantity) * (1 if self.is_long else -1)
    
    @property
    def delta_exposure(self) -> float:
        """Net delta for this leg (delta * quantity)"""
        return self.option.delta * self.quantity
    
    @property
    def vega_exposure(self) -> float:
        """Net vega for this leg (vega * quantity)"""
        return self.option.vega * self.quantity
    
    @property
    def gamma_exposure(self) -> float:
        """Net gamma for this leg (gamma * quantity)"""
        return self.option.gamma * self.quantity
    
    @property
    def theta_exposure(self) -> float:
        """Net theta for this leg (theta * quantity)"""
        return self.option.theta * self.quantity
    
    def calculate_intrinsic_value(self, spot_price: Decimal) -> Decimal:
        """
        Calculate intrinsic value at expiration given spot price.
        
        Args:
            spot_price: Underlying spot price at expiration
            
        Returns:
            Intrinsic value (unsigned, per contract)

        Note:
            TODO : This returns per-contract value, requiring manual
            quantity/direction logic in calculate_payoff(). Consider refactoring
            to return signed, scaled value (like net_premium property) for
            consistency. Would simplify OptionStrategy.calculate_payoff() and
            future mark-to-market calculations.
        """
        if self.option.option_type == 'call':
            # Call: max(spot - strike, 0)
            intrinsic = max(spot_price - self.option.strike, Decimal('0'))
        else:
            # Put: max(strike - spot, 0)
            intrinsic = max(self.option.strike - spot_price, Decimal('0'))
        
        return intrinsic


@dataclass(frozen=True)
class OptionStrategy:
    """
    Generic multi-leg option strategy (straddle, condor, butterfly, etc.).
    
    Represents any combination of option legs. Immutable to ensure
    strategy definition doesn't change after creation. Aggregates
    greeks and premium across all legs.
    """
    ticker: str                      # Underlying ticker
    strategy_type: StrategyType      # Type of strategy
    legs: tuple[OptionLeg, ...]      # Tuple for immutability
    trade_date: date                 # Date strategy was constructed
    
    @property
    def net_premium(self) -> Decimal:
        """
        Total premium - negative for credit, positive for debit.
        
        Credit spread: receive money upfront (negative)
        Debit spread: pay money upfront (positive)
        """
        return sum(leg.net_premium for leg in self.legs)
    
    @property
    def is_credit_spread(self) -> bool:
        """True if net premium is received (credit)"""
        return self.net_premium < 0
    
    @property
    def is_debit_spread(self) -> bool:
        """True if net premium is paid (debit)"""
        return self.net_premium > 0
    
    @property
    def net_delta(self) -> float:
        """Portfolio delta across all legs"""
        return sum(leg.delta_exposure for leg in self.legs)
    
    @property
    def net_vega(self) -> float:
        """Portfolio vega across all legs"""
        return sum(leg.vega_exposure for leg in self.legs)
    
    @property
    def net_gamma(self) -> float:
        """Portfolio gamma across all legs"""
        return sum(leg.gamma_exposure for leg in self.legs)
    
    @property
    def net_theta(self) -> float:
        """Portfolio theta across all legs"""
        return sum(leg.theta_exposure for leg in self.legs)
    
    @property
    def expiry_dates(self) -> set[date]:
        """All unique expiration dates in this strategy"""
        return {leg.option.expiry_date for leg in self.legs}
    
    @property
    def max_expiry(self) -> date:
        """Latest expiration date across all legs"""
        return max(self.expiry_dates)
    
    @property
    def min_expiry(self) -> date:
        """Earliest expiration date across all legs"""
        return min(self.expiry_dates)
    
    @property
    def num_legs(self) -> int:
        """Number of legs in this strategy"""
        return len(self.legs)
    
    def calculate_payoff(self, spot_prices: dict[date, Decimal]) -> Decimal:
        """
        Calculate intrinsic value at expiration.
        
        For strategies with multiple expiries, need spot price at each expiry.
        
        Args:
            spot_prices: Dict mapping expiry_date -> spot_price at that date
            
        Returns:
            Total intrinsic value across all legs
        """
        total_value = Decimal('0')
        
        for leg in self.legs:
            expiry = leg.option.expiry_date
            if expiry not in spot_prices:
                raise ValueError(f"Missing spot price for expiry {expiry}")
            
            spot = spot_prices[expiry]
            intrinsic = leg.calculate_intrinsic_value(spot)  # Unsigned
            
            # Apply leg direction via signed quantity
            total_value += intrinsic * leg.quantity
        
        return total_value


@dataclass(frozen=True)
class Signal:
    """
    Trading signal with conviction score.
    
    Represents a strategy's recommendation to trade a specific ticker
    using a specific option strategy type. Contains feature values
    that generated the signal for explainability.
    """
    ticker: str                         # Ticker to trade
    signal_date: date                   # Date signal was generated
    strategy_type: StrategyType         # What kind of strategy to use
    direction: Literal['long', 'short'] # Overall directional bias
    conviction: float                   # 0.0 to 1.0 signal strength
    features: dict                      # Feature values that generated signal
    metadata: dict                      # Additional context (strikes, expiries, etc.)
    
    def __post_init__(self):
        """Validate signal fields"""
        if not (0.0 <= self.conviction <= 1.0):
            raise ValueError(f"Conviction must be between 0 and 1, got {self.conviction}")


@dataclass(frozen=True)
class Position:
    """
    An open or closed position with P&L tracking.
    
    Represents an actual trade that was executed (or will be executed).
    Immutable once created. To "update" a position (e.g., close it),
    create a new Position instance with exit fields filled.
    
    Phase 1: quantity is float to support fractional contracts in backtests
    Phase 2+: May enforce integer quantities in live trading
    """
    ticker: str                  # Ticker being traded
    entry_date: date             # Date position was entered
    strategy: OptionStrategy     # The option strategy (can be any multi-leg)
    quantity: float              # Number of strategy units (fractional allowed in backtests)
    entry_cost: Decimal          # Total cost/credit for all legs * quantity
                                 # Positive = debit (we pay), Negative = credit (we receive)
    exit_date: date | None = None      # Date position was closed (None if open)
    exit_value: Decimal | None = None  # Exit value when closed
    metadata: dict = field(default_factory=dict)  # Additional context
    
    @property
    def is_open(self) -> bool:
        """True if position is still open"""
        return self.exit_date is None
    
    @property
    def is_closed(self) -> bool:
        """True if position has been closed"""
        return self.exit_date is not None
    
    @property
    def pnl(self) -> Decimal | None:
        """
        Calculate profit/loss.
        
        P&L = exit_value - entry_cost
        - Positive P&L means profit
        - Negative P&L means loss
        
        Returns None if position is still open.
        """
        if not self.is_closed:
            return None
        
        # P&L = what you got out - what you put in
        # For credit spreads: entry_cost is negative (received premium)
        # For debit spreads: entry_cost is positive (paid premium)
        return self.exit_value - self.entry_cost
    
    @property
    def pnl_pct(self) -> float | None:
        """
        Calculate percentage return.
        
        Return % = P&L / |entry_cost|
        
        Returns None if position is still open.
        """
        if not self.is_closed or self.entry_cost == 0:
            return None
        
        pnl_val = self.pnl
        # Use absolute entry cost for percentage calculation
        return float(pnl_val / abs(self.entry_cost))
    
    @property
    def net_delta(self) -> float:
        """
        Position delta (strategy delta * quantity).
        
        Represents overall directional exposure.
        """
        return self.strategy.net_delta * self.quantity
    
    @property
    def net_vega(self) -> float:
        """
        Position vega (strategy vega * quantity).
        
        Represents volatility exposure.
        """
        return self.strategy.net_vega * self.quantity
    
    @property
    def net_gamma(self) -> float:
        """Position gamma"""
        return self.strategy.net_gamma * self.quantity
    
    @property
    def net_theta(self) -> float:
        """Position theta (time decay)"""
        return self.strategy.net_theta * self.quantity
    
    @property
    def strategy_type(self) -> StrategyType:
        """Type of option strategy being traded"""
        return self.strategy.strategy_type
    
    @property
    def holding_period(self) -> int | None:
        """
        Number of days position was held.
        
        Returns None if position is still open.
        """
        if not self.is_closed:
            return None
        return (self.exit_date - self.entry_date).days
