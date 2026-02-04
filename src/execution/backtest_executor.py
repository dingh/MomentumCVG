"""
Backtest executor - simulates trade execution.

Phase 1: Settlement at expiry with intrinsic value.
No slippage, no commissions, mid-price execution.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
from datetime import date
import logging
from typing import Protocol

from src.core.models import Position, OptionStrategy

logger = logging.getLogger(__name__)


class IExecutor(Protocol):
    """
    Executor protocol: Simulates trade execution and calculates P&L.
    
    Separates execution simulation from backtesting logic.
    Enables swapping execution models (mid-price, realistic slippage, pessimistic).
    
    Phase 1: Simple mid-price execution, settlement at expiry.
    Phase 2: Add slippage, commissions, early exit logic.
    """
    
    @property
    def execution_mode(self) -> str:
        """
        Execution model identifier.
        
        Returns:
            Execution mode name:
            - 'mid' (Phase 1): Mid-price, no slippage, no costs
            - 'realistic' (Phase 2): Bid-ask spreads, typical slippage
            - 'pessimistic' (Phase 3): Worst-case execution
        """
        ...
    
    def execute_entry(self, position: Position) -> Position:
        """
        Execute position entry.
        
        Phase 1: No-op (entry_cost already set by optimizer from strategy.net_premium).
        Phase 2: May add slippage, commissions, execution delays.
        
        Args:
            position: Position to enter (entry_cost already calculated by optimizer)
            
        Returns:
            Position (unchanged in Phase 1, may have adjusted entry_cost in Phase 2+)
            
        Notes:
            - In Phase 1, optimizer sets entry_cost based on strategy.net_premium
            - Executor validates but doesn't modify entry_cost
            - Future phases may adjust entry_cost for slippage/commissions
        """
        ...
    
    def execute_exit(
        self,
        position: Position,
        spot_price: Decimal,
        exit_date: date
    ) -> Position:
        """
        Execute position exit and calculate P&L.
        
        Phase 1: Settlement at expiry using intrinsic value.
        Phase 2: May handle early exit, mark-to-market valuation.
        
        Args:
            position: Position to exit (open position with entry_cost set)
            spot_price: Underlying price at exit
            exit_date: Exit date (typically option expiry date)
            
        Returns:
            Position with exit_value and exit_date filled
            
        Exit value calculation:
        - Intrinsic value calculated for each option leg
        - Long position (qty > 0): exit_value = +intrinsic (we receive settlement)
        - Short position (qty < 0): exit_value = -intrinsic (we pay settlement)
        - P&L: position.pnl = exit_value - entry_cost
        
        Example:
            Long straddle entry_cost = $1,250 (we paid premium)
            At expiry: spot=$105, strike=$100
            Intrinsic = max(105-100, 0) + max(100-105, 0) = 5 + 0 = $5
            exit_value = +$500 (qty=1, intrinsic=$5 * 100 shares/contract)
            P&L = $500 - $1,250 = -$750 (loss)
        """
        ...


@dataclass
class BacktestExecutor:
    """
    Phase 1 executor: Mid prices, immediate fills, no costs.
    
    Assumptions:
    - All trades execute at mid price (no slippage)
    - Entry cost already calculated by optimizer
    - Exit at expiry using intrinsic value only
    - No commissions or fees
    - No bid-ask spread impact
    
    Philosophy:
    - Minimal execution noise for signal quality assessment
    - Optimistic but not unrealistic (mid-price is achievable with limit orders)
    - Production constraints added in Phase 2+
    
    Example:
        >>> executor = BacktestExecutor(execution_mode='mid')
        >>> position = optimizer.optimize(...)[0]  # entry_cost already set
        >>> filled = executor.execute_entry(position)  # No-op in Phase 1
        >>> # ... time passes, position expires ...
        >>> closed = executor.execute_exit(filled, spot_price, expiry_date)
        >>> print(closed.pnl)  # exit_value - entry_cost
    """
    
    execution_mode: str = 'mid'
    
    def __post_init__(self):
        """Validate parameters."""
        valid_modes = ['mid', 'realistic', 'pessimistic']
        if self.execution_mode not in valid_modes:
            raise ValueError(
                f"Invalid execution_mode: {self.execution_mode}. "
                f"Must be one of {valid_modes}"
            )
    
    @property
    def name(self) -> str:
        """Executor identifier for logging."""
        return f"BacktestExecutor_{self.execution_mode}"
    
    def execute_entry(self, position: Position) -> Position:
        """
        Execute position entry (no-op in Phase 1).
        
        Phase 1: Optimizer already calculated entry_cost from strategy.net_premium.
        Executor validates that entry_cost is set and returns position unchanged.
        
        Args:
            position: Position with entry_cost already set by optimizer
            
        Returns:
            Same position (unchanged)
            
        Raises:
            ValueError: If entry_cost is None (optimizer didn't set it)
        """
        
        # Validation: entry_cost should be set by optimizer
        if position.entry_cost is None:
            raise ValueError(
                f"Entry cost not set for {position.ticker}. "
                f"Optimizer should set entry_cost before passing to executor."
            )
        
        logger.debug(
            f"{self.name}: Entry {position.ticker} - "
            f"qty={position.quantity:.2f}, "
            f"entry_cost=${position.entry_cost:,.2f} "
            f"({'debit' if position.entry_cost > 0 else 'credit'})"
        )
        
        return position
    
    def execute_exit(
        self,
        position: Position,
        spot_price: Decimal,
        exit_date: date
    ) -> Position:
        """
        Execute position exit using intrinsic value at expiry.
        
        Phase 1: Calculate intrinsic value for each option leg, settle position.
        
        Args:
            position: Open position with entry_cost set
            spot_price: Underlying spot price at exit
            exit_date: Settlement date (expiry date)
            
        Returns:
            Position with exit_value and exit_date filled
            
        Calculation:
        1. Calculate intrinsic value per contract
        2. Apply quantity (signed) to get total exit_value
        3. P&L = exit_value - entry_cost
        
        Example:
            Long straddle (qty=2.5):
            - Strike=$100, Spot at expiry=$105
            - Call intrinsic=$5, Put intrinsic=$0
            - Total intrinsic per straddle=$5 (per share) * 100 = $500
            - exit_value = $500 * 2.5 = $1,250
            - If entry_cost=$1,000, P&L = $1,250 - $1,000 = +$250
        """
        
        # Calculate intrinsic value using existing OptionLeg method
        intrinsic_per_unit = Decimal('0')
        for leg in position.strategy.legs:
            leg_intrinsic = leg.calculate_intrinsic_value(spot_price)
            intrinsic_per_unit += leg_intrinsic * leg.quantity
        
        # Total exit value (signed by position quantity)
        exit_value = intrinsic_per_unit * Decimal(str(position.quantity))
        
        # Create closed position (immutable update)
        # Note: pnl property auto-calculates as exit_value - entry_cost
        closed_position = replace(
            position,
            exit_value=exit_value,
            exit_date=exit_date
        )
        
        logger.debug(
            f"{self.name}: Exit {position.ticker} - "
            f"qty={position.quantity:.2f}, "
            f"spot=${spot_price:,.2f}, "
            f"intrinsic_per_unit=${intrinsic_per_unit:,.2f}, "
            f"exit_value=${exit_value:,.2f}, "
            f"pnl=${closed_position.pnl:,.2f}"
        )
        
        return closed_position
