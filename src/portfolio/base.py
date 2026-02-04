"""
Portfolio optimizer protocol definition.

Defines the interface for position sizing and portfolio construction.
"""

from typing import Protocol, List, Dict, Optional
from decimal import Decimal
from datetime import date

from src.core.models import Signal, Position, OptionStrategy


class IPortfolioOptimizer(Protocol):
    """
    Portfolio optimizer protocol: Converts signals to sized positions.
    
    Separates signal generation (what to trade) from position sizing (how much).
    Enables swapping sizing algorithms without changing strategy logic.
    """
    
    @property
    def name(self) -> str:
        """
        Optimizer identifier for logging and reporting.
        
        Returns:
            Unique optimizer name (e.g., 'EqualWeight_20', 'DeltaNeutral_20')
        """
        ...
    
    def optimize(
        self,
        signals: List[Signal],
        option_strategies: Dict[str, OptionStrategy],
        current_positions: List[Position],
        available_capital: Decimal,
        current_date: date,
        constraints: Optional[Dict] = None
    ) -> List[Position]:
        """
        Convert signals to sized positions respecting constraints.
        
        Args:
            signals: Trading signals from strategy (sorted by conviction descending)
            option_strategies: Built option strategies keyed by ticker
                              {ticker: OptionStrategy} - already priced with greeks
            current_positions: Existing open positions (for rebalancing)
                              Phase 1: Always empty (full turnover each cycle)
                              Phase 2: May contain positions for delta calculation
            available_capital: Cash available for new positions
            current_date: Current trading date
            constraints: Optional constraints dict with keys:
                - 'max_positions': int (default 20)
                - 'long_short_ratio': float (default 1.0 = balanced)
                - 'max_position_pct': float (max % of capital per position)
                - 'min_position_pct': float (min % of capital per position)
                
        Returns:
            List of Position objects with quantities sized according to:
            - Signal conviction scores
            - Portfolio constraints
            - Available capital
            - Strategy-specific sizing rules
            
            Empty list if no tradeable signals or insufficient capital.
            
        Raises:
            ValueError: If signals and option_strategies don't align
            ValueError: If constraints are invalid
            
        Notes:
            - MUST respect available_capital (no over-allocation)
            - MUST handle case where option_strategies missing for signal
            - MAY filter signals (e.g., take top N by conviction)
            - Position quantities signed by direction:
              - Long signals: quantity > 0 (buy straddle)
              - Short signals: quantity < 0 (sell straddle)
        """
        ...
