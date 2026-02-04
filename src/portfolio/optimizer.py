"""
Pure equal-weight portfolio optimizer.

Phase 1 implementation: Trades ALL signals with exact equal weighting.
Allows fractional quantities to achieve perfect notional allocation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from decimal import Decimal
from datetime import date
import logging

from src.core.models import Signal, Position, OptionStrategy

logger = logging.getLogger(__name__)


@dataclass
class EqualWeightOptimizer:
    """
    Pure equal-weight portfolio optimizer.
    
    Phase 1 implementation designed for signal quality assessment:
    - Trades signals where option strategies are available
    - Allocates EXACTLY equal notional to each position
    - Allows fractional quantities (e.g., 2.5 contracts)
    - Long side and short side ALWAYS get equal total notional
    - **Net cash impact ~= 0 when balanced** (longs pay, shorts receive)
    
    Philosophy:
    - Strategy decides WHAT to trade (via signal generation)
    - Option strategy availability determines WHICH tickers are tradeable
    - Optimizer allocates full notional_per_side to each side regardless of signal count
    - Pure assessment of signal quality without execution noise
    
    Example:
        >>> # Strategy generates 30 long, 25 short signals
        >>> # But only 22 long, 18 short have liquid options
        >>> optimizer = EqualWeightOptimizer(notional_per_side=Decimal('10000'))
        >>> positions = optimizer.optimize(...)
        >>> # Result: 22 longs @ $454.55 each = $10k total
        >>> #         18 shorts @ $555.56 each = $10k total
        >>> #         Still balanced, but concentrated in fewer positions
    """
    
    # Notional allocation (NOT capital required)
    notional_per_side: Decimal = Decimal('10000')  # Notional for longs, notional for shorts
    
    def __post_init__(self):
        """Validate parameters."""
        if self.notional_per_side <= 0:
            raise ValueError(f"notional_per_side must be > 0, got {self.notional_per_side}")
    
    @property
    def name(self) -> str:
        """Optimizer identifier."""
        return "PureEqualWeight"
    
    @property
    def total_notional(self) -> Decimal:
        """Total notional exposure (long + short)."""
        return self.notional_per_side * 2
    
    @property
    def expected_net_cash_change(self) -> Decimal:
        """
        Expected net cash change if perfectly balanced.
        
        Returns ~$0 when long_notional == short_notional
        because premium received from shorts offsets premium paid for longs.
        """
        return Decimal('0')  # Balanced portfolio
    
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
        Create pure equal-weighted positions from signals with available strategies.
        
        Algorithm:
        1. Filter signals to only tickers with available option strategies
        2. Separate filtered signals by direction (long/short)
        3. Calculate notional per position = notional_per_side / tradeable_count
        4. For each tradeable signal:
           - Calculate fractional quantity = target_notional / premium
           - Create position with exact equal notional
        5. Return all positions (full notional_per_side allocated to each side)
        
        Key Behavior:
        - If 30 long signals but only 22 have strategies → trade 22 longs @ higher notional each
        - Each side ALWAYS gets full notional_per_side allocation
        - Missing strategies reduce position count but NOT total notional
        
        Cash Flow Mechanics:
        - Long positions: entry_cost > 0 (we PAY premium)
        - Short positions: entry_cost < 0 (we RECEIVE premium)
        - Net cash change ~= 0 when balanced
        
        Phase 1 Philosophy:
        - NO minimum quantity constraint (can be 0.5 contracts)
        - NO capital limit checking (assumes sufficient margin)
        - Trade all available strategies, allocate full notional per side
        - Goal: Pure signal quality assessment without optimizer bias
        """
        
        # === APPLY CUSTOM CONSTRAINTS ===
        if constraints:
            notional_per_side = Decimal(str(constraints.get('notional_per_side', self.notional_per_side)))
        else:
            notional_per_side = self.notional_per_side
        
        # === VALIDATE INPUTS ===
        if not signals:
            logger.info(f"{self.name}: No signals provided")
            return []
        
        if not option_strategies:
            logger.warning(f"{self.name}: No option strategies available")
            return []
        
        # Check for duplicate tickers in signals (strategy bug)
        signal_tickers = [s.ticker for s in signals]
        if len(signal_tickers) != len(set(signal_tickers)):
            duplicates = [t for t in signal_tickers if signal_tickers.count(t) > 1]
            raise ValueError(
                f"Duplicate tickers in signals: {set(duplicates)}. "
                f"Strategy should not generate multiple signals per ticker."
            )
        
        # Phase 1: Warn if insufficient capital (but don't block)
        # For balanced portfolio, net cash ~= 0, but need margin for worst case
        total_required = notional_per_side * 2
        if available_capital < total_required:
            logger.warning(
                f"{self.name}: Low capital for margin! "
                f"Estimated margin need: ${total_required:,.2f}, "
                f"Available: ${available_capital:,.2f}. "
                f"Proceeding anyway for signal quality assessment."
            )
        
        # Phase 1: Log warning about current_positions (should be empty)
        if current_positions:
            logger.warning(
                f"{self.name}: current_positions not empty ({len(current_positions)} positions). "
                f"Phase 1 optimizer assumes full turnover - ignoring existing positions."
            )
        
        # === FILTER SIGNALS BY AVAILABLE STRATEGIES ===
        
        tradeable_signals = [s for s in signals if s.ticker in option_strategies]
        skipped_tickers = [s.ticker for s in signals if s.ticker not in option_strategies]
        
        if skipped_tickers:
            logger.warning(
                f"{self.name}: {len(skipped_tickers)} signals skipped (no option strategy): {skipped_tickers[:10]}..."
            )
        
        if not tradeable_signals:
            logger.warning(f"{self.name}: No tradeable signals after filtering")
            return []
        
        logger.info(
            f"{self.name}: {len(tradeable_signals)} tradeable signals "
            f"(from {len(signals)} total signals, {len(skipped_tickers)} skipped)"
        )
        
        # === SEPARATE TRADEABLE SIGNALS BY DIRECTION ===
        
        long_signals = [s for s in tradeable_signals if s.direction == 'long']
        short_signals = [s for s in tradeable_signals if s.direction == 'short']
        
        logger.info(
            f"{self.name}: Tradeable positions - "
            f"{len(long_signals)} long, {len(short_signals)} short"
        )
        
        # === POSITION SIZING (PURE EQUAL NOTIONAL PER SIDE) ===
        
        positions = []
        
        long_notional_total = Decimal('0')
        short_notional_total = Decimal('0')
        net_cash_change = Decimal('0')
        
        # Process long signals
        if long_signals:
            notional_per_long = notional_per_side / len(long_signals)
            logger.info(
                f"{self.name}: Long side - {len(long_signals)} positions, "
                f"${notional_per_long:,.2f} notional per position "
                f"(total ${notional_per_side:,.2f})"
            )
            
            for signal in long_signals:
                position = self._create_position(
                    signal=signal,
                    strategy=option_strategies[signal.ticker],
                    target_notional=notional_per_long,
                    current_date=current_date
                )
                if position:  # Should always succeed since we pre-filtered
                    positions.append(position)
                    long_notional_total += abs(position.entry_cost)
                    net_cash_change += position.entry_cost  # Positive (we pay)
        else:
            logger.warning(f"{self.name}: No tradeable long signals")
        
        # Process short signals
        if short_signals:
            notional_per_short = notional_per_side / len(short_signals)
            logger.info(
                f"{self.name}: Short side - {len(short_signals)} positions, "
                f"${notional_per_short:,.2f} notional per position "
                f"(total ${notional_per_side:,.2f})"
            )
            
            for signal in short_signals:
                position = self._create_position(
                    signal=signal,
                    strategy=option_strategies[signal.ticker],
                    target_notional=notional_per_short,
                    current_date=current_date
                )
                if position:  # Should always succeed since we pre-filtered
                    positions.append(position)
                    short_notional_total += abs(position.entry_cost)
                    net_cash_change += position.entry_cost  # Negative (we receive)
        else:
            logger.warning(f"{self.name}: No tradeable short signals")
        
        # === SUMMARY LOGGING ===
        
        num_longs_actual = sum(1 for p in positions if p.quantity > 0)
        num_shorts_actual = sum(1 for p in positions if p.quantity < 0)
        
        logger.info(
            f"{self.name}: Created {len(positions)} positions "
            f"({num_longs_actual} long, {num_shorts_actual} short)"
        )
        logger.info(
            f"{self.name}: Notional allocation - "
            f"Long: ${long_notional_total:,.2f}, Short: ${short_notional_total:,.2f}, "
            f"Total: ${long_notional_total + short_notional_total:,.2f}"
        )
        logger.info(
            f"{self.name}: Net cash change: ${net_cash_change:,.2f} "
            f"({'balanced' if abs(net_cash_change) < 100 else 'UNBALANCED'})"
        )
        
        return positions
    
    def _create_position(
        self,
        signal: Signal,
        strategy: OptionStrategy,
        target_notional: Decimal,
        current_date: date
    ) -> Optional[Position]:
        """
        Create a single position with exact equal notional.
        
        Args:
            signal: Trading signal
            strategy: Pre-built option strategy for this ticker
            target_notional: Exact notional to allocate to this position
            current_date: Entry date
        
        Returns:
            Position with fractional quantity, or None if premium invalid
        """
        
        premium_per_unit = abs(strategy.net_premium)  # Premium for 1 straddle (always positive)
        
        # Validate premium (should never be zero if strategy built successfully)
        if premium_per_unit <= 0:
            logger.warning(
                f"{self.name}: Skipping {signal.ticker} - "
                f"zero/negative premium (${premium_per_unit})"
            )
            return None
        
        # Calculate EXACT fractional quantity
        quantity_float = float(target_notional / premium_per_unit)
        
        # Apply sign based on direction
        if signal.direction == 'long':
            # Long straddle: BUY (positive quantity)
            quantity_float = abs(quantity_float)
            entry_cost = target_notional  # Positive (we pay premium)
        else:
            # Short straddle: SELL (negative quantity)
            quantity_float = -abs(quantity_float)
            entry_cost = -target_notional  # Negative (we receive premium)
        
        # Create position
        position = Position(
            ticker=signal.ticker,
            entry_date=current_date,
            strategy=strategy,
            quantity=quantity_float,  # ← Fractional quantity!
            entry_cost=entry_cost,    # ← Signed: positive for long, negative for short
            exit_date=None,
            exit_value=None,
            metadata={
                'signal_conviction': float(signal.conviction),
                'signal_features': signal.features,
                'optimizer': self.name,
                'target_notional': float(target_notional),
                'premium_per_unit': float(premium_per_unit)
            }
        )
        
        logger.debug(
            f"{self.name}: {signal.ticker} {signal.direction} - "
            f"qty={quantity_float:.3f}, "
            f"premium=${premium_per_unit:,.2f}, "
            f"entry_cost=${entry_cost:,.2f} ({'debit' if entry_cost > 0 else 'credit'})"
        )
        
        return position
