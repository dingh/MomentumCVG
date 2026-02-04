"""
Strategy protocol definition.

Defines the interface that all trading strategies must implement.
"""

from typing import Protocol, List
from datetime import date
import pandas as pd

from src.core.models import Signal


class IStrategy(Protocol):
    """
    Strategy protocol: Generates trading signals from features.
    
    All strategies must implement this interface to work with BacktestEngine.
    Strategies are responsible for analyzing pre-computed features and deciding
    which tickers to trade, in which direction, and with what conviction.
    """
    
    @property
    def name(self) -> str:
        """
        Strategy identifier for logging and reporting.
        
        Returns:
            Unique strategy name (e.g., 'MomentumCVG_60_8')
        """
        ...
    
    @property
    def required_features(self) -> List[str]:
        """
        Feature columns required by this strategy.
        
        Engine will validate these exist before calling generate_signals().
        
        Returns:
            List of column names (e.g., ['mom_60_8_mean', 'cvg_60_8'])
        """
        ...
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        current_date: date,
        **kwargs
    ) -> List[Signal]:
        """
        Generate trading signals for given date.
        
        Args:
            features: Pre-computed features for current_date
                     Columns: ['ticker', 'date', <feature_cols>]
                     Guaranteed to have only rows for current_date
                     Guaranteed to have self.required_features columns
            current_date: Trading date (for signal metadata)
            **kwargs: Additional context (e.g., market_regime, volatility)
            
        Returns:
            List of Signal objects, sorted by conviction (descending)
            Returns empty list if no opportunities found
            
        Raises:
            ValueError: If required features are missing
            
        Notes:
            - MUST be stateless (no memory of previous signals)
            - MUST NOT access historical data (use features only)
            - MUST handle missing/NaN values gracefully
            - Should filter out invalid signals (NaN features, etc.)
            
        Example:
            >>> strategy = MomentumCVGStrategy()
            >>> features = pd.read_parquet('features.parquet')
            >>> features_today = features[features['date'] == today]
            >>> signals = strategy.generate_signals(features_today, today)
            >>> for signal in signals[:5]:
            ...     print(f"{signal.ticker} {signal.direction} {signal.conviction:.3f}")
        """
        ...
