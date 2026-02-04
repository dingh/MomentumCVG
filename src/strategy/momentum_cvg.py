"""
Momentum + CVG Strategy Implementation.

Two-stage filtering strategy:
1. Select top/bottom momentum percentiles
2. Filter to most continuous trends (top CVG)

This strategy implements the methodology from the notebook analysis,
combining momentum signals with CVG (Continuous Volatility Gaps) filtering
to identify tickers with persistent trends.
"""

from dataclasses import dataclass
from typing import List
from datetime import date
import pandas as pd

from src.core.models import Signal, StrategyType


@dataclass
class MomentumCVGStrategy:
    """
    Two-stage momentum + CVG filtering strategy.
    
    Stage 1: Rank by momentum, select top/bottom percentiles
    Stage 2: Within selections, filter to top CVG (most continuous trends)
    
    Logic (from notebook):
        - Long candidates: top momentum_long_pct% by momentum
        - Short candidates: bottom momentum_short_pct% by momentum
        - CVG filter: Keep top cvg_filter_pct% by CVG for BOTH
        - Ensures we trade tickers with continuous trends
    
    The CVG filter is applied to BOTH longs and shorts because high CVG
    indicates a continuous trend regardless of direction. A stock can have
    high CVG with negative momentum (continuous downtrend) or positive
    momentum (continuous uptrend).
    
    Example:
        >>> strategy = MomentumCVGStrategy(
        ...     max_lag=60,
        ...     min_lag=8,
        ...     momentum_long_pct=0.10,    # Top 10% momentum for longs
        ...     momentum_short_pct=0.10,   # Bottom 10% momentum for shorts
        ...     cvg_filter_pct=0.50,       # Top 50% CVG within each
        ...     min_count_pct=0.80         # Require 80% of window with data
        ... )
        >>> signals = strategy.generate_signals(features, current_date)
        >>> print(f"Generated {len(signals)} signals")
    """
    
    # Window parameters (used to construct feature column names)
    max_lag: int = 60  # Maximum lag in weeks
    min_lag: int = 8   # Minimum lag in weeks
    
    # Stage 1: Momentum filtering
    momentum_long_pct: float = 0.10   # Top 10% for longs
    momentum_short_pct: float = 0.10  # Bottom 10% for shorts
    
    # Stage 2: CVG filtering
    cvg_filter_pct: float = 0.50      # Top 50% CVG
    
    # Data quality filter
    min_count_pct: float = 0.80       # Require 80% of observations
    
    # Strategy metadata
    strategy_type: StrategyType = StrategyType.STRADDLE
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.max_lag <= self.min_lag:
            raise ValueError(
                f"max_lag ({self.max_lag}) must be greater than min_lag ({self.min_lag})"
            )
        if not 0 < self.momentum_long_pct < 1:
            raise ValueError(
                f"momentum_long_pct must be in (0, 1), got {self.momentum_long_pct}"
            )
        if not 0 < self.momentum_short_pct < 1:
            raise ValueError(
                f"momentum_short_pct must be in (0, 1), got {self.momentum_short_pct}"
            )
        if not 0 < self.cvg_filter_pct <= 1:
            raise ValueError(
                f"cvg_filter_pct must be in (0, 1], got {self.cvg_filter_pct}"
            )
        if not 0 < self.min_count_pct <= 1:
            raise ValueError(
                f"min_count_pct must be in (0, 1], got {self.min_count_pct}"
            )
    
    @property
    def momentum_col(self) -> str:
        """Construct momentum column name from window parameters."""
        return f'mom_{self.max_lag}_{self.min_lag}_mean'
    
    @property
    def cvg_col(self) -> str:
        """Construct CVG column name from window parameters."""
        return f'cvg_{self.max_lag}_{self.min_lag}'
    
    @property
    def count_col(self) -> str:
        """Construct count column name from window parameters."""
        return f'mom_{self.max_lag}_{self.min_lag}_count'
    
    @property
    def name(self) -> str:
        """Strategy identifier."""
        return f"MomentumCVG_{self.max_lag}_{self.min_lag}"
    
    @property
    def required_features(self) -> List[str]:
        """Features needed by this strategy."""
        return [self.momentum_col, self.cvg_col, self.count_col]
    
    def generate_signals(
        self,
        features: pd.DataFrame,
        current_date: date,
        **kwargs
    ) -> List[Signal]:
        """
        Generate momentum + CVG filtered signals.
        
        Two-stage process:
        1. Momentum filter: Select top/bottom percentiles
        2. CVG filter: Keep most continuous trends within each
        
        Args:
            features: Pre-computed features for current_date
            current_date: Trading date
            **kwargs: Additional context (unused in Phase 1)
            
        Returns:
            List of Signal objects sorted by conviction (descending)
        """
        
        # === VALIDATION ===
        if features.empty:
            return []
        
        # Verify required features exist
        missing = set(self.required_features) - set(features.columns)
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Verify ticker column exists
        if 'ticker' not in features.columns:
            raise ValueError("features DataFrame must contain 'ticker' column")
        
        # === DATA QUALITY FILTER ===
        
        # Calculate window size from lag parameters
        window_size = self.max_lag - self.min_lag + 1
        min_count = self.min_count_pct * window_size
        
        # Filter to tickers with sufficient data
        df = features[features[self.count_col] >= min_count].copy()
        
        if df.empty:
            return []
        
        # Remove NaN values in required features
        df = df.dropna(subset=[self.momentum_col, self.cvg_col])
        
        if df.empty:
            return []
        
        # === STAGE 1: MOMENTUM FILTERING ===
        
        # Calculate momentum quantiles
        long_threshold = df[self.momentum_col].quantile(1 - self.momentum_long_pct)
        short_threshold = df[self.momentum_col].quantile(self.momentum_short_pct)
        
        # Select candidates
        long_candidates = df[df[self.momentum_col] >= long_threshold].copy()
        short_candidates = df[df[self.momentum_col] <= short_threshold].copy()
        
        # === STAGE 2: CVG FILTERING ===
        
        signals = []
        
        # Process long candidates
        if not long_candidates.empty:
            cvg_threshold = long_candidates[self.cvg_col].quantile(1 - self.cvg_filter_pct)
            longs = long_candidates[long_candidates[self.cvg_col] >= cvg_threshold]
            
            for _, row in longs.iterrows():
                # Normalize conviction to [0, 1] using CVG value
                # CVG typically in range [0, 2], with mean ~1.24
                # Higher CVG = more continuous = higher conviction
                conviction = min(row[self.cvg_col] / 2.0, 1.0)
                
                signals.append(Signal(
                    ticker=row['ticker'],
                    signal_date=current_date,
                    strategy_type=self.strategy_type,
                    direction='long',
                    conviction=conviction,
                    features={
                        'momentum': float(row[self.momentum_col]),
                        'cvg': float(row[self.cvg_col]),
                        'count': int(row[self.count_col])
                    },
                    metadata={
                        'momentum_rank': float(
                            (row[self.momentum_col] - df[self.momentum_col].min()) / 
                            (df[self.momentum_col].max() - df[self.momentum_col].min())
                        ) if df[self.momentum_col].max() != df[self.momentum_col].min() else 0.5,
                        'cvg_rank': float(
                            (row[self.cvg_col] - long_candidates[self.cvg_col].min()) / 
                            (long_candidates[self.cvg_col].max() - long_candidates[self.cvg_col].min())
                        ) if long_candidates[self.cvg_col].max() != long_candidates[self.cvg_col].min() else 0.5
                    }
                ))
        
        # Process short candidates
        if not short_candidates.empty:
            cvg_threshold = short_candidates[self.cvg_col].quantile(1 - self.cvg_filter_pct)
            shorts = short_candidates[short_candidates[self.cvg_col] >= cvg_threshold]
            
            for _, row in shorts.iterrows():
                conviction = min(row[self.cvg_col] / 2.0, 1.0)
                
                signals.append(Signal(
                    ticker=row['ticker'],
                    signal_date=current_date,
                    strategy_type=self.strategy_type,
                    direction='short',
                    conviction=conviction,
                    features={
                        'momentum': float(row[self.momentum_col]),
                        'cvg': float(row[self.cvg_col]),
                        'count': int(row[self.count_col])
                    },
                    metadata={
                        'momentum_rank': float(
                            (row[self.momentum_col] - df[self.momentum_col].min()) / 
                            (df[self.momentum_col].max() - df[self.momentum_col].min())
                        ) if df[self.momentum_col].max() != df[self.momentum_col].min() else 0.5,
                        'cvg_rank': float(
                            (row[self.cvg_col] - short_candidates[self.cvg_col].min()) / 
                            (short_candidates[self.cvg_col].max() - short_candidates[self.cvg_col].min())
                        ) if short_candidates[self.cvg_col].max() != short_candidates[self.cvg_col].min() else 0.5
                    }
                ))
        
        # Sort by conviction (descending)
        signals.sort(key=lambda s: s.conviction, reverse=True)
        
        return signals
