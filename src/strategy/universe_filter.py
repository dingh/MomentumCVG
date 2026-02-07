"""
Universe filter for backtesting.

Simple pre-processing step in two-stage pipeline:
1. Load pre-computed features
2. Filter to universe (this module)
3. Pass to backtest engine

Supports:
- S&P 500 constituents (time-aware with additions/removals)
- All tickers (no filtering)
"""

from typing import Protocol
import pandas as pd
from pathlib import Path


class IUniverseFilter(Protocol):
    """Protocol for universe filters."""
    
    def filter_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Filter features DataFrame to universe members.
        
        Args:
            features: DataFrame with 'ticker' and 'date' columns
        
        Returns:
            Filtered DataFrame with only universe members
        """
        ...


class SP500UniverseFilter:
    """
    S&P 500 universe filter with time-aware membership.
    
    Handles additions/removals over time (e.g., TSLA added Dec 2020).
    
    Example:
        >>> universe = SP500UniverseFilter('cache/SP500.csv')
        >>> sp500_features = universe.filter_features(features_df)
        >>> print(f"Filtered to {len(sp500_features['ticker'].unique())} S&P 500 stocks")
    """
    
    def __init__(self, sp500_csv_path: str = 'cache/SP500.csv'):
        """
        Initialize S&P 500 filter.
        
        Args:
            sp500_csv_path: Path to S&P 500 membership CSV
                           Expected columns: 'Ticker', 'Date added', 'Date removed'
        """
        self.sp500_csv_path = Path(sp500_csv_path)
        self._membership_df = self._load_membership()
    
    def _load_membership(self) -> pd.DataFrame:
        """Load and parse S&P 500 membership data."""
        if not self.sp500_csv_path.exists():
            raise FileNotFoundError(
                f"S&P 500 membership file not found: {self.sp500_csv_path}"
            )
        
        df = pd.read_csv(self.sp500_csv_path)
        
        # Validate required columns
        required_cols = ['Ticker', 'Date added', 'Date removed']
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"S&P 500 CSV missing columns: {missing}")
        
        # Parse dates
        df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce')
        df['Date removed'] = pd.to_datetime(df['Date removed'], errors='coerce')
        
        # Fill NaN dates
        # NaN in 'Date added' = member since beginning of time
        df['Date added'] = df['Date added'].fillna(pd.Timestamp('1900-01-01'))
        
        # NaN in 'Date removed' = still member (use future date)
        df['Date removed'] = df['Date removed'].fillna(pd.Timestamp('2100-01-01'))
        
        return df
    
    def filter_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Filter features to S&P 500 members, respecting time-varying membership.
        
        Args:
            features: DataFrame with columns:
                     - 'ticker': Stock symbol
                     - 'date': Date of observation (used for time-aware filtering)
        
        Returns:
            Filtered DataFrame with only S&P 500 members on each date
            
        Example:
            >>> features = pd.DataFrame({
            ...     'ticker': ['AAPL', 'TSLA', 'XYZ'],
            ...     'date': [date(2020, 1, 1), date(2020, 1, 1), date(2020, 1, 1)],
            ...     'mom_60_8_mean': [0.05, 0.10, -0.02]
            ... })
            >>> universe = SP500UniverseFilter()
            >>> filtered = universe.filter_features(features)
            >>> # TSLA excluded (not in S&P 500 until Dec 2020)
            >>> # XYZ excluded (not in S&P 500)
            >>> assert 'TSLA' not in filtered['ticker'].values
        """
        if features.empty:
            return features
        
        # Validate required columns
        if 'ticker' not in features.columns:
            raise ValueError("features must contain 'ticker' column")
        if 'date' not in features.columns:
            raise ValueError("features must contain 'date' column")
        
        # Merge with membership data
        merged = features.merge(
            self._membership_df[['Ticker', 'Date added', 'Date removed']],
            left_on='ticker',
            right_on='Ticker',
            how='inner'  # Inner join = only keep S&P 500 tickers
        )
        
        # Time-aware filtering: ticker was member on date
        merged['date'] = pd.to_datetime(merged['date'])
        mask = (
            (merged['Date added'] < merged['date']) &
            (merged['Date removed'] > merged['date'])
        )
        
        filtered = merged[mask].copy()
        
        # Drop membership columns
        filtered = filtered.drop(columns=['Ticker', 'Date added', 'Date removed'])
        
        return filtered


class AllTickersUniverseFilter:
    """
    No-op filter (returns all tickers).
    
    Useful for:
    - Testing strategy on entire market
    - Comparing S&P 500 performance vs broader universe
    
    Example:
        >>> universe = AllTickersUniverseFilter()
        >>> all_features = universe.filter_features(features_df)
        >>> # Returns features_df unchanged
    """
    
    def filter_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return features unchanged (no filtering)."""
        return features