"""
Momentum Feature Calculator

Calculates momentum features from historical straddle returns using rolling windows.
Implements the IFeatureCalculator protocol.

Momentum is calculated as the average return over a lookback window, typically
from t-12 to t-2 weeks (excluding the most recent week to avoid look-ahead bias).

Features computed per window:
- mean: Average return (primary momentum signal)
- sum: Cumulative return
- count: Number of observations
- std: Return volatility

Example:
    >>> from src.features.base import FeatureDataContext
    >>> from src.features.momentum_calculator import MomentumCalculator
    >>> 
    >>> # Setup
    >>> context = FeatureDataContext(
    ...     straddle_history=pd.read_parquet('straddles.parquet')
    ... )
    >>> calculator = MomentumCalculator(windows=[(12, 2)])
    >>> 
    >>> # Single date calculation (live trading)
    >>> features = calculator.calculate(context, date, tickers)
    >>> 
    >>> # Bulk calculation (backtesting)
    >>> all_features = calculator.calculate_bulk(context, dates)
"""

from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

from .base import IFeatureCalculator, FeatureDataContext


class MomentumCalculator:
    """
    Calculate momentum features from historical straddle returns.
    
    Implements IFeatureCalculator protocol to compute windowed momentum
    statistics (mean, sum, count, std) using row-based lookback.
    
    **Important:** Works with sparse data. Each ticker has its own timeline
    of trades which may have gaps due to IPO dates, delistings, liquidity
    issues, or missing data. Rolling windows use row-based lookback (not
    calendar weeks):
    
    - Window (12, 2) at position 14 uses rows [2, 3, ..., 12] from ticker's history
    - These 11 rows may span varying calendar time depending on data gaps
    - Features are NaN if count < min_periods observations
    
    Attributes:
        windows: List of (max_lag, min_lag) tuples defining lookback periods.
            Example: [(12, 2)] means use returns from position t-12 to t-2.
        min_periods: Minimum observations required for valid calculation.
            Default 1 allows single observation; use 3+ for stable statistics.
    
    Example:
        >>> calculator = MomentumCalculator(
        ...     windows=[(12, 2), (8, 1)],  # Two momentum windows
        ...     min_periods=3
        ... )
        >>> calculator.feature_names
        ['mom_12_2_mean', 'mom_12_2_sum', 'mom_12_2_count', 'mom_12_2_std',
         'mom_8_1_mean', 'mom_8_1_sum', 'mom_8_1_count', 'mom_8_1_std']
    """
    
    def __init__(
        self,
        windows: List[Tuple[int, int]] = None,
        min_periods: int = 1
    ):
        """
        Initialize momentum calculator.
        
        Args:
            windows: List of (max_lag, min_lag) tuples. Default [(12, 2)].
                Each tuple defines a lookback window in weeks:
                - max_lag: How far back to start (e.g., 12 weeks ago)
                - min_lag: How far back to end (e.g., 2 weeks ago)
                Using min_lag > 0 avoids look-ahead bias.
            
            min_periods: Minimum observations needed for valid calculation.
                If fewer observations, features will be NaN. Default 3.
                
        Example:
            >>> # Standard momentum: t-12 to t-2 weeks
            >>> calc = MomentumCalculator(windows=[(12, 2)])
            >>> 
            >>> # Multiple windows for different timeframes
            >>> calc = MomentumCalculator(windows=[(12, 2), (8, 1), (20, 4)])
        """
        self.windows = windows if windows is not None else [(12, 2)]
        self.min_periods = min_periods
        
        # Validate windows
        for max_lag, min_lag in self.windows:
            if max_lag <= min_lag:
                raise ValueError(
                    f"max_lag ({max_lag}) must be > min_lag ({min_lag})"
                )
            if min_lag < 0:
                raise ValueError(f"min_lag ({min_lag}) must be >= 0")
    
    @property
    def feature_names(self) -> List[str]:
        """
        List of feature column names produced by this calculator.
        
        Returns:
            List of feature names in format: mom_{max_lag}_{min_lag}_{stat}
            
        Example:
            >>> calculator = MomentumCalculator(windows=[(12, 2)])
            >>> calculator.feature_names
            ['mom_12_2_mean', 'mom_12_2_sum', 'mom_12_2_count', 'mom_12_2_std']
        """
        names = []
        for max_lag, min_lag in self.windows:
            prefix = f'mom_{max_lag}_{min_lag}'
            names.extend([
                f'{prefix}_mean',      # Average return
                f'{prefix}_sum',       # Cumulative return
                f'{prefix}_count',     # Number of observations
                f'{prefix}_std'        # Volatility
            ])
        return names
    
    @property
    def required_data_sources(self) -> List[str]:
        """
        List of required data sources from FeatureDataContext.
        
        Returns:
            ['straddle_history'] - requires historical straddle returns
        """
        return ['straddle_history']
    
    def calculate(
        self,
        context: FeatureDataContext,
        date: datetime,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Calculate momentum features for specified tickers at a single date.
        
        Uses row-based lookback (same as rolling windows in calculate_bulk).
        Works with sparse data - tickers may have gaps in their history due to
        IPO dates, delistings, or data quality issues.
        
        Args:
            context: Data context containing 'straddle_history' DataFrame with:
                - ticker (str): Stock ticker
                - entry_date (datetime/date): Trade entry date (sparse, not all dates)
                - return_pct (float): Realized return percentage
                
            date: Target date for feature calculation (datetime or pandas Timestamp).
                  Compared against entry_date using <= (includes target date).
            
            tickers: List of ticker symbols to calculate features for. Tickers
                     with insufficient history will have NaN features.
            
        Returns:
            DataFrame with columns:
                - ticker: Stock ticker
                - date: Feature calculation date
                - mom_{w1}_{w2}_mean: Average return in window
                - mom_{w1}_{w2}_sum: Cumulative return
                - mom_{w1}_{w2}_count: Number of non-NaN observations
                - mom_{w1}_{w2}_std: Return volatility
                
            NaN values indicate insufficient data (< min_periods observations).
            
        Example:
            >>> features = calculator.calculate(context, datetime(2024, 1, 5), ['AAPL', 'TSLA'])
            >>> print(features)
                ticker       date  mom_12_2_mean  mom_12_2_count
            0    AAPL 2024-01-05          12.5              10
            1    TSLA 2024-01-05           8.3               9
        """
        # Convert tickers to uppercase
        tickers = [t.upper() for t in tickers]
        
        # Get straddle history
        history = context.get('straddle_history')
        
        # Filter to data up to and including target date
        history = history[history['entry_date'] <= date].copy()
        
        # Handle empty history
        if len(history) == 0:
            return pd.DataFrame([
                {'ticker': ticker, 'date': date, **{fn: np.nan for fn in self.feature_names}}
                for ticker in tickers
            ])
        
        # Sort by ticker and date
        history = history.sort_values(['ticker', 'entry_date'])
        
        results = []
        
        for ticker in tickers:
            # Filter to this ticker
            ticker_data = history[history['ticker'] == ticker].copy()
            
            if len(ticker_data) == 0:
                # No history for this ticker
                row = {'ticker': ticker, 'date': date}
                row.update({fn: np.nan for fn in self.feature_names})
                results.append(row)
                continue
            
            # Find the row for the target date
            target_rows = ticker_data[ticker_data['entry_date'] == date]
            
            if len(target_rows) == 0:
                # Target date not in history for this ticker
                row = {'ticker': ticker, 'date': date}
                row.update({fn: np.nan for fn in self.feature_names})
                results.append(row)
                continue
            
            # Get position of target date in ticker's history
            target_position = ticker_data.index.get_loc(target_rows.index[0])
            
            # Calculate features for each window
            row = {'ticker': ticker, 'date': date}
            
            for max_lag, min_lag in self.windows:
                prefix = f'mom_{max_lag}_{min_lag}'
                
                # Row-based lookback:
                # Current row is at position target_position
                # Look back from (target_position - max_lag) to (target_position - min_lag)
                # Example: window (12, 2) at position 14
                #   start = 14 - 12 = 2
                #   end = 14 - 2 = 12
                #   Uses rows [2, 3, 4, ..., 12] (positions 2 through 12, inclusive)
                start_idx = target_position - max_lag
                end_idx = target_position - min_lag
                
                # Ensure indices are valid
                if start_idx < 0:
                    start_idx = 0
                if end_idx <= start_idx:
                    # No valid window
                    row.update({
                        f'{prefix}_mean': np.nan,
                        f'{prefix}_sum': np.nan,
                        f'{prefix}_count': 0,
                        f'{prefix}_std': np.nan
                    })
                    continue
                
                # Get window data (inclusive slice from start_idx to end_idx)
                # iloc[start:end+1] to include end_idx
                window_data = ticker_data.iloc[start_idx:end_idx + 1]
                
                # Calculate features
                features = self._calculate_window_features(window_data, prefix)
                row.update(features)
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def _calculate_window_features(
        self,
        window_data: pd.DataFrame,
        prefix: str
    ) -> dict:
        """
        Calculate momentum features for a single window.
        
        Args:
            window_data: DataFrame with 'return_pct' column
            prefix: Feature name prefix (e.g., 'mom_12_2')
            
        Returns:
            Dict of features with NaN if insufficient data
        """
        # Get returns, excluding NaN values
        returns = window_data['return_pct'].dropna().values
        count = len(returns)  # Count of non-NaN values only
        
        # Initialize features
        features = {
            f'{prefix}_mean': np.nan,
            f'{prefix}_sum': np.nan,
            f'{prefix}_count': count,
            f'{prefix}_std': np.nan
        }
        
        # Need minimum observations for valid statistics
        if count < self.min_periods:
            return features
        
        # Calculate statistics (only on non-NaN returns)
        mean_return = np.mean(returns)
        sum_return = np.sum(returns)
        std_return = np.std(returns, ddof=1) if count > 1 else 0.0
        
        # Update features
        features.update({
            f'{prefix}_mean': mean_return,
            f'{prefix}_sum': sum_return,
            f'{prefix}_std': std_return
        })
        
        return features
    
    def calculate_bulk(
        self,
        context: FeatureDataContext,
        start_date: datetime,
        end_date: datetime,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate momentum features for a date range efficiently (vectorized).
        
        Uses pandas rolling windows + shift for 20-100x speedup vs looping.
        Works with sparse data - each ticker has its own timeline with potential gaps.
        
        Args:
            context: Data context containing 'straddle_history' DataFrame with:
                - ticker (str): Stock ticker
                - entry_date (datetime/date): Trade entry date (sparse timeline)
                - return_pct (float): Realized return percentage
                
            start_date: Start date for feature calculation (inclusive)
            
            end_date: End date for feature calculation (inclusive)
            
            tickers: Optional list of tickers. If None, uses all tickers in data.
            
        Returns:
            DataFrame with columns:
                - ticker (str): Stock ticker
                - date (datetime): Feature calculation date
                - mom_{max_lag}_{min_lag}_mean (float): Average return in window
                - mom_{max_lag}_{min_lag}_sum (float): Cumulative return
                - mom_{max_lag}_{min_lag}_count (int): Number of observations
                - mom_{max_lag}_{min_lag}_std (float): Return volatility
                
            One row per (ticker, date) where ticker has data in the date range.
            NaN values indicate insufficient data (< min_periods).
            
        Example:
            >>> # Calculate for all tickers in 2024
            >>> features = calculator.calculate_bulk(
            ...     context,
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 12, 31)
            ... )
            >>> 
            >>> # Calculate for specific tickers
            >>> features = calculator.calculate_bulk(
            ...     context,
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 12, 31),
            ...     tickers=['AAPL', 'TSLA']
            ... )
        """
        # Get straddle history
        history = context.get('straddle_history').copy()
        
        # Convert dates to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(history['entry_date']):
            history['entry_date'] = pd.to_datetime(history['entry_date'])
        
        # Convert tickers to uppercase if provided
        if tickers is not None:
            tickers = [t.upper() for t in tickers]
            # Filter to requested tickers
            history = history[history['ticker'].isin(tickers)].copy()
        
        # Sort by ticker and date (required for rolling windows)
        history = history.sort_values(['ticker', 'entry_date'])
        
        
        return_col = 'return_pct'
        
        # Calculate features for each window using vectorized rolling + shift
        for max_lag, min_lag in self.windows:
            window_size = max_lag - min_lag + 1
            prefix = f'mom_{max_lag}_{min_lag}'
            
            # Group by ticker for per-ticker rolling calculations
            grouped = history.groupby('ticker')[return_col]
            
            # Sum: rolling sum + shift
            history[f'{prefix}_sum'] = (
                grouped
                .rolling(window=window_size, min_periods=1)
                .sum()
                .shift(min_lag)
                .reset_index(level=0, drop=True)
            )
            
            # Count: rolling count + shift
            history[f'{prefix}_count'] = (
                grouped
                .rolling(window=window_size, min_periods=1)
                .count()
                .shift(min_lag)
                .reset_index(level=0, drop=True)
            )
            
            # Mean: sum / count
            history[f'{prefix}_mean'] = (
                history[f'{prefix}_sum'] / history[f'{prefix}_count']
            )
            
            # Std: rolling std + shift (need at least 2 for std calculation)
            history[f'{prefix}_std'] = (
                grouped
                .rolling(window=window_size, min_periods=max(2, self.min_periods))
                .std(ddof=1)
                .shift(min_lag)
                .reset_index(level=0, drop=True)
            )
            
            # Set features to NaN where count < min_periods
            mask = history[f'{prefix}_count'] < self.min_periods
            for stat in ['mean', 'sum', 'std']:
                history.loc[mask, f'{prefix}_{stat}'] = np.nan
        
        # Filter to target date range (inclusive)
        result = history[
            (history['entry_date'] >= start_date) & 
            (history['entry_date'] <= end_date)
        ].copy()
        
        # Select output columns
        output_cols = ['ticker', 'entry_date'] + self.feature_names
        result = result[output_cols].copy()
        
        # Rename entry_date to date
        result = result.rename(columns={'entry_date': 'date'})
        
        return result
