"""
CVG (Continuous Volatility Gap) Feature Calculator

Calculates CVG features from historical volatility gaps (Realized Vol - Implied Vol).
Implements the IFeatureCalculator protocol.

This implements the methodology from the academic paper on option momentum:
1. Calculate volatility gap: RV - IV
2. Apply cross-sectional median adjustment per date (removes market-wide variation)
3. Calculate cumulative gap (cgap) and percentages of positive/negative gaps
4. Compute DVG (Discontinuous Volatility Gap) and CVG = 1 - DVG

CVG measures trend continuity:
- High CVG (>1): Continuous trend (all gaps same sign as cumulative)
- CVG = 1: Perfectly balanced trend
- Low CVG (<1): Discontinuous/choppy trend (many reversals)

Both longs and shorts prefer HIGH CVG (continuous trends).

Features computed per window:
- cvg: Continuous Volatility Gap (primary signal for trend quality)
- dvg: Discontinuous Volatility Gap (cvg = 1 - dvg)
- cgap: Cumulative adjusted volatility gap
- pct_pos: Percentage of positive adjusted gaps
- pct_neg: Percentage of negative adjusted gaps
- volgap_mean: Mean adjusted volatility gap
- volgap_std: Std dev of adjusted volatility gap
- cvg_count: Number of observations

Example:
    >>> from src.features.base import FeatureDataContext
    >>> from src.features.cvg_calculator import CVGCalculator
    >>> 
    >>> # Setup
    >>> context = FeatureDataContext(
    ...     straddle_history=pd.read_parquet('straddles.parquet')
    ... )
    >>> calculator = CVGCalculator(windows=[(12, 2)])
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


class CVGCalculator:
    """
    Calculate CVG (Continuous Volatility Gap) features from historical vol gaps.
    
    Implements IFeatureCalculator protocol to compute CVG metrics that measure
    the continuity/smoothness of volatility trends.
    
    Attributes:
        windows: List of (max_lag, min_lag) tuples defining lookback periods.
            Example: [(12, 2)] means use data from t-12 to t-2 weeks.
        min_periods: Minimum observations required for valid calculation.
            Default 2 (need at least 2 to determine trend direction).
    
    Example:
        >>> calculator = CVGCalculator(
        ...     windows=[(12, 2), (8, 1)],  # Two CVG windows
        ...     min_periods=2
        ... )
        >>> calculator.feature_names
        ['cvg_12_2', 'dvg_12_2', 'cgap_12_2', 'pct_pos_12_2', 
         'pct_neg_12_2', 'volgap_mean_12_2', 'volgap_std_12_2', 'cvg_count_12_2',
         'cvg_8_1', 'dvg_8_1', ...]
    """
    
    def __init__(
        self,
        windows: List[Tuple[int, int]] = None,
        min_periods: int = 1,
        vol_gap_col: str = 'vol_gap'
    ):
        """
        Initialize CVG calculator.
        
        Args:
            windows: List of (max_lag, min_lag) tuples. Default [(12, 2)].
                Each tuple defines a lookback window in weeks:
                - max_lag: How far back to start (e.g., 12 weeks ago)
                - min_lag: How far back to end (e.g., 2 weeks ago)
                Using min_lag > 0 avoids look-ahead bias.
            
            min_periods: Minimum observations needed for valid calculation.
                Default 1 (need at least 1 to calculate DVG/CVG).
                
            vol_gap_col: Column name for volatility gap (RV - IV).
                Default 'vol_gap'. If missing, will try to compute from
                'realized_volatility' - 'entry_iv'.
                
        Example:
            >>> # Standard CVG: t-12 to t-2 weeks
            >>> calc = CVGCalculator(windows=[(12, 2)])
            >>> 
            >>> # Multiple windows for different timeframes
            >>> calc = CVGCalculator(windows=[(12, 2), (8, 1), (20, 4)])
        """
        self.windows = windows if windows is not None else [(12, 2)]
        self.min_periods = min_periods
        self.vol_gap_col = vol_gap_col
        
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
            List of feature names in format: cvg_{max_lag}_{min_lag}, etc.
            
        Example:
            >>> calculator = CVGCalculator(windows=[(12, 2)])
            >>> calculator.feature_names
            ['cvg_12_2', 'dvg_12_2', 'cgap_12_2', 'pct_pos_12_2',
             'pct_neg_12_2', 'volgap_mean_12_2', 'volgap_std_12_2', 'cvg_count_12_2']
        """
        names = []
        for max_lag, min_lag in self.windows:
            suffix = f'{max_lag}_{min_lag}'
            names.extend([
                f'cvg_{suffix}',          # Continuous Volatility Gap (primary)
                f'dvg_{suffix}',          # Discontinuous Volatility Gap
                f'cgap_{suffix}',         # Cumulative adjusted gap
                f'pct_pos_{suffix}',      # % positive gaps
                f'pct_neg_{suffix}',      # % negative gaps
                f'volgap_mean_{suffix}',  # Mean adjusted gap
                #f'volgap_std_{suffix}',   # Std of adjusted gap
                f'cvg_count_{suffix}'     # Number of observations
            ])
        return names
    
    @property
    def required_data_sources(self) -> List[str]:
        """
        List of required data sources from FeatureDataContext.
        
        Returns:
            ['straddle_history'] - requires historical straddle data with vol gaps
        """
        return ['straddle_history']
    
    def calculate(
        self,
        context: FeatureDataContext,
        date: datetime,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Calculate CVG features for specified tickers at a single date.
        
        For each ticker, applies cross-sectional median adjustment and computes
        CVG metrics from the lookback window.
        
        Args:
            context: Data context containing 'straddle_history' DataFrame with:
                - ticker (str): Stock ticker
                - entry_date (datetime): Trade entry date
                - vol_gap (float): Volatility gap (RV - IV)
                  OR realized_volatility and entry_iv to compute gap
                
            date: Target date for feature calculation (features use data BEFORE this)
            
            tickers: List of ticker symbols to calculate features for
            
        Returns:
            DataFrame with columns:
                - ticker: Stock ticker
                - date: Feature calculation date
                - cvg_{w1}_{w2}: Continuous Volatility Gap
                - dvg_{w1}_{w2}: Discontinuous Volatility Gap
                - cgap_{w1}_{w2}: Cumulative adjusted gap
                - pct_pos_{w1}_{w2}: % positive gaps
                - pct_neg_{w1}_{w2}: % negative gaps
                - volgap_mean_{w1}_{w2}: Mean adjusted gap
                - volgap_std_{w1}_{w2}: Std adjusted gap
                - cvg_count_{w1}_{w2}: Number of observations
                
            NaN values indicate insufficient data (< min_periods observations).
            
        Example:
            >>> features = calculator.calculate(context, datetime(2024, 1, 5), ['AAPL', 'TSLA'])
            >>> print(features)
                ticker       date  cvg_12_2  dvg_12_2  cgap_12_2
            0    AAPL 2024-01-05     1.35     -0.35       2.5
            1    TSLA 2024-01-05     0.82      0.18      -1.2
        """
        # Convert tickers to uppercase
        tickers = [t.upper() for t in tickers]
        
        # Get straddle history
        history = context.get('straddle_history').copy()
        
        # Ensure vol_gap column exists
        if self.vol_gap_col not in history.columns:
            if 'realized_volatility' in history.columns and 'entry_iv' in history.columns:
                history['vol_gap'] = history['realized_volatility'] - history['entry_iv']
                vol_gap_col = 'vol_gap'
            else:
                raise ValueError(
                    f"Need either '{self.vol_gap_col}' column or "
                    "'realized_volatility' and 'entry_iv' columns"
                )
        else:
            vol_gap_col = self.vol_gap_col
        
        # Filter to data before target date
        history = history[history['entry_date'] <= date].copy()
        
        # Handle empty history - return NaN features for all tickers
        if len(history) == 0:
            return pd.DataFrame([
                {'ticker': ticker, 'date': date, **{fn: np.nan for fn in self.feature_names}}
                for ticker in tickers
            ])
        
        # CRITICAL: Apply cross-sectional median adjustment per date
        # This removes market-wide volatility shifts and accounts for skewness
        history['vol_gap_adjusted'] = history.groupby('entry_date')[vol_gap_col].transform(
            lambda x: x - x.median()
        )
        
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
                suffix = f'{max_lag}_{min_lag}'
                
                # Row-based lookback (same as momentum calculator)
                start_idx = target_position - max_lag
                end_idx = target_position - min_lag
                
                # Ensure indices are valid
                if start_idx < 0:
                    start_idx = 0
                if end_idx <= start_idx:
                    # No valid window
                    row.update({
                        f'cvg_{suffix}': np.nan,
                        f'dvg_{suffix}': np.nan,
                        f'cgap_{suffix}': np.nan,
                        f'pct_pos_{suffix}': np.nan,
                        f'pct_neg_{suffix}': np.nan,
                        f'volgap_mean_{suffix}': np.nan,
                        #f'volgap_std_{suffix}': np.nan,
                        f'cvg_count_{suffix}': 0
                    })
                    continue
                
                # Get window data (inclusive)
                window_data = ticker_data.iloc[start_idx:end_idx + 1]
                
                # Calculate features
                suffix = f'{max_lag}_{min_lag}'
                features = self._calculate_window_features(window_data, suffix)
                row.update(features)
            
            results.append(row)
        
        return pd.DataFrame(results)
    
    def _calculate_window_features(
        self,
        window_data: pd.DataFrame,
        suffix: str
    ) -> dict:
        """
        Calculate CVG features for a single window.
        
        Args:
            window_data: DataFrame with 'vol_gap_adjusted' column
            suffix: Feature name suffix (e.g., '12_2')
            
        Returns:
            Dict of features with NaN if insufficient data
        """
        adjusted_gaps = window_data['vol_gap_adjusted'].dropna().values
        count = len(adjusted_gaps)
        
        # Initialize features as NaN
        features = {
            f'cvg_{suffix}': np.nan,
            f'dvg_{suffix}': np.nan,
            f'cgap_{suffix}': np.nan,
            f'pct_pos_{suffix}': np.nan,
            f'pct_neg_{suffix}': np.nan,
            f'volgap_mean_{suffix}': np.nan,
           # f'volgap_std_{suffix}': np.nan,
            f'cvg_count_{suffix}': count
        }
        
        # Need minimum observations for valid statistics
        if count < self.min_periods:
            return features
        
        # Calculate basic statistics
        cgap = np.sum(adjusted_gaps)  # Cumulative adjusted gap
        mean_gap = np.mean(adjusted_gaps)
        std_gap = np.std(adjusted_gaps, ddof=1) if count > 1 else 0.0
        
        # Count positive and negative gaps
        pos_count = np.sum(adjusted_gaps >= 0)
        neg_count = np.sum(adjusted_gaps < 0)
        
        pct_pos = pos_count / count
        pct_neg = neg_count / count
        
        # Calculate DVG based on sign of cumulative gap
        # DVG = sign(cgap) × (%neg - %pos) if cgap > 0
        #     = sign(cgap) × (%pos - %neg) if cgap <= 0
        if cgap > 0:
            dvg = pct_neg - pct_pos  # Positive cumulative case
        else:
            dvg = pct_pos - pct_neg  # Negative cumulative case
        
        # CVG = 1 - DVG
        cvg = 1 - dvg
        
        # Update features
        features.update({
            f'cvg_{suffix}': cvg,
            f'dvg_{suffix}': dvg,
            f'cgap_{suffix}': cgap,
            f'pct_pos_{suffix}': pct_pos,
            f'pct_neg_{suffix}': pct_neg,
            f'volgap_mean_{suffix}': mean_gap,
           # f'volgap_std_{suffix}': std_gap
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
        Calculate CVG features for a date range efficiently (vectorized).
        
        Uses pandas rolling windows + shift (same as add_cvg_features),
        which is 20-100x faster than looping through dates.
        
        CRITICAL: Applies cross-sectional median adjustment per date to remove
        market-wide volatility shifts.
        
        Args:
            context: Data context containing 'straddle_history' DataFrame with:
                - ticker (str): Stock ticker
                - entry_date (datetime): Trade entry date
                - vol_gap (float): Volatility gap (RV - IV)
                  OR realized_volatility and entry_iv to compute gap
                
            start_date: Start date for feature calculation (inclusive)
            
            end_date: End date for feature calculation (inclusive)
            
            tickers: Optional list of tickers. If None, uses all tickers in data.
            
        Returns:
            DataFrame with columns:
                - ticker (str): Stock ticker
                - date (datetime): Feature calculation date
                - cvg_{max_lag}_{min_lag} (float): Continuous Volatility Gap
                - dvg_{max_lag}_{min_lag} (float): Discontinuous Volatility Gap
                - cgap_{max_lag}_{min_lag} (float): Cumulative adjusted gap
                - pct_pos_{max_lag}_{min_lag} (float): % positive gaps
                - pct_neg_{max_lag}_{min_lag} (float): % negative gaps
                - volgap_mean_{max_lag}_{min_lag} (float): Mean adjusted gap
                - volgap_std_{max_lag}_{min_lag} (float): Std adjusted gap
                - cvg_count_{max_lag}_{min_lag} (int): Number of observations
                
            One row per (ticker, date) where ticker has data on that date.
            NaN values indicate insufficient data (< min_periods).
            
        Example:
            >>> # Calculate for all tickers in 2024
            >>> features = calculator.calculate_bulk(
            ...     context,
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 12, 31)
            ... )
            >>> print(f"Generated {len(features):,} feature records")
        """
        # Get straddle history
        history = context.get('straddle_history').copy()
        
        # Convert dates to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(history['entry_date']):
            history['entry_date'] = pd.to_datetime(history['entry_date'])
        
        # Convert tickers to uppercase and filter if provided
        if tickers is not None:
            tickers = [t.upper() for t in tickers]
            history = history[history['ticker'].isin(tickers)].copy()
        
        # Ensure vol_gap column exists
        if self.vol_gap_col not in history.columns:
            if 'realized_volatility' in history.columns and 'entry_iv' in history.columns:
                history['vol_gap'] = history['realized_volatility'] - history['entry_iv']
                vol_gap_col = 'vol_gap'
            else:
                raise ValueError(
                    f"Need either '{self.vol_gap_col}' column or "
                    "'realized_volatility' and 'entry_iv' columns"
                )
        else:
            vol_gap_col = self.vol_gap_col
        
        # Convert dates to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(history['entry_date']):
            history['entry_date'] = pd.to_datetime(history['entry_date'])
        
        # Sort by ticker and date (required for rolling windows)
        history = history.sort_values(['ticker', 'entry_date'])
        
        # CRITICAL: Apply cross-sectional median adjustment per date
        # This removes market-wide volatility shifts and accounts for skewness
        history['vol_gap_adjusted'] = history.groupby('entry_date')[vol_gap_col].transform(
            lambda x: x - x.median()
        )
        
        # Calculate features for each window using vectorized rolling + shift
        for max_lag, min_lag in self.windows:
            window_size = max_lag - min_lag + 1
            suffix = f'{max_lag}_{min_lag}'
            
            # Group by ticker for per-ticker rolling calculations
            grouped = history.groupby('ticker')['vol_gap_adjusted']
            
            # Cumulative gap (cgap): rolling sum + shift
            history[f'cgap_{suffix}'] = (
                grouped
                .rolling(window=window_size, min_periods=1)
                .sum()
                .shift(min_lag)
                .reset_index(level=0, drop=True)
            )
            
            # Count: rolling count + shift
            history[f'cvg_count_{suffix}'] = (
                grouped
                .rolling(window=window_size, min_periods=1)
                .count()
                .shift(min_lag)
                .reset_index(level=0, drop=True)
            )
            
            # Mean: rolling mean + shift
            history[f'volgap_mean_{suffix}'] = (
                grouped
                .rolling(window=window_size, min_periods=1)
                .mean()
                .shift(min_lag)
                .reset_index(level=0, drop=True)
            )
            
            # Std: rolling std + shift
            #history[f'volgap_std_{suffix}'] = (
            #    grouped
            #    .rolling(window=window_size, min_periods=max(2, self.min_periods))
            #    .std(ddof=1)
            #    .shift(min_lag)
            #    .reset_index(level=0, drop=True)
            #)
            
            # Count positive gaps: rolling apply
            history[f'pos_count_{suffix}'] = (
                grouped
                .rolling(window=window_size, min_periods=1)
                .apply(lambda x: (x >= 0).sum(), raw=True)
                .shift(min_lag)
                .reset_index(level=0, drop=True)
            )
            
            # Count negative gaps: rolling apply
            history[f'neg_count_{suffix}'] = (
                grouped
                .rolling(window=window_size, min_periods=1)
                .apply(lambda x: (x < 0).sum(), raw=True)
                .shift(min_lag)
                .reset_index(level=0, drop=True)
            )
            
            # Calculate percentages
            history[f'pct_pos_{suffix}'] = np.where(
                history[f'cvg_count_{suffix}'] > 0,
                history[f'pos_count_{suffix}'] / history[f'cvg_count_{suffix}'],
                np.nan
            )
            
            history[f'pct_neg_{suffix}'] = np.where(
                history[f'cvg_count_{suffix}'] > 0,
                history[f'neg_count_{suffix}'] / history[f'cvg_count_{suffix}'],
                np.nan
            )
            
            # Calculate DVG based on sign of cgap
            # DVG = sign(cgap) × (%neg - %pos) if cgap > 0
            #     = sign(cgap) × (%pos - %neg) if cgap <= 0
            history[f'dvg_{suffix}'] = np.where(
                history[f'cvg_count_{suffix}'] >= self.min_periods,
                np.where(
                    history[f'cgap_{suffix}'] > 0,
                    history[f'pct_neg_{suffix}'] - history[f'pct_pos_{suffix}'],  # Positive cumulative
                    history[f'pct_pos_{suffix}'] - history[f'pct_neg_{suffix}']   # Negative cumulative
                ),
                np.nan
            )
            
            # CVG = 1 - DVG
            history[f'cvg_{suffix}'] = 1 - history[f'dvg_{suffix}']
            
            # Set all features to NaN where count < min_periods
            mask = history[f'cvg_count_{suffix}'] < self.min_periods
            for feat in ['cvg', 'dvg', 'cgap', 'pct_pos', 'pct_neg', 'volgap_mean', 'volgap_std']:
                history.loc[mask, f'{feat}_{suffix}'] = np.nan
            
            # Drop intermediate columns
            history = history.drop(columns=[f'pos_count_{suffix}', f'neg_count_{suffix}'])
        
        # Drop adjusted column
        history = history.drop(columns=['vol_gap_adjusted'])
        
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
