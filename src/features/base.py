"""
Feature Calculator Protocol

Defines the interface for feature calculators that compute trading signals
from various data sources. All feature calculators must implement this
protocol to ensure consistent behavior across the feature engineering pipeline.

The protocol enforces:
1. Consistent input/output contracts
2. Flexible data access via context object
3. Graceful handling of missing data (NaN)
4. Reusable feature calculation logic
5. Easy composition and testing

Usage:
    # Create data context with multiple data sources
    context = FeatureDataContext(
        straddle_history=straddle_df,
        earnings_calendar=earnings_df,
        macro_data=macro_df
    )
    
    # Single date calculation (live trading)
    features = calculator.calculate(context, date, tickers)
    
    # Bulk calculation (backtesting / backfill)
    all_features = calculator.calculate_bulk(context, date_range, tickers)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
import pandas as pd


class FeatureDataContext:
    """
    Container for multiple data sources used by feature calculators.
    
    This class provides a flexible way to pass multiple datasets to feature
    calculators without hardcoding specific data sources in the protocol.
    Calculators can access any data source they need via dictionary-like interface.
    
    Benefits:
    - Decouple calculators from specific data schemas
    - Easy to add new data sources without changing protocol
    - Simple dependency injection for testing
    - Clear documentation of what data each calculator needs
    
    Example:
        >>> # Create context with multiple data sources
        >>> context = FeatureDataContext(
        ...     straddle_history=straddle_df,
        ...     earnings_calendar=earnings_df,
        ...     options_volume=volume_df,
        ...     vix_history=vix_df
        ... )
        >>> 
        >>> # Calculators access data by name
        >>> straddles = context.get('straddle_history')
        >>> earnings = context.get('earnings_calendar')
        >>> 
        >>> # Check if data source exists
        >>> if context.has('earnings_calendar'):
        ...     earnings = context.get('earnings_calendar')
    """
    
    def __init__(self, **data_sources: pd.DataFrame):
        """
        Initialize context with named data sources.
        
        Args:
            **data_sources: Named DataFrames accessible to calculators
                
        Example:
            >>> context = FeatureDataContext(
            ...     straddle_history=pd.read_parquet('straddles.parquet'),
            ...     earnings_calendar=pd.read_parquet('earnings.parquet')
            ... )
        """
        self._data: Dict[str, pd.DataFrame] = data_sources
    
    def get(self, name: str) -> pd.DataFrame:
        """
        Get data source by name.
        
        Args:
            name: Data source name (e.g., 'straddle_history', 'earnings_calendar')
            
        Returns:
            DataFrame for the requested data source
            
        Raises:
            KeyError: If data source not found in context
            
        Example:
            >>> straddles = context.get('straddle_history')
            >>> earnings = context.get('earnings_calendar')
        """
        if name not in self._data:
            available = ', '.join(self._data.keys())
            raise KeyError(
                f"Data source '{name}' not found in context. "
                f"Available sources: {available}"
            )
        return self._data[name]
    
    def has(self, name: str) -> bool:
        """
        Check if data source exists in context.
        
        Args:
            name: Data source name to check
            
        Returns:
            True if data source exists, False otherwise
            
        Example:
            >>> if context.has('earnings_calendar'):
            ...     earnings = context.get('earnings_calendar')
            ... else:
            ...     print("Earnings data not available")
        """
        return name in self._data
    
    def add(self, name: str, data: pd.DataFrame) -> None:
        """
        Add or update data source in context.
        
        Args:
            name: Data source name
            data: DataFrame to add
            
        Example:
            >>> context.add('news_sentiment', sentiment_df)
        """
        self._data[name] = data
    
    @property
    def available_sources(self) -> List[str]:
        """
        List all available data source names.
        
        Returns:
            List of data source names in context
            
        Example:
            >>> print(context.available_sources)
            ['straddle_history', 'earnings_calendar', 'options_volume']
        """
        return list(self._data.keys())


class IFeatureCalculator(Protocol):
    """
    Protocol for feature calculators that compute trading signals from various data sources.
    
    Feature calculators take multiple data sources via FeatureDataContext and compute
    derived features (e.g., momentum, earnings proximity, volatility patterns) that
    can be used by trading strategies to generate signals.
    
    All feature calculators must implement:
    1. feature_names property: List of column names produced
    2. required_data_sources property: List of data sources needed from context
    3. calculate() method: Compute features for single date (live trading)
    4. calculate_bulk() method (optional): Efficient multi-date calculation (backtesting)
    
    The protocol ensures:
    - Consistent DataFrame schemas for feature storage
    - Flexible access to multiple data sources
    - Clear documentation of data dependencies
    - Graceful NaN handling for missing/incomplete data
    - Reusable calculation logic across different feature types
    - Efficient bulk calculations for backtesting
    - Easy testing and composition of multiple feature calculators
    """
    
    @property
    def feature_names(self) -> List[str]:
        """
        List of feature column names produced by this calculator.
        
        These names will be used as DataFrame columns in the output and should:
        - Be descriptive and unique
        - Follow naming convention: {type}_{window}_{stat}
          Example: 'mom_12_2_mean', 'cvg', 'earnings_in_7d', 'vol_spread_30d'
        - Not conflict with reserved columns: 'ticker', 'date'
        
        Returns:
            List of feature column names (e.g., ['mom_12_2_mean', 'mom_12_2_std'])
            
        Example:
            >>> calculator = MomentumCalculator(windows=[(12, 2)])
            >>> calculator.feature_names
            ['mom_12_2_mean', 'mom_12_2_sum', 'mom_12_2_count', 
             'mom_12_2_std', 'mom_12_2_sharpe', 'mom_12_2_win_rate']
        """
        ...
    
    @property
    def required_data_sources(self) -> List[str]:
        """
        List of data source names required from FeatureDataContext.
        
        Declares which data sources this calculator needs to access via context.get().
        This enables:
        - Clear documentation of dependencies
        - Validation before calculation
        - Better error messages if data missing
        
        Returns:
            List of data source names (e.g., ['straddle_history', 'earnings_calendar'])
            
        Example:
            >>> calculator = EarningsFeatureCalculator()
            >>> calculator.required_data_sources
            ['straddle_history', 'earnings_calendar']
            
            >>> # Validate context has required data
            >>> missing = [s for s in calculator.required_data_sources 
            ...           if not context.has(s)]
            >>> if missing:
            ...     raise ValueError(f"Missing data sources: {missing}")
        """
        ...
    
    def calculate(
        self,
        context: FeatureDataContext,
        date: datetime,
        tickers: List[str]
    ) -> pd.DataFrame:
        """
        Calculate features for specified tickers at a single date.
        
        **Use Case:** Live trading - calculate features for current week only.
        
        This method computes features using data from context up to (but not
        including) the specified date. The calculation uses a lookback window
        defined by the feature calculator's parameters.
        
        Args:
            context: Data context containing all required data sources.
                Calculators access data via context.get('source_name').
                Required sources listed in required_data_sources property.
                
            date: Target date for feature calculation (features use data BEFORE this date)
            
            tickers: List of ticker symbols to calculate features for
            
        Returns:
            DataFrame with columns:
                - ticker (str): Stock ticker symbol
                - date (datetime): Feature calculation date (same as input date)
                - {feature_name_1} (float): First feature value (may be NaN)
                - {feature_name_2} (float): Second feature value (may be NaN)
                - ...
                
            Rows: One row per ticker (even if features are NaN)
            
        NaN Handling:
            Features should be NaN when:
            - Insufficient historical data (e.g., < min_periods observations)
            - Ticker not present in required data sources
            - Invalid calculations (e.g., division by zero)
            - Required data source missing from context
            
            Implementations must:
            - Return a row for every ticker in the input list
            - Set features to NaN gracefully (no exceptions)
            - Allow strategies to filter based on completeness
            
        Example (Live Trading):
            >>> # Live trading: calculate features for today only
            >>> context = FeatureDataContext(
            ...     straddle_history=pd.read_parquet('straddles.parquet'),
            ...     earnings_calendar=pd.read_parquet('earnings.parquet')
            ... )
            >>> 
            >>> today = datetime.now()
            >>> active_tickers = ['AAPL', 'TSLA', 'MSFT']
            >>> 
            >>> features = calculator.calculate(
            ...     context=context,
            ...     date=today,
            ...     tickers=active_tickers
            ... )
            >>> 
            >>> print(features)
                ticker       date  mom_12_2_mean  has_earnings
            0    AAPL 2024-01-05          12.5         True
            1    TSLA 2024-01-05           8.3        False
            2    MSFT 2024-01-05          10.1         True
            
        Performance Considerations:
            - Implementations should vectorize calculations when possible
            - Use pandas groupby operations for per-ticker calculations
            - Consider caching if calculate() called multiple times per date
            - Access context data sources once and reuse (avoid repeated .get() calls)
            
        Raises:
            KeyError: If required data source not found in context
            ValueError: If date is invalid (e.g., future date, before history start)
            
        See Also:
            - calculate_bulk(): Efficient multi-date calculation for backtesting
            - MomentumCalculator: Windowed return momentum features
            - CVGCalculator: Continuous volatility gap features
        """
        ...
    
    def calculate_bulk(
        self,
        context: FeatureDataContext,
        start_date: datetime,
        end_date: datetime,
        tickers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate features for a date range efficiently (optimized for backtesting).
        
        **Use Case:** Backtesting / backfill - calculate features for entire date range.
        
        This method is optimized for bulk calculations by:
        - Reusing rolling window computations across dates
        - Avoiding redundant data filtering
        - Vectorizing cross-sectional operations
        
        Default implementation calls calculate() in a loop. Subclasses should
        override with optimized implementations for better performance.
        
        Args:
            context: Data context containing all required data sources
            start_date: Start date for feature calculation (inclusive)
            end_date: End date for feature calculation (inclusive)
            tickers: Optional list of tickers. If None, uses all tickers in data.
            
        Returns:
            DataFrame with columns:
                - ticker (str): Stock ticker symbol
                - date (datetime): Feature calculation date
                - {feature_name_1} (float): First feature value (may be NaN)
                - {feature_name_2} (float): Second feature value (may be NaN)
                - ...
                
            Rows: One row per (ticker, date) combination
            
        Example (Backtesting):
            >>> # Backtesting: calculate features for entire date range
            >>> context = FeatureDataContext(
            ...     straddle_history=pd.read_parquet('straddles.parquet'),
            ...     earnings_calendar=pd.read_parquet('earnings.parquet')
            ... )
            >>> 
            >>> # Calculate for all dates in range (efficient)
            >>> all_features = calculator.calculate_bulk(
            ...     context=context,
            ...     start_date=datetime(2018, 1, 1),
            ...     end_date=datetime(2024, 12, 31),
            ...     tickers=None  # All tickers
            ... )
            >>> 
            >>> print(f"Generated {len(all_features):,} feature records")
            
        Performance Optimization:
            Default implementation:
            ```python
            def calculate_bulk(self, context, start_date, end_date, tickers):
                # Generate dates and call calculate() in loop
                dates = pd.date_range(start_date, end_date, freq='W-FRI')
                results = []
                for date in dates:
                    features = self.calculate(context, date, tickers)
                    results.append(features)
                return pd.concat(results, ignore_index=True)
            ```
            
            Optimized implementation (override in subclass):
            ```python
            def calculate_bulk(self, context, start_date, end_date, tickers):
                # Load data once
                history = context.get('straddle_history')
                
                # Vectorized rolling calculations on entire dataset
                rolling_features = history.groupby('ticker').rolling(
                    window=self.window, min_periods=self.min_periods
                ).agg({'return_pct': ['mean', 'std']})
                
                # Filter to date range
                results = rolling_features[
                    (rolling_features['date'] >= start_date) &
                    (rolling_features['date'] <= end_date)
                ]
                return results
            ```
            
        Raises:
            KeyError: If required data source not found in context
            ValueError: If start_date > end_date or invalid date range
            
        See Also:
            - calculate(): Single date calculation for live trading
        """
        # Default implementation: generate dates and call calculate() in loop
        # Subclasses should override with optimized bulk logic
        if tickers is None:
            # Get all tickers from first required data source
            first_source = self.required_data_sources[0]
            data = context.get(first_source)
            tickers = data['ticker'].unique().tolist()
        
        # Generate weekly dates in range
        dates = pd.date_range(start_date, end_date, freq='W-FRI').tolist()
        
        results = []
        for date in dates:
            features = self.calculate(context, date, tickers)
            results.append(features)
        
        return pd.concat(results, ignore_index=True)


# Type alias for documentation
FeatureDataFrame = pd.DataFrame
"""
Type alias for feature calculator output DataFrame.

Schema:
    ticker: str - Stock ticker symbol
    date: datetime - Feature calculation date
    {feature_names}: float - Feature values (may contain NaN)
    
Example:
    >>> features: FeatureDataFrame = calculator.calculate(context, date, tickers)
"""
