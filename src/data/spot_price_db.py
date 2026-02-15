"""
Spot price database for fast lookups and volatility calculations.

This module provides efficient access to pre-computed spot prices
extracted from ORATS data. Used for:
1. Correct realized volatility calculation (daily returns)
2. Fast spot price lookups during backtesting
3. Performance analytics

Usage:
    >>> spot_db = SpotPriceDB.load('cache/spot_prices_adjusted.csv')
    >>> 
    >>> # Get spot price
    >>> spot = spot_db.get_spot('AAPL', date(2024, 1, 5))
    >>> 
    >>> # Calculate realized volatility (correct method)
    >>> rv = spot_db.calculate_realized_volatility(
    ...     ticker='AAPL',
    ...     start_date=date(2024, 1, 5),
    ...     end_date=date(2024, 1, 12)
    ... )
"""

from typing import Optional, List
from datetime import date, timedelta
from pathlib import Path
import logging

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class SpotPriceDB:
    """
    Efficient spot price database with fast lookups.
    
    Loads pre-computed spot prices into memory for fast queries.
    Uses multi-index for O(1) lookups by (date, ticker).
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize database from DataFrame.
        
        Args:
            df: DataFrame with columns [date, ticker, adj_spot_price]
        """
        # Convert date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Set multi-index for fast lookups: (date, ticker)
        self.df = df.set_index(['date', 'ticker']).sort_index()
        
        # Cache metadata
        self.tickers = sorted(df['ticker'].unique())
        self.date_range = (df['date'].min().date(), df['date'].max().date())
        self.total_records = len(df)
        
        logger.info(f"Loaded spot price database:")
        logger.info(f"  Tickers: {len(self.tickers)}")
        logger.info(f"  Date range: {self.date_range[0]} to {self.date_range[1]}")
        logger.info(f"  Total records: {self.total_records:,}")
    
    @classmethod
    def load(cls, file_path: str) -> 'SpotPriceDB':
        """
        Load spot price database from CSV or Parquet file.
        
        Automatically detects format from file extension.
        
        Args:
            file_path: Path to spot prices file (.csv or .parquet)
            
        Returns:
            SpotPriceDB instance
            
        Example:
            >>> spot_db = SpotPriceDB.load('cache/spot_prices_adjusted.parquet')
            >>> spot_db = SpotPriceDB.load('cache/spot_prices_adjusted.csv')
        """
        logger.info(f"Loading spot prices from {file_path}...")
        
        # Detect format from extension
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, parse_dates=['date'])
        else:
            # Try parquet first (faster), fallback to CSV
            try:
                df = pd.read_parquet(file_path)
            except:
                df = pd.read_csv(file_path, parse_dates=['date'])
        
        return cls(df)
    
    def get_spot(self, ticker: str, trade_date: date) -> Optional[float]:
        """
        Get spot price for ticker on given date.
        
        Args:
            ticker: Stock ticker
            trade_date: Date to get spot price for
            
        Returns:
            Spot price, or None if not found
            
        Example:
            >>> spot = spot_db.get_spot('AAPL', date(2024, 1, 5))
            >>> print(spot)
            185.23
        """
        try:
            # Convert date to pandas Timestamp for indexing
            ts = pd.Timestamp(trade_date)
            spot = self.df.loc[(ts, ticker), 'adj_spot_price']
            return float(spot)
        except KeyError:
            return None
    
    def get_daily_spots(
        self,
        ticker: str,
        start_date: date,
        end_date: date
    ) -> pd.Series:
        """
        Get daily spot prices for ticker over date range.
        
        Args:
            ticker: Stock ticker
            start_date: First date (inclusive)
            end_date: Last date (inclusive)
            
        Returns:
            Series with date index and spot prices
            
        Example:
            >>> spots = spot_db.get_daily_spots(
            ...     'AAPL',
            ...     date(2024, 1, 5),
            ...     date(2024, 1, 12)
            ... )
            >>> print(spots)
            date
            2024-01-05    185.23
            2024-01-08    186.42
            2024-01-09    185.91
            ...
        """
        # Convert to pandas Timestamps
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Slice multi-index: all dates for ticker
        try:
            ticker_data = self.df.xs(ticker, level='ticker')
            
            # Filter to date range
            mask = (ticker_data.index >= start_ts) & (ticker_data.index <= end_ts)
            spots = ticker_data.loc[mask, 'adj_spot_price']
            
            return spots
            
        except KeyError:
            # Ticker not found
            return pd.Series(dtype=float)
    
    def calculate_realized_volatility(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        min_observations: int = 3
    ) -> Optional[float]:
        """
        Calculate annualized realized volatility using sum of squared returns.
        
        Uses standard realized variance formula: RV = sqrt(252 * mean(r_t^2))
        where r_t are daily log returns. This assumes zero mean, which is
        standard practice in realized volatility literature.
        
        Steps:
        1. Get daily spot prices over period
        2. Calculate daily log returns: ln(S_t / S_{t-1})
        3. Calculate RV: sqrt(252 * mean(returns^2))
        
        Args:
            ticker: Stock ticker
            start_date: Period start date
            end_date: Period end date
            min_observations: Minimum daily observations required (default: 3)
            
        Returns:
            Annualized realized volatility, or None if insufficient data
            
        Example:
            >>> # Calculate RV for 7-day straddle holding period
            >>> rv = spot_db.calculate_realized_volatility(
            ...     ticker='AAPL',
            ...     start_date=date(2024, 1, 5),
            ...     end_date=date(2024, 1, 12)
            ... )
            >>> print(f"Realized Vol: {rv:.2%}")
            Realized Vol: 18.45%
        """
        # Get daily spot prices
        spots = self.get_daily_spots(ticker, start_date, end_date)
        
        if len(spots) < min_observations:
            logger.warning(
                f"Insufficient data for {ticker} from {start_date} to {end_date}: "
                f"only {len(spots)} observations (need {min_observations})"
            )
            return None
        
        # Calculate daily log returns
        daily_returns = np.log(spots / spots.shift(1)).dropna()
        
        if len(daily_returns) < (min_observations - 1):
            return None
        
        # Realized volatility: sqrt(252 * mean of squared returns)
        # Standard RV formula assuming zero mean for daily returns
        rv = np.sqrt(252 * np.mean(daily_returns.values**2))
        
        return float(rv)
    
    def calculate_spot_move_pct(
        self,
        ticker: str,
        start_date: date,
        end_date: date
    ) -> Optional[float]:
        """
        Calculate percentage spot move from start to end.
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            
        Returns:
            Percentage move (e.g., 0.05 for +5%), or None if data missing
            
        Example:
            >>> move = spot_db.calculate_spot_move_pct(
            ...     'AAPL',
            ...     date(2024, 1, 5),
            ...     date(2024, 1, 12)
            ... )
            >>> print(f"Spot moved {move:.2%}")
            Spot moved +2.35%
        """
        start_spot = self.get_spot(ticker, start_date)
        end_spot = self.get_spot(ticker, end_date)
        
        if start_spot is None or end_spot is None:
            return None
        
        move_pct = (end_spot - start_spot) / start_spot
        return float(move_pct)
    
    def get_ticker_availability(self, ticker: str) -> tuple[date, date, int]:
        """
        Get availability statistics for a ticker.
        
        Args:
            ticker: Stock ticker
            
        Returns:
            Tuple of (first_date, last_date, num_observations)
            
        Example:
            >>> first, last, count = spot_db.get_ticker_availability('AAPL')
            >>> print(f"AAPL: {first} to {last}, {count} days")
            AAPL: 2018-01-02 to 2025-12-31, 2016 days
        """
        try:
            ticker_data = self.df.xs(ticker, level='ticker')
            first_date = ticker_data.index.min().date()
            last_date = ticker_data.index.max().date()
            count = len(ticker_data)
            return first_date, last_date, count
        except KeyError:
            return None, None, 0
