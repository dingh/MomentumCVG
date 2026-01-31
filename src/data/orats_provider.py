"""
ORATS Data Provider - loads and filters option data from ORATS parquet files.

This provider handles ORATS wide-format data where each row contains BOTH
call and put data for a given strike. It converts this to a list of 
OptionQuote objects (one per option).

Key ORATS data structure insights:
- Wide format: Each row = one strike with both call and put data
- Adjusted prices: Use adj_* columns (split-adjusted)
- Greeks: Row greeks apply to calls, put delta = call delta - 1
- IV: smoothSmvVol is ORATS's proprietary smoothed IV surface
- Mid prices: Use (bid + ask) / 2 since cValue/pValue are not adjusted

File structure expected:
    data_root/YYYY/ORATS_SMV_Strikes_YYYYMMDD.parquet
    
Example:
    c:/ORATS/data/ORATS_Adjusted/2023/ORATS_SMV_Strikes_20230103.parquet
"""

from typing import Protocol, List, Optional, Tuple
from datetime import date
from decimal import Decimal
from pathlib import Path
from functools import lru_cache
import pandas as pd

from src.core.models import OptionQuote


class IDataProvider(Protocol):
    """
    Interface for option data providers.
    
    Defines minimal contract: load option chains and spot prices.
    Implementations can add helper methods but aren't required to.
    """
    
    def get_option_chain(
        self,
        ticker: str,
        trade_date: date,
        expiry_date: date
    ) -> List[OptionQuote]:
        """
        Load option chain for ticker/expiry on given date.
        
        Args:
            ticker: Underlying ticker symbol (e.g., 'AAPL')
            trade_date: Date to get option data for
            expiry_date: Option expiration date
            
        Returns:
            List of OptionQuote objects (both calls and puts)
        """
        ...
    
    def get_spot_price(self, ticker: str, trade_date: date) -> Optional[Decimal]:
        """
        Get underlying spot price for ticker on given date.
        
        Args:
            ticker: Underlying ticker symbol
            trade_date: Date to get spot price for
            
        Returns:
            Spot price as Decimal, or None if not found
        """
        ...
    
    def get_available_expiries(self, ticker: str, trade_date: date) -> List[date]:
        """
        Get list of available expiry dates for ticker on given trade date.
        
        Args:
            ticker: Underlying ticker symbol
            trade_date: Date to get available expiries for
            
        Returns:
            List of available expiry dates, sorted ascending
        """
        ...


class ORATSDataProvider:
    """
    Data provider for ORATS parquet files.
    
    Loads split-adjusted option data from ORATS_Adjusted folder.
    Handles wide-format data (each row contains both call and put).
    
    File structure expected:
        {data_root}/YYYY/ORATS_SMV_Strikes_YYYYMMDD.parquet
    
    ORATS Data Format:
        - Wide format: Each row = strike with BOTH call and put
        - Columns: ticker, adj_strike, adj_cBidPx, adj_cAskPx, adj_pBidPx, adj_pAskPx, ...
        - Greeks: delta, gamma, vega, theta (for calls)
        - IV: smoothSmvVol (ORATS smoothed surface)
        - Mid: (bid + ask) / 2 (cValue/pValue not adjusted for splits)
    """
    
    def __init__(
        self,
        data_root: str = "c:/ORATS/data/ORATS_Adjusted",
        min_volume: int = 10,
        min_open_interest: int = 100,
        min_bid: float = 0.05,
        max_spread_pct: float = 0.50,
        cache_size: int = 5
    ):
        """
        Initialize ORATS data provider with LRU caching.
        
        Args:
            data_root: Path to ORATS_Adjusted folder (contains split-adjusted data)
            min_volume: Minimum option volume filter
            min_open_interest: Minimum open interest filter
            min_bid: Minimum bid price filter (filters out illiquid far OTM)
            max_spread_pct: Maximum bid-ask spread as % of mid (e.g., 0.5 = 50%)
            cache_size: Number of daily files to cache per worker (default: 5)
                       Conservative setting for parallel execution:
                       - 5 files × 50MB = 250MB per worker
                       - For date batching: only 2 dates active (entry + expiry)
                       - 16 workers × 250MB = 4GB total (safe for 32GB RAM)
        """
        self.data_root = Path(data_root)
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest
        self.min_bid = min_bid
        self.max_spread_pct = max_spread_pct
        self._cache_size = cache_size
        
        # LRU cache: stores multiple dates (not shared across workers)
        # Each worker gets its own cache instance
        self._load_cached = lru_cache(maxsize=cache_size)(self._load_day_data_impl)
    
    def _load_day_data_impl(self, trade_date: date) -> pd.DataFrame:
        """
        Load ORATS data file for given date (uncached implementation).
        
        This is the actual loading logic. Wrapped by LRU cache.
        
        Args:
            trade_date: Date to load data for
            
        Returns:
            DataFrame with all tickers/strikes for that date
            
        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        year = trade_date.year
        date_str = trade_date.strftime('%Y%m%d')
        file_path = self.data_root / str(year) / f'ORATS_SMV_Strikes_{date_str}.parquet'
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"ORATS data file not found: {file_path}\n"
                f"Expected format: ORATS_Adjusted/YYYY/ORATS_SMV_Strikes_YYYYMMDD.parquet"
            )
        
        return pd.read_parquet(file_path)
    
    def _load_day_data(self, trade_date: date) -> pd.DataFrame:
        """
        Load data with LRU caching.
        
        Cache behavior (per-worker):
        - First call with date: Loads from disk (cache MISS)
        - Subsequent calls with same date: Returns cached (cache HIT)
        - Keeps last N dates in memory (N = cache_size)
        - Automatically evicts least recently used when full
        
        For date batching:
        - Worker processes 2024-01-05 (10 tickers)
        - Loads entry date (2024-01-05) → cached
        - Loads expiry date (2024-02-02) → cached
        - All 10 tickers reuse these 2 cached dates
        - Cache efficiency: 95%+ hit rate
        
        Args:
            trade_date: Date to load
            
        Returns:
            Cached or freshly loaded DataFrame
        """
        return self._load_cached(trade_date)
    
    def get_cache_info(self):
        """
        Get cache statistics for this worker.
        
        Returns:
            CacheInfo(hits=XXX, misses=YYY, maxsize=5, currsize=ZZZ)
            - hits: Number of cache hits (fast lookups)
            - misses: Number of cache misses (disk loads)
            - maxsize: Maximum cache size
            - currsize: Current number of cached dates
        """
        return self._load_cached.cache_info()
    
    def clear_cache(self):
        """Clear the LRU cache (free memory for this worker)."""
        self._load_cached.cache_clear()
    
    def get_spot_price(self, ticker: str, trade_date: date) -> Optional[Decimal]:
        """
        Get split-adjusted spot price for ticker on given date.
        
        Uses adj_stkPx column which is adjusted for splits/dividends.
        
        Args:
            ticker: Underlying ticker symbol
            trade_date: Date to get spot price for
            
        Returns:
            Adjusted spot price as Decimal, or None if not found
        """
        try:
            df = self._load_day_data(trade_date)
            
            ticker_data = df[df['ticker'] == ticker]
            if ticker_data.empty:
                return None
            
            # Use adjusted stock price (split-adjusted)
            spot = ticker_data['adj_stkPx'].iloc[0]
            return Decimal(str(spot))
            
        except FileNotFoundError:
            # Data file doesn't exist for this date
            return None
        except Exception:
            # Any other error - return None
            return None
    
    def get_available_expiries(self, ticker: str, trade_date: date) -> List[date]:
        """
        Get list of available expiry dates for ticker on given trade date.
        
        Args:
            ticker: Underlying ticker symbol
            trade_date: Date to get available expiries for
            
        Returns:
            List of available expiry dates, sorted ascending
        """
        df = self._load_day_data(trade_date)
        
        # Filter to ticker
        ticker_data = df[df['ticker'] == ticker].copy()
        
        if ticker_data.empty:
            return []
        
        # Get unique expiry dates
        ticker_data['expirDate'] = pd.to_datetime(ticker_data['expirDate']).dt.date
        expiries = sorted(ticker_data['expirDate'].unique())
        
        return expiries
    
    def get_option_chain(
        self,
        ticker: str,
        trade_date: date,
        expiry_date: date
    ) -> List[OptionQuote]:
        """
        Load option chain for ticker/expiry, converting wide format to OptionQuote list.
        
        ORATS Data Structure:
        - Each row contains BOTH call and put for same strike
        - Returns TWO OptionQuote objects per row (one call, one put)
        
        Columns used:
        - adj_strike: Split-adjusted strike price
        - adj_cBidPx, adj_cAskPx: Adjusted call bid/ask
        - adj_pBidPx, adj_pAskPx: Adjusted put bid/ask
        - mid = (bid + ask) / 2: Simple average (cValue/pValue not adjusted)
        - smoothSmvVol: ORATS smoothed IV surface
        - delta, gamma, vega, theta: Greeks (for calls; put delta = call delta - 1)
        - cVolu, cOi: Call volume and open interest
        - pVolu, pOi: Put volume and open interest
        
        Args:
            ticker: Underlying ticker symbol
            trade_date: Date to get option data for
            expiry_date: Option expiration date
            
        Returns:
            List of OptionQuote objects (both calls and puts)
            Empty list if no data or all filtered out
        """
        # Load data (uses cache if same date)
        df = self._load_day_data(trade_date)
        
        # Filter to ticker and expiry
        # Note: expirDate in ORATS data might be datetime, convert for comparison
        mask = (df['ticker'] == ticker)
        
        # Handle expiry date comparison (might be string or datetime in data)
        if 'expirDate' in df.columns:
            df_expiry = pd.to_datetime(df['expirDate']).dt.date
            mask = mask & (df_expiry == expiry_date)
        
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            return []
        
        # Apply liquidity filters (filters in wide format)
        df_filtered = self._apply_liquidity_filters(df_filtered)
        
        if df_filtered.empty:
            return []
        
        # Convert wide format to OptionQuote objects
        # Each row produces TWO quotes (call + put)
        quotes = []
        
        for row in df_filtered.itertuples(index=False):
            # Create CALL option quote
            call_quote = self._create_call_quote(row, trade_date, expiry_date)
            quotes.append(call_quote)
            
            # Create PUT option quote
            put_quote = self._create_put_quote(row, trade_date, expiry_date)
            quotes.append(put_quote)
        
        return quotes
    
    def _create_call_quote(
        self,
        row,  # namedtuple from itertuples()
        trade_date: date,
        expiry_date: date
    ) -> OptionQuote:
        """
        Create OptionQuote for CALL from ORATS wide-format row.
        
        Uses:
        - adj_cBidPx, adj_cAskPx: Adjusted call prices
        - mid = (bid + ask) / 2: Simple average (cValue not adjusted for splits)
        - smoothSmvVol: ORATS smoothed IV
        - delta, gamma, vega, theta: Call greeks from row
        - cVolu, cOi: Call volume and OI
        """
        bid = Decimal(str(getattr(row, 'adj_cBidPx', 0)))
        ask = Decimal(str(getattr(row, 'adj_cAskPx', 0)))
        mid = (bid + ask) / 2
        
        return OptionQuote(
            ticker=getattr(row, 'ticker'),
            trade_date=trade_date,
            expiry_date=expiry_date,
            strike=Decimal(str(getattr(row, 'adj_strike', 0))),
            option_type='call',
            bid=bid,
            ask=ask,
            mid=mid,  # Use (bid + ask) / 2
            iv=float(getattr(row, 'smoothSmvVol', 0)),  # ORATS smoothed IV
            delta=float(getattr(row, 'delta', 0)),  # Call delta from row
            gamma=float(getattr(row, 'gamma', 0)),
            vega=float(getattr(row, 'vega', 0)),
            theta=float(getattr(row, 'theta', 0)),
            volume=int(getattr(row, 'cVolu', 0)),
            open_interest=int(getattr(row, 'cOi', 0))
        )
    
    def _create_put_quote(
        self,
        row,  # namedtuple from itertuples()
        trade_date: date,
        expiry_date: date
    ) -> OptionQuote:
        """
        Create OptionQuote for PUT from ORATS wide-format row.
        
        Uses:
        - adj_pBidPx, adj_pAskPx: Adjusted put prices
        - mid = (bid + ask) / 2: Simple average (pValue not adjusted for splits)
        - smoothSmvVol: ORATS smoothed IV
        - delta - 1: Put delta approximation (call delta - 1)
        - gamma, vega, theta: Same magnitude as call for ATM options
        - pVolu, pOi: Put volume and OI
        """
        bid = Decimal(str(getattr(row, 'adj_pBidPx', 0)))
        ask = Decimal(str(getattr(row, 'adj_pAskPx', 0)))
        mid = (bid + ask) / 2
        
        return OptionQuote(
            ticker=getattr(row, 'ticker'),
            trade_date=trade_date,
            expiry_date=expiry_date,
            strike=Decimal(str(getattr(row, 'adj_strike', 0))),
            option_type='put',
            bid=bid,
            ask=ask,
            mid=mid,  # Use (bid + ask) / 2
            iv=float(getattr(row, 'smoothSmvVol', 0)),  # ORATS smoothed IV
            delta=float(getattr(row, 'delta', 0)) - 1.0,  # Put delta ≈ call delta - 1
            gamma=float(getattr(row, 'gamma', 0)),  # Same as call
            vega=float(getattr(row, 'vega', 0)),    # Same as call
            theta=float(getattr(row, 'theta', 0)),  # Same magnitude as call
            volume=int(getattr(row, 'pVolu', 0)),
            open_interest=int(getattr(row, 'pOi', 0))
        )
    
    def _apply_liquidity_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply liquidity filters to wide-format ORATS data.
        
        Filters applied:
        - Volume: Either call OR put must meet minimum
        - Open Interest: Either call OR put must meet minimum
        - Bid price: Both call AND put must meet minimum
        - Spread: Both call AND put must meet maximum spread %
        
        Args:
            df: DataFrame in ORATS wide format (one row per strike)
            
        Returns:
            Filtered DataFrame
        """
        filters = []
        
        # Volume filter: both call and put must have volume
        # Rationale: If both sides are liquid, the strike is tradeable
        if self.min_volume > 0:
            filters.append(
                (df['cVolu'] >= self.min_volume) & (df['pVolu'] >= self.min_volume)
            )
        
        # Open interest: both call AND put must have OI
        if self.min_open_interest > 0:
            filters.append(
                (df['cOi'] >= self.min_open_interest) & (df['pOi'] >= self.min_open_interest)
            )
        
        # Bid price: Both call AND put must have minimum bid
        # Rationale: Filters out far OTM strikes with near-zero bids
        if self.min_bid > 0:
            filters.append(
                (df['adj_cBidPx'] >= self.min_bid) & (df['adj_pBidPx'] >= self.min_bid)
            )
        
        # Spread filter: Both call AND put spreads must be reasonable
        # Prevents trading options with excessive bid-ask spreads
        if self.max_spread_pct < 1.0:
            # Calculate spread as % of mid price
            # Use (bid + ask) / 2 as denominator (avoid division by zero)
            call_mid = (df['adj_cBidPx'] + df['adj_cAskPx']) / 2
            put_mid = (df['adj_pBidPx'] + df['adj_pAskPx']) / 2
            
            call_spread = (df['adj_cAskPx'] - df['adj_cBidPx']) / call_mid.replace(0, 1)
            put_spread = (df['adj_pAskPx'] - df['adj_pBidPx']) / put_mid.replace(0, 1)
            
            filters.append(
                (call_spread <= self.max_spread_pct) & (put_spread <= self.max_spread_pct)
            )
        
        # Apply all filters
        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter &= f
            return df[combined_filter]
        
        return df
    
    # Helper methods (not in IDataProvider interface)
    # These are convenience methods for common operations
    
    def find_atm_strike(self, chain: List[OptionQuote], spot: Decimal) -> Decimal:
        """
        Find ATM (at-the-money) strike closest to spot price.
        
        Helper method for strategies that need ATM strikes.
        Not part of IDataProvider interface.
        
        Args:
            chain: List of OptionQuote objects
            spot: Current spot price
            
        Returns:
            Strike price closest to spot
            
        Raises:
            ValueError: If chain is empty
        """
        if not chain:
            raise ValueError("Cannot find ATM strike from empty chain")
        
        # Find strike with minimum distance to spot
        atm_strike = min(chain, key=lambda q: abs(q.strike - spot)).strike
        return atm_strike
    
    def get_atm_options(
        self,
        ticker: str,
        trade_date: date,
        expiry_date: date
    ) -> Tuple[Optional[OptionQuote], Optional[OptionQuote]]:
        """
        Get ATM call and put for ticker/expiry.
        
        Convenience method for strategies that trade ATM straddles.
        Not part of IDataProvider interface.
        
        Args:
            ticker: Underlying ticker
            trade_date: Trade date
            expiry_date: Option expiry
            
        Returns:
            Tuple of (atm_call, atm_put) or (None, None) if not found
        """
        # Get full chain
        chain = self.get_option_chain(ticker, trade_date, expiry_date)
        
        if not chain:
            return None, None
        
        # Get spot price - returns None if not found
        spot = self.get_spot_price(ticker, trade_date)
        
        if spot is None:
            return None, None
        
        # Find ATM strike
        atm_strike = self.find_atm_strike(chain, spot)
        
        # Find call and put at ATM strike
        atm_call = next((q for q in chain if q.strike == atm_strike and q.option_type == 'call'), None)
        atm_put = next((q for q in chain if q.strike == atm_strike and q.option_type == 'put'), None)
        
        return atm_call, atm_put