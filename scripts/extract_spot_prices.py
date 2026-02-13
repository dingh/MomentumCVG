"""
Extract historical spot prices for all tickers from ORATS data.

This script creates a pre-computed database of daily adjusted spot prices
for all tickers in the ORATS dataset. The output CSV enables:
1. Correct realized volatility calculation (using daily returns)
2. Fast spot price lookups during backtesting
3. Performance analytics and charting

Output Format:
    date,ticker,adj_spot_price
    2018-01-02,AAPL,42.54
    2018-01-02,MSFT,85.95
    ...

File Size Estimate:
    - 8 years × 500 tickers × 252 days = ~1M rows
    - ~50-100MB uncompressed CSV
    - Loads in ~2 seconds with pandas

Usage:
    python scripts/extract_spot_prices.py --start-year 2018 --end-year 2025
    python scripts/extract_spot_prices.py --year 2024  # Single year
"""

import sys
from pathlib import Path
from datetime import date, datetime
from typing import List, Dict
import logging
import argparse

import pandas as pd

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.orats_provider import ORATSDataProvider


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_trading_dates_from_files(data_root: Path, start_year: int, end_year: int) -> List[date]:
    """
    Get all available trading dates by scanning ORATS file names.
    
    Args:
        data_root: Path to ORATS_Adjusted folder
        start_year: First year to include
        end_year: Last year to include
        
    Returns:
        List of trading dates sorted ascending
    """
    dates = []
    
    for year in range(start_year, end_year + 1):
        year_dir = data_root / str(year)
        
        if not year_dir.exists():
            logger.warning(f"Year directory not found: {year_dir}")
            continue
        
        # Find all ORATS files in year directory
        for file_path in sorted(year_dir.glob('ORATS_SMV_Strikes_*.parquet')):
            # Extract date from filename: ORATS_SMV_Strikes_20240102.parquet
            date_str = file_path.stem.split('_')[-1]  # '20240102'
            
            try:
                trade_date = datetime.strptime(date_str, '%Y%m%d').date()
                dates.append(trade_date)
            except ValueError:
                logger.warning(f"Could not parse date from filename: {file_path.name}")
    
    return sorted(dates)


def extract_spot_prices_for_date(
    provider: ORATSDataProvider,
    trade_date: date
) -> List[Dict]:
    """
    Extract spot prices for all tickers on a given date.
    
    Args:
        provider: ORATS data provider instance
        trade_date: Date to extract spot prices for
        
    Returns:
        List of dicts with {date, ticker, adj_spot_price}
    """
    records = []
    
    try:
        # Load full day data (all tickers)
        df = provider._load_day_data(trade_date)
        
        # Get unique tickers and their spot prices
        # Each ticker appears multiple times (one row per strike), so take first
        ticker_spots = df.groupby('ticker')['adj_stkPx'].first()
        
        # Create records
        for ticker, spot_price in ticker_spots.items():
            records.append({
                'date': trade_date,
                'ticker': ticker,
                'adj_spot_price': float(spot_price)
            })
        
    except FileNotFoundError:
        logger.warning(f"No data file for {trade_date}")
    except Exception as e:
        logger.error(f"Error processing {trade_date}: {e}")
    
    return records


def main():
    """Main extraction logic."""
    parser = argparse.ArgumentParser(
        description='Extract historical spot prices from ORATS data'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='C:/ORATS/data/ORATS_Adjusted',
        help='Path to ORATS_Adjusted folder'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet',
        help='Output parquet file path'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        help='Start year (inclusive)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        help='End year (inclusive)'
    )
    parser.add_argument(
        '--year',
        type=int,
        help='Single year to process (shorthand for --start-year Y --end-year Y)'
    )
    
    args = parser.parse_args()
    
    # Determine year range
    if args.year:
        start_year = end_year = args.year
    else:
        start_year = args.start_year or 2018
        end_year = args.end_year or 2025
    
    logger.info(f"Extracting spot prices from {start_year} to {end_year}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output: {args.output}")
    
    # Initialize provider (no liquidity filters needed for spot prices)
    provider = ORATSDataProvider(
        data_root=args.data_root,
        min_volume=0,           # No filters - we want all tickers
        min_open_interest=0,
        min_bid=0.0,
        max_spread_pct=1.0,
        cache_size=1            # Only cache 1 file at a time
    )
    
    # Get all trading dates
    data_root = Path(args.data_root)
    trading_dates = get_trading_dates_from_files(data_root, start_year, end_year)
    
    if not trading_dates:
        logger.error("No trading dates found! Check data_root path.")
        return
    
    logger.info(f"Found {len(trading_dates)} trading dates")
    logger.info(f"Date range: {trading_dates[0]} to {trading_dates[-1]}")
    
    # Extract spot prices for all dates
    all_records = []
    
    for trade_date in tqdm(trading_dates, desc="Extracting spot prices"):
        records = extract_spot_prices_for_date(provider, trade_date)
        all_records.extend(records)
        
        # Clear cache after each date to free memory
        provider.clear_cache()
    
    # Convert to DataFrame
    df_spots = pd.DataFrame(all_records)
    
    if df_spots.empty:
        logger.error("No spot prices extracted! Check data files.")
        return
    
    logger.info(f"\nExtraction Summary:")
    logger.info(f"  Total records: {len(df_spots):,}")
    logger.info(f"  Unique tickers: {df_spots['ticker'].nunique()}")
    logger.info(f"  Date range: {df_spots['date'].min()} to {df_spots['date'].max()}")
    logger.info(f"  Memory size: {df_spots.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Sort by date, then ticker
    df_spots = df_spots.sort_values(['date', 'ticker'])
    
    # Save to Parquet with compression
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_spots.to_parquet(output_path, index=False, compression='snappy')
    logger.info(f"\nSaved to: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024**2:.1f} MB (compressed)")
    
    # Show sample
    logger.info("\nSample data (first 5 rows):")
    print(df_spots.head().to_string(index=False))
    
    # Show ticker distribution
    ticker_counts = df_spots['ticker'].value_counts()
    logger.info(f"\nTop 10 tickers by observations:")
    print(ticker_counts.head(10))


if __name__ == '__main__':
    main()
