"""
Precompute historical weekly straddle P&L for momentum strategy.

This script:
1. Builds ATM straddles weekly (every Friday, 7 DTE target)
2. Holds until expiry
3. Calculates realized P&L, returns, volatility
4. Tracks tradeability metrics (spreads, volume, OI)
5. Saves to cache/straddle_history_weekly_YYYY_YYYY.parquet

Configuration:
- Frequency: Weekly (hardcoded)
- Target DTE: 7 days (hardcoded)
- Workers: 24 parallel jobs (hardcoded)
- Tickers: S&P 500 from C:\ORATS\data\meta_data\SP500.csv

Usage:
    python scripts/precompute_straddle_history.py
    python scripts/precompute_straddle_history.py --start-year 2020
    python scripts/precompute_straddle_history.py --start-year 2020 --end-year 2024
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import argparse

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.straddle_analyzer import StraddleHistoryBuilder


# HARDCODED CONFIGURATION
# These are fixed for the weekly momentum strategy
FREQUENCY = 'weekly'
DTE_TARGET = 7
N_WORKERS = 24
MAX_SPREAD_PCT = 0.50
MIN_VOLUME = 10
MIN_OI = 0
SP500_FILE = Path('C:/ORATS/data/meta_data/SP500.csv')


def process_date_batch(
    data_root: str,
    trade_date: datetime,
    tickers: List[str]
) -> List[Dict]:
    """
    Process all tickers for ONE date in ONE worker.
    
    This batching strategy ensures that:
    - All tickers on same date share the same cached entry/expiry data
    - LRU cache is effective (only 2 dates per batch: entry + expiry)
    - Memory usage is predictable and conservative
    
    Args:
        data_root: Path to ORATS data
        dte_target: Target DTE
        trade_date: Trade date to process
        tickers: List of tickers to process
        max_spread_pct: Max spread filter
        min_volume: Min volume filter
        min_oi: Min OI filter
    
    Returns:
        List of result dictionaries (one per ticker)
    """
    # Create builder for this worker (gets its own cache)
    builder = StraddleHistoryBuilder(
        data_root=data_root,
        dte_target=DTE_TARGET,
        max_spread_pct=MAX_SPREAD_PCT,
        min_volume=MIN_VOLUME,
        min_oi=MIN_OI
    )
    
    # Initialize once per date (not per ticker)
    builder._init_worker_components()
    
    # Process all tickers - benefits from LRU cache
    results = []
    for ticker in tickers:
        result = builder.process_single_straddle(ticker, trade_date)
        results.append(result)
    
    return results


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/MomentumCVG_env/log/precompute_straddle_history.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_trading_fridays(start_date: datetime, end_date: datetime, data_root: str) -> List[datetime]:
    """
    Get all Fridays that are actual trading days (have data in ORATS).
    If a Friday is a holiday, use the last trading day of that week instead.
    
    Args:
        start_date: Start date
        end_date: End date
        data_root: Path to ORATS data to verify trading days
    
    Returns:
        List of all trading Fridays (or Thursday/Wednesday if Friday is holiday)
    """
    data_path = Path(data_root)
    
    # Get all Fridays in date range
    all_fridays = []
    current = start_date
    while current <= end_date:
        if current.weekday() == 4:  # Friday
            all_fridays.append(current)
        current += timedelta(days=1)
    
    logger.info(f"Found {len(all_fridays)} Fridays in date range")
    
    # For each Friday, find the actual trading day (or fallback to Thu/Wed/Tue/Mon)
    trading_fridays = []
    
    for friday in all_fridays:
        # Try Friday first, then work backwards through the week
        for days_back in range(5):  # Try Fri, Thu, Wed, Tue, Mon
            candidate_date = friday - timedelta(days=days_back)
            
            # Check if data file exists for this date
            # Format: YYYY/ORATS_SMV_Strikes_YYYYMMDD.parquet (e.g., 2023/ORATS_SMV_Strikes_20230106.parquet)
            year_str = candidate_date.strftime('%Y')
            date_str = candidate_date.strftime('%Y%m%d')
            data_file = data_path / year_str / f"ORATS_SMV_Strikes_{date_str}.parquet"
            
            if data_file.exists():
                trading_fridays.append(candidate_date)
                if days_back > 0:
                    logger.info(f"  {friday.date()} (Fri) is holiday -> using {candidate_date.date()} ({candidate_date.strftime('%a')})")
                break
        else:
            # No trading day found in entire week (rare, like Thanksgiving week)
            logger.warning(f"  {friday.date()} - No trading day found in week, skipping")
    
    logger.info(f"Result: {len(trading_fridays)} trading days (from {len(all_fridays)} Fridays)")
    
    return trading_fridays


def sample_fridays_by_frequency(
    all_fridays: List[datetime],
    frequency: str
) -> List[datetime]:
    """
    Sample trading Fridays based on rebalance frequency.
    
    Args:
        all_fridays: List of all trading Fridays (or last trading day of week)
        frequency: 'weekly' or 'monthly'
    
    Returns:
        Sampled list of trade dates
    """
    if frequency == 'weekly':
        # Use all trading Fridays
        return all_fridays
    
    elif frequency == 'monthly':
        # Use first Friday of each month
        selected = []
        fridays_by_month = {}
        
        # Group Fridays by year-month
        for friday in all_fridays:
            key = (friday.year, friday.month)
            if key not in fridays_by_month:
                fridays_by_month[key] = []
            fridays_by_month[key].append(friday)
        
        # Select first Friday of each month
        for (year, month), fridays in fridays_by_month.items():
            if len(fridays) >= 1:
                selected.append(fridays[0])
        
        return sorted(selected)
    
    else:
        raise ValueError(f"Unknown frequency: {frequency}. Must be 'weekly' or 'monthly'")


def generate_trade_dates(
    start_date: datetime,
    end_date: datetime,
    frequency: str,
    data_root: str
) -> List[datetime]:
    """
    Generate trade dates based on frequency, excluding market holidays.
    If a Friday is a holiday, uses the last trading day of that week.
    
    Args:
        start_date: Start date
        end_date: End date
        frequency: 'weekly' or 'monthly'
        data_root: Path to ORATS data to verify trading days
    
    Returns:
        List of trade dates (trading Fridays or fallback days based on frequency)
    """
    # Step 1: Get all trading Fridays (with fallback to Thu/Wed/Tue/Mon if Friday is holiday)
    all_trading_fridays = get_trading_fridays(start_date, end_date, data_root)
    
    # Step 2: Sample based on frequency
    trade_dates = sample_fridays_by_frequency(all_trading_fridays, frequency)
    
    logger.info(f"Generated {len(trade_dates)} {frequency} trade dates from {len(all_trading_fridays)} trading Fridays")
    
    return trade_dates


def save_checkpoint(results: List[Dict], checkpoint_path: Path):
    """Save intermediate results to checkpoint file."""
    if not results:
        logger.warning("No results to checkpoint")
        return
    
    df = pd.DataFrame(results)
    df.to_parquet(checkpoint_path, compression='gzip', index=False)
    logger.info(f"Checkpoint saved: {len(results)} straddles -> {checkpoint_path}")


def main():
    """Main execution function.""" 
    # python ..\MomentumCVG\scripts\precompute_straddle_history.py --start-year 2018 --end-year 2025 
    parser = argparse.ArgumentParser(
        description='Precompute historical weekly straddle P&L (7 DTE target)'
    )
    parser.add_argument('--data-root', type=str, default='C:/ORATS/data/ORATS_Adjusted',
                        help='Path to ORATS data directory (default: C:/ORATS/data/ORATS_Adjusted)')
    parser.add_argument('--start-year', type=int, default=2018,
                        help='Start year (default: 2018)')
    parser.add_argument('--end-year', type=int, default=2024,
                        help='End year (default: 2024)')
    
    args = parser.parse_args()
    
    # Create output directories
    Path('cache').mkdir(exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    
    logger.info("="*80)
    logger.info("Starting weekly straddle history precomputation")
    logger.info("="*80)
    
    # Load tickers from SP500.csv
    if not SP500_FILE.exists():
        logger.error(f"SP500.csv not found: {SP500_FILE}")
        return
    
    df_sp500 = pd.read_csv(SP500_FILE)
    tickers = df_sp500['Ticker'].tolist()
    logger.info(f"Loaded {len(tickers)} tickers from {SP500_FILE}")
    
    logger.info(f"Tickers: {len(tickers)} ({', '.join(tickers[:5])}...)")
    logger.info(f"Frequency: {FREQUENCY}")
    logger.info(f"Target DTE: {DTE_TARGET}")
    
    # Generate trade dates using improved logic (handles holidays)
    start_date = datetime(args.start_year, 1, 1)
    end_date = datetime(args.end_year, 4, 30)
    trade_dates = generate_trade_dates(
        start_date,
        end_date,
        FREQUENCY,
        args.data_root
    )
    
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Trade dates: {len(trade_dates)} (first: {trade_dates[0].date()}, last: {trade_dates[-1].date()})")
    logger.info(f"Expected straddles: {len(tickers)} tickers × {len(trade_dates)} dates = {len(tickers) * len(trade_dates):,}")
    
    logger.info("\n" + "="*80)
    logger.info("PARALLEL PROCESSING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Workers: {N_WORKERS} parallel jobs")
    logger.info(f"Cache per worker: 5 dates × 50MB = 250MB")
    logger.info(f"Total memory budget: {N_WORKERS} × 250MB = {N_WORKERS * 0.25:.1f}GB")
    logger.info(f"Strategy: Date batching (process all tickers per date in one worker)")
    logger.info(f"Expected cache hit rate: 95%+ (2 dates active per batch: entry + expiry)")
    logger.info("="*80)
    
    # Process in parallel BY DATE (not by ticker-date pairs)
    logger.info(f"\nProcessing {len(trade_dates)} dates with {len(tickers)} tickers each...")
    start_time = datetime.now()
    
    all_results = Parallel(n_jobs=N_WORKERS, backend='loky', verbose=0)(
        delayed(process_date_batch)(
            args.data_root,
            trade_date.date(),
            tickers
        )
        for trade_date in tqdm(trade_dates, desc="Processing dates")
    )
    
    # Flatten results: list of lists -> single list
    results = [result for date_results in all_results for result in date_results]
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nProcessed {len(results):,} straddles in {elapsed:.1f}s ({len(results)/elapsed:.1f} straddles/sec)")
    
    logger.info("="*80)
    logger.info("Processing complete!")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Data quality checks
    logger.info("\nData quality checks:")
    logger.info(f"  Rows with null entry_date: {df['entry_date'].isna().sum()}")
    logger.info(f"  Rows with null ticker: {df['ticker'].isna().sum()}")
    logger.info(f"  Unique tickers: {df['ticker'].nunique()} (expected: {len(tickers)})")
    logger.info(f"  Unique dates: {df['entry_date'].nunique()} (expected: {len(trade_dates)})")
    
    # Summary statistics
    total = len(df)
    tradeable = df['is_tradeable'].sum()
    success = df['pnl'].notna().sum()
    
    logger.info(f"\nTotal straddles processed: {total:,}")
    logger.info(f"Successfully calculated P&L: {success:,} ({success/total*100:.1f}%)")
    logger.info(f"Tradeable (passed filters): {tradeable:,} ({tradeable/total*100:.1f}%)")
    
    # Show failure reasons
    if (~df['is_tradeable']).any():
        logger.info("\nFailure reason breakdown:")
        failure_counts = df[~df['is_tradeable']]['failure_reason'].value_counts()
        for reason, count in failure_counts.head(10).items():
            logger.info(f"  {reason}: {count:,} ({count/total*100:.1f}%)")
    
    if success > 0:
        tradeable_df = df[df['is_tradeable'] & df['pnl'].notna()]
        if len(tradeable_df) > 0:
            logger.info(f"Tradeable straddles stats:")
            logger.info(f"  Mean return: {tradeable_df['return_pct'].mean():.2f}%")
            logger.info(f"  Median return: {tradeable_df['return_pct'].median():.2f}%")
            logger.info(f"  Win rate: {(tradeable_df['pnl'] > 0).mean()*100:.1f}%")
            logger.info(f"  Avg spread: {tradeable_df['avg_spread_pct'].mean()*100:.1f}%")
    
    # Save results (include frequency, start year, and end year in filename)
    output_path = Path(f'C:/MomentumCVG_env/cache/straddle_history_{FREQUENCY}_{args.start_year}_{args.end_year}.parquet')
    df.to_parquet(output_path, compression='gzip', index=False)
    
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info("="*80)
    logger.info("Done!")
    

if __name__ == '__main__':
    main()
