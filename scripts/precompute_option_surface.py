"""Precompute a reusable option quote surface for iron fly / condor research.

This script writes two parquet files:

1. **metadata table** — one row per (ticker, entry_date)
   Columns: expiry_date, dte_actual, entry_spot, exit_spot, body_strike,
   spot_move_pct, realized_volatility, has_body_call, has_body_put,
   n_surface_quotes, surface_valid, failure_reason, processing_time.

2. **quote surface table** — many rows per (ticker, entry_date)
   One row per body or OTM option quote that survived the delta-range and
   liquidity filters.  The backtest can assemble straddles, iron flies, and
   iron condors from this surface under arbitrary delta / fill / friction
   assumptions without re-reading the raw ORATS parquet files.

Design intent
-------------
This script is intentionally more primitive than ``precompute_ironfly_history.py``:
it stores raw quotes rather than pre-assembled strategy candidates.  The trade-off
is larger output files but maximum flexibility for downstream strategy research.

Usage
-----
    # Monthly 30-DTE surface for 2018-2024 (default)
    python scripts/precompute_option_surface.py

    # Weekly 7-DTE surface for a single year
    python scripts/precompute_option_surface.py --frequency weekly --start-year 2024 --end-year 2024

    # Narrow the delta window and keep zero-bid quotes for coverage audit
    python scripts/precompute_option_surface.py --min-abs-delta 0.05 --max-abs-delta 0.35 --keep-zero-bid-quotes

    # Custom delta bucket grid
    python scripts/precompute_option_surface.py --delta-buckets "0.10,0.15,0.20,0.25,0.30"
"""
from __future__ import annotations

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.option_surface_analyzer import OptionSurfaceBuilder
from src.data.spot_price_db import SpotPriceDB


# =============================================================================
# Constants (overridable via argparse)
# =============================================================================

N_WORKERS    = 26
SP500_FILE   = Path('C:/MomentumCVG_env/cache/liquid_tickers.csv')
SPOT_DB_PATH = 'C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet'


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/MomentumCVG_env/log/precompute_option_surface.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Date helpers
# =============================================================================

def get_trading_fridays(
    start_date: datetime,
    end_date: datetime,
    data_root: str,
) -> List[datetime]:
    """Return all Fridays in [start_date, end_date] that are actual trading days.

    Confirms each Friday by checking whether an ORATS parquet file exists for
    that date.  If a Friday is a market holiday, falls back to Thursday →
    Wednesday → Tuesday → Monday of the same week.  Weeks with no data file
    on any of the five days are silently dropped.
    """
    data_path = Path(data_root)

    # Collect every calendar Friday in the date range
    all_fridays: List[datetime] = []
    current = start_date
    while current <= end_date:
        if current.weekday() == 4:      # Friday = 4
            all_fridays.append(current)
        current += timedelta(days=1)

    logger.info(f"Found {len(all_fridays)} Fridays in date range")

    # Resolve each Friday to the latest trading day in that week with data
    trading_fridays: List[datetime] = []
    for friday in all_fridays:
        for days_back in range(5):      # Fri → Thu → Wed → Tue → Mon
            candidate = friday - timedelta(days=days_back)
            year_str  = candidate.strftime('%Y')
            date_str  = candidate.strftime('%Y%m%d')
            data_file = data_path / year_str / f"ORATS_SMV_Strikes_{date_str}.parquet"
            if data_file.exists():
                trading_fridays.append(candidate)
                if days_back > 0:
                    logger.info(
                        f"  {friday.date()} (Fri) is holiday "
                        f"-> using {candidate.date()} ({candidate.strftime('%a')})"
                    )
                break
        else:
            logger.warning(f"  {friday.date()} -- no trading day found in week, skipping")

    logger.info(f"Result: {len(trading_fridays)} trading days from {len(all_fridays)} Fridays")
    return trading_fridays


def sample_fridays_by_frequency(
    all_fridays: Sequence[datetime],
    frequency: str,
) -> List[datetime]:
    """Filter the resolved Friday list to the desired sampling frequency.

    - ``'weekly'``  : all Fridays returned as-is
    - ``'monthly'`` : first Friday of each calendar month only
    """
    if frequency == 'weekly':
        return list(all_fridays)
    if frequency == 'monthly':
        # Group by (year, month) and take the earliest Friday in each group
        grouped: Dict[Tuple[int, int], List[datetime]] = {}
        for friday in all_fridays:
            grouped.setdefault((friday.year, friday.month), []).append(friday)
        return sorted(fridays[0] for fridays in grouped.values())
    raise ValueError(f"Unknown frequency: {frequency!r}. Must be 'weekly' or 'monthly'.")


def generate_trade_dates(
    start_date: datetime,
    end_date: datetime,
    frequency: str,
    data_root: str,
) -> List[datetime]:
    """Generate trade entry dates for the given range, frequency, and data root."""
    all_trading_fridays = get_trading_fridays(start_date, end_date, data_root)
    trade_dates = sample_fridays_by_frequency(all_trading_fridays, frequency)
    logger.info(
        f"Generated {len(trade_dates)} {frequency} trade dates "
        f"from {len(all_trading_fridays)} trading Fridays"
    )
    return trade_dates


# =============================================================================
# Worker function (module-level → pickleable by joblib)
# =============================================================================

def process_date_batch(
    data_root: str,
    spot_db_path: str,
    trade_date,             # date object
    tickers: List[str],
    frequency: str,
    dte_target: int,
    min_abs_delta: float,
    max_abs_delta: float,
    delta_buckets: Sequence[float],
    keep_zero_bid_quotes: bool,
) -> Tuple[List[Dict], List[Dict]]:
    """Process all tickers for ONE date inside one worker process.

    Batching by date (rather than by ticker) maximises ORATSDataProvider
    LRU cache hits: the entry-date parquet and expiry-date parquet are
    shared across every ticker in the batch.

    Returns
    -------
    (meta_rows, quote_rows)
        *meta_rows*  — one metadata dict per ticker (success or failure)
        *quote_rows* — many quote dicts per ticker that produced a valid surface
    """
    spot_db = SpotPriceDB.load(spot_db_path)
    builder = OptionSurfaceBuilder(
        data_root=data_root,
        spot_db=spot_db,
        dte_target=dte_target,
        frequency=frequency,
        min_abs_delta=min_abs_delta,
        max_abs_delta=max_abs_delta,
        delta_buckets=delta_buckets,
        keep_zero_bid_quotes=keep_zero_bid_quotes,
    )
    # Initialise the provider once — all tickers in this batch share the same
    # cached parquet reads, which is the primary performance benefit of
    # date-batched parallelism.
    builder._init_worker_components()

    meta_rows: List[Dict] = []
    quote_rows: List[Dict] = []
    for ticker in tickers:
        meta, quotes = builder.process_single_entry(ticker, trade_date)
        meta_rows.append(meta)
        quote_rows.extend(quotes)
    return meta_rows, quote_rows


# =============================================================================
# Argument helpers
# =============================================================================

def _parse_delta_buckets(raw: str) -> List[float]:
    """Parse a comma-separated delta-bucket string into a sorted float list.

    Returns the default bucket grid when *raw* is empty or whitespace-only.
    """
    if not raw or not raw.strip():
        return [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40]
    return sorted(float(x.strip()) for x in raw.split(',') if x.strip())


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute an expiry-level option quote surface for fly / condor backtests."
    )
    parser.add_argument(
        '--data-root', type=str, default='C:/ORATS/data/ORATS_Adjusted',
        help='Path to ORATS adjusted parquet data directory (default: C:/ORATS/data/ORATS_Adjusted)',
    )
    parser.add_argument(
        '--start-year', type=int, default=2018,
        help='Start year inclusive (default: 2018)',
    )
    parser.add_argument(
        '--end-year', type=int, default=2026,
        help='End year inclusive (default: 2026)',
    )
    parser.add_argument(
        '--frequency', choices=['monthly', 'weekly'], default='monthly',
        help="Sampling frequency: 'monthly' (first Friday, ~30 DTE) or 'weekly' (every Friday, ~7 DTE). Default: monthly",
    )
    parser.add_argument(
        '--workers', type=int, default=N_WORKERS,
        help=f'Number of parallel workers (default: {N_WORKERS})',
    )
    parser.add_argument(
        '--min-abs-delta', type=float, default=0.03,
        help='Min abs delta for OTM wing quotes (default: 0.03)',
    )
    parser.add_argument(
        '--max-abs-delta', type=float, default=0.45,
        help='Max abs delta for OTM wing quotes (default: 0.45)',
    )
    parser.add_argument(
        '--delta-buckets',
        type=str,
        default='0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25,0.30,0.35,0.40',
        help='Comma-separated reference delta levels stored on each quote row for bucketing.',
    )
    parser.add_argument(
        '--keep-zero-bid-quotes',
        action='store_true',
        help='Retain quotes with non-positive bid/ask/mid (useful for coverage audits).',
    )
    args = parser.parse_args()

    # DTE target follows frequency — keep these two in sync
    dte_target    = 30 if args.frequency == 'monthly' else 7
    delta_buckets = _parse_delta_buckets(args.delta_buckets)

    if not Path(SPOT_DB_PATH).exists():
        logger.error(f"SpotPriceDB not found: {SPOT_DB_PATH}")
        logger.error("Run scripts/extract_spot_prices.py first.")
        return

    if not SP500_FILE.exists():
        logger.error(f"Ticker universe file not found: {SP500_FILE}")
        return

    Path('C:/MomentumCVG_env/cache').mkdir(parents=True, exist_ok=True)
    Path('C:/MomentumCVG_env/log').mkdir(parents=True, exist_ok=True)

    df_tickers = pd.read_csv(SP500_FILE)
    tickers = df_tickers['Ticker'].tolist()

    start_dt = datetime(args.start_year, 1, 1)
    end_dt = datetime(args.end_year, 2, 20)
    trade_dates = generate_trade_dates(start_dt, end_dt, args.frequency, args.data_root)
    if not trade_dates:
        logger.error("No trade dates generated; aborting.")
        return

    logger.info("=" * 80)
    logger.info("Option Surface Precomputation")
    logger.info("=" * 80)
    logger.info(f"Frequency        : {args.frequency}")
    logger.info(f"DTE target       : {dte_target}")
    logger.info(f"Date range       : {trade_dates[0].date()} -> {trade_dates[-1].date()}")
    logger.info(f"Trade dates      : {len(trade_dates)}")
    logger.info(f"Tickers          : {len(tickers)}")
    logger.info(f"min_abs_delta    : {args.min_abs_delta}")
    logger.info(f"max_abs_delta    : {args.max_abs_delta}")
    logger.info(f"delta_buckets    : {delta_buckets}")
    logger.info(f"keep_zero_bid    : {args.keep_zero_bid_quotes}")
    logger.info(f"Workers          : {args.workers}")
    logger.info(f"SpotPriceDB      : {SPOT_DB_PATH}")
    logger.info(
        f"Expected input   : {len(tickers)} tickers x {len(trade_dates)} dates "
        f"= {len(tickers) * len(trade_dates):,} (ticker, date) pairs"
    )
    logger.info("=" * 80)

    # -- Parallel processing (one job per date, all tickers per job) ----------
    started = datetime.now()
    results = Parallel(n_jobs=args.workers, backend='loky', verbose=0)(
        delayed(process_date_batch)(
            args.data_root,
            SPOT_DB_PATH,
            td.date(),
            tickers,
            args.frequency,
            dte_target,
            args.min_abs_delta,
            args.max_abs_delta,
            delta_buckets,
            args.keep_zero_bid_quotes,
        )
        for td in tqdm(trade_dates, desc='Processing dates')
    )

    # -- Flatten list-of-(meta_list, quote_list) tuples -----------------------
    meta_rows: List[Dict] = []
    quote_rows: List[Dict] = []
    for batch_meta, batch_quotes in results:
        meta_rows.extend(batch_meta)
        quote_rows.extend(batch_quotes)

    elapsed = (datetime.now() - started).total_seconds()

    # -- Build final DataFrames -----------------------------------------------
    meta_df   = pd.DataFrame(meta_rows)
    quotes_df = pd.DataFrame(quote_rows)

    logger.info(
        f"\nProcessed {len(meta_df):,} metadata rows and {len(quotes_df):,} quote rows "
        f"in {elapsed:.1f}s ({(len(meta_df) + len(quotes_df)) / elapsed:.0f} rows/sec)"
    )

    # -- Data quality summary -------------------------------------------------
    if not meta_df.empty:
        n_valid = int(meta_df['surface_valid'].sum())
        n_fail  = int((~meta_df['surface_valid']).sum())
        logger.info(f"Valid surfaces    : {n_valid:,}  ({n_valid/len(meta_df):.1%})")
        logger.info(f"Failure rows      : {n_fail:,}  ({n_fail/len(meta_df):.1%})")
        if (meta_df['failure_reason'].notna()).any():
            logger.info("Failure breakdown:")
            for reason, count in meta_df['failure_reason'].value_counts(dropna=True).items():
                logger.info(f"  {reason}: {count:,}")

    if not quotes_df.empty:
        logger.info(
            f"Quote rows: {len(quotes_df['ticker'].unique())} unique tickers, "
            f"{len(quotes_df[quotes_df['is_body']])} body rows, "
            f"{len(quotes_df[quotes_df['is_otm']])} OTM wing rows"
        )

    # -- Save outputs ---------------------------------------------------------
    output_root = Path('C:/MomentumCVG_env/cache')
    meta_path   = output_root / f"option_surface_meta_{args.frequency}_{args.start_year}_{args.end_year}.parquet"
    quotes_path = output_root / f"option_surface_quotes_{args.frequency}_{args.start_year}_{args.end_year}.parquet"

    meta_df.to_parquet(meta_path,   compression='gzip', index=False)
    quotes_df.to_parquet(quotes_path, compression='gzip', index=False)

    logger.info(f"Saved metadata parquet : {meta_path}  ({meta_path.stat().st_size / 1024 / 1024:.2f} MB)")
    logger.info(f"Saved quote parquet    : {quotes_path}  ({quotes_path.stat().st_size / 1024 / 1024:.2f} MB)")
    logger.info("=" * 80)
    logger.info("Done!")


if __name__ == '__main__':
    main()
