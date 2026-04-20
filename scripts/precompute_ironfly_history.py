"""
Precompute historical iron butterfly wing-candidate P&L.

Key difference from the straddle script: each (ticker, date) produces
MULTIPLE rows — one per symmetric wing pair that passes the quality filters.
This lets analysis scripts sweep over wing widths / delta levels in hindsight
without re-processing raw ORATS data.

Output schema per row:
  Identity    : ticker, entry_date, expiry_date, dte_target, dte_actual,
                entry_spot, body_strike
  Geometry    : wing_width, call_wing_strike, put_wing_strike, avg_wing_delta
  Economics   : net_credit, credit_to_width, total_spread, spread_cost_ratio
  Greeks      : net_delta, net_gamma, net_vega, net_theta
  Leg quotes  : sc/sp/lc/lp _bid/_ask/_iv/_delta
  Exit / P&L  : exit_spot, exit_value, pnl, return_pct_on_width,
                return_pct_on_credit, annualized_return_on_width,
                spot_move_pct, days_held
  Status      : is_tradeable, failure_reason

Configuration:
  Defaults: monthly 30-DTE, permissive filters (max_spread=0.99, min_yield=0.0)
  Tickers:  S&P 500 from C:/ORATS/data/meta_data/SP500.csv

Usage:
    python scripts/precompute_ironfly_history.py
    python scripts/precompute_ironfly_history.py --frequency monthly --start-year 2023 --end-year 2023
    python scripts/precompute_ironfly_history.py --frequency weekly --start-year 2020 --end-year 2024
    python scripts/precompute_ironfly_history.py --max-spread-pct 0.50 --min-yield 0.05
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.ironfly_analyzer import IronFlyHistoryBuilder
from src.data.spot_price_db import SpotPriceDB


# =============================================================================
# Constants (overridable via argparse)
# =============================================================================

DTE_TARGET     = 30
N_WORKERS      = 24
MAX_SPREAD_PCT = 0.99
MIN_YIELD      = 0.00  # net_credit / wing_width floor -- ensures minimum return on capital deployed
MIN_VOLUME     = 0
MIN_OI         = 0
SP500_FILE     = Path('C:/MomentumCVG_env/cache/liquid_tickers.csv')
SPOT_DB_PATH   = 'C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet'


# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('C:/MomentumCVG_env/log/precompute_ironfly_history.log'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Date helpers (same logic as precompute_straddle_history.py)
# =============================================================================

def get_trading_fridays(
    start_date: datetime,
    end_date: datetime,
    data_root: str,
) -> List[datetime]:
    """
    Return all Fridays in [start_date, end_date] that are actual trading days
    (an ORATS parquet file exists for that date).  If a Friday is a holiday,
    falls back to Thursday -> Wednesday -> Tuesday -> Monday of the same week.
    """
    data_path = Path(data_root)

    all_fridays: List[datetime] = []
    current = start_date
    while current <= end_date:
        if current.weekday() == 4:      # Friday = 4
            all_fridays.append(current)
        current += timedelta(days=1)

    logger.info(f"Found {len(all_fridays)} Fridays in date range")

    trading_fridays: List[datetime] = []
    for friday in all_fridays:
        for days_back in range(5):      # Fri -> Thu -> Wed -> Tue -> Mon
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
    all_fridays: List[datetime],
    frequency: str,
) -> List[datetime]:
    """
    Filter the full Friday list to the desired sampling frequency.

    - 'weekly'  : all Fridays
    - 'monthly' : first Friday of each calendar month
    """
    if frequency == 'weekly':
        return list(all_fridays)

    if frequency == 'monthly':
        selected: List[datetime] = []
        by_month: dict = {}
        for friday in all_fridays:
            key = (friday.year, friday.month)
            by_month.setdefault(key, []).append(friday)
        for fridays in by_month.values():
            selected.append(fridays[0])
        return sorted(selected)

    raise ValueError(f"Unknown frequency: {frequency!r}. Must be 'weekly' or 'monthly'.")


def generate_trade_dates(
    start_date: datetime,
    end_date: datetime,
    frequency: str,
    data_root: str,
) -> List[datetime]:
    """Generate trade dates for the given range, frequency, and data root."""
    all_trading_fridays = get_trading_fridays(start_date, end_date, data_root)
    trade_dates = sample_fridays_by_frequency(all_trading_fridays, frequency)
    logger.info(
        f"Generated {len(trade_dates)} {frequency} trade dates "
        f"from {len(all_trading_fridays)} trading Fridays"
    )
    return trade_dates


# =============================================================================
# Worker function (module-level -> pickleable by joblib)
# =============================================================================

def process_date_batch(
    data_root: str,
    spot_db_path: str,
    trade_date,             # date object
    tickers: List[str],
    frequency: str,
    dte_target: int,
    max_spread_pct: float,
    min_yield: float,
    min_volume: int,
    min_oi: int,
) -> List[Dict]:
    """
    Process all tickers for ONE date inside one worker process.

    Batching by date (rather than by ticker) maximises ORATSDataProvider
    LRU cache hits: the entry-date parquet and expiry-date parquet are
    shared across every ticker in the batch.

    Returns:
        Flat list of row dicts -- multiple rows per ticker (one per wing
        candidate) plus one failure row per ticker that failed.
    """
    spot_db = SpotPriceDB.load(spot_db_path)

    builder = IronFlyHistoryBuilder(
        data_root=data_root,
        spot_db=spot_db,
        dte_target=dte_target,
        max_spread_pct=max_spread_pct,
        min_yield_on_capital=min_yield,
        min_volume=min_volume,
        min_oi=min_oi,
        frequency=frequency,
    )

    # Initialise provider once -- all tickers share the same cached parquet
    builder._init_worker_components()

    results: List[Dict] = []
    for ticker in tickers:
        rows = builder.process_single_entry(ticker, trade_date)
        results.extend(rows)

    return results


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute iron butterfly wing-candidate P&L history. "
            "Produces one row per (ticker, date, wing_width) candidate."
        )
    )
    parser.add_argument(
        '--data-root', type=str, default='C:/ORATS/data/ORATS_Adjusted',
        help='Path to ORATS data directory (default: C:/ORATS/data/ORATS_Adjusted)',
    )
    parser.add_argument(
        '--start-year', type=int, default=2018,
        help='Start year inclusive (default: 2018)',
    )
    parser.add_argument(
        '--end-year', type=int, default=2024,
        help='End year inclusive (default: 2024)',
    )
    parser.add_argument(
        '--frequency', choices=['monthly', 'weekly'], default='monthly',
        help=(
            "Sampling frequency: 'monthly' (first Friday, ~30 DTE) or "
            "'weekly' (every Friday, ~7 DTE). Default: monthly"
        ),
    )
    parser.add_argument(
        '--max-spread-pct', type=float, default=MAX_SPREAD_PCT,
        help=f'Max bid-ask spread / mid per leg (default: {MAX_SPREAD_PCT})',
    )
    parser.add_argument(
        '--min-yield', type=float, default=MIN_YIELD,
        help=f'Min net_credit / wing_width floor (default: {MIN_YIELD})',
    )
    parser.add_argument(
        '--workers', type=int, default=N_WORKERS,
        help=f'Number of parallel workers (default: {N_WORKERS})',
    )

    args = parser.parse_args()

    # DTE target follows frequency
    dte_target = 30 if args.frequency == 'monthly' else 7

    # -- Pre-flight checks ---------------------------------------------------
    if not Path(SPOT_DB_PATH).exists():
        logger.error(f"SpotPriceDB not found: {SPOT_DB_PATH}")
        logger.error("Run scripts/extract_spot_prices.py first.")
        return

    if not SP500_FILE.exists():
        logger.error(f"SP500 file not found: {SP500_FILE}")
        return

    # Create output / log directories
    Path('C:/MomentumCVG_env/cache').mkdir(parents=True, exist_ok=True)
    Path('C:/MomentumCVG_env/log').mkdir(parents=True, exist_ok=True)

    # -- Load tickers --------------------------------------------------------
    df_sp500 = pd.read_csv(SP500_FILE)
    tickers: List[str] = df_sp500['Ticker'].tolist()
    logger.info(f"Loaded {len(tickers)} tickers from {SP500_FILE}")

    # -- Generate trade dates ------------------------------------------------
    start_dt = datetime(args.start_year, 1, 1)
    end_dt   = datetime(args.end_year, 2, 20)
    trade_dates = generate_trade_dates(start_dt, end_dt, args.frequency, args.data_root)

    if not trade_dates:
        logger.error("No trade dates generated -- aborting.")
        return

    # -- Summary header ------------------------------------------------------
    logger.info("=" * 80)
    logger.info("Iron Butterfly History Precomputation")
    logger.info("=" * 80)
    logger.info(f"Frequency        : {args.frequency}")
    logger.info(f"DTE target       : {dte_target}")
    logger.info(f"Date range       : {trade_dates[0].date()} -> {trade_dates[-1].date()}")
    logger.info(f"Trade dates      : {len(trade_dates)}")
    logger.info(f"Tickers          : {len(tickers)}")
    logger.info(f"max_spread_pct   : {args.max_spread_pct}")
    logger.info(f"min_yield        : {args.min_yield}")
    logger.info(f"Workers          : {args.workers}")
    logger.info(f"SpotPriceDB      : {SPOT_DB_PATH}")
    logger.info(
        f"Expected input   : {len(tickers)} tickers x {len(trade_dates)} dates "
        f"= {len(tickers) * len(trade_dates):,} (ticker, date) pairs"
    )
    logger.info("=" * 80)

    # -- Parallel processing (one job per date, all tickers per job) ----------
    start_time = datetime.now()

    all_results_nested = Parallel(n_jobs=args.workers, backend='loky', verbose=0)(
        delayed(process_date_batch)(
            args.data_root,
            SPOT_DB_PATH,
            td.date(),
            tickers,
            args.frequency,
            dte_target,
            args.max_spread_pct,
            args.min_yield,
            MIN_VOLUME,
            MIN_OI,
        )
        for td in tqdm(trade_dates, desc="Processing dates")
    )

    # Flatten list-of-lists -> single list
    all_results = [row for date_rows in all_results_nested for row in date_rows]

    elapsed = (datetime.now() - start_time).total_seconds()

    # -- Build final DataFrame -----------------------------------------------
    logger.info(
        f"\nProcessed {len(all_results):,} rows in {elapsed:.1f}s "
        f"({len(all_results)/elapsed:.1f} rows/sec)"
    )

    df = pd.DataFrame(all_results)

    # -- Data quality summary ------------------------------------------------
    total      = len(df)
    body_df    = df[df['row_type'] == 'body']                if 'row_type' in df.columns else df.iloc[0:0]
    cand_df    = df[df['row_type'] == 'ironfly_candidate']   if 'row_type' in df.columns else df.iloc[0:0]
    condor_df  = df[df['row_type'] == 'ironcondor_candidate'] if 'row_type' in df.columns else df.iloc[0:0]
    fail_df    = df[df['row_type'] == 'failure']             if 'row_type' in df.columns else df.iloc[0:0]
    n_body     = len(body_df)
    n_cand     = len(cand_df)
    n_condor   = len(condor_df)
    n_fail     = len(fail_df)

    logger.info("\nData quality summary:")
    logger.info(f"  Total rows           : {total:,}")
    logger.info(f"  Body rows (straddle) : {n_body:,}  ({n_body/total*100:.1f}%)")
    logger.info(f"  Ironfly candidates   : {n_cand:,}  ({n_cand/total*100:.1f}%)")
    logger.info(f"  Condor candidates    : {n_condor:,}  ({n_condor/total*100:.1f}%)")
    logger.info(f"  Failure rows         : {n_fail:,}  ({n_fail/total*100:.1f}%)")

    if n_fail > 0:
        logger.info("\n  Failure reason breakdown:")
        for reason, count in fail_df['failure_reason'].value_counts().items():
            logger.info(f"    {reason}: {count:,}  ({count/total*100:.1f}%)")

    if n_body > 0:
        body_pnl = body_df[body_df['pnl'].notna()]
        if len(body_pnl) > 0:
            logger.info("\n  Body (short straddle) stats:")
            logger.info(f"    Unique tickers        : {body_pnl['ticker'].nunique()}")
            logger.info(f"    Unique dates          : {body_pnl['entry_date'].nunique()}")
            logger.info(
                f"    Mean return_on_credit  : "
                f"{body_pnl['return_pct_on_credit'].mean():.2f}%"
            )
            logger.info(
                f"    Median return_on_credit: "
                f"{body_pnl['return_pct_on_credit'].median():.2f}%"
            )
            logger.info(
                f"    Win rate (pnl > 0)     : "
                f"{(body_pnl['pnl'] > 0).mean()*100:.1f}%"
            )

    if n_cand > 0:
        cand_pnl = cand_df[cand_df['pnl'].notna()]
        if len(cand_pnl) > 0:
            logger.info("\n  Iron fly candidate stats:")
            logger.info(f"    Unique tickers        : {cand_pnl['ticker'].nunique()}")
            logger.info(f"    Unique dates          : {cand_pnl['entry_date'].nunique()}")
            logger.info(f"    Wing widths (unique)  : {cand_pnl['wing_width'].nunique()}")
            logger.info(
                f"    avg_wing_delta range   : "
                f"{cand_pnl['avg_wing_delta'].min():.3f} - "
                f"{cand_pnl['avg_wing_delta'].max():.3f}"
            )
            logger.info(
                f"    Mean return_on_width   : "
                f"{cand_pnl['return_pct_on_width'].mean():.2f}%"
            )
            logger.info(
                f"    Median return_on_width : "
                f"{cand_pnl['return_pct_on_width'].median():.2f}%"
            )
            logger.info(
                f"    Win rate (pnl > 0)     : "
                f"{(cand_pnl['pnl'] > 0).mean()*100:.1f}%"
            )

    if n_condor > 0:
        condor_pnl = condor_df[condor_df['pnl'].notna()]
        if len(condor_pnl) > 0:
            logger.info("\n  Iron condor candidate stats:")
            logger.info(f"    Unique tickers        : {condor_pnl['ticker'].nunique()}")
            logger.info(f"    Unique dates          : {condor_pnl['entry_date'].nunique()}")
            logger.info(f"    Body delta targets    : {sorted(condor_pnl['body_delta_target'].dropna().unique())}")
            logger.info(
                f"    avg_body_delta range   : "
                f"{condor_pnl['avg_body_delta'].min():.3f} - "
                f"{condor_pnl['avg_body_delta'].max():.3f}"
            )
            logger.info(
                f"    avg_wing_delta range   : "
                f"{condor_pnl['avg_wing_delta'].min():.3f} - "
                f"{condor_pnl['avg_wing_delta'].max():.3f}"
            )
            logger.info(
                f"    Mean return_on_width   : "
                f"{condor_pnl['return_pct_on_width'].mean():.2f}%"
            )
            logger.info(
                f"    Median return_on_width : "
                f"{condor_pnl['return_pct_on_width'].median():.2f}%"
            )
            logger.info(
                f"    Win rate (pnl > 0)     : "
                f"{(condor_pnl['pnl'] > 0).mean()*100:.1f}%"
            )

    # -- Save final output ---------------------------------------------------
    output_path = Path(
        f'C:/MomentumCVG_env/cache/'
        f'ironfly_condor_history_{args.frequency}_{args.start_year}_{args.end_year}_liquidity.parquet'
    )
    df.to_parquet(output_path, compression='gzip', index=False)
    logger.info(f"\nFinal output saved: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    logger.info("=" * 80)
    logger.info("Done!")


if __name__ == '__main__':
    main()
