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

    # Safe bounded smoke (isolated output root, dry-run first)
    python scripts/precompute_option_surface.py --frequency weekly \\
        --start-year 2024 --end-year 2024 --start-date 2024-01-01 --end-date 2024-03-31 \\
        --tickers AAPL MSFT NVDA --output-root C:/MomentumCVG_env/cache/c6_smoke --dry-run

    # Narrow the delta window and keep zero-bid quotes for coverage audit
    python scripts/precompute_option_surface.py --min-abs-delta 0.05 --max-abs-delta 0.35 --keep-zero-bid-quotes

    # Custom delta bucket grid
    python scripts/precompute_option_surface.py --delta-buckets "0.10,0.15,0.20,0.25,0.30"
"""
from __future__ import annotations

import sys
import logging
import argparse
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.paths import (
    DEFAULT_ADJUSTED_LIQUID_ROOT,
    DEFAULT_CACHE_ROOT,
    DEFAULT_LIQUID_TICKERS_PATH,
    DEFAULT_PRECOMPUTE_OPTION_SURFACE_LOG,
    DEFAULT_SPOT_PRICES_PATH,
)
from src.data.trading_day import weekly_trade_dates_in_range
from src.features.option_surface_analyzer import OptionSurfaceBuilder
from src.data.spot_price_db import SpotPriceDB


# =============================================================================
# Constants (overridable via argparse)
# =============================================================================

N_WORKERS = 26


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


def configure_logging(log_file: Path | None) -> None:
    """Attach stream and optional file handlers for this script."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


# =============================================================================
# Date helpers
# =============================================================================

def sample_fridays_by_frequency(
    entry_dates: Sequence[date],
    frequency: str,
) -> List[date]:
    """Filter weekly entry dates to the desired sampling frequency.

    - ``'weekly'``  : all entry dates returned as-is
    - ``'monthly'`` : first entry date of each calendar month only
    """
    if frequency == "weekly":
        return list(entry_dates)
    if frequency == "monthly":
        grouped: Dict[Tuple[int, int], List[date]] = {}
        for entry_day in entry_dates:
            grouped.setdefault((entry_day.year, entry_day.month), []).append(entry_day)
        return sorted(days[0] for days in grouped.values())
    raise ValueError(f"Unknown frequency: {frequency!r}. Must be 'weekly' or 'monthly'.")


def generate_trade_dates(
    start_date: date,
    end_date: date,
    frequency: str,
    data_root: str | Path,
) -> List[date]:
    """Generate trade entry dates for the given range, frequency, and data root."""
    weekly_entries = weekly_trade_dates_in_range(start_date, end_date, data_root)
    trade_dates = sample_fridays_by_frequency(weekly_entries, frequency)
    logger.info(
        "Generated %s %s trade dates from %s weekly entry dates",
        len(trade_dates),
        frequency,
        len(weekly_entries),
    )
    return trade_dates


# =============================================================================
# Worker function (module-level → pickleable by joblib)
# =============================================================================

def process_date_batch(
    data_root: str,
    spot_db_path: str,
    trade_date: date,
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
    """Parse a comma-separated delta-bucket string into a sorted float list."""
    if not raw or not raw.strip():
        return [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40]
    return sorted(float(x.strip()) for x in raw.split(",") if x.strip())


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value!r}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute an expiry-level option quote surface for fly / condor backtests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_ADJUSTED_LIQUID_ROOT,
        help="Path to split-adjusted daily parquet root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="Directory for option surface meta/quotes parquets",
    )
    parser.add_argument(
        "--spot-db-path",
        type=Path,
        default=DEFAULT_SPOT_PRICES_PATH,
        help="Spot price parquet used by SpotPriceDB",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="Start year inclusive (used for output filenames and default date bounds)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="End year inclusive (used for output filenames and default date bounds)",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_iso_date,
        default=None,
        help="Inclusive start date (ISO). Default: Jan 1 of --start-year",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_iso_date,
        default=None,
        help="Inclusive end date (ISO). Default: Dec 31 of --end-year",
    )
    ticker_group = parser.add_mutually_exclusive_group()
    ticker_group.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help="Process only these tickers (space-separated)",
    )
    ticker_group.add_argument(
        "--tickers-file",
        type=Path,
        default=None,
        help="CSV with Ticker column; default when --tickers omitted",
    )
    parser.add_argument(
        "--frequency",
        choices=["monthly", "weekly"],
        default="monthly",
        help="Sampling frequency: monthly (first Friday, ~30 DTE) or weekly (~7 DTE)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=N_WORKERS,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--min-abs-delta",
        type=float,
        default=0.03,
        help="Min abs delta for OTM wing quotes",
    )
    parser.add_argument(
        "--max-abs-delta",
        type=float,
        default=0.45,
        help="Max abs delta for OTM wing quotes",
    )
    parser.add_argument(
        "--delta-buckets",
        type=str,
        default="0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25,0.30,0.35,0.40",
        help="Comma-separated reference delta levels stored on each quote row",
    )
    parser.add_argument(
        "--keep-zero-bid-quotes",
        action="store_true",
        help="Retain quotes with non-positive bid/ask/mid (coverage audits)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned run summary and exit without joblib or parquet writes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output parquets",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Per-run log path; use '-' for stderr only; default shared log file",
    )
    return parser.parse_args(argv)


def resolve_date_bounds(args: argparse.Namespace) -> tuple[date, date]:
    start = args.start_date or date(args.start_year, 1, 1)
    end = args.end_date or date(args.end_year, 12, 31)
    if start > end:
        raise ValueError(
            f"start must be on or before end; got {start.isoformat()} > {end.isoformat()}"
        )
    return start, end


def resolve_log_file(raw: str | None) -> Path | None:
    if raw == "-":
        return None
    if raw is None:
        return DEFAULT_PRECOMPUTE_OPTION_SURFACE_LOG
    return Path(raw)


def normalize_tickers(raw_tickers: Sequence[object]) -> list[str]:
    """Strip, uppercase, dedupe, and preserve first-seen order for ticker scope."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in raw_tickers:
        if pd.isna(raw):
            continue
        symbol = str(raw).strip().upper()
        if not symbol:
            continue
        if symbol not in seen:
            seen.add(symbol)
            normalized.append(symbol)
    if not normalized:
        raise ValueError("Ticker scope is empty after normalization")
    return normalized


def load_tickers(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.tickers is not None:
        return normalize_tickers(args.tickers), "inline --tickers"

    tickers_path = args.tickers_file or DEFAULT_LIQUID_TICKERS_PATH
    if not tickers_path.exists():
        raise FileNotFoundError(f"Ticker universe file not found: {tickers_path}")

    df_tickers = pd.read_csv(tickers_path)
    if "Ticker" not in df_tickers.columns:
        raise ValueError(f"Ticker column not found in {tickers_path}")
    tickers = normalize_tickers(df_tickers["Ticker"].tolist())
    return tickers, str(tickers_path)


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_root = Path(args.output_root)
    meta_name = (
        f"option_surface_meta_{args.frequency}_{args.start_year}_{args.end_year}.parquet"
    )
    quotes_name = (
        f"option_surface_quotes_{args.frequency}_{args.start_year}_{args.end_year}.parquet"
    )
    return output_root / meta_name, output_root / quotes_name


def render_dry_run_summary(
    *,
    requested_start: date,
    requested_end: date,
    trade_dates: list[date],
    tickers: list[str],
    ticker_source: str,
    data_root: Path,
    output_root: Path,
    spot_db_path: Path,
    meta_path: Path,
    quotes_path: Path,
    overwrite: bool,
) -> str:
    meta_exists = meta_path.exists()
    quotes_exists = quotes_path.exists()
    outputs_exist = meta_exists or quotes_exists
    overwrite_required = outputs_exist and not overwrite

    schedule_min = trade_dates[0].isoformat() if trade_dates else "(none)"
    schedule_max = trade_dates[-1].isoformat() if trade_dates else "(none)"

    lines = [
        "Option surface precompute — dry run",
        f"requested_start_date: {requested_start.isoformat()}",
        f"requested_end_date: {requested_end.isoformat()}",
        f"resolved_schedule_min: {schedule_min}",
        f"resolved_schedule_max: {schedule_max}",
        f"resolved_entry_date_count: {len(trade_dates)}",
        f"ticker_source: {ticker_source}",
        f"ticker_count: {len(tickers)}",
        f"data_root: {data_root}",
        f"output_root: {output_root}",
        f"spot_db_path: {spot_db_path}",
        f"meta_output_path: {meta_path}",
        f"quotes_output_path: {quotes_path}",
        f"meta_exists: {meta_exists}",
        f"quotes_exists: {quotes_exists}",
        f"overwrite_required_without_flag: {overwrite_required}",
    ]
    if tickers and len(tickers) <= 20:
        lines.append(f"tickers: {', '.join(tickers)}")
    return "\n".join(lines)


def check_overwrite_guard(
    meta_path: Path,
    quotes_path: Path,
    *,
    overwrite: bool,
    dry_run: bool,
) -> int | None:
    """Return exit code 2 when outputs exist and overwrite was not requested."""
    outputs_exist = meta_path.exists() or quotes_path.exists()
    if outputs_exist and not overwrite:
        message = (
            f"Output already exists: {meta_path} and/or {quotes_path}. "
            "Pass --overwrite to replace."
        )
        if dry_run:
            logger.warning(message)
            return None
        logger.error(message)
        return 2
    return None


# =============================================================================
# Main
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(resolve_log_file(args.log_file))

    try:
        requested_start, requested_end = resolve_date_bounds(args)
        tickers, ticker_source = load_tickers(args)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 2

    delta_buckets = _parse_delta_buckets(args.delta_buckets)
    dte_target = 30 if args.frequency == "monthly" else 7
    meta_path, quotes_path = output_paths(args)
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    spot_db_path = Path(args.spot_db_path)

    trade_dates = generate_trade_dates(
        requested_start, requested_end, args.frequency, data_root
    )

    summary = render_dry_run_summary(
        requested_start=requested_start,
        requested_end=requested_end,
        trade_dates=trade_dates,
        tickers=tickers,
        ticker_source=ticker_source,
        data_root=data_root,
        output_root=output_root,
        spot_db_path=spot_db_path,
        meta_path=meta_path,
        quotes_path=quotes_path,
        overwrite=args.overwrite,
    )

    if args.dry_run:
        print(summary)
        return 0

    guard_code = check_overwrite_guard(
        meta_path, quotes_path, overwrite=args.overwrite, dry_run=False
    )
    if guard_code is not None:
        return guard_code

    if not spot_db_path.exists():
        logger.error("SpotPriceDB not found: %s", spot_db_path)
        logger.error("Run scripts/extract_spot_prices.py first.")
        return 2

    if not trade_dates:
        logger.error("No trade dates generated; aborting.")
        return 2

    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Option Surface Precomputation")
    logger.info("=" * 80)
    logger.info("Frequency        : %s", args.frequency)
    logger.info("DTE target       : %s", dte_target)
    logger.info(
        "Date range       : %s -> %s",
        trade_dates[0].isoformat(),
        trade_dates[-1].isoformat(),
    )
    logger.info("Trade dates      : %s", len(trade_dates))
    logger.info("Tickers          : %s", len(tickers))
    logger.info("min_abs_delta    : %s", args.min_abs_delta)
    logger.info("max_abs_delta    : %s", args.max_abs_delta)
    logger.info("delta_buckets    : %s", delta_buckets)
    logger.info("keep_zero_bid    : %s", args.keep_zero_bid_quotes)
    logger.info("Workers          : %s", args.workers)
    logger.info("SpotPriceDB      : %s", spot_db_path)
    logger.info(
        "Expected input   : %s tickers x %s dates = %s (ticker, date) pairs",
        len(tickers),
        len(trade_dates),
        len(tickers) * len(trade_dates),
    )
    logger.info("=" * 80)

    started = datetime.now()
    results = Parallel(n_jobs=args.workers, backend="loky", verbose=0)(
        delayed(process_date_batch)(
            str(data_root),
            str(spot_db_path),
            trade_day,
            tickers,
            args.frequency,
            dte_target,
            args.min_abs_delta,
            args.max_abs_delta,
            delta_buckets,
            args.keep_zero_bid_quotes,
        )
        for trade_day in tqdm(trade_dates, desc="Processing dates")
    )

    meta_rows: List[Dict] = []
    quote_rows: List[Dict] = []
    for batch_meta, batch_quotes in results:
        meta_rows.extend(batch_meta)
        quote_rows.extend(batch_quotes)

    elapsed = (datetime.now() - started).total_seconds()

    meta_df = pd.DataFrame(meta_rows)
    quotes_df = pd.DataFrame(quote_rows)

    logger.info(
        "\nProcessed %s metadata rows and %s quote rows in %.1fs (%.0f rows/sec)",
        f"{len(meta_df):,}",
        f"{len(quotes_df):,}",
        elapsed,
        (len(meta_df) + len(quotes_df)) / elapsed if elapsed > 0 else 0,
    )

    if not meta_df.empty:
        n_valid = int(meta_df["surface_valid"].sum())
        n_fail = int((~meta_df["surface_valid"]).sum())
        logger.info(
            "Valid surfaces    : %s  (%.1f%%)",
            f"{n_valid:,}",
            100.0 * n_valid / len(meta_df),
        )
        logger.info(
            "Failure rows      : %s  (%.1f%%)",
            f"{n_fail:,}",
            100.0 * n_fail / len(meta_df),
        )
        if meta_df["failure_reason"].notna().any():
            logger.info("Failure breakdown:")
            for reason, count in meta_df["failure_reason"].value_counts(dropna=True).items():
                logger.info("  %s: %s", reason, f"{count:,}")

    if not quotes_df.empty:
        logger.info(
            "Quote rows: %s unique tickers, %s body rows, %s OTM wing rows",
            len(quotes_df["ticker"].unique()),
            len(quotes_df[quotes_df["is_body"]]),
            len(quotes_df[quotes_df["is_otm"]]),
        )

    meta_df.to_parquet(meta_path, compression="gzip", index=False)
    quotes_df.to_parquet(quotes_path, compression="gzip", index=False)

    logger.info(
        "Saved metadata parquet : %s  (%.2f MB)",
        meta_path,
        meta_path.stat().st_size / 1024 / 1024,
    )
    logger.info(
        "Saved quote parquet    : %s  (%.2f MB)",
        quotes_path,
        quotes_path.stat().st_size / 1024 / 1024,
    )
    logger.info("=" * 80)
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
