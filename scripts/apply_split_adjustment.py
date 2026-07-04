"""
Apply split adjustments to all raw ORATS ZIP files.

Reads every ORATS_SMV_Strikes_YYYYMMDD.zip in raw_root/{YYYY}/,
looks up each ticker's cumulative split factor from splits_hist.parquet,
and writes split-adjusted parquet files to adj_root/{YYYY}/.

Already-existing output files are skipped by default (safe to re-run).
Ctrl-C safe — partially processed years are fine since each file is
written independently.

Usage:
    # Default paths, all years, skip existing
    python scripts/apply_split_adjustment.py

    # Specific years only
    python scripts/apply_split_adjustment.py --years 2022 2023 2024

    # Custom paths
    python scripts/apply_split_adjustment.py ^
        --raw-root  C:/ORATS/data/ORATS_Data ^
        --adj-root  C:/ORATS/data/ORATS_Adjusted ^
        --splits    C:/MomentumCVG_env/cache/splits_hist.parquet

    # Force re-process even if output already exists
    python scripts/apply_split_adjustment.py --overwrite

    # Limit parallel workers (default: half of CPU count)
    python scripts/apply_split_adjustment.py --workers 4
    # Re-adjust only specific tickers (e.g. after new splits were fetched)
    python scripts/apply_split_adjustment.py --tickers NVDA TSLA AAPL

    # Ignore splits before a custom cutoff (default 2014-01-01 = earliest data)
    python scripts/apply_split_adjustment.py --min-split-date 2014-01-01

    # Filtered (C5) mode: write ONLY C4-liquidity-universe tickers to a
    # dedicated adjusted root. --adj-root is REQUIRED when --ticker-universe is
    # set so the full ORATS_Adjusted mirror is never overwritten by accident.
    python scripts/apply_split_adjustment.py ^
        --ticker-universe C:/MomentumCVG_env/input/liquidity/liquid_tickers.csv ^
        --adj-root        C:/MomentumCVG_env/input/adjusted_liquid"""

import argparse
import logging
import sys
from pathlib import Path

# ── project root on sys.path ──────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.split_adjuster import SplitAdjuster  # noqa: E402
from src.data.ticker_universe import load_ticker_universe  # noqa: E402

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RAW_ROOT    = Path("C:/ORATS/data/ORATS_Data")
DEFAULT_ADJ_ROOT    = Path("C:/ORATS/data/ORATS_Adjusted")
DEFAULT_SPLITS_PATH = Path("C:/MomentumCVG_env/cache/splits_hist.parquet")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply split adjustments to ORATS raw ZIP files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Root directory containing per-year ZIP files.",
    )
    p.add_argument(
        "--adj-root",
        type=Path,
        default=None,
        help=(
            "Output root for adjusted parquet files. "
            f"When omitted, defaults to the full mirror {DEFAULT_ADJ_ROOT}. "
            "REQUIRED explicitly when --ticker-universe is set (filtered mode) "
            "so the full mirror is never overwritten by accident."
        ),
    )
    p.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS_PATH,
        dest="splits_path",
        help="Parquet with split history (ticker, split_date, divisor).",
    )
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        metavar="YYYY",
        help="Restrict processing to these years only.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-process files even if parquet already exists.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel worker processes (default: CPU count // 2).",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="TICKER",
        help=(
            "Re-adjust only these tickers (e.g. after new splits were fetched). "
            "Implies --overwrite for those files."
        ),
    )
    p.add_argument(
        "--min-split-date",
        default="2014-01-01",
        metavar="YYYY-MM-DD",
        help=(
            "Ignore splits that occurred before this date. "
            "Splits before your earliest raw data are never relevant. "
            "Default: 2014-01-01 (earliest ORATS data)."
        ),
    )
    p.add_argument(
        "--ticker-universe",
        type=Path,
        default=None,
        dest="ticker_universe",
        metavar="PATH",
        help=(
            "Optional C4 liquidity universe file (CSV or parquet with a "
            "'Ticker'/'ticker' column). When set, only rows for these tickers "
            "are written (filtered C5 mode) and --adj-root MUST be given "
            "explicitly (e.g. C:/MomentumCVG_env/input/adjusted_liquid/)."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── filtered-mode safety guard (fail fast) ────────────────────────────────
    # The old default --adj-root is the *full* mirror. Filtered (C5) output must
    # be written to a separate root deliberately, never on top of the full
    # adjusted cache. So --ticker-universe requires an explicit --adj-root.
    if args.ticker_universe is not None and args.adj_root is None:
        msg = (
            "Filtered mode requires an explicit --adj-root. "
            "When --ticker-universe is set, pass --adj-root explicitly so the "
            "full adjusted mirror is never overwritten. Use a dedicated root "
            "such as C:/MomentumCVG_env/input/adjusted_liquid/."
        )
        logger.error(msg)
        sys.exit(msg)

    # No universe → preserve the historical default (full mirror).
    adj_root = args.adj_root if args.adj_root is not None else DEFAULT_ADJ_ROOT

    # ── validate inputs ───────────────────────────────────────────────────────
    if not args.raw_root.exists():
        logger.error("raw_root does not exist: %s", args.raw_root)
        sys.exit(1)
    if not args.splits_path.exists():
        logger.error(
            "Splits file not found: %s\n"
            "Run scripts/fetch_splits.py first.",
            args.splits_path,
        )
        sys.exit(1)

    # ── resolve optional ticker universe (filtered C5 mode) ───────────────────
    ticker_universe: list[str] | None = None
    if args.ticker_universe is not None:
        ticker_universe = load_ticker_universe(args.ticker_universe)
        logger.info(
            "ticker_universe: loaded %d tickers from %s",
            len(ticker_universe),
            args.ticker_universe,
        )

    logger.info("raw_root       : %s", args.raw_root)
    logger.info("adj_root       : %s", adj_root)
    logger.info("splits_path    : %s", args.splits_path)
    logger.info("min_split_date : %s", args.min_split_date)
    logger.info("years          : %s", args.years or "all")
    logger.info("tickers        : %s", args.tickers or "all")
    logger.info("ticker_universe: %s", args.ticker_universe or "none (full)")
    logger.info("overwrite      : %s", args.overwrite)
    logger.info("workers        : %s", args.workers or "auto (CPU//2)")

    # ── run ───────────────────────────────────────────────────────────────────
    adjuster = SplitAdjuster(
        raw_root=args.raw_root,
        adj_root=adj_root,
        splits_path=args.splits_path,
        overwrite=args.overwrite,
        min_split_date=args.min_split_date,
        ticker_universe=ticker_universe,
    )

    if args.tickers:
        # Targeted re-adjustment for specific tickers only
        adjuster.readjust_tickers(
            tickers=args.tickers,
            years=args.years,
            max_workers=args.workers,
        )
    else:
        adjuster.run(years=args.years, max_workers=args.workers)


if __name__ == "__main__":
    main()
