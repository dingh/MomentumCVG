"""
Fetch ORATS earnings history for S&P 500 tickers.

Reads the ticker list from SP500.csv (Ticker, Date added, Date removed),
calls the ORATS datav2 /hist/earnings endpoint for each ticker, and saves
combined results to a parquet file.  Checkpoint/resume is built-in — safe
to Ctrl-C and re-run; already-fetched tickers are skipped automatically.

By default only **currently active** S&P 500 members are fetched (rows where
"Date removed" is blank).  Pass --include-historical to fetch all tickers
that have ever been in the index.

Usage:
    # Required: pass your ORATS API token
    python scripts/fetch_earnings.py --token YOUR_TOKEN

    # Custom paths
    python scripts/fetch_earnings.py ^
        --token    YOUR_TOKEN ^
        --sp500    C:/MomentumCVG_env/cache/SP500.csv ^
        --output   C:/MomentumCVG_env/cache/earnings_hist.parquet

    # Include all historical S&P 500 members (not just current)
    python scripts/fetch_earnings.py --token YOUR_TOKEN --include-historical

    # Slow down requests (default 0.7 s between tickers)
    python scripts/fetch_earnings.py --token YOUR_TOKEN --rate-limit 1.5

    # Resume an interrupted run (just re-run the same command — checkpoint is automatic)
    python scripts/fetch_earnings.py --token YOUR_TOKEN
"""

import argparse
import logging
import sys
from pathlib import Path

# ── resolve project root so src/ imports work regardless of cwd ───────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent     # scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent                  # MomentumCVG/
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.corporate_actions import OratsCorporateActionsFetcher  # noqa: E402

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_SP500_PATH  = Path("C:/MomentumCVG_env/cache/SP500.csv")
DEFAULT_OUTPUT_PATH = Path("C:/MomentumCVG_env/cache/earnings_hist.parquet")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_sp500_tickers(csv_path: Path, include_historical: bool) -> list[str]:
    """Read SP500.csv and return the relevant ticker list.

    Parameters
    ----------
    csv_path : Path
        Path to SP500.csv with columns: Ticker, Date added, Date removed.
    include_historical : bool
        If False (default), only return currently active members
        (rows where 'Date removed' is blank / NaN).
        If True, return every ticker that has ever been in the index.

    Returns
    -------
    list[str]
        Sorted, deduplicated list of tickers.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Normalise column names — strip whitespace, handle case
    df.columns = df.columns.str.strip()
    ticker_col   = next(c for c in df.columns if c.lower() == "ticker")
    removed_col  = next((c for c in df.columns if "removed" in c.lower()), None)

    if not include_historical and removed_col:
        # Keep only rows where Date removed is blank (still in the index)
        active_mask = df[removed_col].isna() | (df[removed_col].astype(str).str.strip() == "")
        df = df[active_mask]
        logger.info(
            "Active S&P 500 members: %d  (use --include-historical for all %d ever-members)",
            len(df),
            pd.read_csv(csv_path)[ticker_col].nunique(),
        )
    else:
        logger.info("Using all historical S&P 500 members: %d tickers", df[ticker_col].nunique())

    tickers = sorted(df[ticker_col].dropna().str.strip().unique().tolist())
    return tickers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch ORATS earnings history for S&P 500 tickers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--token",
        required=True,
        help="ORATS API token.",
    )
    p.add_argument(
        "--sp500",
        type=Path,
        default=DEFAULT_SP500_PATH,
        help="Path to SP500.csv (Ticker, Date added, Date removed).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output parquet path for earnings history. Also used as checkpoint.",
    )
    p.add_argument(
        "--include-historical",
        action="store_true",
        default=False,
        help="Include all tickers that have ever been in the S&P 500, not just current members.",
    )
    p.add_argument(
        "--rate-limit",
        type=float,
        default=0.7,
        metavar="SECONDS",
        help="Seconds to sleep between API requests.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retry attempts per request on 429 / 5xx responses.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── validate SP500 file ───────────────────────────────────────────────────
    if not args.sp500.exists():
        logger.error("SP500 file not found: %s", args.sp500)
        sys.exit(1)

    tickers = load_sp500_tickers(args.sp500, args.include_historical)
    logger.info("Loaded %d tickers from %s", len(tickers), args.sp500)

    # ── show resume status ────────────────────────────────────────────────────
    import pandas as pd

    if args.output.exists():
        existing = pd.read_parquet(args.output)
        already_done = existing["ticker"].nunique() if "ticker" in existing.columns else 0
        remaining = len(tickers) - already_done
        logger.info(
            "Checkpoint found — %d tickers already fetched, %d remaining.",
            already_done, remaining,
        )
    else:
        logger.info("No checkpoint found — starting fresh.")

    # ── fetch ─────────────────────────────────────────────────────────────────
    fetcher = OratsCorporateActionsFetcher(
        token=args.token,
        rate_limit=args.rate_limit,
        max_retries=args.max_retries,
    )

    earnings_df = fetcher.fetch_all_earnings(
        tickers=tickers,
        checkpoint_path=args.output,   # saves after every ticker; resumes on re-run
    )

    # ── summary ───────────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("Done.")
    logger.info("  Total earnings records   : %d", len(earnings_df))
    logger.info("  Tickers with earnings    : %d / %d", earnings_df["ticker"].nunique(), len(tickers))
    logger.info("  Saved to                 : %s", args.output)
    if not earnings_df.empty and "earn_date" in earnings_df.columns:
        dates = pd.to_datetime(earnings_df["earn_date"], errors="coerce").dropna()
        if not dates.empty:
            logger.info(
                "  Date range               : %s -> %s",
                dates.min().date(), dates.max().date(),
            )


if __name__ == "__main__":
    main()
