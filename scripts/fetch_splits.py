"""
Fetch ORATS stock split history for every ticker in the ticker universe.

Reads the ticker list from all_tickers.parquet (built by build_ticker_universe.py),
calls the ORATS datav2 /hist/splits endpoint for each ticker, and saves combined
results to a parquet file.  Checkpoint/resume is built-in — safe to Ctrl-C and
re-run; already-fetched tickers are skipped automatically.

Usage:
    # Required: pass your ORATS API token
    python scripts/fetch_splits.py --token YOUR_TOKEN

    # Custom paths
    python scripts/fetch_splits.py ^
        --token    YOUR_TOKEN ^
        --tickers  C:/MomentumCVG_env/cache/all_tickers.parquet ^
        --output   C:/MomentumCVG_env/cache/splits_hist.parquet

    # Slow down requests (default 0.7 s between tickers)
    python scripts/fetch_splits.py --token YOUR_TOKEN --rate-limit 1.5

    # Resume an interrupted run (just re-run the same command — checkpoint is automatic)
    python scripts/fetch_splits.py --token YOUR_TOKEN
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
DEFAULT_TICKERS_PATH = Path("C:/MomentumCVG_env/cache/all_tickers.parquet")
DEFAULT_OUTPUT_PATH  = Path("C:/MomentumCVG_env/cache/splits_hist.parquet")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch ORATS stock split history for a ticker universe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--token",
        required=True,
        help="ORATS API token.",
    )
    p.add_argument(
        "--tickers",
        type=Path,
        default=DEFAULT_TICKERS_PATH,
        help="Path to all_tickers.parquet (single 'ticker' column).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output parquet path for split history. Also used as checkpoint.",
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

    # ── validate tickers file ─────────────────────────────────────────────────
    if not args.tickers.exists():
        logger.error(
            "Tickers file not found: %s\n"
            "Run build_ticker_universe.py first.",
            args.tickers,
        )
        sys.exit(1)

    import pandas as pd

    tickers_df = pd.read_parquet(args.tickers)
    if "ticker" not in tickers_df.columns:
        logger.error("Expected a 'ticker' column in %s", args.tickers)
        sys.exit(1)

    tickers = tickers_df["ticker"].dropna().unique().tolist()
    logger.info("Loaded %d tickers from %s", len(tickers), args.tickers)

    # ── show resume status ────────────────────────────────────────────────────
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

    splits_df = fetcher.fetch_all_splits(
        tickers=tickers,
        checkpoint_path=args.output,   # saves after every ticker; resumes on re-run
    )

    # ── summary ───────────────────────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("Done.")
    logger.info("  Total split records : %d", len(splits_df))
    logger.info("  Tickers with splits : %d / %d", splits_df["ticker"].nunique(), len(tickers))
    logger.info("  Saved to            : %s", args.output)


if __name__ == "__main__":
    main()
