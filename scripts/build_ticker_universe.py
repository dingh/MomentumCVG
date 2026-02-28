"""
Build the ticker universe by scanning all ORATS raw ZIP files.

Reads every *.zip in DATA_ROOT/{YYYY}/ across all year directories,
extracts unique ticker values, and writes a sorted parquet file.

Usage:
    # Default paths
    python scripts/build_ticker_universe.py

    # Custom data root and output
    python scripts/build_ticker_universe.py \\
        --data-root C:/ORATS/data/ORATS_Data \\
        --output    C:/MomentumCVG_env/cache/all_tickers.parquet

    # Scan only specific years
    python scripts/build_ticker_universe.py \\
        --years 2022 2023 2024
"""

import argparse
import logging
import sys
from pathlib import Path

# ── resolve project root so src/ imports work regardless of cwd ───────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent          # scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent                       # MomentumCVG/
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.corporate_actions import get_all_unique_tickers  # noqa: E402

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_ROOT = Path("C:/ORATS/data/ORATS_Data")
DEFAULT_OUTPUT    = Path("C:/MomentumCVG_env/cache/all_tickers.parquet")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scan ORATS raw ZIP store and build a unique ticker universe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing per-year subdirs of ORATS ZIP files.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet path for the ticker universe.",
    )
    p.add_argument(
        "--ticker-col",
        default="ticker",
        help="Column name inside each CSV that holds the ticker symbol.",
    )
    p.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=None,
        metavar="YYYY",
        help="Restrict scan to these years only (default: all years found).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_root: Path = args.data_root
    output:    Path = args.output
    ticker_col: str = args.ticker_col

    if not data_root.exists():
        logger.error("DATA_ROOT does not exist: %s", data_root)
        sys.exit(1)

    # ── optionally restrict to a subset of years ──────────────────────────────
    effective_root = data_root
    if args.years:
        missing = [y for y in args.years if not (data_root / str(y)).is_dir()]
        if missing:
            logger.warning("Year directories not found (skipped): %s", missing)
        logger.info("Scanning years: %s", sorted(args.years))

        # Build a temporary symlink-free approach: pass the original root and
        # let get_all_unique_tickers handle it, but we pre-filter by renaming
        # This is handled inside get_all_unique_tickers which scans all years;
        # since it doesn't accept a years filter directly, we create a thin
        # wrapper here that temporarily patches the year list in the function.
        # Simpler: point at a filtered set via a dedicated code path below.

        import zipfile
        import pandas as pd
        from tqdm import tqdm

        year_dirs = sorted(
            [d for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()
             and int(d.name) in args.years]
        )

        all_zip_files = []
        for yd in year_dirs:
            all_zip_files.extend(sorted(yd.glob("*.zip")))

        total_files = len(all_zip_files)
        logger.info("Total ZIP files to scan: %d", total_files)

        tickers: set[str] = set()
        output.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output.with_suffix(".tmp.parquet")

        with tqdm(total=total_files, unit="file", desc="Scanning ZIPs") as pbar:
            for zip_path in all_zip_files:
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        csv_name = next(
                            n for n in zf.namelist()
                            if n.endswith((".csv", ".txt"))
                        )
                        with zf.open(csv_name) as f:
                            df = pd.read_csv(f, usecols=[ticker_col], dtype=str)
                            tickers.update(df[ticker_col].dropna().unique())
                except Exception as exc:
                    logger.warning("Skipping %s — %s", zip_path.name, exc)
                pbar.update(1)
                pbar.set_postfix(tickers=len(tickers))

        result = sorted(tickers)
        pd.DataFrame({"ticker": result}).to_parquet(output, index=False)
        if temp_path.exists():
            temp_path.unlink()

        logger.info("Saved %d tickers → %s", len(result), output)
        return

    # ── default: scan all years via the shared utility ────────────────────────
    tickers = get_all_unique_tickers(
        data_root=effective_root,
        output_path=output,
        ticker_col=ticker_col,
    )

    logger.info("Done — %d unique tickers saved to %s", len(tickers), output)


if __name__ == "__main__":
    main()
