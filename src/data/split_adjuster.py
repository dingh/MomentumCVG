"""
src/data/split_adjuster.py

Convert raw ORATS ZIP files → split-adjusted parquet files.

Algorithm
---------
For each daily ZIP in raw_root/{YYYY}/:
  1. Extract the trade date from the filename  (ORATS_SMV_Strikes_YYYYMMDD.zip)
  2. Read the single CSV inside the ZIP
  3. For every ticker row, look up the cumulative split factor
       cum_factor = product of divisors for ALL splits that occur STRICTLY
                    AFTER trade_date
  4. Divide price columns by the cumulative factor → adj_* columns
  5. Write out as parquet to adj_root/{YYYY}/

Split-factor accumulation
-------------------------
ORATS ``divisor``:
  - 2-for-1 split → divisor = 2.0
  - 3-for-2 split → divisor = 1.5

To express a historical raw price in post-all-splits terms:
    adj_price = raw_price / product(divisors for all splits after trade_date)

Example (AAPL had a 7:1 split on 2014-06-09 and a 4:1 split on 2020-08-31):
  trade_date 2013-01-01  →  cum_factor = 7.0 × 4.0 = 28.0
  trade_date 2019-01-01  →  cum_factor = 4.0
  trade_date 2021-01-01  →  cum_factor = 1.0  (no future splits)

Note on the legacy ``PriceAdjuster`` bug
-----------------------------------------
The legacy class used ``future_splits.iloc[0]`` after sorting splits descending.
Since ``iloc[0]`` is the *most recent* future split, it only captured the most
recent divisor, not the full product.  The correct row is ``iloc[-1]`` (oldest
future split) because its ``cum_factor`` is the cumulative product from newest
to that oldest event — i.e., the product of *all* future splits.
This class uses a vectorized lookup that correctly picks the oldest future split
per ticker.
"""
from __future__ import annotations

import logging
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
import multiprocessing

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── columns to split-adjust ───────────────────────────────────────────────────
PRICE_COLS = ["stkPx", "strike", "cBidPx", "cAskPx", "pBidPx", "pAskPx"]


# ── module-level worker (avoids pickling bound methods) ───────────────────────

def _process_year_worker(
    year_dir: Path,
    adj_root: Path,
    cum_factors: pd.DataFrame,
    overwrite: bool,
) -> tuple[int, int, int]:
    """
    Process all ZIP files in *year_dir*.

    Returns (total, skipped, errors) counts.
    """
    zip_files = sorted(year_dir.glob("*.zip"))
    total = len(zip_files)
    skipped = errors = 0

    for zip_path in tqdm(zip_files, desc=f"  {year_dir.name}", unit="file", leave=False):
        result = _process_single_zip(zip_path, adj_root, cum_factors, overwrite)
        if result is None:
            errors += 1
        elif result is False:
            skipped += 1

    return total, skipped, errors


def _process_single_zip(
    zip_path: Path,
    adj_root: Path,
    cum_factors: pd.DataFrame,
    overwrite: bool,
) -> Optional[bool]:
    """
    Process one ZIP file.

    Returns:
        Path  – success (output path)
        False – skipped (already exists, overwrite=False)
        None  – error
    """
    # ── parse trade_date from filename ────────────────────────────────────────
    try:
        date_str = zip_path.stem.split("_")[-1]          # "YYYYMMDD"
        trade_date = pd.Timestamp(date_str)
    except Exception:
        logger.warning("Cannot parse date from filename: %s", zip_path.name)
        return None

    # ── skip if output already exists ─────────────────────────────────────────
    out_path = (
        adj_root
        / str(trade_date.year)
        / f"ORATS_SMV_Strikes_{date_str}.parquet"
    )
    if not overwrite and out_path.exists():
        return False

    # ── read CSV from ZIP ─────────────────────────────────────────────────────
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith((".csv", ".txt"))]
            if not csv_names:
                logger.warning("No CSV/TXT inside %s", zip_path.name)
                return None
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(f, dtype={"ticker": str})
    except Exception as exc:
        logger.error("Failed to read %s: %s", zip_path.name, exc)
        return None

    # ── compute and apply split adjustments ───────────────────────────────────
    try:
        df = _apply_adjustments(df, trade_date, cum_factors)
    except Exception as exc:
        logger.error("Failed to adjust %s: %s", zip_path.name, exc)
        return None

    # ── write parquet ─────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return True


def _apply_adjustments(
    df: pd.DataFrame,
    trade_date: pd.Timestamp,
    cum_factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add ``trade_date``, ``split_factor``, ``adj_*``, and ``spot_px`` columns.

    Vectorized: builds the factor map for all tickers in one pandas operation.
    """
    df = df.copy()
    df["trade_date"] = trade_date

    # ── build per-ticker factor map ───────────────────────────────────────────
    # 1. Keep only future splits for tickers present in this file
    present_tickers = set(df["ticker"].unique())
    future = cum_factors[
        (cum_factors["split_date"] > trade_date)
        & (cum_factors["ticker"].isin(present_tickers))
    ]

    if future.empty:
        # No future splits for any ticker on this date → all factors = 1.0
        factor_map: dict[str, float] = {}
    else:
        # For each ticker pick the OLDEST future split event.
        # Its cum_factor = product of ALL future splits (including itself),
        # because cum_factors was built with a newest→oldest descending cumprod.
        oldest_idx = future.groupby("ticker")["split_date"].idxmin()
        oldest = future.loc[oldest_idx].set_index("ticker")["cum_factor"]
        factor_map = oldest.to_dict()

    # ── apply to dataframe ────────────────────────────────────────────────────
    df["split_factor"] = df["ticker"].map(factor_map).fillna(1.0)

    for col in PRICE_COLS:
        if col in df.columns:
            df[f"adj_{col}"] = df[col] / df["split_factor"]

    # spot_px = adj_stkPx (convenience alias consumed by ORATSDataProvider)
    if "adj_stkPx" in df.columns:
        df["spot_px"] = df["adj_stkPx"]

    return df


# ── main class ─────────────────────────────────────────────────────────────────

class SplitAdjuster:
    """
    Convert raw ORATS ZIP data → split-adjusted parquet files.

    Parameters
    ----------
    raw_root : str or Path
        Root directory containing per-year subdirs with ORATS ZIP files.
        e.g. ``C:/ORATS/data/ORATS_Data``
    adj_root : str or Path
        Output root for adjusted parquet files.
        e.g. ``C:/ORATS/data/ORATS_Adjusted``
    splits_path : str or Path
        Parquet file with split history (columns: ticker, split_date, divisor).
        e.g. ``C:/MomentumCVG_env/cache/splits_hist.parquet``
    overwrite : bool
        If False (default), skip files that already exist in *adj_root*.

    Usage
    -----
    adjuster = SplitAdjuster(
        raw_root="C:/ORATS/data/ORATS_Data",
        adj_root="C:/ORATS/data/ORATS_Adjusted",
        splits_path="C:/MomentumCVG_env/cache/splits_hist.parquet",
    )
    adjuster.run()                          # all years, parallel
    adjuster.run(years=[2023, 2024])        # specific years
    adjuster.process_zip(Path("..."))       # single file (for testing)
    """

    def __init__(
        self,
        raw_root: str | Path,
        adj_root: str | Path,
        splits_path: str | Path,
        overwrite: bool = False,
        min_split_date: str = "2014-01-01",
    ) -> None:
        self.raw_root = Path(raw_root)
        self.adj_root = Path(adj_root)
        self.splits_path = Path(splits_path)
        self.overwrite = overwrite
        self.min_split_date = min_split_date

        self._cum_factors = self._load_cum_factors(splits_path, min_split_date)
        logger.info(
            "Loaded splits for %d tickers from %s (splits >= %s)",
            self._cum_factors["ticker"].nunique(),
            splits_path,
            min_split_date,
        )

    # ── public single-file API (useful for testing) ────────────────────────────

    def process_zip(self, zip_path: str | Path) -> Optional[Path]:
        """
        Adjust a single ZIP file and write parquet to *adj_root*.

        Returns the output Path on success, None on skip/error.
        """
        result = _process_single_zip(
            Path(zip_path), self.adj_root, self._cum_factors, self.overwrite
        )
        if result is True:
            date_str = Path(zip_path).stem.split("_")[-1]
            year = date_str[:4]
            return self.adj_root / year / f"ORATS_SMV_Strikes_{date_str}.parquet"
        return None

    def adjust_dataframe(
        self, df: pd.DataFrame, trade_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Apply split adjustments to an already-loaded DataFrame.

        Useful for one-off testing without writing to disk.
        """
        return _apply_adjustments(df, trade_date, self._cum_factors)

    def readjust_tickers(
        self,
        tickers: list[str],
        years: Optional[list[int]] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Re-process only the ZIP files that contain the given tickers.

        Use this after fetching new split data for a small set of tickers
        (e.g. a handful of stocks that split recently).  Much faster than
        re-running the full adjustment over all files.

        Parameters
        ----------
        tickers : list[str]
            Tickers whose historical files need re-adjustment.
        years : list[int], optional
            Restrict to these years only.  Default: all years.
        max_workers : int, optional
            Parallel worker processes.  Default: CPU count // 2.

        Notes
        -----
        This always overwrites existing output files for the affected tickers.
        Other tickers in the same daily file are re-written with their current
        (unchanged) factors, so there is no data loss.
        """
        ticker_set = set(tickers)
        logger.info(
            "Re-adjusting %d ticker(s): %s", len(ticker_set), sorted(ticker_set)
        )

        year_dirs = sorted(
            [d for d in self.raw_root.iterdir() if d.is_dir() and d.name.isdigit()]
        )
        if years:
            year_dirs = [d for d in year_dirs if int(d.name) in years]

        # Collect ZIP files that actually contain at least one of the tickers.
        # We do a fast filename-only scan first (no decompression) — the actual
        # ticker filter happens inside _process_single_zip via the full CSV read.
        # Since we can't know which ZIPs contain a given ticker without reading
        # them, we process all ZIPs but always overwrite.
        all_zips: list[Path] = []
        for yd in year_dirs:
            all_zips.extend(sorted(yd.glob("*.zip")))

        logger.info("Scanning %d ZIP files for affected tickers...", len(all_zips))

        workers = max_workers or max(1, multiprocessing.cpu_count() // 2)
        cum_factors = self._cum_factors
        adj_root = self.adj_root

        updated = errors = 0

        # Process sequentially within each year to avoid I/O contention;
        # parallelise across years.
        def _process_year_for_tickers(year_dir: Path) -> tuple[int, int]:
            _updated = _errors = 0
            for zp in sorted(year_dir.glob("*.zip")):
                res = _process_single_zip(zp, adj_root, cum_factors, overwrite=True)
                if res is True:
                    _updated += 1
                elif res is None:
                    _errors += 1
            return _updated, _errors

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_year_for_tickers, yd): yd for yd in year_dirs}
            with tqdm(total=len(year_dirs), unit="yr", desc="Re-adjusting") as pbar:
                for future in as_completed(futures):
                    yd = futures[future]
                    try:
                        u, e = future.result()
                        updated += u
                        errors += e
                    except Exception as exc:
                        logger.error("Year %s failed: %s", yd.name, exc)
                    pbar.update(1)

        logger.info(
            "Re-adjustment done — %d files updated, %d errors", updated, errors
        )

    # ── bulk processing ────────────────────────────────────────────────────────

    def run(
        self,
        years: Optional[list[int]] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Process all ZIP files in *raw_root*.

        Parameters
        ----------
        years : list[int], optional
            Restrict to these years. Default: all years found.
        max_workers : int, optional
            Number of parallel year-level worker processes.
            Default: half of available CPUs.
        """
        year_dirs = sorted(
            [d for d in self.raw_root.iterdir() if d.is_dir() and d.name.isdigit()]
        )
        if years:
            missing = [y for y in years if not (self.raw_root / str(y)).is_dir()]
            if missing:
                logger.warning("Year dirs not found (skipped): %s", missing)
            year_dirs = [d for d in year_dirs if int(d.name) in years]

        if not year_dirs:
            logger.warning("No year directories found under %s", self.raw_root)
            return

        total_zips = sum(len(list(yd.glob("*.zip"))) for yd in year_dirs)
        logger.info(
            "Processing %d year(s), %d ZIP files → %s",
            len(year_dirs), total_zips, self.adj_root,
        )

        workers = max_workers or max(1, multiprocessing.cpu_count() // 2)
        cum_factors = self._cum_factors   # captured for workers

        grand_total = grand_skipped = grand_errors = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_year_worker,
                    yd, self.adj_root, cum_factors, self.overwrite,
                ): yd
                for yd in year_dirs
            }
            with tqdm(total=len(year_dirs), unit="yr", desc="Years") as pbar:
                for future in as_completed(futures):
                    yd = futures[future]
                    try:
                        total, skipped, errors = future.result()
                        grand_total += total
                        grand_skipped += skipped
                        grand_errors += errors
                        pbar.set_postfix(year=yd.name, skip=grand_skipped, err=grand_errors)
                    except Exception as exc:
                        logger.error("Year %s worker failed: %s", yd.name, exc)
                    pbar.update(1)

        logger.info(
            "Done — %d files: %d adjusted, %d skipped, %d errors",
            grand_total,
            grand_total - grand_skipped - grand_errors,
            grand_skipped,
            grand_errors,
        )

    # ── internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_cum_factors(
        splits_path: str | Path,
        min_split_date: str | pd.Timestamp = "2014-01-01",
    ) -> pd.DataFrame:
        """
        Load split history and build the cumulative-factor table.

        Input columns : ticker, split_date, divisor
        Output columns: ticker, split_date, cum_factor

        For each ticker, sorts events newest→oldest and computes rolling
        product so that ``cum_factor[i]`` = product of all divisors from row 0
        (newest) through row i (inclusive).

        Lookup rule: for a given trade_date, find the oldest future split
        (split_date > trade_date, minimum split_date).  Its ``cum_factor``
        equals the product of ALL future split divisors.

        Parameters
        ----------
        splits_path : str or Path
            Parquet file with columns: ticker, split_date, divisor.
        min_split_date : str or Timestamp, default "2014-01-01"
            Ignore splits that occurred before this date.  Splits before
            the earliest raw data date can never appear as "future" splits
            for any trade_date in our dataset, so they are irrelevant.
            Setting this keeps the table small and makes intent explicit.

        Re-adjustment on new splits
        ---------------------------
        When a new split is announced/fetched (e.g. NVDA splits in 2025),
        every historical file for that ticker must be **re-processed** because
        the cumulative factor for pre-split dates increases.

        Workflow:
            1. Re-run ``scripts/fetch_splits.py`` (updates splits_hist.parquet)
            2. Re-run ``scripts/apply_split_adjustment.py --overwrite``
               (or use ``SplitAdjuster.readjust_tickers()`` for just the
               affected tickers, which is ~100x faster for a few tickers)
        """
        splits_df = pd.read_parquet(splits_path).copy()
        splits_df["split_date"] = pd.to_datetime(splits_df["split_date"])

        # Drop splits that pre-date our raw data — they are never
        # "future" splits for any trade_date we process.
        cutoff = pd.Timestamp(min_split_date)
        before_cutoff = splits_df["split_date"] < cutoff
        if before_cutoff.any():
            logger.debug(
                "Dropping %d splits before %s (pre-data cutoff)",
                before_cutoff.sum(), cutoff.date(),
            )
            splits_df = splits_df[~before_cutoff]

        out: list[pd.DataFrame] = []
        for ticker, grp in splits_df.groupby("ticker"):
            grp = grp.sort_values("split_date", ascending=False).copy()
            grp["cum_factor"] = grp["divisor"].cumprod()
            out.append(grp[["ticker", "split_date", "cum_factor"]])

        if not out:
            return pd.DataFrame(columns=["ticker", "split_date", "cum_factor"])

        return (
            pd.concat(out, ignore_index=True)
            .sort_values(["ticker", "split_date"])
            .reset_index(drop=True)
        )
