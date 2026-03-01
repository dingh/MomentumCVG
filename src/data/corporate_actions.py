"""
src/data/corporate_actions.py

Fetch and store earnings + split history from the ORATS datav2 historical API.

API docs:
  - Earnings: https://api.orats.io/datav2/hist/earnings?token=...&ticker=...
  - Splits:   https://api.orats.io/datav2/hist/splits?token=...&ticker=...

Typical usage (from scripts/fetch_orats.py):
    fetcher = OratsCorporateActionsFetcher()          # reads ORATS_API_TOKEN env var
    splits  = fetcher.fetch_all_splits(tickers, checkpoint_path=SPLITS_PATH)
    earnings = fetcher.fetch_all_earnings(tickers, checkpoint_path=EARNINGS_PATH)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from time import sleep
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── default output paths ──────────────────────────────────────────────────────
DEFAULT_CACHE_DIR    = Path("C:/MomentumCVG_env/cache")
DEFAULT_SPLITS_PATH  = DEFAULT_CACHE_DIR / "splits_hist.parquet"
DEFAULT_EARNINGS_PATH = DEFAULT_CACHE_DIR / "earnings_hist.parquet"

# ── API base URL ───────────────────────────────────────────────────────────────
_BASE_URL = "https://api.orats.io/datav2/hist"


class OratsCorporateActionsFetcher:
    """
    Fetch earnings + split history from the ORATS datav2 historical endpoints.

    Token resolution order:
      1. `token` kwarg (explicit)
      2. ``ORATS_API_TOKEN`` environment variable

    Parameters
    ----------
    token : str, optional
        ORATS API token. If omitted, reads from ``ORATS_API_TOKEN`` env var.
    rate_limit : float
        Seconds to sleep between ticker requests (default 0.7 s).
    max_retries : int
        Maximum retry attempts per request on 429 / 5xx responses.
    timeout : int
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        rate_limit: float = 0.7,
        max_retries: int = 5,
        timeout: int = 30,
    ) -> None:
        resolved = token # or os.environ.get("ORATS_API_TOKEN")
        if not resolved:
            raise ValueError(
                "ORATS API token not provided. Pass `token=` or set the "
                "ORATS_API_TOKEN environment variable."
            )
        self.token = resolved
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()

    # ── HTTP layer ─────────────────────────────────────────────────────────────

    def _get(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """GET with retries + exponential backoff (handles 429 / 5xx)."""
        for attempt in range(self.max_retries):
            try:
                r = self.session.get(url, params=params, timeout=self.timeout)

                if r.status_code == 200:
                    return r.json()

                if r.status_code in (429, 500, 502, 503, 504):
                    backoff = min(60.0, (2 ** attempt) * 1.0)
                    retry_after = r.headers.get("Retry-After")
                    if retry_after:
                        try:
                            backoff = max(backoff, float(retry_after))
                        except ValueError:
                            pass
                    logger.warning(
                        "HTTP %s — backing off %.1fs (attempt %d/%d)",
                        r.status_code, backoff, attempt + 1, self.max_retries,
                    )
                    time.sleep(backoff)
                    continue

                # Non-retryable error
                logger.error(
                    "HTTP %s for %s params=%s body=%s",
                    r.status_code, url, params, r.text[:200],
                )
                return None

            except requests.RequestException as exc:
                backoff = min(60.0, (2 ** attempt) * 1.0)
                logger.warning(
                    "Request error: %s — backing off %.1fs (attempt %d/%d)",
                    exc, backoff, attempt + 1, self.max_retries,
                )
                time.sleep(backoff)

        logger.error("Failed after %d retries: %s params=%s", self.max_retries, url, params)
        return None

    # ── per-ticker fetchers ────────────────────────────────────────────────────

    def fetch_splits_for_ticker(self, ticker: str) -> pd.DataFrame:
        """Return split history for a single ticker as a DataFrame.

        Columns: ticker, split_date, divisor
        """
        url = f"{_BASE_URL}/splits"
        payload = self._get(url, {"token": self.token, "ticker": ticker})

        if not payload or "data" not in payload or not payload["data"]:
            return pd.DataFrame()

        rows = [
            {
                "ticker":     row.get("ticker", ticker),
                "split_date": pd.to_datetime(row.get("splitDate")).date(),
                "divisor":    pd.to_numeric(row.get("divisor"), errors="coerce"),
            }
            for row in payload["data"]
        ]
        return pd.DataFrame(rows)

    def fetch_earnings_for_ticker(self, ticker: str) -> pd.DataFrame:
        """Return earnings history for a single ticker as a DataFrame.

        Columns: ticker, earn_date, annc_tod, updated_at
        """
        url = f"{_BASE_URL}/earnings"
        payload = self._get(url, {"token": self.token, "ticker": ticker})

        if not payload or "data" not in payload or not payload["data"]:
            return pd.DataFrame()

        rows = [
            {
                "ticker":     row.get("ticker", ticker),
                "earn_date":  pd.to_datetime(row.get("earnDate")).date(),
                "annc_tod":   row.get("anncTod"),          # e.g. "1630" (HHMM string)
                "updated_at": pd.to_datetime(row.get("updatedAt"), errors="coerce"),
            }
            for row in payload["data"]
        ]
        return pd.DataFrame(rows)

    # ── checkpoint helpers ─────────────────────────────────────────────────────

    def _load_checkpoint(
        self, checkpoint_path: Optional[str | Path]
    ) -> tuple[list[pd.DataFrame], set[str]]:
        """Load an existing parquet checkpoint and return (dfs, processed_tickers)."""
        dfs: list[pd.DataFrame] = []
        processed: set[str] = set()

        if checkpoint_path and Path(checkpoint_path).exists():
            existing = pd.read_parquet(checkpoint_path)
            dfs.append(existing)
            if "ticker" in existing.columns:
                processed = set(existing["ticker"].unique())
            logger.info(
                "Loaded %d previously processed tickers from checkpoint: %s",
                len(processed), checkpoint_path,
            )
        return dfs, processed

    @staticmethod
    def _save_parquet(df: pd.DataFrame, path: str | Path, dedup_cols: list[str]) -> None:
        """Deduplicate and save DataFrame to parquet, creating directories as needed."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = df.drop_duplicates(subset=dedup_cols)
        df.to_parquet(path, index=False)
        logger.info("Saved %d rows to %s", len(df), path)

    # ── bulk fetchers ──────────────────────────────────────────────────────────

    def fetch_all_splits(
        self,
        tickers: List[str],
        checkpoint_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """Fetch split history for all tickers with checkpoint/resume support.

        Parameters
        ----------
        tickers : list[str]
            Universe of tickers to fetch.
        checkpoint_path : str or Path, optional
            If provided, existing results are loaded as a starting point and
            results are saved here after each ticker (and on interrupt).

        Returns
        -------
        pd.DataFrame
            Combined splits DataFrame with columns: ticker, split_date, divisor.
        """
        all_rows, processed = self._load_checkpoint(checkpoint_path)

        try:
            for ticker in tqdm(tickers, desc="Fetching ORATS splits"):
                if ticker in processed:
                    continue
                df = self.fetch_splits_for_ticker(ticker)
                if not df.empty:
                    all_rows.append(df)
                processed.add(ticker)
                sleep(self.rate_limit)

        except KeyboardInterrupt:
            logger.warning("Interrupted — saving checkpoint before exit.")

        finally:
            combined = (
                pd.concat(all_rows, ignore_index=True)
                .sort_values(["ticker", "split_date"])
                if all_rows
                else pd.DataFrame(columns=["ticker", "split_date", "divisor"])
            )
            if checkpoint_path:
                self._save_parquet(
                    combined, checkpoint_path, dedup_cols=["ticker", "split_date"]
                )

        return combined

    def fetch_all_earnings(
        self,
        tickers: List[str],
        checkpoint_path: Optional[str | Path] = None,
    ) -> pd.DataFrame:
        """Fetch earnings history for all tickers with checkpoint/resume support.

        Parameters
        ----------
        tickers : list[str]
            Universe of tickers to fetch.
        checkpoint_path : str or Path, optional
            If provided, existing results are loaded as a starting point and
            results are saved here after each ticker (and on interrupt).

        Returns
        -------
        pd.DataFrame
            Combined earnings DataFrame with columns:
            ticker, earn_date, annc_tod, updated_at.
        """
        all_rows, processed = self._load_checkpoint(checkpoint_path)

        try:
            for ticker in tqdm(tickers, desc="Fetching ORATS earnings"):
                if ticker in processed:
                    continue
                df = self.fetch_earnings_for_ticker(ticker)
                if not df.empty:
                    all_rows.append(df)
                processed.add(ticker)
                sleep(self.rate_limit)

        except KeyboardInterrupt:
            logger.warning("Interrupted — saving checkpoint before exit.")

        finally:
            combined = (
                pd.concat(all_rows, ignore_index=True)
                .sort_values(["ticker", "earn_date"])
                if all_rows
                else pd.DataFrame(
                    columns=["ticker", "earn_date", "annc_tod", "updated_at"]
                )
            )
            if checkpoint_path:
                self._save_parquet(
                    combined, checkpoint_path, dedup_cols=["ticker", "earn_date"]
                )

        return combined


# ── universe helpers ───────────────────────────────────────────────────────────

def get_all_unique_tickers(
    data_root: str | Path,
    output_path: Optional[str | Path] = None,
    ticker_col: str = "ticker",
) -> List[str]:
    """Scan every ZIP in a partitioned ORATS raw-data store and return unique tickers.

    Walks ``data_root/{YYYY}/ORATS_SMV_Strikes_*.zip``, opens each ZIP in-memory,
    and reads only the ticker column from the CSV inside for efficiency.
    Intended as a one-time scan; results can be persisted via ``output_path``.

    Parameters
    ----------
    data_root : str or Path
        Root directory of the ORATS raw data store,
        e.g. ``C:/ORATS/data/ORATS_Data``.
    output_path : str or Path, optional
        If provided, saves the resulting ticker list as a single-column parquet
        file at this path (directories are created as needed).
    ticker_col : str
        Name of the ticker column inside the CSV (default ``"ticker"``).

    Returns
    -------
    list[str]
        Sorted list of unique tickers found across all files.
    """
    import zipfile

    data_root = Path(data_root)
    tickers: set[str] = set()

    year_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    # Pre-count total ZIP files across all years so the outer bar shows accurate ETA
    all_zip_files: list[tuple[Path, list[Path]]] = []
    for year_dir in year_dirs:
        zips = sorted(year_dir.glob("*.zip"))
        if zips:
            all_zip_files.append((year_dir, zips))

    total_files = sum(len(zips) for _, zips in all_zip_files)

    # Temp file sits next to the final output (or in cwd if no output_path given)
    if output_path:
        temp_path = Path(output_path).with_suffix(".tmp.parquet")
    else:
        temp_path = data_root / "_tickers_tmp.parquet"

    outer = tqdm(all_zip_files, desc="Years", unit="yr", position=0, leave=True)
    inner = tqdm(total=total_files, desc="Files", unit="file", position=1, leave=True)

    try:
        for year_dir, zip_files in outer:
            outer.set_postfix(year=year_dir.name, tickers=len(tickers))

            for zip_path in zip_files:
                inner.set_postfix(file=zip_path.name)
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        csv_names = [n for n in zf.namelist() if n.endswith((".csv", ".txt"))]
                        if not csv_names:
                            logger.warning("No CSV/TXT found inside %s", zip_path.name)
                        else:
                            with zf.open(csv_names[0]) as f:
                                df = pd.read_csv(f, usecols=[ticker_col], dtype={ticker_col: str})
                                tickers.update(df[ticker_col].dropna().unique())
                except Exception as exc:
                    logger.warning("Failed to read %s: %s", zip_path.name, exc)
                finally:
                    inner.update(1)

            # ── save incremental result after each year ────────────────────────
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"ticker": sorted(tickers)}).to_parquet(temp_path, index=False)
            logger.info("Year %s done — %d tickers so far (temp saved)", year_dir.name, len(tickers))

    finally:
        outer.close()
        inner.close()

    result = sorted(tickers)
    logger.info("Scan complete — %d unique tickers across %s", len(result), data_root)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticker": result}).to_parquet(output_path, index=False)
        logger.info("Ticker list saved to %s", output_path)
        # Clean up temp file once final output is written
        if temp_path.exists():
            temp_path.unlink()

    return result
