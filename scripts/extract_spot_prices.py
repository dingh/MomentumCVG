"""
Extract historical spot prices for all tickers from adjusted ORATS chains.

Sprint 004 C8.2 — hardened, fail-closed spot extraction.

This script builds a pre-computed database of daily spot prices (raw and
split-adjusted) from the adjusted daily parquet store. It exits successfully
and publishes an output parquet only when:

1. Every expected adjusted-chain date (discovered from filenames under the
   requested year range) was processed successfully. Weekend-dated files are
   excluded from the expected inventory when empty (non-trading days); a
   weekend file that contains data fails the run.
2. The combined output contains exactly one valid spot row for every
   retained normalized ticker in every adjusted daily file. A ticker-date
   whose repeated source spot values disagree beyond the tolerance (e.g.
   cash-index tickers carrying per-expiry forward levels) is dropped from
   the output with a logged warning; the downstream effect is simply that
   no surface is extracted for that ticker-date. A date where every ticker
   is dropped still fails the run.
3. Every published spot value is finite and strictly positive.
4. The output was written through a same-directory temporary parquet,
   read back, revalidated, and atomically published via ``os.replace``.

Failure never publishes a partial dataset and never damages an existing
output file.

Output schema (column names are a compatibility contract with
``src/data/spot_price_db.py`` and ``scripts/precompute_option_surface.py``):

    date, ticker, adj_spot_price, spot_price

Exit codes:
    0 = complete valid output published
    1 = extraction, validation, or output-write failure
    2 = invalid arguments or invalid path configuration

Usage:
    python scripts/extract_spot_prices.py --year 2024
    python scripts/extract_spot_prices.py \
        --data-root <adjusted-root> --output <out.parquet> \
        --start-year 2018 --end-year 2026
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.orats_provider import ORATSDataProvider
from src.data.paths import DEFAULT_ADJUSTED_LIQUID_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# C8.2 spot contract tolerance: repeated underlying spot fields in one daily
# chain should be numerically identical apart from serialization noise.
SPOT_CONSISTENCY_REL_TOL = 1e-9
SPOT_CONSISTENCY_ABS_TOL = 1e-8

REQUIRED_SOURCE_COLUMNS = ("ticker", "stkPx", "adj_stkPx")
OUTPUT_COLUMNS = ("date", "ticker", "adj_spot_price", "spot_price")

_ADJUSTED_FILENAME_RE = re.compile(r"^ORATS_SMV_Strikes_(\d{8})\.parquet$")


class SpotExtractionError(RuntimeError):
    """Raised for any data, validation, or output-write failure (exit 1)."""


class UsageError(RuntimeError):
    """Raised for invalid arguments or path configuration (exit 2)."""


@dataclass(frozen=True)
class DateSpotResult:
    """Validated one-date extraction result.

    ``expected_tickers`` is the retained ticker set (after dropping tickers
    with inconsistent repeated spot values); ``dropped_tickers`` records the
    ticker-date entries excluded from the output for that reason.
    """

    trade_date: date
    expected_tickers: frozenset[str]
    records: tuple[dict[str, object], ...]
    dropped_tickers: frozenset[str] = frozenset()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract historical spot prices from adjusted ORATS chains"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(DEFAULT_ADJUSTED_LIQUID_ROOT),
        help=(
            "Path to split-adjusted daily parquet root "
            f"(default: {DEFAULT_ADJUSTED_LIQUID_ROOT})"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="C:/MomentumCVG_env/cache/spot_prices_adjusted.parquet",
        help="Output parquet file path",
    )
    parser.add_argument("--start-year", type=int, help="Start year (inclusive)")
    parser.add_argument("--end-year", type=int, help="End year (inclusive)")
    parser.add_argument(
        "--year",
        type=int,
        help="Single year to process (shorthand for --start-year Y --end-year Y)",
    )
    return parser.parse_args(argv)


def ensure_output_outside_data_root(
    data_root: Path,
    output_path: Path,
) -> None:
    """Reject an output path that resolves inside the adjusted input root.

    The extractor must never overwrite or create files anywhere inside the
    adjusted-chain input root (including the root itself, any year directory,
    and any adjusted daily parquet). Uses resolved path ancestry, not string
    prefixes, so sibling directories with a shared prefix remain valid.

    Raises:
        UsageError: if the resolved output equals or lies inside the resolved
            data root.
    """
    resolved_data_root = data_root.resolve()
    resolved_output = output_path.resolve(strict=False)

    try:
        resolved_output.relative_to(resolved_data_root)
    except ValueError:
        return

    raise UsageError(
        f"output must be outside data root: {resolved_output}"
    )


def resolve_year_range(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve and validate the requested year range.

    Raises:
        UsageError: if --year is combined with --start-year/--end-year or the
            resolved range is inverted.
    """
    if args.year is not None:
        if args.start_year is not None or args.end_year is not None:
            raise UsageError(
                "--year cannot be combined with --start-year or --end-year"
            )
        return args.year, args.year

    start_year = args.start_year if args.start_year is not None else 2018
    end_year = args.end_year if args.end_year is not None else 2026

    if start_year > end_year:
        raise UsageError(
            f"start year {start_year} must be <= end year {end_year}"
        )
    return start_year, end_year


def discover_adjusted_dates(
    data_root: Path,
    start_year: int,
    end_year: int,
) -> list[date]:
    """Discover expected trading dates from adjusted parquet filenames.

    Weekend-dated files (Saturday/Sunday) are not trading days: they are
    excluded from the expected inventory after verifying they contain no
    rows. A weekend file with data fails the run so real rows can never be
    silently skipped. Year-directory membership is validated before weekend
    exclusion, so an empty weekend file in the wrong year directory still
    fails.

    Fails closed on a missing data root, a missing requested year directory,
    a malformed filename, a duplicate date, a file date outside its containing
    year directory, a non-empty weekend file, or an empty inventory.
    """
    if not data_root.is_dir():
        raise SpotExtractionError(f"data root does not exist: {data_root}")

    discovered: list[tuple[date, int]] = []
    weekend_excluded: list[date] = []
    for year in range(start_year, end_year + 1):
        year_dir = data_root / str(year)
        if not year_dir.is_dir():
            raise SpotExtractionError(
                f"requested year directory missing: {year_dir}"
            )
        for file_path in sorted(year_dir.glob("ORATS_SMV_Strikes_*.parquet")):
            match = _ADJUSTED_FILENAME_RE.match(file_path.name)
            if match is None:
                raise SpotExtractionError(
                    f"malformed adjusted filename: {file_path}"
                )
            try:
                trade_date = datetime.strptime(match.group(1), "%Y%m%d").date()
            except ValueError as exc:
                raise SpotExtractionError(
                    f"invalid date in adjusted filename: {file_path}"
                ) from exc
            # Year membership must be checked before weekend exclusion so an
            # empty weekend file in the wrong year directory still fails.
            if trade_date.year != year:
                raise SpotExtractionError(
                    f"adjusted file date {trade_date.isoformat()} does not belong "
                    f"to its containing year directory {year}"
                )
            if trade_date.weekday() >= 5:  # Saturday/Sunday: not a trading day
                try:
                    weekend_frame = pd.read_parquet(file_path)
                except Exception as exc:
                    raise SpotExtractionError(
                        f"failed to read weekend-dated file {file_path}: {exc}"
                    ) from exc
                if not weekend_frame.empty:
                    raise SpotExtractionError(
                        f"weekend-dated file contains data: {file_path}"
                    )
                weekend_excluded.append(trade_date)
                continue
            discovered.append((trade_date, year))

    if weekend_excluded:
        logger.info(
            "Excluded %d empty weekend-dated files from the expected "
            "inventory: %s",
            len(weekend_excluded),
            ", ".join(d.isoformat() for d in sorted(weekend_excluded)),
        )

    if not discovered:
        raise SpotExtractionError(
            f"no adjusted dates discovered under {data_root} "
            f"for years {start_year}-{end_year}"
        )

    seen: set[date] = set()
    for trade_date, _ in discovered:
        if trade_date in seen:
            raise SpotExtractionError(
                f"duplicate adjusted date discovered: {trade_date.isoformat()}"
            )
        seen.add(trade_date)

    return sorted(d for d, _ in discovered)


def _normalize_tickers(raw: pd.Series, trade_date: date) -> pd.Series:
    """Normalize tickers (strip + uppercase) and reject null/blank/'nan'."""
    if raw.isna().any():
        raise SpotExtractionError(
            f"{trade_date.isoformat()}: null ticker value in source frame"
        )
    normalized = raw.astype(str).str.strip().str.upper()
    if (normalized == "").any():
        raise SpotExtractionError(
            f"{trade_date.isoformat()}: blank ticker value in source frame"
        )
    if (normalized == "NAN").any():
        raise SpotExtractionError(
            f"{trade_date.isoformat()}: 'nan' ticker produced from missing "
            "values in source frame"
        )
    return normalized


def _validated_numeric(
    raw: pd.Series, column: str, trade_date: date
) -> pd.Series:
    """Convert a spot column to numeric; reject null/non-numeric/nonfinite/<=0."""
    values = pd.to_numeric(raw, errors="coerce")
    if values.isna().any():
        raise SpotExtractionError(
            f"{trade_date.isoformat()}: null or non-numeric {column} value"
        )
    values = values.astype(float)
    if np.isinf(values.to_numpy()).any():
        raise SpotExtractionError(
            f"{trade_date.isoformat()}: infinite {column} value"
        )
    if (values <= 0).any():
        raise SpotExtractionError(
            f"{trade_date.isoformat()}: zero or negative {column} value"
        )
    return values


def _find_inconsistent_tickers(frame: pd.DataFrame, column: str) -> set[str]:
    """Return tickers whose source rows disagree on ``column``.

    Values are already finite and positive. Each value is compared against
    the ticker's first value with ``math.isclose``. Exactly-equal repeated
    values (the expected production case for equities) take a vectorized
    fast path. Cash-index tickers routinely carry per-expiry forward levels
    in this column, so disagreement marks the ticker-date for exclusion
    rather than failing the whole date.
    """
    inconsistent: set[str] = set()
    for ticker, values in frame.groupby("ticker")[column]:
        arr = values.to_numpy(dtype=float)
        reference = float(arr[0])
        if (arr == reference).all():
            continue
        for value in arr:
            if not math.isclose(
                value,
                reference,
                rel_tol=SPOT_CONSISTENCY_REL_TOL,
                abs_tol=SPOT_CONSISTENCY_ABS_TOL,
            ):
                inconsistent.add(str(ticker))
                break
    return inconsistent


def extract_spot_prices_for_date(
    provider: ORATSDataProvider,
    trade_date: date,
) -> DateSpotResult:
    """Extract exactly one validated spot record per normalized ticker.

    Tickers whose repeated source spot values disagree beyond the tolerance
    (raw or adjusted) are dropped from the result and reported via
    ``dropped_tickers``; every retained ticker produces exactly one record.

    Raises:
        SpotExtractionError: on read failure, empty/missing frame, missing
            columns, invalid tickers, invalid values, or when every ticker
            on the date is dropped. Failures are never converted into empty
            results.
    """
    try:
        df = provider._load_day_data(trade_date)
    except SpotExtractionError:
        raise
    except Exception as exc:
        raise SpotExtractionError(
            f"{trade_date.isoformat()}: failed to load adjusted daily frame: "
            f"{exc}"
        ) from exc

    if df is None or df.empty:
        raise SpotExtractionError(
            f"{trade_date.isoformat()}: adjusted daily frame is empty or missing"
        )

    missing_columns = [c for c in REQUIRED_SOURCE_COLUMNS if c not in df.columns]
    if missing_columns:
        raise SpotExtractionError(
            f"{trade_date.isoformat()}: missing required source columns: "
            f"{', '.join(missing_columns)}"
        )

    work = pd.DataFrame(
        {
            "ticker": _normalize_tickers(df["ticker"], trade_date),
            "stkPx": _validated_numeric(df["stkPx"], "stkPx", trade_date),
            "adj_stkPx": _validated_numeric(
                df["adj_stkPx"], "adj_stkPx", trade_date
            ),
        }
    )

    # Repeated-value consistency must be established before selecting one
    # row per ticker. Inconsistent tickers (e.g. cash indices with
    # per-expiry forward levels) are dropped from this date's output.
    dropped = _find_inconsistent_tickers(work, "stkPx")
    dropped |= _find_inconsistent_tickers(work, "adj_stkPx")

    if dropped:
        logger.warning(
            "%s: dropping %d ticker(s) with inconsistent repeated spot "
            "values: %s",
            trade_date.isoformat(),
            len(dropped),
            ", ".join(sorted(dropped)),
        )
        work = work[~work["ticker"].isin(dropped)]

    if work.empty:
        raise SpotExtractionError(
            f"{trade_date.isoformat()}: no valid tickers remain after "
            "dropping inconsistent spot values"
        )

    per_ticker = work.groupby("ticker", sort=True)[["adj_stkPx", "stkPx"]].first()

    expected_tickers = frozenset(work["ticker"])
    records = tuple(
        {
            "date": trade_date,
            "ticker": ticker,
            "adj_spot_price": float(row["adj_stkPx"]),
            "spot_price": float(row["stkPx"]),
        }
        for ticker, row in per_ticker.iterrows()
    )

    result_tickers = frozenset(r["ticker"] for r in records)
    if result_tickers != expected_tickers:
        raise SpotExtractionError(
            f"{trade_date.isoformat()}: extracted ticker set does not equal "
            "source ticker set"
        )

    return DateSpotResult(
        trade_date=trade_date,
        expected_tickers=expected_tickers,
        records=records,
        dropped_tickers=frozenset(dropped),
    )


def validate_complete_spot_frame(
    frame: pd.DataFrame,
    expected_dates: list[date],
    expected_tickers_by_date: dict[date, set[str]],
) -> pd.DataFrame:
    """Independently validate the combined spot frame before any write.

    Returns the frame sorted by (date, ticker) with a reset index.

    Raises:
        SpotExtractionError: on any schema, null, uniqueness, value, or
            date/ticker completeness violation. Invalid rows are never
            silently dropped.
    """
    if frame is None or frame.empty:
        raise SpotExtractionError("combined spot frame is empty")

    actual_columns = list(frame.columns)
    if set(actual_columns) != set(OUTPUT_COLUMNS):
        raise SpotExtractionError(
            f"combined spot frame has wrong schema: {actual_columns} "
            f"(expected {list(OUTPUT_COLUMNS)})"
        )

    for column in OUTPUT_COLUMNS:
        if frame[column].isna().any():
            raise SpotExtractionError(
                f"combined spot frame has null values in column {column}"
            )

    tickers = frame["ticker"]
    normalized = tickers.astype(str).str.strip().str.upper()
    if (normalized == "").any():
        raise SpotExtractionError("combined spot frame has blank tickers")
    if not tickers.equals(normalized):
        raise SpotExtractionError(
            "combined spot frame has non-normalized ticker values"
        )

    for column in ("adj_spot_price", "spot_price"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or np.isinf(values.to_numpy()).any():
            raise SpotExtractionError(
                f"combined spot frame has non-finite {column} values"
            )
        if (values <= 0).any():
            raise SpotExtractionError(
                f"combined spot frame has non-positive {column} values"
            )

    if frame.duplicated(subset=["date", "ticker"]).any():
        raise SpotExtractionError(
            "combined spot frame has duplicate (date, ticker) rows"
        )

    actual_dates = set(frame["date"])
    expected_date_set = set(expected_dates)
    missing_dates = expected_date_set - actual_dates
    unexpected_dates = actual_dates - expected_date_set
    if missing_dates:
        raise SpotExtractionError(
            "combined spot frame is missing expected dates: "
            + ", ".join(sorted(d.isoformat() for d in missing_dates))
        )
    if unexpected_dates:
        raise SpotExtractionError(
            "combined spot frame has unexpected dates: "
            + ", ".join(sorted(d.isoformat() for d in unexpected_dates))
        )

    if set(expected_tickers_by_date.keys()) != expected_date_set:
        raise SpotExtractionError(
            "expected ticker map does not cover the expected date set"
        )

    for trade_date, group in frame.groupby("date"):
        actual_tickers = set(group["ticker"])
        expected_tickers = expected_tickers_by_date[trade_date]
        missing = expected_tickers - actual_tickers
        unexpected = actual_tickers - expected_tickers
        if missing:
            raise SpotExtractionError(
                f"{trade_date.isoformat()}: missing expected tickers "
                f"({len(missing)}), e.g. {sorted(missing)[:5]}"
            )
        if unexpected:
            raise SpotExtractionError(
                f"{trade_date.isoformat()}: unexpected tickers "
                f"({len(unexpected)}), e.g. {sorted(unexpected)[:5]}"
            )

    result = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
    return result[list(OUTPUT_COLUMNS)]


def read_parquet_for_validation(path: Path) -> pd.DataFrame:
    """Read the temporary parquet back for pre-publication verification."""
    return pd.read_parquet(path)


def _verify_readback(frame: pd.DataFrame, readback: pd.DataFrame) -> None:
    """Verify the read-back parquet matches the validated in-memory frame."""
    if list(readback.columns) != list(OUTPUT_COLUMNS):
        raise SpotExtractionError(
            f"read-back parquet has wrong columns: {list(readback.columns)}"
        )
    if len(readback) != len(frame):
        raise SpotExtractionError(
            f"read-back parquet row count {len(readback)} does not match "
            f"expected {len(frame)}"
        )

    rb = readback.copy()
    rb["date"] = pd.to_datetime(rb["date"]).dt.date

    if rb.duplicated(subset=["date", "ticker"]).any():
        raise SpotExtractionError(
            "read-back parquet has duplicate (date, ticker) rows"
        )

    rb = rb.sort_values(["date", "ticker"]).reset_index(drop=True)
    if list(rb["date"]) != list(frame["date"]):
        raise SpotExtractionError("read-back parquet date values do not match")
    if list(rb["ticker"]) != list(frame["ticker"]):
        raise SpotExtractionError("read-back parquet ticker values do not match")

    for column in ("adj_spot_price", "spot_price"):
        rb_values = rb[column].to_numpy(dtype=float)
        if not np.isfinite(rb_values).all() or (rb_values <= 0).any():
            raise SpotExtractionError(
                f"read-back parquet has invalid {column} values"
            )
        if not np.array_equal(rb_values, frame[column].to_numpy(dtype=float)):
            raise SpotExtractionError(
                f"read-back parquet {column} values do not match"
            )


def write_parquet_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    """Write the validated frame via temp file + read-back + ``os.replace``.

    On any failure the temporary file (if it was created) is removed and an
    existing output file is left untouched. Output-directory creation,
    temporary write, read-back validation, and the ``os.replace`` publication
    all share the same controlled error handling.
    """
    temp_path: Path | None = None

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = output_path.parent / (
            f"{output_path.stem}.tmp-{uuid.uuid4().hex}.parquet"
        )

        frame.to_parquet(temp_path, index=False, compression="snappy")
        readback = read_parquet_for_validation(temp_path)
        _verify_readback(frame, readback)
        os.replace(temp_path, output_path)
    except SpotExtractionError:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise SpotExtractionError(
            f"failed to write output parquet {output_path}: {exc}"
        ) from exc


def log_output_size_best_effort(output_path: Path) -> None:
    """Log the published output size; never fail after successful publication.

    The output has already been atomically published when this runs, so a
    ``stat`` failure must not turn the success into an apparent failure.
    """
    try:
        output_size_mb = output_path.stat().st_size / 1024**2
    except OSError as exc:
        logger.warning(
            "Output published successfully, but size could not be read: %s",
            exc,
        )
    else:
        logger.info("  Output size: %.2f MB", output_size_mb)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        start_year, end_year = resolve_year_range(args)

        data_root = Path(args.data_root)
        if not data_root.is_dir():
            raise UsageError(
                f"data root does not exist or is not a directory: {data_root}"
            )

        output_path = Path(args.output)
        if output_path.suffix.lower() != ".parquet":
            raise UsageError(
                f"output must be a parquet path: {output_path}"
            )

        ensure_output_outside_data_root(data_root, output_path)
    except UsageError as exc:
        logger.error("usage error: %s", exc)
        return 2

    logger.info("Extracting spot prices from %d to %d", start_year, end_year)
    logger.info("Data root: %s", data_root)
    logger.info("Output: %s", output_path)

    try:
        expected_dates = discover_adjusted_dates(data_root, start_year, end_year)
    except SpotExtractionError as exc:
        logger.error("date inventory failure: %s", exc)
        return 1

    logger.info("Found %d expected trading dates", len(expected_dates))
    logger.info(
        "Date range: %s to %s",
        expected_dates[0].isoformat(),
        expected_dates[-1].isoformat(),
    )

    provider = ORATSDataProvider(
        data_root=str(data_root),
        min_volume=0,           # No filters - we want all tickers
        min_open_interest=0,
        min_bid=0.0,
        max_spread_pct=1.0,
        cache_size=1,           # Only cache 1 file at a time
    )

    results: list[DateSpotResult] = []
    failures: list[tuple[date, str]] = []

    for trade_date in tqdm(expected_dates, desc="Extracting spot prices"):
        try:
            results.append(extract_spot_prices_for_date(provider, trade_date))
        except SpotExtractionError as exc:
            failures.append((trade_date, str(exc)))
        except Exception as exc:  # pragma: no cover - defensive catch-all
            failures.append((trade_date, f"unexpected error: {exc}"))
        finally:
            provider.clear_cache()

    if failures:
        logger.error(
            "extraction failed for %d of %d expected dates; "
            "no output was created or replaced",
            len(failures),
            len(expected_dates),
        )
        for failed_date, reason in failures:
            logger.error("  %s: %s", failed_date.isoformat(), reason)
        return 1

    expected_tickers_by_date = {
        result.trade_date: set(result.expected_tickers) for result in results
    }
    all_records = [record for result in results for record in result.records]
    combined = pd.DataFrame(all_records, columns=list(OUTPUT_COLUMNS))

    try:
        validated = validate_complete_spot_frame(
            combined, expected_dates, expected_tickers_by_date
        )
        write_parquet_atomically(validated, output_path)
    except SpotExtractionError as exc:
        logger.error("output validation/write failure: %s", exc)
        return 1

    dropped_entries = sum(len(result.dropped_tickers) for result in results)
    dropped_unique = sorted(
        {ticker for result in results for ticker in result.dropped_tickers}
    )

    logger.info("Extraction summary:")
    logger.info("  Expected dates: %d", len(expected_dates))
    logger.info("  Processed dates: %d", len(results))
    logger.info("  Rows: %d", len(validated))
    logger.info("  Tickers: %d", validated["ticker"].nunique())
    if dropped_entries:
        logger.info(
            "  Dropped inconsistent ticker-date entries: %d "
            "(%d unique tickers: %s)",
            dropped_entries,
            len(dropped_unique),
            ", ".join(dropped_unique[:20])
            + (", ..." if len(dropped_unique) > 20 else ""),
        )
    logger.info(
        "  Date range: %s to %s",
        expected_dates[0].isoformat(),
        expected_dates[-1].isoformat(),
    )
    logger.info("  Output: %s", output_path)
    log_output_size_best_effort(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
