"""Load ticker universe lists from CSV or parquet files (Sprint 004 C5.3).

Used by scoped split fetch, filtered split adjustment, and split output audit.
Accepts C4 ``liquid_tickers.csv`` (column ``Ticker``) or parquet files with
``ticker`` / ``Ticker`` columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_SUPPORTED_EXTENSIONS = frozenset({".csv", ".parquet"})
_QUALIFICATION_COLUMN_NAMES = ("snapshots_qualified", "months_qualified")


def _resolve_ticker_column(columns: pd.Index) -> str:
    if "Ticker" in columns:
        return "Ticker"
    if "ticker" in columns:
        return "ticker"
    raise ValueError(
        "No ticker column found; expected 'Ticker' or 'ticker', "
        f"got {list(columns)}"
    )


def _resolve_qualification_column(columns: pd.Index) -> str:
    for name in _QUALIFICATION_COLUMN_NAMES:
        if name in columns:
            return name
    raise ValueError(
        "No qualification column found; expected one of "
        f"{list(_QUALIFICATION_COLUMN_NAMES)} when min_snapshots_qualified "
        f"is set, got {list(columns)}"
    )


def load_ticker_universe(
    path: str | Path,
    *,
    min_snapshots_qualified: int | None = None,
) -> list[str]:
    """Load a cleaned, deduplicated, alphabetically sorted ticker list.

    By default (``min_snapshots_qualified=None``), no qualification filtering
    is applied. The full C4 ``liquid_tickers.csv`` historical precompute
    universe is loaded — every ticker row in the file after standard cleaning.

    ``min_snapshots_qualified`` is an explicit optional narrowing filter. Use
    it only when HD intentionally wants a smaller precompute universe than the
    full C4 superset. When set, rows are kept only when the qualification
    count is greater than or equal to this value. Filtering uses
    ``snapshots_qualified`` when that column is present; ``months_qualified``
    is used only as a fallback when ``snapshots_qualified`` is absent (C4
    legacy alias).

    Parameters
    ----------
    path:
        CSV (``.csv``) or parquet (``.parquet``) file containing a ``Ticker`` or
        ``ticker`` column. When both columns exist, ``Ticker`` is preferred.
    min_snapshots_qualified:
        Optional minimum week-qualification count. ``None`` (default) loads the
        full universe with no count-based filtering.

    Returns
    -------
    list[str]
        Uppercase ticker symbols, sorted alphabetically.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        Unsupported file extension, missing ticker column, missing qualification
        column when ``min_snapshots_qualified`` is set, or no valid tickers
        after cleaning (null, blank, or whitespace-only values are dropped).
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Ticker universe file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported ticker universe file extension {suffix!r}; "
            f"expected one of {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_parquet(file_path)

    if min_snapshots_qualified is not None:
        qual_col = _resolve_qualification_column(df.columns)
        counts = pd.to_numeric(df[qual_col], errors="coerce")
        df = df[counts >= min_snapshots_qualified]

    col = _resolve_ticker_column(df.columns)
    series = df[col].dropna().astype(str).str.strip()
    series = series[series != ""].str.upper()
    tickers = sorted(series.unique())

    if not tickers:
        raise ValueError(f"No valid tickers remain after cleaning: {file_path}")

    return tickers
