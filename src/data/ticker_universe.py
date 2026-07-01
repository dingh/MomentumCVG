"""Load ticker universe lists from CSV or parquet files (Sprint 004 C5.3).

Used by scoped split fetch, filtered split adjustment, and split output audit.
Accepts C4 ``liquid_tickers.csv`` (column ``Ticker``) or parquet files with
``ticker`` / ``Ticker`` columns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_SUPPORTED_EXTENSIONS = frozenset({".csv", ".parquet"})
_COUNT_COLUMN_NAMES = ("snapshots_qualified", "months_qualified")


def _resolve_ticker_column(columns: pd.Index) -> str:
    if "Ticker" in columns:
        return "Ticker"
    if "ticker" in columns:
        return "ticker"
    raise ValueError(
        "No ticker column found; expected 'Ticker' or 'ticker', "
        f"got {list(columns)}"
    )


def _resolve_count_column(columns: pd.Index) -> str:
    if "snapshots_qualified" in columns:
        return "snapshots_qualified"
    if "months_qualified" in columns:
        return "months_qualified"
    raise ValueError(
        "No qualification count column found; expected 'snapshots_qualified' "
        f"or 'months_qualified' when threshold is set, got {list(columns)}"
    )


def load_ticker_universe(
    path: str | Path,
    *,
    threshold: int | None = None,
) -> list[str]:
    """Load a cleaned, deduplicated, alphabetically sorted ticker list.

    Parameters
    ----------
    path:
        CSV (``.csv``) or parquet (``.parquet``) file containing a ``Ticker`` or
        ``ticker`` column. When both columns exist, ``Ticker`` is preferred.
    threshold:
        When set, keep only tickers whose ``snapshots_qualified`` count is
        greater than or equal to this value. ``months_qualified`` is used as a
        fallback when ``snapshots_qualified`` is absent (C4 alias). Ignored
        when ``None``.

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
        count column when ``threshold`` is set, or no valid tickers after
        cleaning (null, blank, or whitespace-only values are dropped).
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

    if threshold is not None:
        count_col = _resolve_count_column(df.columns)
        counts = pd.to_numeric(df[count_col], errors="coerce")
        df = df[counts >= threshold]

    col = _resolve_ticker_column(df.columns)
    series = df[col].dropna().astype(str).str.strip()
    series = series[series != ""].str.upper()
    tickers = sorted(series.unique())

    if not tickers:
        raise ValueError(f"No valid tickers remain after cleaning: {file_path}")

    return tickers
