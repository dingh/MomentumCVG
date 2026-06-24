"""ORATS daily parquet path helpers and ``--as-of`` trading-day resolution (Sprint 004 C2).

Background
----------
The weekly input CLI accepts an operator ``--as-of`` calendar date (often a Friday,
but not always). Downstream Stage A scripts need the *latest ORATS adjusted daily
store* that exists on or before that date — not a separate exchange calendar.

This module centralizes that rule so the CLI, future audits (C6), and tests share
one resolver. It intentionally does **not** import ``get_trading_fridays`` from
precompute scripts; holidays and weekends are inferred from parquet presence.

Contract (HD-004-2)
-------------------
``resolve_as_of_trading_day`` returns the greatest calendar day ``t`` such that
``t <= as_of`` and the expected daily parquet exists under ``orats_adj_root``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path


def orats_daily_parquet_path(orats_adj_root: Path | str, day: date) -> Path:
    """Build the canonical on-disk path for one ORATS adjusted daily SMV file.

    Layout matches the existing ORATS adjusted cache tree used by Stage A scripts::

        {orats_adj_root}/{YYYY}/ORATS_SMV_Strikes_{YYYYMMDD}.parquet

    Example::

        C:/ORATS/data/ORATS_Adjusted/2026/ORATS_SMV_Strikes_20260626.parquet
    """
    # Reject datetime — callers must pass a plain date so path keys stay unambiguous.
    if isinstance(day, datetime) or not isinstance(day, date):
        raise ValueError(f"Expected date, got {day!r}")

    root = Path(orats_adj_root)
    yyyy = f"{day.year:04d}"
    yyyymmdd = day.strftime("%Y%m%d")
    return root / yyyy / f"ORATS_SMV_Strikes_{yyyymmdd}.parquet"


def resolve_as_of_trading_day(
    as_of: date | str,
    orats_adj_root: Path | str,
    *,
    max_lookback_days: int = 10,
    exists_fn: Callable[[Path], bool] | None = None,
) -> date:
    """Resolve ``as_of`` to the latest ORATS day with a daily parquet on disk.

    Algorithm
    ---------
    1. Parse ``as_of`` (ISO ``YYYY-MM-DD`` if string).
    2. For ``offset`` in ``0 .. max_lookback_days - 1``, check
       ``as_of - offset`` days via ``orats_daily_parquet_path``.
    3. Return the first candidate whose path passes ``exists_fn``.
    4. If none exist, raise ``ValueError`` (CLI maps this to exit code 2).

    Weekends / holidays
        No NYSE calendar is used. If Friday is a holiday, Thursday's file wins when
        the operator passes Saturday or the holiday itself — same pattern as existing
        precompute scripts that scan for the first existing parquet in a range.

    Args:
        as_of: Operator request date (``date`` or ISO string).
        orats_adj_root: Root of the ORATS *adjusted* daily parquet store.
        max_lookback_days: Calendar days to walk back (default 10 per HD-C2-2).
        exists_fn: Injectable existence check; defaults to ``Path.is_file``.
            Unit tests pass a synthetic map so no real ORATS mount is required.

    Raises:
        ValueError: Invalid date, empty root, bad lookback, or no file in window.
    """
    if max_lookback_days <= 0:
        raise ValueError(f"max_lookback_days must be positive; got {max_lookback_days}")

    root_text = str(orats_adj_root).strip()
    if not root_text:
        raise ValueError("orats_adj_root must be a non-empty path")

    as_of_day = _parse_as_of(as_of)
    # Production uses real filesystem checks; tests inject a stub map.
    checker = exists_fn if exists_fn is not None else Path.is_file

    for offset in range(max_lookback_days):
        candidate = as_of_day - timedelta(days=offset)
        path = orats_daily_parquet_path(orats_adj_root, candidate)
        if checker(path):
            return candidate

    raise ValueError(
        "No ORATS adjusted daily parquet found within "
        f"{max_lookback_days} calendar day(s) on or before {as_of_day.isoformat()} "
        f"under {root_text}"
    )


def _parse_as_of(as_of: date | str) -> date:
    """Normalize CLI / resolver input to a plain ``date`` (not ``datetime``)."""
    if isinstance(as_of, date) and not isinstance(as_of, datetime):
        return as_of
    if isinstance(as_of, str):
        try:
            return date.fromisoformat(as_of)
        except ValueError as exc:
            raise ValueError(f"Invalid ISO date for as_of: {as_of!r}") from exc
    raise ValueError(f"Invalid as_of value: {as_of!r}")
