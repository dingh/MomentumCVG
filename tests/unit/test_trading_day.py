"""Unit tests for ORATS trading-day resolution (Sprint 004 C2).

These tests use synthetic ``exists_fn`` maps — no real ORATS mount required.
Each case documents the resolver contract operators rely on for ``--as-of``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.data.trading_day import orats_daily_parquet_path, resolve_as_of_trading_day


# Fixed root for path construction assertions; existence is mocked per test.
ROOT = Path("C:/ORATS/data/ORATS_Adjusted")


class TestOratsDailyParquetPath:
    """Path builder must match the on-disk ORATS adjusted layout."""

    def test_returns_expected_pattern(self):
        day = date(2026, 6, 26)
        path = orats_daily_parquet_path(ROOT, day)
        assert path == ROOT / "2026" / "ORATS_SMV_Strikes_20260626.parquet"


class TestResolveAsOfTradingDay:
    """Walk-back resolver: latest calendar day on or before ``as_of`` with a file."""

    def test_friday_with_file_returns_same_date(self):
        # Baseline: operator passes a trading Friday and that day's parquet exists.
        friday = date(2026, 6, 26)
        existing = {orats_daily_parquet_path(ROOT, friday)}

        def exists_fn(path: Path) -> bool:
            return path in existing

        resolved = resolve_as_of_trading_day(friday, ROOT, exists_fn=exists_fn)
        assert resolved == friday

    def test_saturday_with_friday_file_returns_previous_friday(self):
        # Weekend operator date should resolve to the prior existing ORATS day.
        friday = date(2026, 6, 26)
        saturday = date(2026, 6, 27)
        existing = {orats_daily_parquet_path(ROOT, friday)}

        def exists_fn(path: Path) -> bool:
            return path in existing

        resolved = resolve_as_of_trading_day(saturday, ROOT, exists_fn=exists_fn)
        assert resolved == friday

    def test_missing_friday_with_thursday_file_returns_thursday(self):
        # Holiday / missing Friday: walk back to Thursday when only that file exists.
        thursday = date(2026, 6, 25)
        friday = date(2026, 6, 26)
        existing = {orats_daily_parquet_path(ROOT, thursday)}

        def exists_fn(path: Path) -> bool:
            return path in existing

        resolved = resolve_as_of_trading_day(friday, ROOT, exists_fn=exists_fn)
        assert resolved == thursday

    def test_invalid_iso_date_raises_value_error(self):
        # CLI maps this to exit 2 after ``build_cli_context`` catches ValueError.
        with pytest.raises(ValueError, match="Invalid ISO date"):
            resolve_as_of_trading_day("2026-13-40", ROOT, exists_fn=lambda _p: False)

    def test_no_file_in_lookback_raises_value_error(self):
        # Short lookback + no files → operator must fix as-of or ORATS root.
        with pytest.raises(ValueError, match="No ORATS adjusted daily parquet found"):
            resolve_as_of_trading_day(
                date(2026, 6, 26),
                ROOT,
                max_lookback_days=3,
                exists_fn=lambda _p: False,
            )

    def test_non_positive_max_lookback_raises_value_error(self):
        with pytest.raises(ValueError, match="max_lookback_days must be positive"):
            resolve_as_of_trading_day(date(2026, 6, 26), ROOT, max_lookback_days=0)

    def test_empty_orats_adj_root_raises_value_error(self):
        with pytest.raises(ValueError, match="orats_adj_root must be a non-empty path"):
            resolve_as_of_trading_day(date(2026, 6, 26), "   ", exists_fn=lambda _p: False)
