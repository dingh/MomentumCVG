"""Unit tests for ORATS trading-day resolution (Sprint 004 C2).

These tests use synthetic ``exists_fn`` maps — no real ORATS mount required.
Each case documents the resolver contract operators rely on for ``--as-of``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from src.data.trading_day import (
    orats_daily_parquet_path,
    resolve_as_of_trading_day,
    resolve_weekly_entry_date,
    target_weekly_expiry_from_schedule,
    weekly_trade_dates_in_range,
)


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


class TestResolveWeeklyEntryDate:
    def test_friday_file_returns_friday(self):
        friday = date(2024, 1, 5)
        existing = {orats_daily_parquet_path(ROOT, friday)}

        resolved = resolve_weekly_entry_date(
            friday, ROOT, exists_fn=lambda path: path in existing
        )
        assert resolved == friday

    def test_missing_friday_with_thursday_returns_thursday(self):
        thursday = date(2024, 1, 4)
        friday = date(2024, 1, 5)
        existing = {orats_daily_parquet_path(ROOT, thursday)}

        resolved = resolve_weekly_entry_date(
            friday, ROOT, exists_fn=lambda path: path in existing
        )
        assert resolved == thursday

    def test_no_files_in_week_returns_none(self):
        friday = date(2024, 1, 5)
        assert (
            resolve_weekly_entry_date(friday, ROOT, exists_fn=lambda _p: False) is None
        )

    def test_non_friday_anchor_raises(self):
        with pytest.raises(ValueError, match="Expected a Friday anchor"):
            resolve_weekly_entry_date(date(2024, 1, 4), ROOT, exists_fn=lambda _p: False)


class TestWeeklyTradeDatesInRange:
    def _exists_for(self, days: set[date]):
        existing = {orats_daily_parquet_path(ROOT, day) for day in days}

        def exists_fn(path: Path) -> bool:
            return path in existing

        return exists_fn

    def test_normal_fridays_in_range(self):
        fridays = [date(2024, 1, 5), date(2024, 1, 12), date(2024, 1, 19)]
        schedule = weekly_trade_dates_in_range(
            date(2024, 1, 1),
            date(2024, 1, 31),
            ROOT,
            exists_fn=self._exists_for(set(fridays)),
        )
        assert schedule == fridays

    def test_holiday_friday_falls_back_to_thursday(self):
        thursday = date(2024, 1, 4)
        friday = date(2024, 1, 5)
        schedule = weekly_trade_dates_in_range(
            date(2024, 1, 1),
            date(2024, 1, 7),
            ROOT,
            exists_fn=self._exists_for({thursday}),
        )
        assert schedule == [thursday]

    def test_missing_week_is_omitted(self):
        friday_week1 = date(2024, 1, 5)
        schedule = weekly_trade_dates_in_range(
            date(2024, 1, 1),
            date(2024, 1, 14),
            ROOT,
            exists_fn=self._exists_for({friday_week1}),
        )
        assert schedule == [friday_week1]

    def test_inclusive_bounds_filter_resolved_dates(self):
        friday = date(2024, 1, 5)
        schedule = weekly_trade_dates_in_range(
            date(2024, 1, 5),
            date(2024, 1, 5),
            ROOT,
            exists_fn=self._exists_for({friday}),
        )
        assert schedule == [friday]

        schedule_before = weekly_trade_dates_in_range(
            date(2024, 1, 6),
            date(2024, 1, 10),
            ROOT,
            exists_fn=self._exists_for({friday}),
        )
        assert schedule_before == []

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="start must be on or before end"):
            weekly_trade_dates_in_range(
                date(2024, 2, 1),
                date(2024, 1, 1),
                ROOT,
                exists_fn=lambda _p: False,
            )


class TestTargetWeeklyExpiryFromSchedule:
    def test_returns_next_schedule_entry(self):
        schedule = [date(2024, 1, 5), date(2024, 1, 12), date(2024, 1, 19)]
        assert target_weekly_expiry_from_schedule(date(2024, 1, 5), schedule) == date(
            2024, 1, 12
        )

    def test_last_entry_returns_none(self):
        schedule = [date(2024, 1, 5), date(2024, 1, 12)]
        assert target_weekly_expiry_from_schedule(date(2024, 1, 12), schedule) is None

    def test_unknown_entry_returns_none(self):
        schedule = [date(2024, 1, 5), date(2024, 1, 12)]
        assert target_weekly_expiry_from_schedule(date(2024, 1, 6), schedule) is None
