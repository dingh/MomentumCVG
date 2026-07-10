"""Unit tests for C6.1C strict calendar-paired weekly expiry."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.core.models import OptionQuote
from src.features.option_surface_analyzer import OptionSurfaceBuilder


SCHEDULE = [date(2024, 1, 5), date(2024, 1, 12), date(2024, 1, 19)]


def _quote(
    *,
    strike: str = "100",
    option_type: str = "call",
    bid: str = "1.0",
    ask: str = "1.2",
    expiry: date = date(2024, 1, 12),
) -> OptionQuote:
    bid_dec = Decimal(bid)
    ask_dec = Decimal(ask)
    return OptionQuote(
        ticker="AAPL",
        trade_date=date(2024, 1, 5),
        expiry_date=expiry,
        strike=Decimal(strike),
        option_type=option_type,
        bid=bid_dec,
        ask=ask_dec,
        mid=(bid_dec + ask_dec) / 2,
        iv=0.2,
        delta=0.5 if option_type == "call" else -0.5,
        gamma=0.01,
        vega=0.1,
        theta=-0.05,
        volume=100,
        open_interest=1000,
    )


@pytest.fixture
def spot_db() -> MagicMock:
    db = MagicMock()
    db.calculate_spot_move_pct.return_value = 0.01
    db.calculate_realized_volatility.return_value = 0.2
    return db


@pytest.fixture
def weekly_builder(spot_db: MagicMock) -> OptionSurfaceBuilder:
    builder = OptionSurfaceBuilder(
        data_root="C:/unused",
        spot_db=spot_db,
        dte_target=7,
        frequency="weekly",
        weekly_schedule=SCHEDULE,
    )
    builder.provider = MagicMock()
    return builder


@pytest.fixture
def monthly_builder(spot_db: MagicMock) -> OptionSurfaceBuilder:
    builder = OptionSurfaceBuilder(
        data_root="C:/unused",
        spot_db=spot_db,
        dte_target=30,
        frequency="monthly",
    )
    builder.provider = MagicMock()
    return builder


class TestStrictWeeklyExpirySelection:
    def test_exact_target_expiry_is_selected(self, weekly_builder: OptionSurfaceBuilder) -> None:
        weekly_builder.provider.get_spot_price.side_effect = (
            lambda ticker, day: Decimal("100")
        )
        weekly_builder.provider.get_available_expiries.return_value = [
            date(2024, 1, 12),
            date(2024, 1, 19),
        ]
        weekly_builder.provider.get_option_chain.return_value = [
            _quote(option_type="call"),
            _quote(option_type="put"),
        ]

        meta, quotes = weekly_builder.process_single_entry("AAPL", date(2024, 1, 5))

        assert meta["expiry_date"] == date(2024, 1, 12)
        assert meta["surface_valid"] is True
        assert meta["failure_reason"] is None
        assert len(quotes) >= 2
        weekly_builder.provider.get_option_chain.assert_called_with(
            ticker="AAPL",
            trade_date=date(2024, 1, 5),
            expiry_date=date(2024, 1, 12),
        )

    def test_no_fallback_when_target_missing(self, weekly_builder: OptionSurfaceBuilder) -> None:
        weekly_builder.provider.get_spot_price.return_value = Decimal("100")
        # Nearby 2024-01-19 exists, but exact target 2024-01-12 does not.
        weekly_builder.provider.get_available_expiries.return_value = [
            date(2024, 1, 19),
            date(2024, 1, 26),
        ]

        meta, quotes = weekly_builder.process_single_entry("AAPL", date(2024, 1, 5))

        assert meta["surface_valid"] is False
        assert meta["failure_reason"] == "target_weekly_expiry_not_listed"
        assert meta["expiry_date"] is None
        assert quotes == []
        weekly_builder.provider.get_option_chain.assert_not_called()

    def test_no_expiries_on_entry_chain(self, weekly_builder: OptionSurfaceBuilder) -> None:
        weekly_builder.provider.get_spot_price.return_value = Decimal("100")
        weekly_builder.provider.get_available_expiries.return_value = []

        meta, quotes = weekly_builder.process_single_entry("AAPL", date(2024, 1, 5))

        assert meta["surface_valid"] is False
        assert meta["failure_reason"] == "no_expiries_on_entry_chain"
        assert quotes == []

    def test_no_successor_schedule(self, spot_db: MagicMock) -> None:
        builder = OptionSurfaceBuilder(
            data_root="C:/unused",
            spot_db=spot_db,
            dte_target=7,
            frequency="weekly",
            weekly_schedule=[date(2024, 1, 5)],  # last item — no schedule[i+1]
        )
        builder.provider = MagicMock()
        builder.provider.get_spot_price.return_value = Decimal("100")
        builder.provider.get_available_expiries.return_value = [date(2024, 1, 12)]

        meta, quotes = builder.process_single_entry("AAPL", date(2024, 1, 5))

        assert meta["surface_valid"] is False
        assert meta["failure_reason"] == "no_target_weekly_expiry"
        assert quotes == []
        builder.provider.get_available_expiries.assert_not_called()

    def test_target_listed_but_body_not_quotable(
        self, weekly_builder: OptionSurfaceBuilder
    ) -> None:
        weekly_builder.provider.get_spot_price.side_effect = (
            lambda ticker, day: Decimal("100")
        )
        weekly_builder.provider.get_available_expiries.return_value = [date(2024, 1, 12)]
        weekly_builder.provider.get_option_chain.return_value = [
            _quote(option_type="call", bid="0", ask="1.2"),
            _quote(option_type="put", bid="1.0", ask="1.2"),
        ]

        meta, quotes = weekly_builder.process_single_entry("AAPL", date(2024, 1, 5))

        assert meta["expiry_date"] == date(2024, 1, 12)
        assert meta["has_body_call"] is False
        assert meta["has_body_put"] is True
        assert meta["surface_valid"] is False
        assert meta["failure_reason"] == "target_weekly_body_not_quotable"


class TestMonthlyExpiryUnchanged:
    def test_monthly_still_uses_find_best_expiry(
        self, monthly_builder: OptionSurfaceBuilder
    ) -> None:
        monthly_builder.provider.get_spot_price.side_effect = (
            lambda ticker, day: Decimal("100")
        )
        # Standard monthly path: Thu/Fri in next calendar month
        monthly_expiry = date(2024, 2, 16)  # Friday in February
        monthly_builder.provider.get_available_expiries.return_value = [
            date(2024, 1, 19),
            monthly_expiry,
            date(2024, 3, 15),
        ]
        monthly_builder.provider.get_option_chain.return_value = [
            _quote(option_type="call", expiry=monthly_expiry),
            _quote(option_type="put", expiry=monthly_expiry),
        ]

        meta, _quotes = monthly_builder.process_single_entry("AAPL", date(2024, 1, 5))

        assert meta["expiry_date"] == monthly_expiry
        assert meta["failure_reason"] is None
        assert meta["surface_valid"] is True

    def test_find_best_expiry_weekly_path_still_exists(
        self, weekly_builder: OptionSurfaceBuilder
    ) -> None:
        # Legacy permissive helper remains for comparison / monthly callers.
        weekly_builder.provider.get_available_expiries.return_value = [
            date(2024, 1, 11),  # Thursday
            date(2024, 1, 12),  # Friday exact 7 DTE
        ]
        result = weekly_builder._find_best_expiry("AAPL", date(2024, 1, 5), target_dte=7)
        assert result == date(2024, 1, 12)
