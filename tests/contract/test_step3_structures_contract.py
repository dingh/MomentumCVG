"""
Contract: S3 — step3_get_eligible_structures.

See docs/surface_engine_data_contract.md § S3.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.backtest.option_surface import StrategyAssemblyResult
from src.backtest.pipeline import step3_get_eligible_structures
from tests.contract.conftest import (
    CONTRACT_IRON_NET_CREDIT,
    CONTRACT_TICK_BAD,
    CONTRACT_TICK_LONG,
    CONTRACT_TICK_SHORT,
    CONTRACT_TRADE_DATE,
    SIGNALS_OUT_COLS,
    make_contract_config,
)


def test_one_row_per_signal_preserves_signal_columns(
    contract_surface_db, contract_signals_two_sides
):
    cfg = make_contract_config()
    out = step3_get_eligible_structures(
        CONTRACT_TRADE_DATE, contract_signals_two_sides, contract_surface_db, cfg
    )
    assert len(out) == len(contract_signals_two_sides)
    for col in SIGNALS_OUT_COLS:
        assert col in out.columns


def test_step3_does_not_set_earnings_flag(
    contract_surface_db, contract_signals_two_sides
):
    cfg = make_contract_config()
    out = step3_get_eligible_structures(
        CONTRACT_TRADE_DATE, contract_signals_two_sides, contract_surface_db, cfg
    )
    assert "had_earnings_nearby" not in out.columns


def test_invalid_surface_metadata_error(
    contract_surface_db, contract_signals_with_bad_surface
):
    cfg = make_contract_config()
    out = step3_get_eligible_structures(
        CONTRACT_TRADE_DATE, contract_signals_with_bad_surface, contract_surface_db, cfg
    )
    row = out.iloc[0]
    assert row["ticker"] == CONTRACT_TICK_BAD
    assert bool(row["structure_ok"]) is False
    assert "_assembly" not in row or row.get("_assembly") is None
    assert "metadata_error" in str(row["failure_reason"])


def test_short_iron_fly_build_ok_and_assembly(
    contract_surface_db, contract_signals_two_sides
):
    cfg = make_contract_config(short_structure="ironfly")
    out = step3_get_eligible_structures(
        CONTRACT_TRADE_DATE, contract_signals_two_sides, contract_surface_db, cfg
    )
    short = out[out["ticker"] == CONTRACT_TICK_SHORT].iloc[0]
    assert bool(short["structure_ok"]) is True
    assert short["instrument_type"] == "iron_fly"
    assert short["_assembly"] is not None
    assert isinstance(short["_assembly"], StrategyAssemblyResult)
    assert short["net_credit_per_share"] == pytest.approx(CONTRACT_IRON_NET_CREDIT)
    assert short["max_loss_per_share"] == pytest.approx(0.90)
    assert short["trade_date"] == CONTRACT_TRADE_DATE


def test_long_routes_to_long_straddle(
    contract_surface_db, contract_signals_two_sides
):
    cfg = make_contract_config()
    out = step3_get_eligible_structures(
        CONTRACT_TRADE_DATE, contract_signals_two_sides, contract_surface_db, cfg
    )
    long_row = out[out["ticker"] == CONTRACT_TICK_LONG].iloc[0]
    assert bool(long_row["structure_ok"]) is True
    assert long_row["instrument_type"] == "long_straddle"
    assert long_row["exit_spot"] == pytest.approx(102.0)


def test_settle_on_assembly_matches_exit_spot_golden(
    contract_surface_db, contract_signals_two_sides
):
    """L2: iron fly settle at meta exit_spot → hand-calculated +4.10 per share."""
    cfg = make_contract_config()
    out = step3_get_eligible_structures(
        CONTRACT_TRADE_DATE, contract_signals_two_sides, contract_surface_db, cfg
    )
    short = out[out["ticker"] == CONTRACT_TICK_SHORT].iloc[0]
    pos = short["_assembly"].settle(exit_spot=Decimal(str(short["exit_spot"])))
    assert float(pos.pnl) == pytest.approx(CONTRACT_IRON_NET_CREDIT)


def test_empty_signals_returns_empty_frame(
    contract_surface_db, contract_signals_two_sides
):
    cfg = make_contract_config()
    empty = contract_signals_two_sides.iloc[0:0]
    out = step3_get_eligible_structures(
        CONTRACT_TRADE_DATE, empty, contract_surface_db, cfg
    )
    assert out.empty
