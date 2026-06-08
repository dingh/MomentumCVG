"""
Contract: S7 — StrategyAssemblyResult.settle (hold to expiry).

Settlement is invoked from the runner's select/size path via assembly.settle(exit_spot).
These tests pin payoff sign conventions on the same synthetic surface as S3.

See docs/surface_engine_data_contract.md § S7.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from src.backtest.pipeline import step3_get_eligible_structures
from tests.contract.conftest import (
    CONTRACT_EXIT_IRON,
    CONTRACT_EXIT_STRADDLE,
    CONTRACT_IRON_NET_CREDIT,
    CONTRACT_LONG_SETTLE_PNL,
    CONTRACT_TICK_LONG,
    CONTRACT_TICK_SHORT,
    CONTRACT_TRADE_DATE,
    make_contract_config,
)


def test_iron_fly_settle_at_exit_spot_from_meta(
    contract_surface_db, contract_signals_two_sides
):
    cfg = make_contract_config()
    structures = step3_get_eligible_structures(
        CONTRACT_TRADE_DATE, contract_signals_two_sides, contract_surface_db, cfg
    )
    short = structures[structures["ticker"] == CONTRACT_TICK_SHORT].iloc[0]
    assembly = short["_assembly"]
    pos = assembly.settle(exit_spot=Decimal(str(CONTRACT_EXIT_IRON)))
    assert pos.pnl is not None
    assert float(pos.pnl) == pytest.approx(CONTRACT_IRON_NET_CREDIT)


def test_long_straddle_settle_at_exit_spot_from_meta(
    contract_surface_db, contract_signals_two_sides
):
    cfg = make_contract_config()
    structures = step3_get_eligible_structures(
        CONTRACT_TRADE_DATE, contract_signals_two_sides, contract_surface_db, cfg
    )
    long_row = structures[structures["ticker"] == CONTRACT_TICK_LONG].iloc[0]
    assembly = long_row["_assembly"]
    pos = assembly.settle(exit_spot=Decimal(str(CONTRACT_EXIT_STRADDLE)))
    assert float(pos.pnl) == pytest.approx(CONTRACT_LONG_SETTLE_PNL)


def test_settle_uses_expiry_as_default_exit_date(
    contract_surface_db, contract_signals_two_sides
):
    cfg = make_contract_config()
    structures = step3_get_eligible_structures(
        CONTRACT_TRADE_DATE, contract_signals_two_sides, contract_surface_db, cfg
    )
    short = structures[structures["ticker"] == CONTRACT_TICK_SHORT].iloc[0]
    pos = short["_assembly"].settle(exit_spot=Decimal(str(CONTRACT_EXIT_IRON)))
    assert pos.exit_date == short["expiry_date"]
