"""
Contract: S4 — step4_apply_exclusions.

See docs/surface_engine_data_contract.md § S4.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.pipeline import step3_get_eligible_structures, step4_apply_exclusions
from tests.contract.conftest import (
    CONTRACT_EXPIRY_IRON,
    CONTRACT_TICK_SHORT,
    CONTRACT_TRADE_DATE,
    make_contract_config,
)


@pytest.fixture
def structures_with_expiry(contract_surface_db, contract_signals_two_sides):
    cfg = make_contract_config()
    return step3_get_eligible_structures(
        CONTRACT_TRADE_DATE, contract_signals_two_sides, contract_surface_db, cfg
    )


def test_adds_had_earnings_nearby_column(structures_with_expiry):
    cfg = make_contract_config()
    out = step4_apply_exclusions(structures_with_expiry, None, cfg)
    assert "had_earnings_nearby" in out.columns
    assert len(out) == len(structures_with_expiry)


def test_no_earnings_table_all_false(structures_with_expiry):
    cfg = make_contract_config(earnings_exclusion_days=5)
    out = step4_apply_exclusions(structures_with_expiry, None, cfg)
    assert not out["had_earnings_nearby"].any()


def test_exclusion_days_zero_all_false(structures_with_expiry):
    earnings = pd.DataFrame(
        [
            {
                "ticker": CONTRACT_TICK_SHORT,
                "earnings_date": pd.Timestamp(CONTRACT_EXPIRY_IRON),
            }
        ]
    )
    cfg = make_contract_config(earnings_exclusion_days=0)
    out = step4_apply_exclusions(structures_with_expiry, earnings, cfg)
    assert not out["had_earnings_nearby"].any()


def test_earnings_inside_window_flags_ticker(structures_with_expiry):
    # Window [expiry - 5d, expiry] for SHORT1 iron fly expiry.
    earnings = pd.DataFrame(
        [
            {
                "ticker": CONTRACT_TICK_SHORT,
                "earnings_date": pd.Timestamp(CONTRACT_EXPIRY_IRON) - pd.Timedelta(days=2),
            },
            {
                "ticker": "OTHER",
                "earnings_date": pd.Timestamp(CONTRACT_EXPIRY_IRON) - pd.Timedelta(days=2),
            },
        ]
    )
    cfg = make_contract_config(earnings_exclusion_days=5)
    out = step4_apply_exclusions(structures_with_expiry, earnings, cfg)
    short = out[out["ticker"] == CONTRACT_TICK_SHORT].iloc[0]
    long_row = out[out["ticker"] != CONTRACT_TICK_SHORT].iloc[0]
    assert bool(short["had_earnings_nearby"]) is True
    assert bool(long_row["had_earnings_nearby"]) is False


def test_earnings_before_window_not_flagged(structures_with_expiry):
    earnings = pd.DataFrame(
        [
            {
                "ticker": CONTRACT_TICK_SHORT,
                "earnings_date": pd.Timestamp(CONTRACT_EXPIRY_IRON) - pd.Timedelta(days=10),
            }
        ]
    )
    cfg = make_contract_config(earnings_exclusion_days=5)
    out = step4_apply_exclusions(structures_with_expiry, earnings, cfg)
    short = out[out["ticker"] == CONTRACT_TICK_SHORT].iloc[0]
    assert bool(short["had_earnings_nearby"]) is False


def test_preserves_structure_columns(structures_with_expiry):
    cfg = make_contract_config()
    out = step4_apply_exclusions(structures_with_expiry, None, cfg)
    for col in structures_with_expiry.columns:
        assert col in out.columns
