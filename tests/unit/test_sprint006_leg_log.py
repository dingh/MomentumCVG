"""Sprint 006 D3 Commit 2 — included-trade leg completeness and reconciliation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.backtest.option_surface import build_straddle_from_surface
from src.backtest.surface_decision_report import (
    DecisionMetricsError,
    assert_included_trade_legs,
)
from src.backtest.surface_run_config import SurfaceDataPaths
from src.backtest.surface_runner import SurfaceRunner, serialize_constructable_legs
from tests.unit.test_surface_runner_data_flow import (
    TICK_LONG,
    TRADE_DATE,
    _build_features,
    _build_liquidity_panel,
    _build_surface_parquets,
    _make_config,
)


@pytest.fixture
def synthetic_runner(tmp_path: Path) -> SurfaceRunner:
    meta_path, quotes_path = _build_surface_parquets(tmp_path)
    liquidity_path = _build_liquidity_panel(tmp_path)
    features_dir = _build_features(tmp_path).parent
    return SurfaceRunner(
        data_paths=SurfaceDataPaths(
            cache_dir=tmp_path,
            features_dir=features_dir,
            liquidity_panel_path=liquidity_path,
            surface_meta_path=meta_path,
            surface_quotes_path=quotes_path,
            earnings_path=None,
        )
    )


def _included_trade_row(**overrides) -> dict:
    row = {
        "run_id": "t",
        "fill_label": "mid",
        "trade_date": TRADE_DATE,
        "ticker": "AAA",
        "direction": "long",
        "included_in_portfolio": True,
        "structure_ok": True,
        "instrument_type": "long_straddle",
        "entry_cost_per_share": 4.2,
        "pnl_per_share": -2.2,
        "pnl_total": -220.0,
    }
    row.update(overrides)
    return row


def _leg_row(*, leg_index: int, **overrides) -> dict:
    row = {
        "run_id": "t",
        "fill_label": "mid",
        "trade_date": TRADE_DATE,
        "ticker": "AAA",
        "direction": "long",
        "leg_index": leg_index,
        "included_in_portfolio": True,
        "entry_cash_per_unit": 2.1,
        "expiry_payoff_per_unit": 1.0,
        "pnl_per_unit": -1.1,
        "pnl_total_leg": -110.0,
    }
    row.update(overrides)
    return row


def test_missing_legs_abort():
    trades = pd.DataFrame([_included_trade_row()])
    legs = pd.DataFrame(columns=["trade_date", "ticker", "direction", "leg_index"])
    with pytest.raises(DecisionMetricsError, match="no matching leg rows"):
        assert_included_trade_legs(trades, legs, run_id="t", fill_label="mid")


def test_incomplete_straddle_legs_abort():
    trades = pd.DataFrame([_included_trade_row()])
    legs = pd.DataFrame([_leg_row(leg_index=0)])
    with pytest.raises(DecisionMetricsError, match="unexpected"):
        assert_included_trade_legs(trades, legs, run_id="t", fill_label="mid")


def test_duplicate_leg_index_aborts():
    trades = pd.DataFrame([_included_trade_row()])
    legs = pd.DataFrame([_leg_row(leg_index=0), _leg_row(leg_index=0)])
    with pytest.raises(DecisionMetricsError, match="duplicate"):
        assert_included_trade_legs(trades, legs, run_id="t", fill_label="mid")


def test_unexpected_instrument_aborts():
    trades = pd.DataFrame([_included_trade_row(instrument_type="iron_condor")])
    legs = pd.DataFrame([_leg_row(leg_index=i) for i in range(4)])
    with pytest.raises(DecisionMetricsError, match="unsupported instrument_type"):
        assert_included_trade_legs(trades, legs, run_id="t", fill_label="mid")


def test_structure_not_ok_included_aborts():
    trades = pd.DataFrame([_included_trade_row(structure_ok=False)])
    legs = pd.DataFrame([_leg_row(leg_index=0), _leg_row(leg_index=1)])
    with pytest.raises(DecisionMetricsError, match="structure_ok"):
        assert_included_trade_legs(trades, legs, run_id="t", fill_label="mid")


def test_reconciliation_mismatch_aborts():
    trades = pd.DataFrame([_included_trade_row(entry_cost_per_share=99.0)])
    legs = pd.DataFrame(
        [
            _leg_row(leg_index=0, entry_cash_per_unit=2.1, expiry_payoff_per_unit=1.0, pnl_per_unit=-1.1, pnl_total_leg=-110.0),
            _leg_row(leg_index=1, entry_cash_per_unit=2.1, expiry_payoff_per_unit=1.0, pnl_per_unit=-1.1, pnl_total_leg=-110.0),
        ]
    )
    with pytest.raises(DecisionMetricsError, match="entry cash"):
        assert_included_trade_legs(trades, legs, run_id="t", fill_label="mid")


def test_matching_straddle_legs_pass():
    trades = pd.DataFrame([_included_trade_row()])
    legs = pd.DataFrame(
        [
            _leg_row(leg_index=0),
            _leg_row(leg_index=1),
        ]
    )
    assert_included_trade_legs(trades, legs, run_id="t", fill_label="mid")


def test_iron_fly_requires_four_indices():
    trades = pd.DataFrame(
        [
            _included_trade_row(
                direction="short",
                instrument_type="iron_fly",
                entry_cost_per_share=-4.0,
                pnl_per_share=4.1,
                pnl_total=410.0,
            )
        ]
    )
    legs = pd.DataFrame(
        [
            _leg_row(
                leg_index=i,
                direction="short",
                entry_cash_per_unit=-1.0,
                expiry_payoff_per_unit=0.025,
                pnl_per_unit=1.025,
                pnl_total_leg=102.5,
            )
            for i in range(4)
        ]
    )
    assert_included_trade_legs(trades, legs, run_id="t", fill_label="mid")


def test_extra_unmatched_included_leg_key_aborts():
    trades = pd.DataFrame([_included_trade_row()])
    legs = pd.DataFrame(
        [
            _leg_row(leg_index=0),
            _leg_row(leg_index=1),
            _leg_row(leg_index=0, ticker="OTHER"),
            _leg_row(leg_index=1, ticker="OTHER"),
        ]
    )
    with pytest.raises(DecisionMetricsError, match="unmatched trade"):
        assert_included_trade_legs(trades, legs, run_id="t", fill_label="mid")


def test_excluded_constructable_has_unit_economics_null_portfolio(synthetic_runner):
    config = _make_config()
    assembly = build_straddle_from_surface(
        surface_db=synthetic_runner.surface_db,
        ticker=TICK_LONG,
        entry_date=TRADE_DATE,
        direction="long",
        fill=config.fill,
    )
    s5 = pd.DataFrame(
        [
            {
                "trade_date": TRADE_DATE,
                "ticker": TICK_LONG,
                "direction": "long",
                "structure_ok": True,
                "included_in_portfolio": False,
                "quantity": float("nan"),
                "exit_spot": 102.0,
                "_assembly": assembly,
            }
        ]
    )
    rows = serialize_constructable_legs(s5, config)
    assert len(rows) == 2
    for row in rows:
        assert row["included_in_portfolio"] is False
        assert row["portfolio_quantity"] is None
        assert row["pnl_total_leg"] is None
        assert row["entry_cash_per_unit"] is not None
        assert row["expiry_payoff_per_unit"] is not None
        assert row["pnl_per_unit"] is not None
        assert row["unit_quantity"] > 0
