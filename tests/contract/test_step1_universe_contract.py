"""
Contract: S1 — step1_get_universe (PIT liquidity universe).

Invariants pinned here:
  I1. Point-in-time snapshot = the latest month_date <= trade_date.
  I2. Rows with has_valid_atm_pair=False (or NaN metrics) are excluded.
  I3. dvol and spread filters are applied with AND logic.
  I4. Output schema is exactly [ticker, dvol_rank_pct, spread_rank_pct].
  I5. No future snapshot leaks (trade_date before any snapshot → empty).

See docs/surface_engine_data_contract.md § S1.
"""
from __future__ import annotations

from datetime import date

from src.backtest.pipeline import step1_get_universe
from tests.contract.conftest import UNIVERSE_OUT_COLS, make_contract_config


def test_output_schema(liquidity_panel_two_snapshots):
    cfg = make_contract_config()
    out = step1_get_universe(date(2024, 1, 15), liquidity_panel_two_snapshots, cfg)
    assert list(out.columns) == UNIVERSE_OUT_COLS


def test_pit_uses_latest_snapshot_at_or_before_trade_date(liquidity_panel_two_snapshots):
    cfg = make_contract_config()
    # Jan trade date → Jan snapshot (A, B, C; D dropped for invalid ATM pair).
    jan = step1_get_universe(date(2024, 1, 15), liquidity_panel_two_snapshots, cfg)
    assert set(jan["ticker"]) == {"A", "B", "C"}
    # Feb trade date → Feb snapshot (A, B only).
    feb = step1_get_universe(date(2024, 2, 15), liquidity_panel_two_snapshots, cfg)
    assert set(feb["ticker"]) == {"A", "B"}


def test_invalid_atm_pair_excluded(liquidity_panel_two_snapshots):
    cfg = make_contract_config()
    out = step1_get_universe(date(2024, 1, 15), liquidity_panel_two_snapshots, cfg)
    # D had has_valid_atm_pair=False in the Jan snapshot.
    assert "D" not in set(out["ticker"])


def test_and_filter_intersects_both_metrics(liquidity_panel_two_snapshots):
    # Keep top 50% by dollar volume; spread filter open. Among Jan {A,B,C}
    # (D already dropped), dvol ranks are A=1.0, B=0.667, C=0.333 → keep A, B.
    cfg = make_contract_config(dvol_top_pct=0.5, spread_bottom_pct=1.0)
    out = step1_get_universe(date(2024, 1, 15), liquidity_panel_two_snapshots, cfg)
    assert set(out["ticker"]) == {"A", "B"}


def test_no_future_snapshot_leak(liquidity_panel_two_snapshots):
    cfg = make_contract_config()
    # Trade date before the earliest snapshot → empty universe, correct schema.
    out = step1_get_universe(date(2023, 12, 1), liquidity_panel_two_snapshots, cfg)
    assert out.empty
    assert list(out.columns) == UNIVERSE_OUT_COLS


def test_rank_pct_in_unit_interval(liquidity_panel_two_snapshots):
    cfg = make_contract_config()
    out = step1_get_universe(date(2024, 1, 15), liquidity_panel_two_snapshots, cfg)
    assert ((out["dvol_rank_pct"] >= 0) & (out["dvol_rank_pct"] <= 1)).all()
    assert ((out["spread_rank_pct"] >= 0) & (out["spread_rank_pct"] <= 1)).all()
