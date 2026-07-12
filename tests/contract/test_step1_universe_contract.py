"""
Contract: S1 — step1_get_universe (PIT liquidity universe).

Invariants pinned here:
  I1. Point-in-time snapshot = the latest global month_date STRICTLY BEFORE
      trade_date (max(month_date < trade_date)). Same-day and future snapshots
      are never selected (C7.2 strict prior-snapshot rule).
  I2. Rows with has_valid_atm_pair=False (or NaN metrics) are excluded.
  I3. dvol and spread filters are applied with AND logic.
  I4. Output schema is exactly [ticker, dvol_rank_pct, spread_rank_pct].
  I5. trade_date on or before every snapshot → empty universe.

See docs/surface_engine_data_contract.md § S1.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src.backtest.pipeline import step1_get_universe
from tests.contract.conftest import UNIVERSE_OUT_COLS, make_contract_config


def test_output_schema(liquidity_panel_two_snapshots):
    cfg = make_contract_config()
    out = step1_get_universe(date(2024, 1, 15), liquidity_panel_two_snapshots, cfg)
    assert list(out.columns) == UNIVERSE_OUT_COLS


def test_pit_selects_latest_prior_snapshot(liquidity_panel_two_snapshots):
    cfg = make_contract_config()
    # Jan trade date → Jan snapshot (A, B, C; D dropped for invalid ATM pair).
    jan = step1_get_universe(date(2024, 1, 15), liquidity_panel_two_snapshots, cfg)
    assert set(jan["ticker"]) == {"A", "B", "C"}
    # Feb trade date (after Feb snapshot) → Feb snapshot (A, B only).
    feb = step1_get_universe(date(2024, 2, 15), liquidity_panel_two_snapshots, cfg)
    assert set(feb["ticker"]) == {"A", "B"}


def test_same_day_snapshot_not_selected(liquidity_panel_two_snapshots):
    cfg = make_contract_config()
    # trade_date == Feb snapshot date (2024-02-01). Strict `<` excludes the
    # same-day Feb snapshot and falls back to the prior Jan snapshot {A, B, C}.
    out = step1_get_universe(date(2024, 2, 1), liquidity_panel_two_snapshots, cfg)
    assert set(out["ticker"]) == {"A", "B", "C"}


def test_friday_trade_uses_prior_friday_snapshot():
    cfg = make_contract_config()
    # Two Friday snapshots; a Friday trade on the second must use the first.
    fri_1 = pd.Timestamp("2024-02-02")
    fri_2 = pd.Timestamp("2024-02-09")
    panel = pd.DataFrame(
        [
            dict(month_date=fri_1, ticker="A", atm_straddle_dollar_vol=4_000_000, atm_spread_pct=0.010, has_valid_atm_pair=True),
            dict(month_date=fri_1, ticker="B", atm_straddle_dollar_vol=3_000_000, atm_spread_pct=0.012, has_valid_atm_pair=True),
            dict(month_date=fri_2, ticker="C", atm_straddle_dollar_vol=5_000_000, atm_spread_pct=0.009, has_valid_atm_pair=True),
            dict(month_date=fri_2, ticker="D", atm_straddle_dollar_vol=4_500_000, atm_spread_pct=0.011, has_valid_atm_pair=True),
        ]
    )
    panel["month_date"] = pd.to_datetime(panel["month_date"])
    # Friday 2024-02-09 trade → prior Friday 2024-02-02 snapshot {A, B}.
    out = step1_get_universe(date(2024, 2, 9), panel, cfg)
    assert set(out["ticker"]) == {"A", "B"}


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


def test_trade_date_on_first_snapshot_returns_empty(liquidity_panel_two_snapshots):
    cfg = make_contract_config()
    # trade_date == earliest snapshot (2024-01-01). No month_date < trade_date → empty.
    out = step1_get_universe(date(2024, 1, 1), liquidity_panel_two_snapshots, cfg)
    assert out.empty
    assert list(out.columns) == UNIVERSE_OUT_COLS


def test_trade_date_before_all_snapshots_returns_empty(liquidity_panel_two_snapshots):
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
