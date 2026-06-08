"""
Contract: Stage A inputs (A1 meta, A2 quotes, A3 liquidity panel, A4 features).

Stage A (precompute) is treated as *given* for Sprint 002. These tests pin the
required-column schemas that downstream consumers depend on. They validate the
producer code paths in src/features/option_surface_analyzer.py and the synthetic
consumer fixtures, so drift between producer output and consumer expectation
fails loudly.

See docs/surface_engine_data_contract.md § Stage A.
"""
from __future__ import annotations

from datetime import date

from src.features.option_surface_analyzer import (
    _metadata_failure_row,
    _metadata_success_row,
)
from tests.contract.conftest import (
    FEATURES_REQUIRED,
    LIQUIDITY_PANEL_REQUIRED,
    SURFACE_META_REQUIRED,
    SURFACE_QUOTES_REQUIRED,
)


# --- A1: surface meta ------------------------------------------------------

def test_meta_success_row_has_required_columns():
    row = _metadata_success_row(
        ticker="A",
        entry_date=date(2024, 1, 5),
        expiry_date=date(2024, 1, 12),
        dte_target=7,
        frequency="weekly",
        entry_spot=100,
        exit_spot=101,
        body_strike=100,
        spot_move_pct=0.01,
        realized_volatility=0.2,
        has_body_call=True,
        has_body_put=True,
        n_surface_quotes=6,
        processing_time=0.01,
    )
    missing = SURFACE_META_REQUIRED - set(row.keys())
    assert not missing, f"meta success row missing required keys: {missing}"


def test_meta_failure_row_shares_schema_with_success_row():
    success = _metadata_success_row(
        ticker="A",
        entry_date=date(2024, 1, 5),
        expiry_date=date(2024, 1, 12),
        dte_target=7,
        frequency="weekly",
        entry_spot=100,
        exit_spot=101,
        body_strike=100,
        spot_move_pct=0.01,
        realized_volatility=0.2,
        has_body_call=True,
        has_body_put=True,
        n_surface_quotes=6,
        processing_time=0.01,
    )
    failure = _metadata_failure_row(
        ticker="A",
        entry_date=date(2024, 1, 5),
        dte_target=7,
        frequency="weekly",
        failure_reason="no_spot_price",
        processing_time=0.01,
    )
    # Failure and success rows must coexist in the same table → identical keys.
    assert set(success.keys()) == set(failure.keys())


def test_meta_surface_valid_requires_both_body_legs():
    # surface_valid is the primary downstream filter; it must be False when a
    # body leg is missing even if quotes exist.
    row = _metadata_success_row(
        ticker="A",
        entry_date=date(2024, 1, 5),
        expiry_date=date(2024, 1, 12),
        dte_target=7,
        frequency="weekly",
        entry_spot=100,
        exit_spot=101,
        body_strike=100,
        spot_move_pct=0.01,
        realized_volatility=0.2,
        has_body_call=True,
        has_body_put=False,
        n_surface_quotes=6,
        processing_time=0.01,
    )
    assert row["surface_valid"] is False


# --- A2: surface quotes ----------------------------------------------------

def test_quotes_required_columns_documented():
    # Guard against silent shrinking of the documented quote schema.
    # (Producer columns are asserted in option_surface unit tests; here we pin
    #  the consumer-required subset.)
    assert {"bid", "ask", "mid", "delta", "is_body"} <= SURFACE_QUOTES_REQUIRED


# --- A3: liquidity panel ---------------------------------------------------

def test_liquidity_panel_fixture_has_required_columns(liquidity_panel_two_snapshots):
    missing = LIQUIDITY_PANEL_REQUIRED - set(liquidity_panel_two_snapshots.columns)
    assert not missing, f"liquidity panel missing required columns: {missing}"


# --- A4: features ----------------------------------------------------------

def test_features_fixture_has_required_columns(features_four_tickers):
    missing = FEATURES_REQUIRED - set(features_four_tickers.columns)
    assert not missing, f"features missing required columns: {missing}"
    # configured signal columns must be present
    for col in ("mom_42_8_mean", "cvg_42_8", "mom_42_8_count"):
        assert col in features_four_tickers.columns
