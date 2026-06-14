"""
Shared fixtures and canonical schemas for Sprint 002 contract tests.

Contract tests assert that each surface-engine component honours the data
contract in `docs/surface_engine_data_contract.md`. They use small synthetic
DataFrames so every expected value is hand-auditable.

These tests are intentionally about *schema + invariants*, not full backtest
correctness. Numeric/golden checks live in the option_surface unit tests and
later sprint tests.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.option_surface import FillAssumption, OptionSurfaceDB
from src.backtest.run_config import BacktestRunConfig


# =============================================================================
# Canonical required-column sets (Stage A inputs — see contract § Stage A)
# =============================================================================
# These mirror the producer schemas in:
#   - src/features/option_surface_analyzer.py (_metadata_*_row, _build_quote_rows)
#   - pipeline.step1_get_universe / step2_score_signals docstrings
# A contract test failure here means producer output and consumer expectation
# have drifted apart.

SURFACE_META_REQUIRED = {
    "ticker",
    "entry_date",
    "expiry_date",
    "dte_actual",
    "entry_spot",
    "exit_spot",
    "body_strike",
    "surface_valid",
    "failure_reason",
}

SURFACE_QUOTES_REQUIRED = {
    "ticker",
    "entry_date",
    "expiry_date",
    "entry_spot",
    "body_strike",
    "side",
    "is_body",
    "is_otm",
    "strike",
    "bid",
    "ask",
    "mid",
    "spread_pct",
    "iv",
    "delta",
    "abs_delta",
    "gamma",
    "vega",
    "theta",
    "volume",
    "open_interest",
}

LIQUIDITY_PANEL_REQUIRED = {
    "month_date",
    "ticker",
    "atm_straddle_dollar_vol",
    "atm_spread_pct",
    "has_valid_atm_pair",
}

FEATURES_REQUIRED = {
    "date",
    "ticker",
    # plus the configured momentum_col, cvg_col, count_col
}

# Output contracts
UNIVERSE_OUT_COLS = ["ticker", "dvol_rank_pct", "spread_rank_pct"]
SIGNALS_OUT_COLS = [
    "ticker",
    "direction",
    "signal_score",
    "signal_rank_pct",
    "cvg_score",
    "cvg_rank_pct",
]

# Session B — synthetic surface (aligned with test_surface_runner_data_flow.py)
CONTRACT_TRADE_DATE = date(2024, 1, 5)
CONTRACT_BODY = 100.0
CONTRACT_EXPIRY_IRON = date(2024, 2, 2)
CONTRACT_EXPIRY_STRADDLE = date(2024, 1, 12)
CONTRACT_TICK_SHORT = "SHORT1"
CONTRACT_TICK_LONG = "LONG1"
CONTRACT_TICK_BAD = "BAD1"
CONTRACT_EXIT_IRON = 100.0
CONTRACT_EXIT_STRADDLE = 102.0
CONTRACT_IRON_NET_CREDIT = 4.10
CONTRACT_LONG_SETTLE_PNL = -2.20


# =============================================================================
# Config factory
# =============================================================================

def make_contract_config(**overrides) -> BacktestRunConfig:
    """A minimal valid BacktestRunConfig for contract tests.

    Defaults choose simple thresholds so a small synthetic cross-section
    produces predictable, disjoint long/short pools.
    """
    defaults = dict(
        run_id="contract_test",
        momentum_col="mom_42_8_mean",
        cvg_col="cvg_42_8",
        count_col="mom_42_8_count",
        min_count_pct=0.5,
        long_top_pct=0.25,
        short_bottom_pct=0.5,
        cvg_filter_pct=1.0,
        dvol_top_pct=1.0,
        spread_bottom_pct=1.0,
        short_structure="ironfly",
        wing_selection_rule="closest_delta",
        wing_delta_target=0.25,
        max_names_per_side=10,
        max_loss_budget_per_trade=500.0,
        earnings_exclusion_days=0,
        cost_model="mid",
        start_date=date(2024, 1, 5),
        end_date=date(2024, 1, 6),
        fill=FillAssumption.mid(),
        include_diagnostics=True,
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_short_budget=10_000.0,
        tier_a_long_budget=10_000.0,
    )
    defaults.update(overrides)
    return BacktestRunConfig(**defaults)


# =============================================================================
# Stage A input fixtures
# =============================================================================

@pytest.fixture
def liquidity_panel_two_snapshots() -> pd.DataFrame:
    """Liquidity panel with two PIT month snapshots.

    Jan snapshot: A, B, C, D (one with invalid ATM pair).
    Feb snapshot: A, B only (different membership to prove PIT selection).
    """
    jan = pd.Timestamp("2024-01-01")
    feb = pd.Timestamp("2024-02-01")
    rows = [
        # Jan snapshot — descending dollar vol, ascending spread
        dict(month_date=jan, ticker="A", atm_straddle_dollar_vol=4_000_000, atm_spread_pct=0.010, has_valid_atm_pair=True),
        dict(month_date=jan, ticker="B", atm_straddle_dollar_vol=3_000_000, atm_spread_pct=0.012, has_valid_atm_pair=True),
        dict(month_date=jan, ticker="C", atm_straddle_dollar_vol=2_000_000, atm_spread_pct=0.014, has_valid_atm_pair=True),
        dict(month_date=jan, ticker="D", atm_straddle_dollar_vol=1_000_000, atm_spread_pct=0.016, has_valid_atm_pair=False),
        # Feb snapshot — only A and B
        dict(month_date=feb, ticker="A", atm_straddle_dollar_vol=5_000_000, atm_spread_pct=0.009, has_valid_atm_pair=True),
        dict(month_date=feb, ticker="B", atm_straddle_dollar_vol=4_500_000, atm_spread_pct=0.011, has_valid_atm_pair=True),
    ]
    df = pd.DataFrame(rows)
    df["month_date"] = pd.to_datetime(df["month_date"])
    return df


@pytest.fixture
def features_four_tickers() -> pd.DataFrame:
    """Features for one trade date, four tickers, monotone momentum.

    Momentum ranks (pct): A=1.0, B=0.75, C=0.5, D=0.25 (after ranking).
    """
    td = pd.Timestamp("2024-01-05")
    rows = [
        dict(date=td, ticker="A", mom_42_8_mean=4.0, cvg_42_8=1.0, mom_42_8_count=35),
        dict(date=td, ticker="B", mom_42_8_mean=3.0, cvg_42_8=1.0, mom_42_8_count=35),
        dict(date=td, ticker="C", mom_42_8_mean=2.0, cvg_42_8=1.0, mom_42_8_count=35),
        dict(date=td, ticker="D", mom_42_8_mean=1.0, cvg_42_8=1.0, mom_42_8_count=35),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def universe_four_tickers() -> pd.DataFrame:
    """A step1-shaped universe output covering the four feature tickers."""
    return pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "dvol_rank_pct": [1.0, 0.75, 0.5, 0.25],
            "spread_rank_pct": [1.0, 0.75, 0.5, 0.25],
        }
    )


# =============================================================================
# Session B — surface DB + signals for S3 / S4 / S7
# =============================================================================

def _contract_quote_row(
    ticker: str,
    side: str,
    strike: float,
    bid: float,
    ask: float,
    delta: float,
    *,
    is_body: bool = False,
    is_otm: bool = False,
    expiry: date = CONTRACT_EXPIRY_IRON,
) -> dict:
    mid = (bid + ask) / 2
    return dict(
        ticker=ticker,
        entry_date=pd.Timestamp(CONTRACT_TRADE_DATE),
        expiry_date=pd.Timestamp(expiry),
        entry_spot=CONTRACT_BODY,
        body_strike=CONTRACT_BODY,
        side=side,
        strike=float(strike),
        bid=bid,
        ask=ask,
        mid=mid,
        iv=0.22,
        delta=delta,
        gamma=0.04,
        vega=0.09,
        theta=-0.02,
        volume=500,
        open_interest=2000,
        abs_delta=abs(delta),
        spread_pct=(ask - bid) / mid if mid > 0 else 0.0,
        is_body=is_body,
        is_otm=is_otm,
    )


def build_contract_surface_db() -> OptionSurfaceDB:
    """Minimal surface: SHORT1 iron fly, LONG1 long straddle, BAD1 invalid meta."""
    meta_rows = [
        {
            "ticker": CONTRACT_TICK_SHORT,
            "entry_date": pd.Timestamp(CONTRACT_TRADE_DATE),
            "expiry_date": pd.Timestamp(CONTRACT_EXPIRY_IRON),
            "surface_valid": True,
            "failure_reason": None,
            "entry_spot": CONTRACT_BODY,
            "body_strike": CONTRACT_BODY,
            "exit_spot": CONTRACT_EXIT_IRON,
            "dte_actual": 28,
        },
        {
            "ticker": CONTRACT_TICK_LONG,
            "entry_date": pd.Timestamp(CONTRACT_TRADE_DATE),
            "expiry_date": pd.Timestamp(CONTRACT_EXPIRY_STRADDLE),
            "surface_valid": True,
            "failure_reason": None,
            "entry_spot": CONTRACT_BODY,
            "body_strike": CONTRACT_BODY,
            "exit_spot": CONTRACT_EXIT_STRADDLE,
            "dte_actual": 7,
        },
        {
            "ticker": CONTRACT_TICK_BAD,
            "entry_date": pd.Timestamp(CONTRACT_TRADE_DATE),
            "expiry_date": pd.Timestamp(CONTRACT_EXPIRY_IRON),
            "surface_valid": False,
            "failure_reason": "synthetic_invalid_surface",
            "entry_spot": CONTRACT_BODY,
            "body_strike": CONTRACT_BODY,
            "exit_spot": CONTRACT_EXIT_IRON,
            "dte_actual": 28,
        },
    ]
    quote_rows = [
        _contract_quote_row(CONTRACT_TICK_SHORT, "call", 100, 3.00, 3.40, 0.50, is_body=True),
        _contract_quote_row(CONTRACT_TICK_SHORT, "put", 100, 2.80, 3.20, -0.50, is_body=True),
        _contract_quote_row(CONTRACT_TICK_SHORT, "call", 105, 1.00, 1.20, 0.25, is_otm=True),
        _contract_quote_row(CONTRACT_TICK_SHORT, "put", 95, 0.90, 1.10, -0.25, is_otm=True),
        _contract_quote_row(
            CONTRACT_TICK_LONG, "call", 100, 2.00, 2.40, 0.50,
            is_body=True, expiry=CONTRACT_EXPIRY_STRADDLE,
        ),
        _contract_quote_row(
            CONTRACT_TICK_LONG, "put", 100, 1.80, 2.20, -0.50,
            is_body=True, expiry=CONTRACT_EXPIRY_STRADDLE,
        ),
    ]
    return OptionSurfaceDB(pd.DataFrame(meta_rows), pd.DataFrame(quote_rows))


@pytest.fixture
def contract_surface_db() -> OptionSurfaceDB:
    return build_contract_surface_db()


@pytest.fixture
def contract_signals_two_sides() -> pd.DataFrame:
    """Two signal rows: one long, one short (iron fly path)."""
    return pd.DataFrame(
        [
            {
                "ticker": CONTRACT_TICK_SHORT,
                "direction": "short",
                "signal_score": 1.0,
                "signal_rank_pct": 0.25,
                "cvg_score": 1.0,
                "cvg_rank_pct": 1.0,
            },
            {
                "ticker": CONTRACT_TICK_LONG,
                "direction": "long",
                "signal_score": 4.0,
                "signal_rank_pct": 1.0,
                "cvg_score": 1.0,
                "cvg_rank_pct": 1.0,
            },
        ]
    )


@pytest.fixture
def contract_signals_with_bad_surface() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": CONTRACT_TICK_BAD,
                "direction": "short",
                "signal_score": 2.0,
                "signal_rank_pct": 0.5,
                "cvg_score": 1.0,
                "cvg_rank_pct": 1.0,
            },
        ]
    )
