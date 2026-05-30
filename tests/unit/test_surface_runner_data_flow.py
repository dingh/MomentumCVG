"""
Session B — synthetic SurfaceRunner.run_single_config() data-flow verification.

Exercises the canonical surface backtest path with tiny parquet fixtures:
liquidity panel → features → surface meta/quotes → universe → signals →
assembly → selection/settlement → trade log + summaries.

Hand-calculated settlement values reuse the same quote layout as
test_option_surface_ironfly.py and test_option_surface_straddle.py.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from src.backtest.option_surface import FillAssumption
from src.backtest.run_config import BacktestRunConfig
from src.backtest.surface_run_config import SurfaceDataPaths
from src.backtest.surface_runner import SurfaceRunner


# =============================================================================
# Constants
# =============================================================================

TRADE_DATE = date(2024, 1, 5)
MONTH_SNAPSHOT = pd.Timestamp("2024-01-01")
EXPIRY_IRON = date(2024, 2, 2)
EXPIRY_STRADDLE = date(2024, 1, 12)
BODY = 100.0

TICK_LONG = "LONG1"    # top momentum → long straddle
TICK_SHORT = "SHORT1"  # bottom momentum → short iron fly
TICK_BAD = "BAD1"      # short pool, invalid surface row
TICK_MID = "MID3"      # middle momentum; keeps BAD1 in short-only pool

# Hand-calculated at expiry (mid fill), matching unit surface tests:
#   iron fly @ exit_spot=100 → pnl_per_share = +4.10
#   long straddle @ exit_spot=102 → pnl_per_share = -2.20
EXPECTED_SHORT_PNL = 4.10
EXPECTED_LONG_PNL = -2.20
EXIT_SPOT_IRON = 100.0
EXIT_SPOT_STRADDLE = 102.0


# =============================================================================
# Synthetic surface helpers
# =============================================================================

def _quote_row(
    ticker: str,
    side: str,
    strike: float,
    bid: float,
    ask: float,
    delta: float,
    abs_delta: float,
    *,
    is_body: bool = False,
    is_otm: bool = False,
    expiry: date = EXPIRY_IRON,
) -> dict:
    mid = (bid + ask) / 2
    return dict(
        ticker=ticker,
        entry_date=pd.Timestamp(TRADE_DATE),
        expiry_date=pd.Timestamp(expiry),
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
        abs_delta=abs_delta,
        spread_pct=(ask - bid) / mid if mid > 0 else 0.0,
        is_body=is_body,
        is_otm=is_otm,
    )


def _ironfly_meta(ticker: str, exit_spot: float) -> dict:
    return {
        "ticker": ticker,
        "entry_date": pd.Timestamp(TRADE_DATE),
        "expiry_date": pd.Timestamp(EXPIRY_IRON),
        "surface_valid": True,
        "failure_reason": None,
        "entry_spot": BODY,
        "body_strike": BODY,
        "exit_spot": exit_spot,
        "spot_move_pct": 0.0,
        "realized_volatility": 0.20,
        "dte_actual": 28,
    }


def _straddle_meta(ticker: str, exit_spot: float) -> dict:
    return {
        "ticker": ticker,
        "entry_date": pd.Timestamp(TRADE_DATE),
        "expiry_date": pd.Timestamp(EXPIRY_STRADDLE),
        "surface_valid": True,
        "failure_reason": None,
        "entry_spot": BODY,
        "body_strike": BODY,
        "exit_spot": exit_spot,
        "spot_move_pct": (exit_spot - BODY) / BODY * 100,
        "realized_volatility": 0.18,
        "dte_actual": 7,
    }


def _ironfly_quotes(ticker: str) -> list[dict]:
    return [
        _quote_row(ticker, "call", 100, 3.00, 3.40, +0.50, 0.50, is_body=True),
        _quote_row(ticker, "put", 100, 2.80, 3.20, -0.50, 0.50, is_body=True),
        _quote_row(ticker, "call", 105, 1.00, 1.20, +0.25, 0.25, is_otm=True),
        _quote_row(ticker, "put", 95, 0.90, 1.10, -0.25, 0.25, is_otm=True),
    ]


def _straddle_quotes(ticker: str) -> list[dict]:
    return [
        _quote_row(
            ticker, "call", 100, 2.00, 2.40, +0.50, 0.50,
            is_body=True, expiry=EXPIRY_STRADDLE,
        ),
        _quote_row(
            ticker, "put", 100, 1.80, 2.20, -0.50, 0.50,
            is_body=True, expiry=EXPIRY_STRADDLE,
        ),
    ]


def _build_surface_parquets(tmp_path: Path) -> tuple[Path, Path]:
    meta_rows = [
        _ironfly_meta(TICK_SHORT, EXIT_SPOT_IRON),
        _straddle_meta(TICK_LONG, EXIT_SPOT_STRADDLE),
        {
            **_ironfly_meta(TICK_BAD, EXIT_SPOT_IRON),
            "surface_valid": False,
            "failure_reason": "synthetic_invalid_surface",
        },
    ]
    quote_rows = _ironfly_quotes(TICK_SHORT) + _straddle_quotes(TICK_LONG)
    meta_path = tmp_path / "surface_meta.parquet"
    quotes_path = tmp_path / "surface_quotes.parquet"
    pd.DataFrame(meta_rows).to_parquet(meta_path, index=False)
    pd.DataFrame(quote_rows).to_parquet(quotes_path, index=False)
    return meta_path, quotes_path


def _build_liquidity_panel(tmp_path: Path) -> Path:
    rows = []
    for i, ticker in enumerate((TICK_SHORT, TICK_BAD, TICK_MID, TICK_LONG)):
        rows.append(
            {
                "month_date": MONTH_SNAPSHOT,
                "ticker": ticker,
                "atm_straddle_dollar_vol": 1_000_000 - i * 100_000,
                "atm_spread_pct": 0.01 + i * 0.001,
                "has_valid_atm_pair": True,
            }
        )
    path = tmp_path / "liquidity.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _build_features(tmp_path: Path) -> Path:
    # Momentum ranks (4 names): SHORT1=0.25, BAD1=0.50, MID3=0.75, LONG1=1.0.
    # long_top_pct=0.25 → long pool MID3+LONG1; short_bottom_pct=0.5 → SHORT1+BAD1.
    df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(TRADE_DATE),
                "ticker": TICK_SHORT,
                "mom_42_8_mean": 1.0,
                "cvg_42_8": 1.0,
                "mom_42_8_count": 35,
            },
            {
                "date": pd.Timestamp(TRADE_DATE),
                "ticker": TICK_BAD,
                "mom_42_8_mean": 2.0,
                "cvg_42_8": 1.0,
                "mom_42_8_count": 35,
            },
            {
                "date": pd.Timestamp(TRADE_DATE),
                "ticker": TICK_MID,
                "mom_42_8_mean": 3.0,
                "cvg_42_8": 1.0,
                "mom_42_8_count": 35,
            },
            {
                "date": pd.Timestamp(TRADE_DATE),
                "ticker": TICK_LONG,
                "mom_42_8_mean": 4.0,
                "cvg_42_8": 1.0,
                "mom_42_8_count": 35,
            },
        ]
    )
    path = tmp_path / "features" / "features_42_8.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _make_config(**overrides) -> BacktestRunConfig:
    defaults = dict(
        run_id="session_b_synthetic",
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
        start_date=TRADE_DATE,
        end_date=date(2024, 1, 6),  # must be > start_date; features exist only on TRADE_DATE
        fill=FillAssumption.mid(),
        include_diagnostics=True,
    )
    defaults.update(overrides)
    return BacktestRunConfig(**defaults)


@pytest.fixture
def synthetic_runner(tmp_path: Path) -> SurfaceRunner:
    meta_path, quotes_path = _build_surface_parquets(tmp_path)
    liquidity_path = _build_liquidity_panel(tmp_path)
    features_dir = _build_features(tmp_path).parent

    data_paths = SurfaceDataPaths(
        cache_dir=tmp_path,
        features_dir=features_dir,
        liquidity_panel_path=liquidity_path,
        surface_meta_path=meta_path,
        surface_quotes_path=quotes_path,
        earnings_path=None,
    )
    return SurfaceRunner(data_paths=data_paths)


@pytest.fixture
def run_result(synthetic_runner: SurfaceRunner):
    return synthetic_runner.run_single_config(_make_config())


# =============================================================================
# Data-flow tests
# =============================================================================

class TestSurfaceRunnerDataFlow:
    """End-to-end synthetic run through SurfaceRunner.run_single_config()."""

    def test_produces_trade_log_and_summaries(self, run_result):
        assert not run_result.trade_log.empty
        assert not run_result.date_summary.empty
        assert run_result.run_summary.get("n_traded_rows", 0) >= 1

    def test_pit_universe_uses_month_snapshot(self, run_result):
        traded = run_result.trade_log[
            run_result.trade_log["included_in_portfolio"] == True  # noqa: E712
        ]
        assert set(traded["ticker"].unique()) <= {TICK_LONG, TICK_SHORT}

    def test_long_and_short_routing(self, run_result):
        traded = run_result.trade_log[
            run_result.trade_log["included_in_portfolio"] == True  # noqa: E712
        ]
        long_row = traded[traded["ticker"] == TICK_LONG].iloc[0]
        short_row = traded[traded["ticker"] == TICK_SHORT].iloc[0]
        assert long_row["direction"] == "long"
        assert long_row["instrument_type"] == "long_straddle"
        assert short_row["direction"] == "short"
        assert short_row["instrument_type"] == "iron_fly"

    def test_short_iron_fly_pnl_per_share(self, run_result):
        short_row = run_result.trade_log[
            (run_result.trade_log["ticker"] == TICK_SHORT)
            & (run_result.trade_log["included_in_portfolio"] == True)  # noqa: E712
        ].iloc[0]
        assert short_row["pnl_per_share"] == pytest.approx(EXPECTED_SHORT_PNL)

    def test_long_straddle_pnl_per_share(self, run_result):
        long_row = run_result.trade_log[
            (run_result.trade_log["ticker"] == TICK_LONG)
            & (run_result.trade_log["included_in_portfolio"] == True)  # noqa: E712
        ].iloc[0]
        assert long_row["pnl_per_share"] == pytest.approx(EXPECTED_LONG_PNL)

    def test_invalid_surface_row_excluded_with_reason(self, run_result):
        bad_rows = run_result.trade_log[run_result.trade_log["ticker"] == TICK_BAD]
        assert len(bad_rows) == 1
        bad = bad_rows.iloc[0]
        assert not bool(bad["included_in_portfolio"])
        assert not bool(bad["structure_ok"])
        assert bad["exclusion_reason"] == "no_tradeable_structure"
        assert "metadata_error" in str(bad.get("failure_reason", ""))


class TestSurfaceRunnerV1Gaps:
    """Document missing v1 engine fields (expected until Sprint 002 build)."""

    def test_contracts_not_implemented(self, run_result):
        assert "contracts" not in run_result.trade_log.columns

    def test_pnl_dollars_not_implemented(self, run_result):
        assert "pnl_dollars" not in run_result.trade_log.columns

    def test_return_on_max_loss_not_implemented(self, run_result):
        assert "return_on_max_loss" not in run_result.trade_log.columns
