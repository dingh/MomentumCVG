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
                "cvg_count_42_8": 35,
            },
            {
                "date": pd.Timestamp(TRADE_DATE),
                "ticker": TICK_BAD,
                "mom_42_8_mean": 2.0,
                "cvg_42_8": 1.0,
                "mom_42_8_count": 35,
                "cvg_count_42_8": 35,
            },
            {
                "date": pd.Timestamp(TRADE_DATE),
                "ticker": TICK_MID,
                "mom_42_8_mean": 3.0,
                "cvg_42_8": 1.0,
                "mom_42_8_count": 35,
                "cvg_count_42_8": 35,
            },
            {
                "date": pd.Timestamp(TRADE_DATE),
                "ticker": TICK_LONG,
                "mom_42_8_mean": 4.0,
                "cvg_42_8": 1.0,
                "mom_42_8_count": 35,
                "cvg_count_42_8": 35,
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
        sizing_mode="conceptual",
        tier_a_mode="equal_premium",
        tier_a_short_budget=10_000.0,
        tier_a_long_budget=10_000.0,
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
        assert run_result.run_summary
        assert run_result.run_summary.get("n_traded_rows", 0) >= 1
        assert run_result.run_summary.get("n_trade_dates", 0) >= 1

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


class TestSurfaceRunnerS5Economics:
    """S5 columns from pipeline.step5_select_and_size appear in the trade log."""

    S5_COLUMNS = [
        "quantity",
        "sizing_mode",
        "pnl_per_share",
        "pnl_total",
        "capital_at_risk_dollars",
        "return_on_premium",
        "return_on_max_loss",
        "return_on_atm_straddle",
        "fill_label",
    ]
    S5_FINITE_ON_ALL_INCLUDED = [
        c for c in S5_COLUMNS if c != "return_on_max_loss"
    ]

    def test_s5_columns_present_on_traded_rows(self, run_result):
        traded = run_result.trade_log[
            run_result.trade_log["included_in_portfolio"] == True  # noqa: E712
        ]
        for col in self.S5_COLUMNS:
            assert col in run_result.trade_log.columns
        for col in self.S5_FINITE_ON_ALL_INCLUDED:
            assert traded[col].notna().all()
        assert (traded["capital_at_risk_dollars"] > 0).all()
        short_traded = traded[traded["direction"] == "short"]
        assert short_traded["return_on_max_loss"].notna().all()

    def test_cycle_metrics_from_s5_economics(self, run_result):
        assert not run_result.date_summary.empty
        cycle = run_result.date_summary.iloc[0]["cycle_return_on_capital_at_risk"]
        assert cycle == pytest.approx(
            run_result.run_summary["mean_cycle_return_on_capital_at_risk"]
        )
        assert cycle == pytest.approx(
            run_result.date_summary.iloc[0]["cycle_pnl_total"]
            / run_result.date_summary.iloc[0]["cycle_capital_at_risk"]
        )


class TestSurfaceRunnerDateStatus:
    """Sprint 006 D2 — A1 expected calendar and date_status partition."""

    def test_traded_date_status(self, run_result):
        assert list(run_result.date_status.columns) == ["trade_date", "status", "reason"]
        assert len(run_result.date_status) == 1
        row = run_result.date_status.iloc[0]
        assert row["trade_date"] == TRADE_DATE
        assert row["status"] == "traded"
        assert row["reason"] is None or (isinstance(row["reason"], float) and pd.isna(row["reason"]))
        assert run_result.run_summary["n_expected_dates"] == 1
        assert run_result.run_summary["n_traded_dates"] == 1
        assert run_result.run_summary["has_unresolved_failures"] is False

    def test_missing_feature_date_is_failed(self, tmp_path: Path):
        extra_date = date(2024, 1, 12)
        meta_path, quotes_path = _build_surface_parquets(tmp_path)
        meta = pd.read_parquet(meta_path)
        extra = meta.iloc[[0]].copy()
        extra["entry_date"] = pd.Timestamp(extra_date)
        extra["surface_valid"] = False
        extra["failure_reason"] = "synthetic_a1_only"
        pd.concat([meta, extra], ignore_index=True).to_parquet(meta_path, index=False)

        liquidity_path = _build_liquidity_panel(tmp_path)
        features_dir = _build_features(tmp_path).parent
        runner = SurfaceRunner(
            data_paths=SurfaceDataPaths(
                cache_dir=tmp_path,
                features_dir=features_dir,
                liquidity_panel_path=liquidity_path,
                surface_meta_path=meta_path,
                surface_quotes_path=quotes_path,
                earnings_path=None,
            )
        )
        result = runner.run_single_config(
            _make_config(start_date=TRADE_DATE, end_date=extra_date)
        )
        statuses = {
            row["trade_date"]: (row["status"], row["reason"])
            for _, row in result.date_status.iterrows()
        }
        assert statuses[TRADE_DATE][0] == "traded"
        assert statuses[extra_date] == ("failed", "missing_features")
        assert result.run_summary["n_failed_dates"] == 1
        assert result.run_summary["has_unresolved_failures"] is True
        assert set(result.date_status["trade_date"]) == {TRADE_DATE, extra_date}

    def test_invalid_surface_only_date_remains_expected(self, tmp_path: Path):
        """A1 dates that appear only on surface_valid=False rows stay in the calendar."""
        only_invalid = date(2024, 1, 12)
        meta_path, quotes_path = _build_surface_parquets(tmp_path)
        meta = pd.read_parquet(meta_path)
        extra = {
            "ticker": "ONLYBAD",
            "entry_date": pd.Timestamp(only_invalid),
            "expiry_date": pd.Timestamp(EXPIRY_IRON),
            "surface_valid": False,
            "failure_reason": "synthetic_invalid_only",
            "entry_spot": BODY,
            "body_strike": BODY,
            "exit_spot": BODY,
            "spot_move_pct": 0.0,
            "realized_volatility": 0.18,
            "dte_actual": 7,
        }
        pd.concat([meta, pd.DataFrame([extra])], ignore_index=True).to_parquet(
            meta_path, index=False
        )
        # Also give the invalid date feature coverage so it is not missing_features.
        features_path = _build_features(tmp_path)
        feats = pd.read_parquet(features_path)
        extra_feat = feats.iloc[[0]].copy()
        extra_feat["date"] = pd.Timestamp(only_invalid)
        extra_feat["ticker"] = "ONLYBAD"
        pd.concat([feats, extra_feat], ignore_index=True).to_parquet(
            features_path, index=False
        )

        runner = SurfaceRunner(
            data_paths=SurfaceDataPaths(
                cache_dir=tmp_path,
                features_dir=features_path.parent,
                liquidity_panel_path=_build_liquidity_panel(tmp_path),
                surface_meta_path=meta_path,
                surface_quotes_path=quotes_path,
                earnings_path=None,
            )
        )
        result = runner.run_single_config(
            _make_config(start_date=TRADE_DATE, end_date=only_invalid)
        )
        assert only_invalid in set(result.date_status["trade_date"])
        row = result.date_status[
            result.date_status["trade_date"] == only_invalid
        ].iloc[0]
        # ONLYBAD is not in the liquidity universe → empty signals on that date.
        assert row["status"] == "valid_no_trade"
        assert row["reason"] == "empty_signals"


class TestSurfaceRunnerFunnelAndLegs:
    """Sprint 006 D3 Commit 2 — funnel counts and constructable leg log."""

    def test_pinned_empty_schemas_when_no_expected_dates(self, synthetic_runner):
        result = synthetic_runner.run_single_config(
            _make_config(start_date=date(2099, 1, 1), end_date=date(2099, 1, 31))
        )
        from src.backtest.surface_runner import FUNNEL_SUMMARY_COLUMNS, LEG_LOG_COLUMNS

        assert list(result.funnel_summary.columns) == FUNNEL_SUMMARY_COLUMNS
        assert result.funnel_summary.empty
        assert list(result.leg_log.columns) == LEG_LOG_COLUMNS
        assert result.leg_log.empty

    def test_normal_funnel_counts_and_side_splits(self, run_result):
        from src.backtest.surface_runner import FUNNEL_SUMMARY_COLUMNS

        assert list(run_result.funnel_summary.columns) == FUNNEL_SUMMARY_COLUMNS
        assert len(run_result.funnel_summary) == 1
        row = run_result.funnel_summary.iloc[0]
        assert row["n_expected"] == 1
        assert row["n_feature_covered"] == 1
        assert row["n_universe"] == 4
        assert row["n_jointly_eligible"] == 4
        assert row["n_post_signal"] == 4
        assert row["n_post_signal_long"] == 2
        assert row["n_post_signal_short"] == 2
        assert row["n_constructable"] == 2
        assert row["n_constructable_long"] == 1
        assert row["n_constructable_short"] == 1
        assert row["n_included"] == 2
        assert row["n_included_long"] == 1
        assert row["n_included_short"] == 1
        assert row["date_status"] == "traded"
        assert row["date_reason"] is None or pd.isna(row["date_reason"])
        assert row["n_post_signal"] <= row["n_jointly_eligible"]

    def test_missing_feature_funnel_nulls(self, tmp_path: Path):
        extra_date = date(2024, 1, 12)
        meta_path, quotes_path = _build_surface_parquets(tmp_path)
        meta = pd.read_parquet(meta_path)
        extra = meta.iloc[[0]].copy()
        extra["entry_date"] = pd.Timestamp(extra_date)
        extra["surface_valid"] = False
        extra["failure_reason"] = "synthetic_a1_only"
        pd.concat([meta, extra], ignore_index=True).to_parquet(meta_path, index=False)
        runner = SurfaceRunner(
            data_paths=SurfaceDataPaths(
                cache_dir=tmp_path,
                features_dir=_build_features(tmp_path).parent,
                liquidity_panel_path=_build_liquidity_panel(tmp_path),
                surface_meta_path=meta_path,
                surface_quotes_path=quotes_path,
                earnings_path=None,
            )
        )
        result = runner.run_single_config(
            _make_config(start_date=TRADE_DATE, end_date=extra_date)
        )
        failed = result.funnel_summary[
            result.funnel_summary["trade_date"] == extra_date
        ].iloc[0]
        assert failed["n_expected"] == 1
        assert failed["n_feature_covered"] == 0
        for col in (
            "n_universe",
            "n_jointly_eligible",
            "n_post_signal",
            "n_constructable",
            "n_included",
            "n_post_signal_long",
            "n_constructable_short",
            "n_included_long",
        ):
            assert pd.isna(failed[col])
        assert failed["date_status"] == "failed"
        assert failed["date_reason"] == "missing_features"

    def test_empty_s2_funnel_zeros(self, tmp_path: Path):
        features_path = _build_features(tmp_path)
        feats = pd.read_parquet(features_path)
        feats["mom_42_8_mean"] = float("nan")
        feats.to_parquet(features_path, index=False)
        meta_path, quotes_path = _build_surface_parquets(tmp_path)
        runner = SurfaceRunner(
            data_paths=SurfaceDataPaths(
                cache_dir=tmp_path,
                features_dir=features_path.parent,
                liquidity_panel_path=_build_liquidity_panel(tmp_path),
                surface_meta_path=meta_path,
                surface_quotes_path=quotes_path,
                earnings_path=None,
            )
        )
        result = runner.run_single_config(_make_config())
        row = result.funnel_summary.iloc[0]
        assert row["date_status"] == "valid_no_trade"
        assert row["date_reason"] == "empty_signals"
        assert row["n_feature_covered"] == 1
        assert row["n_universe"] == 4
        assert row["n_jointly_eligible"] == 0
        assert row["n_post_signal"] == 0
        assert row["n_post_signal_long"] == 0
        assert row["n_post_signal_short"] == 0
        assert row["n_constructable"] == 0
        assert row["n_included"] == 0
        assert result.leg_log.empty

    def test_structure_failures_have_no_leg_rows(self, run_result):
        failed = run_result.trade_log[
            run_result.trade_log["structure_ok"] != True  # noqa: E712
        ]
        assert not failed.empty
        failed_keys = set(zip(failed["ticker"], failed["direction"]))
        if run_result.leg_log.empty:
            return
        leg_keys = set(zip(run_result.leg_log["ticker"], run_result.leg_log["direction"]))
        assert failed_keys.isdisjoint(leg_keys)

    def test_straddle_and_iron_fly_leg_counts_and_signs(self, run_result):
        from src.backtest.surface_decision_report import assert_included_trade_legs

        legs = run_result.leg_log
        assert not legs.empty
        long_legs = legs[legs["ticker"] == TICK_LONG].sort_values("leg_index")
        short_legs = legs[legs["ticker"] == TICK_SHORT].sort_values("leg_index")
        assert list(long_legs["leg_index"]) == [0, 1]
        assert list(short_legs["leg_index"]) == [0, 1, 2, 3]
        assert list(short_legs["unit_quantity"]) == [1, -1, -1, 1]
        short_trade = run_result.trade_log[
            (run_result.trade_log["ticker"] == TICK_SHORT)
            & (run_result.trade_log["included_in_portfolio"] == True)  # noqa: E712
        ].iloc[0]
        qty_mag = abs(float(short_trade["quantity"]))
        assert qty_mag > 0
        assert list(short_legs["portfolio_quantity"]) == [
            pytest.approx(qty_mag * q) for q in (1, -1, -1, 1)
        ]
        assert_included_trade_legs(
            run_result.trade_log,
            run_result.leg_log,
            run_id=run_result.config.run_id,
            fill_label=run_result.config.fill.label,
        )
        assert "_assembly" not in run_result.trade_log.columns
