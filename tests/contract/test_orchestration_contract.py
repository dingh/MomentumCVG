"""
Contract: ORCH — SurfaceRunner thin S1→S8 orchestration.

Verifies that ``SurfaceRunner.run_single_config`` delegates S5 to
``pipeline.step5_select_and_size``, preserves S5 economics in the trade log,
and feeds S8 metrics without duplicating inline select/size/settle logic.

See docs/surface_engine_data_contract.md § ORCH and Sprint 003 Phase 6.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtest.pipeline import step5_select_and_size
from src.backtest.surface_run_config import SurfaceDataPaths
from src.backtest.surface_runner import SurfaceRunner

from tests.contract.test_run_metrics_contract import DATE_SUMMARY_COLUMNS

# Reuse Session B synthetic layout from unit data-flow tests.
from tests.unit.test_surface_runner_data_flow import (
    TICK_BAD,
    TICK_SHORT,
    _build_features,
    _build_liquidity_panel,
    _build_surface_parquets,
    _make_config,
)

S5_OUTPUT_COLUMNS = [
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

# M2 (return_on_max_loss) is defined-risk shorts only; long rows may be NaN.
S5_FINITE_ON_ALL_INCLUDED = [
    c for c in S5_OUTPUT_COLUMNS if c != "return_on_max_loss"
]


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


class TestRunnerDelegatesToPipelineS5:
    def test_step5_called_once_per_trade_date_with_nonempty_inputs(
        self, synthetic_runner: SurfaceRunner
    ):
        config = _make_config()
        calls: list[dict] = []
        real_step5 = step5_select_and_size

        def _spy(signals, structures, config):
            calls.append(
                {
                    "n_signals": len(signals),
                    "n_structures": len(structures),
                    "signals_nonempty": not signals.empty,
                    "structures_nonempty": not structures.empty,
                }
            )
            return real_step5(signals, structures, config)

        with patch(
            "src.backtest.surface_runner.step5_select_and_size",
            side_effect=_spy,
        ):
            synthetic_runner.run_single_config(config)

        assert len(calls) == 1
        assert calls[0]["n_signals"] > 0
        assert calls[0]["n_structures"] > 0
        assert calls[0]["signals_nonempty"]
        assert calls[0]["structures_nonempty"]

    def test_inline_select_size_and_settle_removed(self):
        assert not hasattr(SurfaceRunner, "_select_size_and_settle")
        assert not hasattr(SurfaceRunner, "_resolve_max_loss_per_share")


class TestS5ColumnsSurviveIntoTradeLog:
    def test_included_rows_carry_s5_economics(self, synthetic_runner: SurfaceRunner):
        result = synthetic_runner.run_single_config(_make_config())
        traded = result.trade_log[
            result.trade_log["included_in_portfolio"] == True  # noqa: E712
        ]
        assert not traded.empty
        for col in S5_OUTPUT_COLUMNS:
            assert col in result.trade_log.columns, f"missing column {col}"
        for col in S5_FINITE_ON_ALL_INCLUDED:
            assert traded[col].notna().all(), f"{col} has NaN on included rows"
        short_traded = traded[traded["direction"] == "short"]
        assert short_traded["return_on_max_loss"].notna().all()

    def test_assembly_not_leaked_into_trade_log(self, synthetic_runner: SurfaceRunner):
        result = synthetic_runner.run_single_config(_make_config())
        assert "_assembly" not in result.trade_log.columns


class TestS8ReceivesValidS5Economics:
    def test_cycle_return_finite_from_runner_output(self, synthetic_runner: SurfaceRunner):
        result = synthetic_runner.run_single_config(_make_config())

        assert not result.date_summary.empty
        cycle = result.date_summary.iloc[0]["cycle_return_on_capital_at_risk"]
        assert np.isfinite(cycle)

        mean_cycle = result.run_summary.get("mean_cycle_return_on_capital_at_risk")
        assert mean_cycle is not None and np.isfinite(mean_cycle)

    def test_excluded_rows_do_not_affect_cycle_return(self, synthetic_runner: SurfaceRunner):
        """Diagnostics rows stay in the log but S8 aggregates included rows only."""
        result = synthetic_runner.run_single_config(_make_config(include_diagnostics=True))
        included = result.trade_log[
            result.trade_log["included_in_portfolio"] == True  # noqa: E712
        ]
        excluded = result.trade_log[
            result.trade_log["included_in_portfolio"] == False  # noqa: E712
        ]
        assert not excluded.empty

        expected_pnl = included["pnl_total"].sum()
        expected_car = included["capital_at_risk_dollars"].sum()
        row = result.date_summary.iloc[0]
        assert row["cycle_pnl_total"] == pytest.approx(expected_pnl)
        assert row["cycle_capital_at_risk"] == pytest.approx(expected_car)
        assert row["cycle_return_on_capital_at_risk"] == pytest.approx(
            expected_pnl / expected_car
        )


class TestDiagnosticsBehavior:
    def test_include_diagnostics_true_keeps_excluded_rows(
        self, synthetic_runner: SurfaceRunner
    ):
        result = synthetic_runner.run_single_config(_make_config(include_diagnostics=True))
        bad = result.trade_log[result.trade_log["ticker"] == TICK_BAD]
        assert len(bad) == 1
        assert not bool(bad.iloc[0]["included_in_portfolio"])

    def test_include_diagnostics_false_drops_excluded_rows(
        self, synthetic_runner: SurfaceRunner
    ):
        result = synthetic_runner.run_single_config(
            _make_config(include_diagnostics=False)
        )
        assert TICK_BAD not in set(result.trade_log["ticker"].tolist())
        assert (result.trade_log["included_in_portfolio"] == True).all()  # noqa: E712


class TestEmptyInputs:
    def test_no_trade_dates_yields_empty_summaries(self, synthetic_runner: SurfaceRunner):
        config = _make_config(
            start_date=date(2099, 1, 1),
            end_date=date(2099, 1, 31),
        )
        result = synthetic_runner.run_single_config(config)

        assert result.trade_log.empty
        assert list(result.date_summary.columns) == DATE_SUMMARY_COLUMNS
        assert result.date_summary.empty
        assert result.run_summary.get("n_traded_rows", 0) == 0
        assert np.isnan(result.run_summary.get("mean_cycle_return_on_capital_at_risk"))

    def test_empty_signals_date_skipped_without_s5_call(
        self, synthetic_runner: SurfaceRunner
    ):
        """A date with no qualifying signals should not invoke S5."""
        calls: list[int] = []

        def _count(signals, structures, config):
            calls.append(1)
            return step5_select_and_size(signals, structures, config)

        with (
            patch(
                "src.backtest.surface_runner.step2_score_signals",
                return_value=pd.DataFrame(),
            ),
            patch(
                "src.backtest.surface_runner.step5_select_and_size",
                side_effect=_count,
            ),
        ):
            result = synthetic_runner.run_single_config(_make_config())

        assert len(calls) == 0
        assert result.trade_log.empty
