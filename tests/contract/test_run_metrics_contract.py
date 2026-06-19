"""
Contract tests for S8 run metrics (surface_metrics.py).

Authoritative design: docs/surface_engine_portfolio_metrics_design.md § S8.
S8 aggregates S5 trade-log economics only — it must not recompute trade PnL.

Function → test mapping
-----------------------
build_date_summary
  TestBuildDateSummarySchemaAndLegacy
  TestCycleReturnOneDateMixedSides
  TestExcludedRowsIgnored
  TestEmptySide
  TestZeroOrMissingDenominator
  TestNoIncludedRows

summarize_trade_log
  TestSummarizeTradeLogRunLevel
  TestMultiDateSeries (also exercises build_date_summary)

_compute_max_drawdown
  TestComputeMaxDrawdown

rank_run_summaries
  TestRankRunSummaries

Private helpers (_cycle_economics_complete, _cycle_aggregate, _safe_cycle_return,
_mean_return_on_body_credit) are exercised on primary paths through
build_date_summary integration tests (not isolated unit tests).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.surface_metrics import (
    _compute_max_drawdown,
    build_date_summary,
    rank_run_summaries,
    summarize_trade_log,
)

# Expected output schema from build_date_summary (empty-input contract).
DATE_SUMMARY_COLUMNS = [
    "trade_date",
    "n_candidates",
    "n_traded",
    "cycle_pnl_total",
    "cycle_capital_at_risk",
    "cycle_return_on_capital_at_risk",
    "short_cycle_pnl_total",
    "short_cycle_capital_at_risk",
    "short_cycle_return",
    "long_cycle_pnl_total",
    "long_cycle_capital_at_risk",
    "long_cycle_return",
    "date_return_on_body_credit",
    "long_n_candidates",
    "short_n_candidates",
    "long_n_traded",
    "short_n_traded",
    "long_date_return_on_body_credit",
    "short_date_return_on_body_credit",
]


def _trade_row(
    trade_date: date,
    ticker: str,
    direction: str,
    *,
    included: bool = True,
    pnl_total: float = 0.0,
    capital_at_risk_dollars: float = 0.0,
    pnl_per_share: float = 0.0,
    body_credit_per_share: float = 2.0,
    spread_cost_ratio: float | None = None,
    leg_spread_to_credit_ratio: float | None = None,
) -> dict:
    """Build one minimal S5-shaped trade-log row for hand-auditable metrics tests."""
    row = {
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": direction,
        "included_in_portfolio": included,
        "quantity": 100.0 if direction == "long" else -100.0,
        "pnl_per_share": pnl_per_share,
        "pnl_total": pnl_total,
        "capital_at_risk_dollars": capital_at_risk_dollars,
        "body_credit_per_share": body_credit_per_share,
    }
    if spread_cost_ratio is not None:
        row["spread_cost_ratio"] = spread_cost_ratio
    if leg_spread_to_credit_ratio is not None:
        row["leg_spread_to_credit_ratio"] = leg_spread_to_credit_ratio
    return row


# ---------------------------------------------------------------------------
# build_date_summary — schema, counts, legacy body-credit
# ---------------------------------------------------------------------------

class TestBuildDateSummarySchemaAndLegacy:
    """Tests for build_date_summary: non-cycle columns and empty-input behavior."""

    def test_empty_trade_log_returns_expected_schema(self):
        """build_date_summary: empty input returns the full column schema with zero rows."""
        out = build_date_summary(pd.DataFrame())
        assert list(out.columns) == DATE_SUMMARY_COLUMNS
        assert out.empty

    def test_counts_and_body_credit_returns(self):
        """build_date_summary: legacy date_return_on_body_credit = mean(per-trade ROBC), not Σ/Σ.

        S1: 1/2=0.5, L1: 3/4=0.75 → date mean = 0.625 (equal weight per name).
        This is NOT (1+3)/(2+4)=0.667 and NOT dollar-weighted by quantity.
        Portfolio return on a date uses cycle_return_on_capital_at_risk instead.
        """
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [
                _trade_row(
                    td, "S1", "short",
                    pnl_total=500.0, capital_at_risk_dollars=2000.0,
                    pnl_per_share=1.0, body_credit_per_share=2.0,
                ),
                _trade_row(
                    td, "L1", "long",
                    pnl_total=300.0, capital_at_risk_dollars=1500.0,
                    pnl_per_share=3.0, body_credit_per_share=4.0,
                ),
                _trade_row(
                    td, "X", "long", included=False,
                    pnl_per_share=9.0, body_credit_per_share=1.0,
                ),
            ]
        )

        row = build_date_summary(trade_log).iloc[0]

        assert row["n_candidates"] == 3
        assert row["n_traded"] == 2
        assert row["long_n_candidates"] == 2
        assert row["short_n_candidates"] == 1
        assert row["long_n_traded"] == 1
        assert row["short_n_traded"] == 1
        # Legacy equal-weight mean: (1/2 + 3/4) / 2 = 0.625 — not Σpnl/Σbody_credit.
        assert row["date_return_on_body_credit"] == pytest.approx((0.5 + 0.75) / 2)
        assert row["short_date_return_on_body_credit"] == pytest.approx(0.5)
        assert row["long_date_return_on_body_credit"] == pytest.approx(0.75)

    def test_body_credit_zero_denominator_is_nan(self):
        """build_date_summary: date_return_on_body_credit is NaN when body_credit_per_share <= 0."""
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [_trade_row(td, "S1", "short", pnl_per_share=1.0, body_credit_per_share=0.0)]
        )
        row = build_date_summary(trade_log).iloc[0]
        assert np.isnan(row["date_return_on_body_credit"])


# ---------------------------------------------------------------------------
# summarize_trade_log — run-level aggregation
# ---------------------------------------------------------------------------

class TestSummarizeTradeLogRunLevel:
    """Tests for summarize_trade_log: run-level keys beyond per-date cycle returns."""

    def test_empty_trade_log_returns_default_summary(self):
        """summarize_trade_log: empty input returns the documented zero/NaN default dict."""
        summary = summarize_trade_log(pd.DataFrame())
        assert summary["n_trade_dates"] == 0
        assert summary["n_candidate_rows"] == 0
        assert summary["n_traded_rows"] == 0
        assert summary["availability_rate"] == 0.0
        assert summary["avg_trades_per_date"] == 0.0
        assert np.isnan(summary["annualized_sharpe"])
        assert np.isnan(summary["max_drawdown"])
        assert np.isnan(summary["robust_score"])
        assert np.isnan(summary["mean_cycle_return_on_capital_at_risk"])
        assert np.isnan(summary["mean_trade_return_on_body_credit"])

    def test_availability_hit_rate_and_robust_score(self):
        """summarize_trade_log: availability_rate, hit_rate, and robust_score = sharpe × availability."""
        d1 = date(2024, 1, 5)
        d2 = date(2024, 1, 12)
        trade_log = pd.DataFrame(
            [
                _trade_row(d1, "S1", "short", pnl_total=500.0, capital_at_risk_dollars=2000.0, pnl_per_share=1.0),
                _trade_row(d1, "GHOST", "long", included=False, pnl_per_share=-9.0),
                _trade_row(d2, "L1", "long", pnl_total=300.0, capital_at_risk_dollars=1500.0, pnl_per_share=-1.0),
            ]
        )

        summary = summarize_trade_log(trade_log)

        assert summary["n_trade_dates"] == 2
        assert summary["n_candidate_rows"] == 3
        assert summary["n_traded_rows"] == 2
        assert summary["availability_rate"] == pytest.approx(2 / 3)
        assert summary["avg_trades_per_date"] == pytest.approx(1.0)
        assert summary["hit_rate"] == pytest.approx(0.5)
        assert summary["long_n_traded_rows"] == 1
        assert summary["short_n_traded_rows"] == 1
        assert np.isfinite(summary["annualized_sharpe"])
        assert summary["robust_score"] == pytest.approx(
            summary["annualized_sharpe"] * summary["availability_rate"]
        )

    def test_single_date_sharpe_is_nan(self):
        """summarize_trade_log: annualized_sharpe and robust_score need >= 2 finite cycle returns."""
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [_trade_row(td, "S1", "short", pnl_total=500.0, capital_at_risk_dollars=2000.0)]
        )
        summary = summarize_trade_log(trade_log)
        assert np.isnan(summary["annualized_sharpe"])
        assert np.isnan(summary["robust_score"])

    def test_legacy_body_credit_and_spread_averages(self):
        """summarize_trade_log: legacy body-credit means, side cycle averages, and spread ratios."""
        d1 = date(2024, 1, 5)
        d2 = date(2024, 1, 12)
        trade_log = pd.DataFrame(
            [
                _trade_row(
                    d1, "S1", "short",
                    pnl_total=500.0, capital_at_risk_dollars=2000.0,
                    pnl_per_share=2.0, body_credit_per_share=4.0,
                    spread_cost_ratio=0.10, leg_spread_to_credit_ratio=0.20,
                ),
                _trade_row(
                    d2, "L1", "long",
                    pnl_total=300.0, capital_at_risk_dollars=1500.0,
                    pnl_per_share=1.0, body_credit_per_share=2.0,
                    spread_cost_ratio=0.30, leg_spread_to_credit_ratio=0.40,
                ),
            ]
        )

        summary = summarize_trade_log(trade_log)
        date_summary = build_date_summary(trade_log)

        assert summary["mean_trade_return_on_body_credit"] == pytest.approx(0.5)
        assert summary["median_trade_return_on_body_credit"] == pytest.approx(0.5)
        assert summary["avg_long_cycle_return"] == pytest.approx(
            date_summary["long_cycle_return"].iloc[1]
        )
        assert summary["avg_short_cycle_return"] == pytest.approx(
            date_summary["short_cycle_return"].iloc[0]
        )
        assert summary["avg_long_return_on_body_credit"] == pytest.approx(0.5)
        assert summary["avg_short_return_on_body_credit"] == pytest.approx(0.5)
        assert summary["avg_spread_cost_ratio"] == pytest.approx(0.20)
        assert summary["avg_leg_spread_to_credit_ratio"] == pytest.approx(0.30)

    def test_hit_rate_ignores_nan_pnl_per_share(self):
        """summarize_trade_log: hit_rate excludes NaN pnl_per_share; all-NaN → NaN."""
        d1 = date(2024, 1, 5)
        d2 = date(2024, 1, 12)
        trade_log = pd.DataFrame(
            [
                _trade_row(d1, "A", "short", pnl_total=100.0, capital_at_risk_dollars=1000.0, pnl_per_share=1.0),
                _trade_row(d1, "B", "long", pnl_total=50.0, capital_at_risk_dollars=500.0, pnl_per_share=float("nan")),
                _trade_row(d1, "C", "short", pnl_total=-50.0, capital_at_risk_dollars=500.0, pnl_per_share=-1.0),
                _trade_row(d2, "D", "long", pnl_total=10.0, capital_at_risk_dollars=100.0, pnl_per_share=float("nan")),
                _trade_row(d2, "E", "long", pnl_total=20.0, capital_at_risk_dollars=200.0, pnl_per_share=float("nan")),
            ]
        )

        summary = summarize_trade_log(trade_log)
        assert summary["hit_rate"] == pytest.approx(0.5)

        all_nan = pd.DataFrame(
            [
                _trade_row(d1, "A", "short", pnl_total=100.0, capital_at_risk_dollars=1000.0, pnl_per_share=float("nan")),
            ]
        )
        assert np.isnan(summarize_trade_log(all_nan)["hit_rate"])

    def test_all_nan_cycle_returns_yield_nan_drawdown(self):
        """summarize_trade_log: max_drawdown is NaN when no finite cycle returns exist."""
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [_trade_row(td, "S1", "short", pnl_total=100.0, capital_at_risk_dollars=0.0)]
        )
        summary = summarize_trade_log(trade_log)
        assert np.isnan(summary["max_drawdown"])
        assert np.isnan(summary["mean_cycle_return_on_capital_at_risk"])


# ---------------------------------------------------------------------------
# _compute_max_drawdown — drawdown helper used by summarize_trade_log
# ---------------------------------------------------------------------------

class TestComputeMaxDrawdown:
    """Tests for _compute_max_drawdown (drawdown on a per-date return series)."""

    def test_empty_series_returns_nan(self):
        """_compute_max_drawdown: empty input series returns NaN."""
        assert np.isnan(_compute_max_drawdown(pd.Series(dtype=float)))

    def test_known_drawdown_path(self):
        """_compute_max_drawdown: matches hand-computed cumulative simple-return drawdown."""
        returns = pd.Series([0.10, -0.05, 0.02])
        curve = (1.0 + returns).cumprod()
        expected = float((curve / curve.cummax() - 1.0).min())
        assert _compute_max_drawdown(returns) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# rank_run_summaries — config search ranking
# ---------------------------------------------------------------------------

class TestRankRunSummaries:
    """Tests for rank_run_summaries (sort config comparison table by metric)."""

    def test_empty_dataframe_returns_copy(self):
        """rank_run_summaries: empty input returns an empty copy with columns preserved."""
        empty = pd.DataFrame(columns=["robust_score", "run_id"])
        out = rank_run_summaries(empty)
        assert out.empty
        assert list(out.columns) == ["robust_score", "run_id"]

    def test_sorts_descending_by_metric(self):
        """rank_run_summaries: default metric robust_score sorts highest-first."""
        df = pd.DataFrame(
            {"run_id": ["a", "b", "c"], "robust_score": [1.0, 3.0, 2.0]}
        )
        ranked = rank_run_summaries(df)
        assert ranked["run_id"].tolist() == ["b", "c", "a"]

    def test_custom_metric_column(self):
        """rank_run_summaries: metric= parameter selects a different sort column."""
        df = pd.DataFrame(
            {"run_id": ["a", "b"], "annualized_sharpe": [0.5, 1.5]}
        )
        ranked = rank_run_summaries(df, metric="annualized_sharpe")
        assert ranked["run_id"].tolist() == ["b", "a"]

    def test_missing_metric_raises_key_error(self):
        """rank_run_summaries: unknown metric column raises KeyError."""
        df = pd.DataFrame({"run_id": ["a"], "robust_score": [1.0]})
        with pytest.raises(KeyError, match="not_a_metric"):
            rank_run_summaries(df, metric="not_a_metric")


# ---------------------------------------------------------------------------
# build_date_summary — S8 primary cycle metrics (Sprint 003 Phase 5)
# ---------------------------------------------------------------------------

class TestCycleReturnOneDateMixedSides:
    """Tests for build_date_summary: primary cycle_return_on_capital_at_risk + side splits."""

    def test_cycle_return_and_side_splits(self):
        """build_date_summary: one date, long+short included — hand-calc cycle and side returns."""
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [
                _trade_row(td, "S1", "short", pnl_total=500.0, capital_at_risk_dollars=2000.0, pnl_per_share=5.0),
                _trade_row(td, "L1", "long", pnl_total=300.0, capital_at_risk_dollars=1500.0, pnl_per_share=3.0),
            ]
        )

        summary = build_date_summary(trade_log)
        row = summary.iloc[0]

        assert row["cycle_pnl_total"] == pytest.approx(800.0)
        assert row["cycle_capital_at_risk"] == pytest.approx(3500.0)
        assert row["cycle_return_on_capital_at_risk"] == pytest.approx(800.0 / 3500.0)

        assert row["short_cycle_pnl_total"] == pytest.approx(500.0)
        assert row["short_cycle_capital_at_risk"] == pytest.approx(2000.0)
        assert row["short_cycle_return"] == pytest.approx(0.25)

        assert row["long_cycle_pnl_total"] == pytest.approx(300.0)
        assert row["long_cycle_capital_at_risk"] == pytest.approx(1500.0)
        assert row["long_cycle_return"] == pytest.approx(0.20)


class TestExcludedRowsIgnored:
    """Tests for build_date_summary: excluded rows must not affect cycle aggregation."""

    def test_excluded_row_does_not_affect_cycle_metrics(self):
        """build_date_summary: included_in_portfolio=False rows are omitted from cycle sums."""
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [
                _trade_row(td, "S1", "short", pnl_total=500.0, capital_at_risk_dollars=2000.0),
                _trade_row(
                    td,
                    "GHOST",
                    "long",
                    included=False,
                    pnl_total=9_999.0,
                    capital_at_risk_dollars=9_999.0,
                ),
            ]
        )

        row = build_date_summary(trade_log).iloc[0]

        assert row["cycle_pnl_total"] == pytest.approx(500.0)
        assert row["cycle_capital_at_risk"] == pytest.approx(2000.0)
        assert row["cycle_return_on_capital_at_risk"] == pytest.approx(0.25)
        assert row["long_cycle_pnl_total"] == pytest.approx(0.0)
        assert np.isnan(row["long_cycle_return"])


class TestEmptySide:
    """Tests for build_date_summary: empty long or short side within a cycle."""

    def test_long_only_cycle_metrics(self):
        """build_date_summary: long-only book — short_cycle_return is NaN; book return computes."""
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [_trade_row(td, "L1", "long", pnl_total=300.0, capital_at_risk_dollars=1500.0)]
        )
        row = build_date_summary(trade_log).iloc[0]

        assert row["cycle_return_on_capital_at_risk"] == pytest.approx(0.20)
        assert row["long_cycle_return"] == pytest.approx(0.20)
        assert row["short_cycle_pnl_total"] == pytest.approx(0.0)
        assert row["short_cycle_capital_at_risk"] == pytest.approx(0.0)
        assert np.isnan(row["short_cycle_return"])

    def test_short_only_cycle_metrics(self):
        """build_date_summary: short-only book — long_cycle_return is NaN; book return computes."""
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [_trade_row(td, "S1", "short", pnl_total=500.0, capital_at_risk_dollars=2000.0)]
        )
        row = build_date_summary(trade_log).iloc[0]

        assert row["cycle_return_on_capital_at_risk"] == pytest.approx(0.25)
        assert row["short_cycle_return"] == pytest.approx(0.25)
        assert np.isnan(row["long_cycle_return"])


class TestZeroOrMissingDenominator:
    """Tests for build_date_summary: zero/NaN/missing capital_at_risk_dollars → NaN return."""

    @pytest.mark.parametrize(
        "capital_at_risk",
        [0.0, np.nan],
        ids=["zero", "nan"],
    )
    def test_zero_or_nan_capital_at_risk_yields_nan_return(self, capital_at_risk):
        """build_date_summary: cycle return is NaN when Σ capital_at_risk is zero or NaN."""
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [_trade_row(td, "S1", "short", pnl_total=100.0, capital_at_risk_dollars=capital_at_risk)]
        )
        row = build_date_summary(trade_log).iloc[0]

        assert np.isnan(row["cycle_return_on_capital_at_risk"])
        assert np.isnan(row["short_cycle_return"])

    def test_missing_capital_column_yields_nan_return(self):
        """build_date_summary: missing capital_at_risk_dollars column → NaN cycle return."""
        td = date(2024, 1, 5)
        row_dict = _trade_row(td, "S1", "short", pnl_total=100.0, capital_at_risk_dollars=1000.0)
        del row_dict["capital_at_risk_dollars"]
        trade_log = pd.DataFrame([row_dict])

        row = build_date_summary(trade_log).iloc[0]

        assert row["cycle_capital_at_risk"] == pytest.approx(0.0)
        assert np.isnan(row["cycle_return_on_capital_at_risk"])


class TestInvalidCycleEconomics:
    """Tests for build_date_summary: partial missing S5 economics invalidate cycle returns."""

    def test_included_row_with_missing_car_invalidates_book_and_long_side(self):
        """build_date_summary: one bad CAR row → book and long return NaN; short side valid."""
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [
                _trade_row(td, "S1", "short", pnl_total=500.0, capital_at_risk_dollars=2000.0),
                _trade_row(td, "L1", "long", pnl_total=300.0, capital_at_risk_dollars=float("nan")),
            ]
        )

        row = build_date_summary(trade_log).iloc[0]

        assert row["cycle_pnl_total"] == pytest.approx(800.0)
        assert row["cycle_capital_at_risk"] == pytest.approx(2000.0)
        assert np.isnan(row["cycle_return_on_capital_at_risk"])
        assert row["short_cycle_return"] == pytest.approx(0.25)
        assert np.isnan(row["long_cycle_return"])
        assert row["long_cycle_pnl_total"] == pytest.approx(300.0)
        assert row["long_cycle_capital_at_risk"] == pytest.approx(0.0)

    def test_included_row_with_missing_pnl_invalidates_cycle_return(self):
        """build_date_summary: NaN pnl_total on an included row → cycle return NaN."""
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [_trade_row(td, "S1", "short", pnl_total=float("nan"), capital_at_risk_dollars=2000.0)]
        )
        row = build_date_summary(trade_log).iloc[0]

        assert np.isnan(row["cycle_return_on_capital_at_risk"])
        assert np.isnan(row["short_cycle_return"])

    def test_missing_capital_column_invalidates_cycle_return(self):
        """build_date_summary: absent capital_at_risk_dollars column → cycle return NaN."""
        td = date(2024, 1, 5)
        row_dict = _trade_row(td, "S1", "short", pnl_total=100.0, capital_at_risk_dollars=1000.0)
        del row_dict["capital_at_risk_dollars"]
        row = build_date_summary(pd.DataFrame([row_dict])).iloc[0]

        assert row["cycle_capital_at_risk"] == pytest.approx(0.0)
        assert np.isnan(row["cycle_return_on_capital_at_risk"])


class TestMultiDateSeries:
    """Tests for build_date_summary + summarize_trade_log: multi-date cycle return series."""

    def test_independent_date_returns_and_run_level_sharpe(self):
        """build_date_summary: per-date returns independent; summarize_trade_log Sharpe/drawdown on cycle series."""
        d1 = date(2024, 1, 5)
        d2 = date(2024, 1, 12)
        trade_log = pd.DataFrame(
            [
                _trade_row(d1, "S1", "short", pnl_total=500.0, capital_at_risk_dollars=2000.0),
                _trade_row(d1, "L1", "long", pnl_total=300.0, capital_at_risk_dollars=1500.0),
                _trade_row(d2, "S2", "short", pnl_total=200.0, capital_at_risk_dollars=1000.0),
            ]
        )

        date_summary = build_date_summary(trade_log)
        assert len(date_summary) == 2

        r1 = 800.0 / 3500.0
        r2 = 200.0 / 1000.0
        assert date_summary.iloc[0]["cycle_return_on_capital_at_risk"] == pytest.approx(r1)
        assert date_summary.iloc[1]["cycle_return_on_capital_at_risk"] == pytest.approx(r2)

        run_summary = summarize_trade_log(trade_log)
        per_date = date_summary["cycle_return_on_capital_at_risk"].dropna()
        expected_sharpe = float(
            per_date.mean() / per_date.std(ddof=1) * np.sqrt(52.0)
        )
        assert run_summary["annualized_sharpe"] == pytest.approx(expected_sharpe)
        assert run_summary["mean_cycle_return_on_capital_at_risk"] == pytest.approx(per_date.mean())

        curve = (1.0 + per_date).cumprod()
        dd = float((curve / curve.cummax() - 1.0).min())
        assert run_summary["max_drawdown"] == pytest.approx(dd)


class TestNoIncludedRows:
    """Tests for build_date_summary: rebalance dates with no included trades."""

    def test_only_excluded_rows_on_date(self):
        """build_date_summary: all-excluded date — zero sums, NaN cycle return (zero denominator)."""
        td = date(2024, 1, 5)
        trade_log = pd.DataFrame(
            [
                _trade_row(
                    td,
                    "X",
                    "short",
                    included=False,
                    pnl_total=500.0,
                    capital_at_risk_dollars=2000.0,
                ),
                _trade_row(
                    td,
                    "Y",
                    "long",
                    included=False,
                    pnl_total=300.0,
                    capital_at_risk_dollars=1500.0,
                ),
            ]
        )

        row = build_date_summary(trade_log).iloc[0]

        assert row["n_traded"] == 0
        assert row["cycle_pnl_total"] == pytest.approx(0.0)
        assert row["cycle_capital_at_risk"] == pytest.approx(0.0)
        assert np.isnan(row["cycle_return_on_capital_at_risk"])
        assert np.isnan(row["short_cycle_return"])
        assert np.isnan(row["long_cycle_return"])
