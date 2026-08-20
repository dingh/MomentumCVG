"""Synthetic tests for Sprint 006 D3 decision-report calculations."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.surface_decision_report import (
    CANDIDATE_VIEW_COLUMNS,
    PRIMARY_END,
    PRIMARY_START,
    SELECTION_BIAS_NOTICE,
    DecisionMetricsError,
    assert_report_preconditions,
    build_candidate_view,
    build_decision_report,
    classify_structure_reason_code,
    compute_activity,
    compute_fill_assumption_sensitivity,
    compute_long_short_attribution,
    compute_top5_abs_pnl_concentration,
    compute_view_a,
    compute_view_b,
    compute_weekly_outcomes,
    dumps_decision_report,
    evaluate_fill_window,
    filter_to_window,
    render_decision_report_markdown,
    summarize_funnel,
)


def _status_rows(rows: list[tuple[date, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"trade_date": d, "status": st, "reason": None} for d, st in rows]
    )


def _summary_row(
    trade_date: date,
    *,
    pnl: float,
    cap: float,
    n_traded: int = 1,
    long_n: int = 1,
    short_n: int = 0,
    long_cycle: float | None = None,
    short_cycle: float | None = None,
) -> dict:
    car = pnl / cap
    return {
        "trade_date": trade_date,
        "n_candidates": n_traded,
        "n_traded": n_traded,
        "cycle_pnl_total": pnl,
        "cycle_capital_at_risk": cap,
        "cycle_return_on_capital_at_risk": car,
        "long_n_traded": long_n,
        "short_n_traded": short_n,
        "long_cycle_return": np.nan if long_cycle is None else long_cycle,
        "short_cycle_return": np.nan if short_cycle is None else short_cycle,
        "long_cycle_pnl_total": pnl if long_n else 0.0,
        "short_cycle_pnl_total": pnl if short_n else 0.0,
        "long_cycle_capital_at_risk": cap if long_n else 0.0,
        "short_cycle_capital_at_risk": cap if short_n else 0.0,
    }


def _included_trade(
    trade_date: date,
    ticker: str,
    direction: str,
    pnl: float,
    cap: float,
    *,
    spread: float = 0.1,
    leg_spread: float = 0.2,
) -> dict:
    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": direction,
        "included_in_portfolio": True,
        "pnl_total": pnl,
        "capital_at_risk_dollars": cap,
        "spread_cost_ratio": spread,
        "leg_spread_to_credit_ratio": leg_spread,
    }


D1 = date(2020, 1, 3)
D2 = date(2020, 1, 10)
D3 = date(2020, 1, 17)
D4 = date(2020, 1, 24)


def test_view_a_excludes_valid_no_trade():
    status = _status_rows([(D1, "traded"), (D2, "valid_no_trade"), (D3, "traded")])
    summary = pd.DataFrame(
        [
            _summary_row(D1, pnl=10.0, cap=100.0),
            _summary_row(D3, pnl=30.0, cap=100.0),
        ]
    )
    assert_report_preconditions(status, summary, pd.DataFrame())
    view_a = compute_view_a(status, summary)
    assert view_a["n_traded_dates"] == 2
    assert view_a["n_valid_no_trade_dates"] == 1
    assert view_a["mean_cycle_car"] == pytest.approx(0.20)
    # 0.10 and 0.30; inserting a 0 for D2 would pull the mean to ~0.133.
    assert view_a["mean_cycle_car"] != pytest.approx(0.4 / 3.0)


def test_view_b_includes_valid_no_trade_as_zero():
    status = _status_rows([(D1, "traded"), (D2, "valid_no_trade"), (D3, "traded")])
    summary = pd.DataFrame(
        [
            _summary_row(D1, pnl=10.0, cap=100.0),
            _summary_row(D3, pnl=30.0, cap=100.0),
        ]
    )
    view_b = compute_view_b(status, summary)
    assert view_b["complete"] is True
    expected = (1.1 * 1.0 * 1.3) - 1.0
    assert view_b["compounded_return"] == pytest.approx(expected)


def test_hand_calculated_compounding_and_annualization():
    status = _status_rows([(D1, "traded"), (D2, "traded")])
    summary = pd.DataFrame(
        [
            _summary_row(D1, pnl=10.0, cap=100.0),
            _summary_row(D2, pnl=20.0, cap=100.0),
        ]
    )
    view_b = compute_view_b(status, summary)
    compounded = 1.1 * 1.2 - 1.0
    assert view_b["compounded_return"] == pytest.approx(compounded)
    expected_ann = (1.0 + compounded) ** (52.0 / 2.0) - 1.0
    assert view_b["annualized_return"] == pytest.approx(expected_ann)


def test_sharpe_fewer_than_two_or_zero_variance():
    one = _status_rows([(D1, "traded")])
    one_sum = pd.DataFrame([_summary_row(D1, pnl=10.0, cap=100.0)])
    view_a = compute_view_a(one, one_sum)
    assert np.isnan(view_a["annualized_sharpe"])

    zeros = _status_rows([(D1, "traded"), (D2, "traded")])
    zero_sum = pd.DataFrame(
        [
            _summary_row(D1, pnl=0.0, cap=100.0),
            _summary_row(D2, pnl=0.0, cap=100.0),
        ]
    )
    view_a_z = compute_view_a(zeros, zero_sum)
    assert np.isnan(view_a_z["annualized_sharpe"])
    view_b_z = compute_view_b(zeros, zero_sum)
    assert np.isnan(view_b_z["annualized_sharpe"])


def test_failed_dates_incomplete_and_never_zero_filled():
    status = _status_rows([(D1, "traded"), (D2, "failed"), (D3, "traded")])
    summary = pd.DataFrame(
        [
            _summary_row(D1, pnl=10.0, cap=100.0),
            _summary_row(D3, pnl=30.0, cap=100.0),
        ]
    )
    assert_report_preconditions(status, summary, pd.DataFrame())
    result = evaluate_fill_window(
        status, summary, pd.DataFrame(), start=D1, end=D3
    )
    assert result["result_complete"] is False
    assert result["view_b"]["complete"] is False
    assert result["view_b"]["compounded_return"] is None
    assert result["view_b"]["annualized_return"] is None
    assert result["view_b"]["annualized_sharpe"] is None
    assert result["view_b"]["max_drawdown"] is None
    # View A still uses traded dates only (0.10, 0.30), not a 0 for the failed date.
    assert result["view_a"]["mean_cycle_car"] == pytest.approx(0.20)
    assert result["activity"]["turnover"]["complete"] is False
    assert result["activity"]["turnover"]["mean_included_names"] is None


def test_missing_traded_date_summary_aborts():
    status = _status_rows([(D1, "traded"), (D2, "traded")])
    summary = pd.DataFrame([_summary_row(D1, pnl=10.0, cap=100.0)])
    with pytest.raises(DecisionMetricsError, match="2020-01-10"):
        assert_report_preconditions(status, summary, pd.DataFrame())


def test_duplicate_traded_date_summary_aborts():
    status = _status_rows([(D1, "traded")])
    summary = pd.DataFrame(
        [
            _summary_row(D1, pnl=10.0, cap=100.0),
            _summary_row(D1, pnl=12.0, cap=100.0),
        ]
    )
    with pytest.raises(DecisionMetricsError, match="2020-01-03"):
        assert_report_preconditions(status, summary, pd.DataFrame())


def test_non_finite_pnl_aborts():
    status = _status_rows([(D1, "traded")])
    row = _summary_row(D1, pnl=10.0, cap=100.0)
    row["cycle_pnl_total"] = float("nan")
    with pytest.raises(DecisionMetricsError, match="2020-01-03"):
        assert_report_preconditions(status, pd.DataFrame([row]), pd.DataFrame())


def test_non_finite_car_aborts():
    status = _status_rows([(D1, "traded")])
    row = _summary_row(D1, pnl=10.0, cap=100.0)
    row["cycle_return_on_capital_at_risk"] = float("nan")
    with pytest.raises(DecisionMetricsError, match="2020-01-03"):
        assert_report_preconditions(status, pd.DataFrame([row]), pd.DataFrame())


def test_non_positive_or_non_finite_capital_aborts():
    status = _status_rows([(D1, "traded")])
    for cap in (0.0, -5.0, float("nan")):
        row = _summary_row(D1, pnl=10.0, cap=100.0)
        row["cycle_capital_at_risk"] = cap
        with pytest.raises(DecisionMetricsError, match="2020-01-03"):
            assert_report_preconditions(status, pd.DataFrame([row]), pd.DataFrame())


def test_valid_no_trade_with_included_trade_aborts():
    status = _status_rows([(D1, "valid_no_trade")])
    log = pd.DataFrame([_included_trade(D1, "AAA", "long", 1.0, 10.0)])
    with pytest.raises(DecisionMetricsError, match="2020-01-03"):
        assert_report_preconditions(status, pd.DataFrame(), log)


def test_win_rate_ignores_calendar_no_trade_zeros():
    status = _status_rows([(D1, "traded"), (D2, "valid_no_trade"), (D3, "traded")])
    summary = pd.DataFrame(
        [
            _summary_row(D1, pnl=10.0, cap=100.0),
            _summary_row(D3, pnl=-5.0, cap=100.0),
        ]
    )
    weekly = compute_weekly_outcomes(status, summary)
    # One win, one loss among traded weeks. A 0-fill D2 must not count as a win or a third week.
    assert weekly["win_rate"] == pytest.approx(0.5)
    assert weekly["no_trade_frequency"] == pytest.approx(1.0 / 3.0)


def test_profit_factor_ordinary_no_loss_and_all_zero():
    status = _status_rows([(D1, "traded"), (D2, "traded")])
    mixed = pd.DataFrame(
        [
            _summary_row(D1, pnl=20.0, cap=100.0),
            _summary_row(D2, pnl=-5.0, cap=100.0),
        ]
    )
    assert compute_weekly_outcomes(status, mixed)["profit_factor"] == pytest.approx(4.0)

    no_loss = pd.DataFrame(
        [
            _summary_row(D1, pnl=20.0, cap=100.0),
            _summary_row(D2, pnl=5.0, cap=100.0),
        ]
    )
    assert compute_weekly_outcomes(status, no_loss)["profit_factor"] == float("inf")

    zeros = pd.DataFrame(
        [
            _summary_row(D1, pnl=0.0, cap=100.0),
            _summary_row(D2, pnl=0.0, cap=100.0),
        ]
    )
    assert np.isnan(compute_weekly_outcomes(status, zeros)["profit_factor"])


def test_closed_window_filtering_and_yearly_grouping():
    early = date(2019, 6, 7)
    late = date(2020, 6, 5)
    outside = date(2021, 1, 8)
    status = _status_rows(
        [
            (early, "traded"),
            (late, "traded"),
            (outside, "traded"),
        ]
    )
    summary = pd.DataFrame(
        [
            _summary_row(early, pnl=10.0, cap=100.0),
            _summary_row(late, pnl=20.0, cap=100.0),
            _summary_row(outside, pnl=40.0, cap=100.0),
        ]
    )
    log = pd.DataFrame(
        [
            _included_trade(early, "AAA", "long", 10.0, 100.0),
            _included_trade(late, "BBB", "long", 20.0, 100.0),
            _included_trade(outside, "CCC", "long", 40.0, 100.0),
        ]
    )
    result = evaluate_fill_window(
        status, summary, log, start=PRIMARY_START, end=PRIMARY_END
    )
    assert result["date_class_counts"]["n_expected_dates"] == 2
    assert result["date_class_counts"]["first_date"] == late
    assert result["date_class_counts"]["last_date"] == outside
    years = {item["year"] for item in result["yearly"]}
    assert years == {2020, 2021}
    y2020 = next(item for item in result["yearly"] if item["year"] == 2020)
    assert y2020["date_class_counts"]["n_traded_dates"] == 1
    assert y2020["view_a"]["mean_cycle_car"] == pytest.approx(0.20)


def test_missing_side_not_zero_filled_into_the_other():
    status = _status_rows([(D1, "traded"), (D2, "traded")])
    summary = pd.DataFrame(
        [
            _summary_row(D1, pnl=10.0, cap=100.0, long_n=1, short_n=0, long_cycle=0.10),
            _summary_row(D2, pnl=20.0, cap=100.0, long_n=0, short_n=1, short_cycle=0.20),
        ]
    )
    log = pd.DataFrame(
        [
            _included_trade(D1, "AAA", "long", 10.0, 100.0),
            _included_trade(D2, "BBB", "short", 20.0, 100.0),
        ]
    )
    attr = compute_long_short_attribution(status, summary, log)
    assert attr["long"]["n_traded_rows"] == 1
    assert attr["short"]["n_traded_rows"] == 1
    assert attr["long"]["mean_cycle_return"] == pytest.approx(0.10)
    assert attr["short"]["mean_cycle_return"] == pytest.approx(0.20)
    # A 0-fill of the missing side on D1 would pull short mean to 0.10.


def test_mid_cross_unmatched_dates_and_candidates_disclosed():
    cross_status = _status_rows([(D1, "traded"), (D2, "traded"), (D3, "traded")])
    mid_status = _status_rows([(D1, "traded"), (D2, "valid_no_trade"), (D4, "traded")])
    cross_sum = pd.DataFrame(
        [
            _summary_row(D1, pnl=10.0, cap=100.0),
            _summary_row(D2, pnl=20.0, cap=100.0),
            _summary_row(D3, pnl=5.0, cap=100.0),
        ]
    )
    mid_sum = pd.DataFrame(
        [
            _summary_row(D1, pnl=8.0, cap=100.0),
            _summary_row(D4, pnl=7.0, cap=100.0),
        ]
    )
    cross_log = pd.DataFrame(
        [
            _included_trade(D1, "AAA", "long", 10.0, 100.0, spread=0.2),
            _included_trade(D2, "BBB", "short", 20.0, 100.0, spread=0.3),
            _included_trade(D3, "CCC", "long", 5.0, 100.0, spread=0.1),
        ]
    )
    mid_log = pd.DataFrame(
        [
            _included_trade(D1, "AAA", "long", 8.0, 100.0, spread=0.05),
            _included_trade(D4, "DDD", "long", 7.0, 100.0, spread=0.04),
        ]
    )
    sens = compute_fill_assumption_sensitivity(
        cross_date_status=cross_status,
        cross_date_summary=cross_sum,
        cross_trade_log=cross_log,
        mid_date_status=mid_status,
        mid_date_summary=mid_sum,
        mid_trade_log=mid_log,
        start=D1,
        end=D4,
    )
    assert sens["n_dates_both_traded"] == 1
    assert sens["dates_both_traded"] == [D1]
    assert sens["dates_cross_only"] == [D2, D3]
    assert sens["dates_mid_only"] == [D4]
    assert sens["mean_cross_minus_mid_car_both_traded"] == pytest.approx(0.02)
    assert sens["mean_cross_minus_mid_pnl_both_traded"] == pytest.approx(2.0)
    assert (D2, "BBB", "short") in sens["candidates_cross_only"]
    assert (D3, "CCC", "long") in sens["candidates_cross_only"]
    assert (D4, "DDD", "long") in sens["candidates_mid_only"]
    assert sens["n_candidates_both_included"] == 1


def test_top_five_concentration_share_sum_at_most_one():
    status = _status_rows([(D1, "traded")])
    summary = pd.DataFrame([_summary_row(D1, pnl=100.0, cap=100.0, n_traded=6)])
    log = pd.DataFrame(
        [
            _included_trade(D1, "A", "long", 40.0, 10.0),
            _included_trade(D1, "B", "long", -30.0, 10.0),
            _included_trade(D1, "C", "short", 20.0, 10.0),
            _included_trade(D1, "D", "short", -5.0, 10.0),
            _included_trade(D1, "E", "long", 3.0, 10.0),
            _included_trade(D1, "F", "long", 2.0, 10.0),
        ]
    )
    conc = compute_top5_abs_pnl_concentration(log)
    assert len(conc["top5"]) == 5
    assert conc["top5_share_sum"] <= 1.0 + 1e-12
    assert conc["top5_share_sum"] == pytest.approx((40 + 30 + 20 + 5 + 3) / 100.0)
    result = evaluate_fill_window(status, summary, log, start=D1, end=D1)
    assert result["concentration"]["top5_share_sum"] <= 1.0 + 1e-12


def test_turnover_zero_only_for_valid_no_trade_never_failed():
    status = _status_rows(
        [(D1, "traded"), (D2, "valid_no_trade"), (D3, "failed")]
    )
    summary = pd.DataFrame(
        [_summary_row(D1, pnl=10.0, cap=100.0, n_traded=4, long_n=2, short_n=2)]
    )
    activity = compute_activity(status, summary)
    assert activity["turnover"]["complete"] is False
    assert activity["turnover"]["mean_included_names"] is None
    # Diagnostic mean uses traded=4 and VNT=0 only — never a failed 0 that would make 4/3.
    assert activity["turnover"]["diagnostic_mean_included_names_traded_and_vnt"] == pytest.approx(2.0)
    assert activity["turnover"]["n_failed_dates_excluded"] == 1

    complete_status = _status_rows([(D1, "traded"), (D2, "valid_no_trade")])
    complete_activity = compute_activity(complete_status, summary)
    assert complete_activity["turnover"]["complete"] is True
    assert complete_activity["turnover"]["mean_included_names"] == pytest.approx(2.0)


def test_window_constants_match_frozen_d0():
    assert PRIMARY_START == date(2020, 1, 1)
    assert PRIMARY_END == date(2026, 7, 10)
    filtered_status, _, _ = filter_to_window(
        _status_rows(
            [
                (date(2019, 12, 27), "traded"),
                (date(2020, 1, 3), "traded"),
                (date(2026, 7, 10), "traded"),
                (date(2026, 7, 17), "traded"),
            ]
        ),
        pd.DataFrame(),
        pd.DataFrame(),
        PRIMARY_START,
        PRIMARY_END,
    )
    assert [d for d in filtered_status["trade_date"]] == [
        date(2020, 1, 3),
        date(2026, 7, 10),
    ]


# =============================================================================
# Commit 3 — candidate view, funnel totals, report serialization
# =============================================================================

def _empty_legs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "run_id",
            "fill_label",
            "trade_date",
            "ticker",
            "direction",
            "leg_index",
            "included_in_portfolio",
            "entry_cash_per_unit",
            "expiry_payoff_per_unit",
            "pnl_per_unit",
            "pnl_total_leg",
        ]
    )


def _trade_row(
    trade_date: date,
    *,
    ticker: str = "AAA",
    direction: str = "long",
    included: bool = True,
    structure_ok: bool = True,
    failure_reason=None,
    exclusion_reason=None,
    pnl_total: float = 10.0,
    fill_label: str = "cross",
    run_id: str = "run_cross",
) -> dict:
    return {
        "run_id": run_id,
        "fill_label": fill_label,
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": direction,
        "included_in_portfolio": included,
        "structure_ok": structure_ok,
        "failure_reason": failure_reason,
        "exclusion_reason": exclusion_reason,
        "pnl_total": pnl_total,
        "instrument_type": "long_straddle",
        "entry_cost_per_share": 1.0,
        "pnl_per_share": 0.1,
        "capital_at_risk_dollars": 100.0,
    }


def _funnel_row(trade_date: date, **overrides) -> dict:
    row = {
        "run_id": "run_cross",
        "fill_label": "cross",
        "trade_date": trade_date,
        "n_expected": 1,
        "n_feature_covered": 1,
        "n_universe": 10,
        "n_jointly_eligible": 8,
        "n_post_signal": 4,
        "n_post_signal_long": 2,
        "n_post_signal_short": 2,
        "n_constructable": 3,
        "n_constructable_long": 2,
        "n_constructable_short": 1,
        "n_included": 2,
        "n_included_long": 1,
        "n_included_short": 1,
        "date_status": "traded",
        "date_reason": None,
    }
    row.update(overrides)
    return row


def _pack(
    *,
    fill_label: str,
    run_id: str,
    status: pd.DataFrame,
    summary: pd.DataFrame,
    trade_log: pd.DataFrame,
    funnel: pd.DataFrame | None = None,
    legs: pd.DataFrame | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "fill_label": fill_label,
        "date_status": status,
        "date_summary": summary,
        "trade_log": trade_log,
        "funnel_summary": funnel if funnel is not None else pd.DataFrame([_funnel_row(D1)]),
        "leg_log": legs if legs is not None else _empty_legs(),
    }


class TestCandidateViewMappings:
    def test_traded_structure_failed_and_portfolio_excluded(self):
        log = pd.DataFrame(
            [
                _trade_row(D1, included=True, structure_ok=True),
                _trade_row(
                    D1,
                    ticker="BBB",
                    included=False,
                    structure_ok=False,
                    failure_reason="metadata_error: surface_valid=False",
                ),
                _trade_row(
                    D1,
                    ticker="CCC",
                    included=False,
                    structure_ok=True,
                    exclusion_reason="max_names_cap",
                ),
            ]
        )
        view = build_candidate_view(log, run_id="run_cross", fill_label="cross")
        assert list(view.columns) == CANDIDATE_VIEW_COLUMNS
        by_ticker = view.set_index("ticker")
        assert by_ticker.loc["AAA", "decision_status"] == "traded"
        assert by_ticker.loc["AAA", "stage"] == "traded"
        assert pd.isna(by_ticker.loc["AAA", "reason_code"])
        assert by_ticker.loc["BBB", "stage"] == "structure_failed"
        assert by_ticker.loc["BBB", "decision_status"] == "no_trade"
        assert by_ticker.loc["BBB", "reason_code"] == "metadata_error"
        assert by_ticker.loc["BBB", "reason_raw"].startswith("metadata_error:")
        assert by_ticker.loc["CCC", "stage"] == "portfolio_excluded"
        assert by_ticker.loc["CCC", "reason_code"] == "max_names_cap"

    @pytest.mark.parametrize(
        "failure,code",
        [
            ("No quote surface rows for X", "missing_quotes_or_body"),
            ("No eligible quotes available", "missing_quotes_or_body"),
            ("Missing body call/put for X", "missing_quotes_or_body"),
            ("Missing tradeable body call/put for X", "missing_quotes_or_body"),
            ("No quotes with abs_delta for X", "wing_or_liquidity_selection"),
            ("Iron fly spread_cost_ratio=1.2 exceeds 1.0", "wing_or_liquidity_selection"),
            ("unexpected builder failure", "other_structure"),
        ],
    )
    def test_structure_reason_prefixes(self, failure, code):
        assert classify_structure_reason_code(failure) == code

    def test_structure_failure_not_mapped_to_no_tradeable_structure(self):
        log = pd.DataFrame(
            [
                _trade_row(
                    D1,
                    included=False,
                    structure_ok=False,
                    failure_reason="No eligible quotes",
                    exclusion_reason="no_tradeable_structure",
                )
            ]
        )
        view = build_candidate_view(log, run_id="r", fill_label="cross")
        assert view.iloc[0]["reason_code"] == "missing_quotes_or_body"
        assert view.iloc[0]["reason_raw"] == "No eligible quotes"


class TestFunnelAndReportAssembly:
    def test_funnel_averages_skip_nulls(self):
        funnel = pd.DataFrame(
            [
                _funnel_row(D1, n_jointly_eligible=10, n_included=2),
                _funnel_row(
                    D2,
                    n_feature_covered=0,
                    n_universe=None,
                    n_jointly_eligible=None,
                    n_post_signal=None,
                    n_constructable=None,
                    n_included=None,
                    date_status="failed",
                    date_reason="missing_features",
                ),
                _funnel_row(D3, n_jointly_eligible=0, n_included=0, date_status="valid_no_trade"),
            ]
        )
        totals = summarize_funnel(funnel)
        assert totals["n_expected_dates"] == 3
        assert totals["n_feature_covered_dates"] == 2
        assert totals["n_dates_with_jointly_eligible"] == 2
        assert totals["mean_jointly_eligible"] == pytest.approx(5.0)
        assert totals["sum_included"] == pytest.approx(2.0)
        assert SELECTION_BIAS_NOTICE in totals["selection_bias_notice"]

    def _complete_packs(self):
        status = _status_rows([(D1, "traded"), (D2, "valid_no_trade")])
        summary = pd.DataFrame([_summary_row(D1, pnl=50.0, cap=100.0)])
        # No included rows → empty legs satisfy integrity; structure failures still counted.
        cross_log = pd.DataFrame(
            [
                _trade_row(
                    D1,
                    ticker="ZZZ",
                    included=False,
                    structure_ok=False,
                    failure_reason="metadata_error: bad",
                    fill_label="cross",
                    run_id="run_cross",
                    pnl_total=0.0,
                ),
                _trade_row(
                    D1,
                    ticker="YYY",
                    included=False,
                    structure_ok=True,
                    exclusion_reason="max_names_cap",
                    fill_label="cross",
                    run_id="run_cross",
                    pnl_total=0.0,
                ),
            ]
        )
        mid_log = pd.DataFrame(
            [
                _trade_row(
                    D1,
                    ticker="ZZZ",
                    included=False,
                    structure_ok=False,
                    failure_reason="No eligible quotes",
                    fill_label="mid",
                    run_id="run_mid",
                    pnl_total=0.0,
                )
            ]
        )
        funnel_cross = pd.DataFrame(
            [_funnel_row(D1), _funnel_row(D2, n_included=0, date_status="valid_no_trade")]
        )
        funnel_mid = funnel_cross.copy()
        funnel_mid["fill_label"] = "mid"
        funnel_mid["run_id"] = "run_mid"
        return (
            _pack(
                fill_label="mid",
                run_id="run_mid",
                status=status,
                summary=summary,
                trade_log=mid_log,
                funnel=funnel_mid,
            ),
            _pack(
                fill_label="cross",
                run_id="run_cross",
                status=status,
                summary=summary,
                trade_log=cross_log,
                funnel=funnel_cross,
            ),
        )

    def test_deterministic_json_and_infinity_serialization(self):
        mid, cross = self._complete_packs()
        report = build_decision_report(
            mid=mid,
            cross=cross,
            experiment_id="sprint006_baseline_v1",
            contract_id="sprint006_baseline_v1",
            repo_sha="a" * 40,
        )
        text1 = dumps_decision_report(report)
        text2 = dumps_decision_report(report)
        assert text1 == text2
        payload = json.loads(text1)
        assert payload["result_complete"] is True
        assert payload["has_unresolved_failures"] is False
        assert payload["by_fill"]["cross"]["full_history"]["weekly"]["profit_factor"] == "Infinity"
        assert "NaN" not in text1
        assert payload["concentration_primary_cross_top5"]["top5_share_sum"] <= 1.0 + 1e-12
        md = render_decision_report_markdown(report)
        assert md.startswith("# Sprint 006 baseline decision report")
        assert "INCOMPLETE RESULT" not in md
        assert SELECTION_BIAS_NOTICE in md
        assert "Cross (primary)" in md

    def test_markdown_renders_injected_sentinel_values(self):
        """Inject distinctive values into a built report and assert exact Markdown cells."""
        mid, cross = self._complete_packs()
        report = build_decision_report(
            mid=mid,
            cross=cross,
            experiment_id="sprint006_baseline_v1",
            contract_id="sprint006_baseline_v1",
            repo_sha="a" * 40,
        )

        # Distinctive sentinels — unlikely to appear from the synthetic pack.
        view_b_ann = 0.271828
        yearly_ann = 0.161803
        yearly_dd = -0.041421
        distx_abs = 1234.5
        distx_share = 0.314159
        top5_sum = 0.618033
        car_delta = -0.012345
        pnl_delta = -67.89
        scr_cross = 0.111111
        scr_mid = 0.222222
        lscr_cross = 0.333333
        lscr_mid = 0.444444

        report["by_fill"]["cross"]["full_history"]["view_b_calendar"][
            "annualized_return"
        ] = view_b_ann
        report["by_fill"]["cross"]["primary"]["yearly"] = [
            {
                "year": 2020,
                "date_class_counts": {
                    "n_expected_dates": 2,
                    "n_traded_dates": 1,
                },
                "view_b": {
                    "compounded_return": 0.05,
                    "annualized_return": yearly_ann,
                    "annualized_sharpe": 0.9,
                    "max_drawdown": yearly_dd,
                },
            }
        ]
        report["concentration_primary_cross_top5"] = {
            "top5": [
                {"ticker": "DISTX", "abs_pnl": distx_abs, "share": distx_share},
            ],
            "top5_share_sum": top5_sum,
            "total_abs_pnl": distx_abs,
        }
        for window_name in ("full_history", "primary"):
            sens = report["fill_assumption_sensitivity"][window_name]
            sens["mean_cross_minus_mid_car_both_traded"] = car_delta
            sens["mean_cross_minus_mid_pnl_both_traded"] = pnl_delta
            sens["mean_spread_cost_ratio_cross"] = scr_cross
            sens["mean_spread_cost_ratio_mid"] = scr_mid
            sens["mean_leg_spread_to_credit_ratio_cross"] = lscr_cross
            sens["mean_leg_spread_to_credit_ratio_mid"] = lscr_mid

        md = render_decision_report_markdown(report)

        assert f"| full_history | B calendar |" in md
        assert f"{view_b_ann:.6g}" in md
        assert (
            f"| 2020 | 2 | 1 | 0.05 | {yearly_ann:.6g} | 0.9 | {yearly_dd:.6g} |"
            in md
        )
        assert f"| DISTX | {distx_abs:.6g} | {distx_share:.6g} |" in md
        assert f"top-5 |PnL| aggregate share (primary cross): {top5_sum:.6g}" in md
        assert f"{car_delta:.6g}" in md
        assert f"{pnl_delta:.6g}" in md
        assert f"{scr_cross:.6g}" in md
        assert f"{scr_mid:.6g}" in md
        assert f"{lscr_cross:.6g}" in md
        assert f"{lscr_mid:.6g}" in md
        # Sensitivity row must carry all six diagnostics together.
        assert (
            f"| full_history | 1 | 0 | 0 | 0 | 0 | "
            f"{car_delta:.6g} | {pnl_delta:.6g} | "
            f"{scr_cross:.6g} | {scr_mid:.6g} | "
            f"{lscr_cross:.6g} | {lscr_mid:.6g} |"
        ) in md

    def test_incomplete_results_banner_when_failed_dates_exist(self):
        mid, cross = self._complete_packs()
        failed_status = _status_rows(
            [(D1, "traded"), (D2, "valid_no_trade"), (D3, "failed")]
        )
        mid["date_status"] = failed_status
        cross["date_status"] = failed_status
        report = build_decision_report(
            mid=mid,
            cross=cross,
            experiment_id="sprint006_baseline_v1",
            contract_id="sprint006_baseline_v1",
            repo_sha="b" * 40,
        )
        assert report["result_complete"] is False
        assert report["has_unresolved_failures"] is True
        md = render_decision_report_markdown(report)
        assert "INCOMPLETE RESULT" in md

    def test_integrity_check_aborts_before_report(self):
        mid, cross = self._complete_packs()
        cross["trade_log"] = pd.DataFrame(
            [_trade_row(D1, fill_label="cross", run_id="run_cross", included=True)]
        )
        cross["leg_log"] = _empty_legs()
        with pytest.raises(DecisionMetricsError, match="no matching leg rows"):
            build_decision_report(
                mid=mid,
                cross=cross,
                experiment_id="sprint006_baseline_v1",
                contract_id="sprint006_baseline_v1",
                repo_sha="c" * 40,
            )

    def test_fill_sensitivity_and_structure_counts_present(self):
        mid, cross = self._complete_packs()
        report = build_decision_report(
            mid=mid,
            cross=cross,
            experiment_id="sprint006_baseline_v1",
            contract_id="sprint006_baseline_v1",
            repo_sha="d" * 40,
        )
        sens = report["fill_assumption_sensitivity"]["full_history"]
        assert sens["n_dates_both_traded"] == 1
        assert "mean_cross_minus_mid_car_both_traded" in sens
        counts = report["by_fill"]["cross"]["full_history"]["structure_failure_counts"]
        assert counts["metadata_error"] == 1
        assert counts["other_structure"] == 0
        assert SELECTION_BIAS_NOTICE in report["limitations"]
