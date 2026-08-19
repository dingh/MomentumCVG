"""Synthetic tests for Sprint 006 D3 Commit 1 decision-report calculations."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.surface_decision_report import (
    PRIMARY_END,
    PRIMARY_START,
    DecisionMetricsError,
    assert_report_preconditions,
    compute_activity,
    compute_fill_assumption_sensitivity,
    compute_long_short_attribution,
    compute_top5_abs_pnl_concentration,
    compute_view_a,
    compute_view_b,
    compute_weekly_outcomes,
    evaluate_fill_window,
    filter_to_window,
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
