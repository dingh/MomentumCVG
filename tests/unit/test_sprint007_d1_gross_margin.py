"""Sprint 007 D1 gross-margin gate tests (synthetic mid frames only)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.sprint007_d1_gross_margin import (
    BREADTH_TOP_N,
    EXPECTED_INCLUDED_TRADES,
    VERDICT_CONTINUE,
    VERDICT_STOP,
    MidPrimaryBundle,
    compute_d1_gate_scorecard,
    pnl_excluding_top_groups,
    reconcile_mid_primary,
    select_included_traded_rows,
    yearly_pnl_table,
)


def _trade(
    *,
    trade_date: date,
    ticker: str,
    direction: str,
    pnl_total: float,
    included: bool = True,
    capital: float = 1000.0,
) -> dict:
    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": direction,
        "included_in_portfolio": included,
        "pnl_total": pnl_total,
        "capital_at_risk_dollars": capital,
        "fill_label": "mid",
    }


def _bundle(trades: list[dict]) -> MidPrimaryBundle:
    trade_log = pd.DataFrame(trades)
    dates = sorted({row["trade_date"] for row in trades})
    date_status = pd.DataFrame(
        {"trade_date": dates, "status": ["traded"] * len(dates), "reason": [""] * len(dates)}
    )
    summary_rows = []
    for d in dates:
        day = trade_log[
            (trade_log["trade_date"] == d) & (trade_log["included_in_portfolio"] == True)  # noqa: E712
        ]
        pnl = float(day["pnl_total"].sum())
        capital = float(day["capital_at_risk_dollars"].sum())
        long_day = day[day["direction"] == "long"]
        short_day = day[day["direction"] == "short"]
        summary_rows.append(
            {
                "trade_date": d,
                "cycle_pnl_total": pnl,
                "cycle_capital_at_risk": capital,
                "cycle_return_on_capital_at_risk": pnl / capital if capital else 0.0,
                "short_cycle_pnl_total": float(short_day["pnl_total"].sum()),
                "short_cycle_return": 0.0,
                "long_cycle_pnl_total": float(long_day["pnl_total"].sum()),
                "long_cycle_return": 0.0,
            }
        )
    date_summary = pd.DataFrame(summary_rows)
    return MidPrimaryBundle(
        run_dir=None,
        date_status=date_status,
        date_summary=date_summary,
        trade_log=trade_log,
        included=select_included_traded_rows(trade_log, date_status),
    )


def _broad_profitable_trades() -> list[dict]:
    """Two years, many tickers, both sides positive — passes all four gate parts."""
    trades: list[dict] = []
    for year in (2021, 2022):
        for day in range(1, 13):
            trade_date = date(year, 1, day)
            for idx in range(8):
                trades.append(
                    _trade(
                        trade_date=trade_date,
                        ticker=f"T{idx}",
                        direction="long" if idx % 2 == 0 else "short",
                        pnl_total=100.0,
                    )
                )
    return trades


def test_select_included_traded_rows_drops_excluded_and_untraded() -> None:
    trade_log = pd.DataFrame(
        [
            _trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", pnl_total=10.0),
            _trade(
                trade_date=date(2021, 1, 4),
                ticker="BBB",
                direction="short",
                pnl_total=5.0,
                included=False,
            ),
            _trade(trade_date=date(2021, 1, 5), ticker="CCC", direction="long", pnl_total=7.0),
        ]
    )
    date_status = pd.DataFrame(
        {
            "trade_date": [date(2021, 1, 4), date(2021, 1, 5)],
            "status": ["traded", "valid_no_trade"],
            "reason": ["", ""],
        }
    )
    included = select_included_traded_rows(trade_log, date_status)
    assert len(included) == 1
    assert included.iloc[0]["ticker"] == "AAA"


def test_pnl_excluding_top_groups_removes_five_highest() -> None:
    included = pd.DataFrame(
        [
            _trade(trade_date=date(2021, 1, 4), ticker=f"T{i}", direction="long", pnl_total=pnl)
            for i, pnl in enumerate([500.0, 400.0, 300.0, 200.0, 100.0, 10.0, 10.0])
        ]
    )
    remaining, excluded = pnl_excluding_top_groups(included, "ticker")
    assert len(excluded) == BREADTH_TOP_N
    assert remaining == pytest.approx(20.0)


def test_yearly_pnl_table_buckets_by_calendar_year() -> None:
    included = pd.DataFrame(
        [
            _trade(trade_date=date(2021, 3, 1), ticker="AAA", direction="long", pnl_total=10.0),
            _trade(trade_date=date(2022, 3, 1), ticker="AAA", direction="long", pnl_total=-4.0),
        ]
    )
    table = yearly_pnl_table(included)
    assert table["year"].tolist() == [2021, 2022]
    assert table["year_pnl"].tolist() == pytest.approx([10.0, -4.0])


def test_scorecard_continues_when_all_four_gate_parts_pass() -> None:
    scorecard = compute_d1_gate_scorecard(_bundle(_broad_profitable_trades()))
    assert scorecard.verdict == VERDICT_CONTINUE
    assert all(part.passed for part in scorecard.parts)


def test_sign_gate_fails_on_negative_total_pnl() -> None:
    trades = [
        _trade(trade_date=date(2021, 1, d), ticker="AAA", direction="long", pnl_total=-50.0)
        for d in range(1, 6)
    ]
    scorecard = compute_d1_gate_scorecard(_bundle(trades))
    sign = next(p for p in scorecard.parts if p.part_id == "G-Sign")
    assert not sign.passed
    assert scorecard.verdict == VERDICT_STOP


def test_breadth_gate_fails_when_profit_is_a_five_ticker_artifact() -> None:
    trades = [
        _trade(trade_date=date(2021, 1, 4), ticker=f"WIN{i}", direction="long", pnl_total=1000.0)
        for i in range(5)
    ]
    trades += [
        _trade(trade_date=date(2022, 1, 4), ticker=f"LOSE{i}", direction="short", pnl_total=-10.0)
        for i in range(5)
    ]
    scorecard = compute_d1_gate_scorecard(_bundle(trades))
    breadth = next(p for p in scorecard.parts if p.part_id == "G-Breadth")
    assert not breadth.passed
    assert breadth.metrics["total_pnl_excl_top5_tickers"] < 0.0
    assert scorecard.verdict == VERDICT_STOP


def test_breadth_gate_fails_when_profit_is_a_five_date_artifact() -> None:
    trades = [
        _trade(trade_date=date(2021, 1, d), ticker=f"T{d}", direction="long", pnl_total=1000.0)
        for d in range(1, 6)
    ]
    trades += [
        _trade(trade_date=date(2022, 2, d), ticker=f"S{d}", direction="short", pnl_total=-50.0)
        for d in range(1, 7)
    ]
    scorecard = compute_d1_gate_scorecard(_bundle(trades))
    breadth = next(p for p in scorecard.parts if p.part_id == "G-Breadth")
    assert not breadth.passed
    assert breadth.metrics["total_pnl_excl_top5_dates"] < 0.0


def test_location_gate_fails_when_profitable_side_is_below_trade_share() -> None:
    trades = [
        _trade(trade_date=date(2021, 1, 4), ticker="WIN", direction="long", pnl_total=5000.0)
    ]
    trades += [
        _trade(trade_date=date(2022, 1, 4), ticker=f"L{i}", direction="short", pnl_total=-10.0)
        for i in range(29)
    ]
    scorecard = compute_d1_gate_scorecard(_bundle(trades))
    location = next(p for p in scorecard.parts if p.part_id == "G-Location")
    assert not location.passed
    assert location.metrics["qualifying_sides"] == []


def test_location_gate_passes_when_one_side_carries_margin() -> None:
    scorecard = compute_d1_gate_scorecard(_bundle(_broad_profitable_trades()))
    location = next(p for p in scorecard.parts if p.part_id == "G-Location")
    assert location.passed
    assert set(location.metrics["qualifying_sides"]) == {"long", "short"}


def test_stability_gate_fails_on_single_positive_year() -> None:
    trades = [
        _trade(trade_date=date(2021, 1, d), ticker=f"T{d}", direction="long", pnl_total=400.0)
        for d in range(1, 11)
    ]
    trades += [
        _trade(trade_date=date(2022, 1, d), ticker=f"S{d}", direction="short", pnl_total=-100.0)
        for d in range(1, 11)
    ]
    scorecard = compute_d1_gate_scorecard(_bundle(trades))
    stability = next(p for p in scorecard.parts if p.part_id == "G-Stability")
    assert not stability.passed
    assert stability.metrics["n_years_with_positive_pnl"] == 1
    assert stability.metrics["best_year"] == 2021


def test_stability_gate_fails_when_best_year_carries_all_margin() -> None:
    trades = [
        _trade(trade_date=date(2021, 1, d), ticker=f"T{d}", direction="long", pnl_total=1000.0)
        for d in range(1, 11)
    ]
    trades += [
        _trade(trade_date=date(2022, 1, d), ticker=f"U{d}", direction="long", pnl_total=1.0)
        for d in range(1, 11)
    ]
    trades += [
        _trade(trade_date=date(2023, 1, d), ticker=f"S{d}", direction="short", pnl_total=-100.0)
        for d in range(1, 11)
    ]
    scorecard = compute_d1_gate_scorecard(_bundle(trades))
    stability = next(p for p in scorecard.parts if p.part_id == "G-Stability")
    assert stability.metrics["n_years_with_positive_pnl"] == 2
    assert stability.metrics["total_pnl_excl_best_year"] < 0.0
    assert not stability.passed


def _reference_report(bundle: MidPrimaryBundle) -> dict:
    included = bundle.included
    long_rows = included[included["direction"] == "long"]
    short_rows = included[included["direction"] == "short"]
    return {
        "long_short": {
            "long": {
                "pnl_total": float(long_rows["pnl_total"].sum()),
                "n_traded_rows": int(len(long_rows)),
            },
            "short": {
                "pnl_total": float(short_rows["pnl_total"].sum()),
                "n_traded_rows": int(len(short_rows)),
            },
        },
        "view_a_conditional": {
            "mean_cycle_car": float(
                bundle.date_summary["cycle_return_on_capital_at_risk"].mean()
            )
        },
        "date_class_counts": {"n_traded_dates": int(len(bundle.date_status))},
    }


def test_reconciliation_flags_dollar_mismatch_above_one_cent() -> None:
    bundle = _bundle(_broad_profitable_trades())
    report = _reference_report(bundle)
    report["long_short"]["long"]["pnl_total"] += 0.05
    rows = {row.metric: row for row in reconcile_mid_primary(bundle, report).rows}
    assert not rows["total_pnl"].passed


def test_reconciliation_tolerates_sub_cent_dollar_drift() -> None:
    bundle = _bundle(_broad_profitable_trades())
    report = _reference_report(bundle)
    report["long_short"]["long"]["pnl_total"] += 0.005
    rows = {row.metric: row for row in reconcile_mid_primary(bundle, report).rows}
    assert rows["total_pnl"].passed


def test_reconciliation_requires_expected_included_trade_count() -> None:
    bundle = _bundle(_broad_profitable_trades())
    report = _reference_report(bundle)
    rows = {row.metric: row for row in reconcile_mid_primary(bundle, report).rows}
    assert rows["n_included_trades"].passed
    assert not rows["expected_included_trades"].passed
    assert rows["expected_included_trades"].reference == EXPECTED_INCLUDED_TRADES
