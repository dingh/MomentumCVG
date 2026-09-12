"""Sprint 008 D2 tests — synthetic frames only. Do not open official evaluation outcomes."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.sprint008_d0_input_readiness import BUDGET_B
from src.backtest.sprint008_d1_cost_diagnosis import (
    CostDiagnosisError,
    attach_decomposition_columns,
    build_portfolio_comparison,
    fixed_budget_max_drawdown,
    require_all_executed_outcomes,
)
from src.backtest.sprint008_d1_measurement_validation import attach_scenario_economics
from src.backtest.sprint008_d1_within_date_followup import (
    bonferroni_adjust_p,
    select_within_date_groups,
)
from src.backtest.sprint008_d2_fixed_exclusion_validation import (
    ADJUSTED_CI_LEVEL,
    D2_REPORTING_PERIODS,
    EVAL_END,
    EVAL_START,
    FAMILY_SIZE,
    D2ValidationError,
    assign_eval_u_labels,
    classify_evaluation_calendar,
    interpret_adjusted_interval,
    run_d2_inference,
)


def _row(
    *,
    trade_date: date,
    ticker: str,
    m1: float,
    x: float = 1.0,
    crossed: bool = False,
) -> dict:
    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": "long",
        "in_N": True,
        "structure_ok": True,
        "M": 1.0,
        "H": 0.1,
        "X": x,
        "S0": 100.0,
        "M1": m1,
        "M2": 0.01 * m1,
        "outcome_finite": np.isfinite(x),
        "analysis_eligible": not crossed,
        "crossed_quote_excluded": crossed,
    }


def _labeled(dates: list[date], *, x_for_u: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for d in dates:
        for i, ticker in enumerate(list("ABCDE"), start=1):
            x = x_for_u if ticker == "E" else (1.4 if ticker == "A" else 1.0)
            rows.append(_row(trade_date=d, ticker=ticker, m1=float(i), x=x))
    econ = attach_decomposition_columns(attach_scenario_economics(pd.DataFrame(rows), 1.0))
    econ, paired, _excluded = assign_eval_u_labels(econ, "M1")
    return econ, paired


def _calendar(dates: list[date], n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": dates,
            "n_in_N": [n] * len(dates),
            "long_book_class": ["has_long_candidates"] * len(dates),
        }
    )


def _status(dates: list[date], status: str = "valid_no_trade", reason: str | None = "no_included_names") -> pd.DataFrame:
    return pd.DataFrame({"trade_date": dates, "status": status, "reason": reason})


def _funnel(dates: list[date], *, n_con: int, n_inc: int, n_post: int = 0, n_short: int = 0, status: str = "valid_no_trade", reason: str | None = "no_included_names") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": dates,
            "date_status": status,
            "date_reason": reason,
            "n_post_signal_long": n_post,
            "n_constructable_long": n_con,
            "n_included_long": n_inc,
            "n_included_short": n_short,
        }
    )


def test_evaluation_dates_retained_and_outside_window_excluded() -> None:
    inside = [date(2024, 1, 8), date(2025, 1, 6), date(2026, 1, 5)]
    outside = [date(2023, 12, 29), date(2026, 7, 17)]
    econ, paired = _labeled(inside + outside)
    cal = _calendar(inside)
    port, summary = build_portfolio_comparison(
        econ,
        paired,
        "M1",
        window_start=EVAL_START,
        window_end=EVAL_END,
        entry_calendar=cal,
        reporting_periods=D2_REPORTING_PERIODS,
    )
    assert list(port["trade_date"]) == inside
    assert date(2023, 12, 29) not in set(port["trade_date"])
    assert date(2026, 7, 17) not in set(port["trade_date"])
    assert summary["period_reconcile_ok"] is True
    periods = summary["reporting_periods"]
    assert periods["2024"]["pnl_baseline"] + periods["2025"]["pnl_baseline"] + periods["2026_partial"]["pnl_baseline"] == pytest.approx(
        summary["total_pnl_baseline"]
    )
    assert periods["2024"]["pnl_filtered"] + periods["2025"]["pnl_filtered"] + periods["2026_partial"]["pnl_filtered"] == pytest.approx(
        summary["total_pnl_filtered"]
    )
    assert periods["2024"]["losses_avoided"] + periods["2025"]["losses_avoided"] + periods["2026_partial"]["losses_avoided"] == pytest.approx(
        summary["losses_avoided"]
    )
    assert periods["2024"]["winning_profits_sacrificed"] + periods["2025"]["winning_profits_sacrificed"] + periods["2026_partial"]["winning_profits_sacrificed"] == pytest.approx(
        summary["winning_profits_sacrificed"]
    )
    assert "half_periods" not in summary


def test_verified_n0_absent_from_panel_is_cash_on_both_series() -> None:
    d_trade = date(2024, 2, 5)
    d_zero = date(2024, 2, 12)
    econ, paired = _labeled([d_trade])
    cal = pd.concat(
        [
            _calendar([d_trade]),
            pd.DataFrame(
                {
                    "trade_date": [d_zero],
                    "n_in_N": [0],
                    "long_book_class": ["verified_zero_long"],
                }
            ),
        ],
        ignore_index=True,
    )
    ports = {}
    for measurement, score in (("M1", "M1"), ("M2", "M2")):
        work = econ.copy()
        work["M2"] = work["M1"]
        work, paired_m, _ex = assign_eval_u_labels(work, measurement)
        port, _summary = build_portfolio_comparison(
            work,
            paired_m,
            measurement,
            window_start=EVAL_START,
            window_end=EVAL_END,
            entry_calendar=cal,
            reporting_periods=D2_REPORTING_PERIODS,
        )
        ports[measurement] = port
        zero = port.loc[port["trade_date"] == d_zero].iloc[0]
        assert zero["pnl_baseline"] == 0.0
        assert zero["pnl_filtered"] == 0.0
        assert zero["uplift"] == 0.0
        assert zero["cash_frac_filtered"] == 1.0
        assert zero["n_in_N"] == 0
    assert ports["M1"]["trade_date"].tolist() == ports["M2"]["trade_date"].tolist()
    assert list(ports["M1"]["trade_date"]) == [d_trade, d_zero]


def test_missing_data_does_not_become_cash() -> None:
    d0 = date(2024, 3, 4)
    status = _status([d0], status="failed", reason="missing_features")
    funnel = _funnel([d0], n_con=None, n_inc=None, n_post=None, n_short=None, status="failed", reason="missing_features")
    with pytest.raises(D2ValidationError, match="missing or failed"):
        classify_evaluation_calendar(status, status.copy(), funnel, pd.DataFrame())

    status_ok = _status([d0])
    funnel_candidates = _funnel([d0], n_con=5, n_inc=5, n_post=5)
    with pytest.raises(D2ValidationError, match="no panel rows|reconstructed in_N"):
        classify_evaluation_calendar(status_ok, status_ok.copy(), funnel_candidates, pd.DataFrame())

    disagree = funnel_candidates.copy()
    disagree["date_status"] = "traded"
    with pytest.raises(D2ValidationError, match="disagree"):
        classify_evaluation_calendar(status_ok, status_ok.copy(), disagree, pd.DataFrame())


def test_panel_date_absent_from_calendar_raises() -> None:
    d0 = date(2024, 4, 1)
    d1 = date(2024, 4, 8)
    econ, paired = _labeled([d0, d1])
    cal = _calendar([d0])
    with pytest.raises(CostDiagnosisError, match="missing from entry_calendar"):
        build_portfolio_comparison(
            econ,
            paired,
            "M1",
            window_start=EVAL_START,
            window_end=EVAL_END,
            entry_calendar=cal,
        )


def test_verified_zero_classifier_and_group_is_entry_only() -> None:
    d0 = date(2024, 5, 6)
    status = _status([d0], status="valid_no_trade", reason="empty_signals")
    funnel = _funnel(
        [d0],
        n_con=0,
        n_inc=0,
        n_post=0,
        n_short=0,
        status="valid_no_trade",
        reason="empty_signals",
    )
    cal = classify_evaluation_calendar(status, status.copy(), funnel, pd.DataFrame())
    assert cal.iloc[0]["long_book_class"] == "verified_zero_long"
    assert int(cal.iloc[0]["n_in_N"]) == 0

    rows = [_row(trade_date=d0, ticker=t, m1=float(i), x=9.0 if t == "E" else 0.1) for i, t in enumerate(list("ABCDE"), start=1)]
    day = pd.DataFrame(rows)
    sel1 = select_within_date_groups(day, "M1")
    swapped = day.copy()
    swapped.loc[swapped["ticker"] == "A", "X"] = 9.0
    swapped.loc[swapped["ticker"] == "E", "X"] = 0.1
    sel2 = select_within_date_groups(swapped, "M1")
    assert list(sel1["high"]["ticker"]) == list(sel2["high"]["ticker"]) == ["E"]
    assert int(sel1["k"]) == 1

    econ = attach_decomposition_columns(attach_scenario_economics(day, 1.0))
    econ, paired, excluded = assign_eval_u_labels(econ, "M1")
    assert excluded.empty
    assert list(econ.loc[econ["group_M1"] == "U", "ticker"]) == ["E"]
    # No valid split when too few scored names: retain baseline.
    short = econ.loc[econ["ticker"].isin(list("AB"))].copy()
    _work, paired_short, excluded_short = assign_eval_u_labels(short, "M1")
    assert paired_short.empty
    assert excluded_short.iloc[0]["reason"] == "fewer_than_five_scored"


def test_original_n_cash_and_pnl_identity() -> None:
    d0 = date(2024, 6, 3)
    rows = [_row(trade_date=d0, ticker=t, m1=float(i), x=0.2 if t == "E" else 1.5) for i, t in enumerate(list("ABCDE"), start=1)]
    rows.append(_row(trade_date=d0, ticker="Z", m1=0.01, x=1.0, crossed=True))
    panel = pd.DataFrame(rows)
    econ = attach_decomposition_columns(attach_scenario_economics(panel, 1.0))
    q_before = econ.set_index("ticker")["q_h"].copy()
    econ, paired, _ex = assign_eval_u_labels(econ, "M1")
    retained = econ.loc[econ["group_M1"] != "U"]
    assert retained.set_index("ticker")["q_h"].equals(q_before.loc[retained["ticker"]])
    assert float(econ["stake_dollars"].iloc[0]) == pytest.approx(BUDGET_B / 6.0)
    cal = _calendar([d0], n=6)
    port, summary = build_portfolio_comparison(
        econ,
        paired,
        "M1",
        window_start=EVAL_START,
        window_end=EVAL_END,
        entry_calendar=cal,
        reporting_periods=D2_REPORTING_PERIODS,
    )
    assert summary["total_pnl_improvement"] == pytest.approx(
        summary["losses_avoided"] - summary["winning_profits_sacrificed"]
    )
    assert port.iloc[0]["cash_frac_filtered"] == pytest.approx(2.0 / 6.0)
    assert port.iloc[0]["n_in_N"] == 6


def test_missing_middle_outcome_raises() -> None:
    d0 = date(2024, 7, 1)
    rows = [_row(trade_date=d0, ticker=t, m1=float(i)) for i, t in enumerate(list("ABCDE"), start=1)]
    rows[2]["X"] = np.nan
    rows[2]["outcome_finite"] = False
    econ = attach_decomposition_columns(attach_scenario_economics(pd.DataFrame(rows), 1.0))
    with pytest.raises(CostDiagnosisError, match="Missing required outcomes"):
        require_all_executed_outcomes(econ)


def test_drawdown_includes_initial_zero() -> None:
    assert fixed_budget_max_drawdown(np.array([-100.0])) == pytest.approx(-100.0)
    assert fixed_budget_max_drawdown(np.array([-50.0, 200.0, -20.0])) == pytest.approx(-50.0)


def test_two_contrast_inference_family() -> None:
    assert FAMILY_SIZE == 2
    assert ADJUSTED_CI_LEVEL == pytest.approx(0.975)
    assert bonferroni_adjust_p(0.01, family_size=2) == pytest.approx(0.02)
    assert bonferroni_adjust_p(0.6, family_size=2) == pytest.approx(1.0)
    assert interpret_adjusted_interval(0.01, 0.02) == "relative_benefit"
    assert interpret_adjusted_interval(-0.02, -0.01) == "relative_harm"
    assert interpret_adjusted_interval(-0.01, 0.01) == "inconclusive"
    dates = [date(2024, 1, 8), date(2024, 1, 15), date(2024, 1, 22), date(2024, 1, 29)]
    cal = _calendar(dates, n=5)
    ports = {}
    for measurement, uplift in (("M1", [0.01, 0.02, -0.005, 0.0]), ("M2", [0.0, 0.0, 0.0, 0.0])):
        ports[measurement] = pd.DataFrame(
            {
                "measurement": measurement,
                "trade_date": dates,
                "uplift": uplift,
            }
        )
    inferred = run_d2_inference(ports, cal)
    assert inferred["adjusted_ci_level"] == pytest.approx(0.975)
    assert inferred["family_size"] == 2
    assert inferred["contrasts"]["M2"]["label"] == "inconclusive"
    assert EVAL_START == date(2024, 1, 1)
    assert EVAL_END == date(2026, 7, 10)
