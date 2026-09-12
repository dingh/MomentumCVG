"""Sprint 008 D1 cost-diagnosis tests — synthetic frames only."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.sprint008_d0_input_readiness import BUDGET_B
from src.backtest.sprint008_d1_measurement_validation import (
    EVAL_START,
    attach_scenario_economics,
    filter_development_panel,
)
from src.backtest.sprint008_d1_cost_diagnosis import (
    ADJUSTED_CI_LEVEL,
    FAMILY_SIZE,
    CostDiagnosisError,
    attach_decomposition_columns,
    build_portfolio_comparison,
    fixed_budget_max_drawdown,
    require_all_executed_outcomes,
    select_groups_with_middle,
    summarize_decomposition,
)
from src.backtest.sprint008_d1_within_date_followup import bonferroni_adjust_p


def _row(
    *,
    trade_date: date,
    ticker: str,
    M: float,
    H: float,
    X: float,
    S0: float = 100.0,
    M1: float | None = None,
    crossed: bool = False,
) -> dict:
    m1 = (H / M) if M1 is None and M > 0 else (np.nan if M1 is None else M1)
    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": "long",
        "in_N": True,
        "structure_ok": True,
        "M": M,
        "H": H,
        "X": X,
        "S0": S0,
        "M1": m1,
        "M2": H / S0,
        "M3": 0.5,
        "outcome_finite": np.isfinite(X),
        "analysis_eligible": not crossed,
        "crossed_quote_excluded": crossed,
        "delta_M_vs_stored": 0.0,
        "payoff_reconcile_ok": True,
    }


def test_decomposition_identity() -> None:
    d0 = date(2020, 1, 6)
    rows = []
    for i, t in enumerate(list("ABCDEFGHIJ"), start=1):
        # Vary H so M1 ranks with i; X so L better net than U
        x = 2.0 if i <= 2 else (0.5 if i >= 9 else 1.2)
        rows.append(_row(trade_date=d0, ticker=t, M=1.0, H=0.05 * i, X=x, M1=0.05 * i))
    panel = pd.DataFrame(rows)
    econ = attach_scenario_economics(panel, 1.0)
    econ = attach_decomposition_columns(econ)
    # r = g - a
    assert (econ["g"] - econ["a"] - econ["r"]).abs().max() < 1e-12

    scored = econ.copy()
    sel = select_groups_with_middle(scored, "M1")
    assert sel["eligible"]
    assert sel["k"] == 2
    assert len(sel["middle"]) == 6
    low, high = sel["low"], sel["high"]
    d_net = float(low["r"].mean() - high["r"].mean())
    d_gross = float(low["g"].mean() - high["g"].mean())
    spread_saving = float(high["a"].mean() - low["a"].mean())
    assert d_net == pytest.approx(d_gross + spread_saving)


def test_entry_only_grouping_and_middle() -> None:
    d0 = date(2020, 1, 6)
    rows = [
        _row(trade_date=d0, ticker=t, M=1.0, H=0.1 * i, X=99.0 if t == "E" else 1.0, M1=0.1 * i)
        for i, t in enumerate(list("ABCDE"), start=1)
    ]
    day = pd.DataFrame(rows)
    sel1 = select_groups_with_middle(day, "M1")
    day2 = day.copy()
    day2.loc[day2["ticker"] == "A", "X"] = 99.0
    day2.loc[day2["ticker"] == "E", "X"] = 0.0
    sel2 = select_groups_with_middle(day2, "M1")
    assert list(sel1["low"]["ticker"]) == list(sel2["low"]["ticker"]) == ["A"]
    assert list(sel1["high"]["ticker"]) == list(sel2["high"]["ticker"]) == ["E"]
    assert list(sel1["middle"]["ticker"]) == list(sel2["middle"]["ticker"])


def test_original_n_cash_under_exclusion() -> None:
    d0 = date(2020, 2, 3)
    rows = [
        _row(trade_date=d0, ticker=t, M=1.0, H=0.1 * i, X=1.1, M1=0.1 * i)
        for i, t in enumerate(list("ABCDE"), start=1)
    ]
    rows.append(_row(trade_date=d0, ticker="Z", M=1.0, H=0.01, X=1.0, M1=0.01, crossed=True))
    panel = pd.DataFrame(rows)
    econ = attach_scenario_economics(panel, 1.0)
    econ = attach_decomposition_columns(econ)
    assert int((econ["in_N"] == True).sum()) == 6  # noqa: E712
    assert float(econ["stake_dollars"].iloc[0]) == pytest.approx(BUDGET_B / 6.0)
    assert float(econ.loc[econ["ticker"] == "Z", "q_h"].iloc[0]) == 0.0

    # Label groups manually for portfolio path
    scored = econ.loc[econ["analysis_eligible"] == True].copy()  # noqa: E712
    sel = select_groups_with_middle(scored, "M1")
    econ["group_M1"] = pd.NA
    econ.loc[sel["low"].index, "group_M1"] = "L"
    econ.loc[sel["high"].index, "group_M1"] = "U"
    econ.loc[sel["middle"].index, "group_M1"] = "middle"
    paired = pd.DataFrame(
        [
            {
                "measurement": "M1",
                "trade_date": d0,
                "d_net": 0.0,
                "winrate_net_diff_LU": 0.0,
            }
        ]
    )
    port, summary = build_portfolio_comparison(econ, paired, "M1")
    assert summary["n_excluded_u_trades"] == 1
    # Filtered invests 4/6 of budget (4 executed retained; Z cash + 1 U cash)
    assert port.iloc[0]["invested_frac_filtered"] == pytest.approx(4.0 / 6.0)
    assert port.iloc[0]["cash_frac_filtered"] == pytest.approx(2.0 / 6.0)


def test_winner_retention_na_when_zero() -> None:
    d0 = date(2020, 3, 2)
    # All losses
    rows = [
        _row(trade_date=d0, ticker=t, M=1.0, H=0.1 * i, X=0.5, M1=0.1 * i)
        for i, t in enumerate(list("ABCDE"), start=1)
    ]
    panel = pd.DataFrame(rows)
    econ = attach_scenario_economics(panel, 1.0)
    econ = attach_decomposition_columns(econ)
    sel = select_groups_with_middle(econ, "M1")
    econ["group_M1"] = pd.NA
    econ.loc[sel["low"].index, "group_M1"] = "L"
    econ.loc[sel["high"].index, "group_M1"] = "U"
    econ.loc[sel["middle"].index, "group_M1"] = "middle"
    paired = pd.DataFrame([{"measurement": "M1", "trade_date": d0}])
    _, summary = build_portfolio_comparison(econ, paired, "M1")
    assert summary["winning_profit_retention"] is None
    assert summary["top5_profit_retention"] is None


def test_pnl_improvement_identity() -> None:
    d0 = date(2020, 4, 6)
    rows = []
    for i, t in enumerate(list("ABCDE"), start=1):
        # U (highest M1=E) is a big loser; L is small winner
        x = 1.5 if t == "A" else (0.2 if t == "E" else 1.0)
        rows.append(_row(trade_date=d0, ticker=t, M=1.0, H=0.1 * i, X=x, M1=0.1 * i))
    panel = pd.DataFrame(rows)
    econ = attach_scenario_economics(panel, 1.0)
    econ = attach_decomposition_columns(econ)
    sel = select_groups_with_middle(econ, "M1")
    econ["group_M1"] = pd.NA
    econ.loc[sel["low"].index, "group_M1"] = "L"
    econ.loc[sel["high"].index, "group_M1"] = "U"
    econ.loc[sel["middle"].index, "group_M1"] = "middle"
    paired = pd.DataFrame([{"measurement": "M1", "trade_date": d0}])
    _, summary = build_portfolio_comparison(econ, paired, "M1")
    assert summary["identity_improvement_ok"] is True
    assert summary["total_pnl_improvement"] == pytest.approx(
        summary["losses_avoided"] - summary["winning_profits_sacrificed"]
    )


def test_missing_middle_outcome_fails() -> None:
    d0 = date(2020, 5, 4)
    rows = [
        _row(trade_date=d0, ticker=t, M=1.0, H=0.1 * i, X=1.0, M1=0.1 * i)
        for i, t in enumerate(list("ABCDE"), start=1)
    ]
    # Middle ticker C missing outcome
    rows[2]["X"] = np.nan
    rows[2]["outcome_finite"] = False
    panel = pd.DataFrame(rows)
    econ = attach_scenario_economics(panel, 1.0)
    econ = attach_decomposition_columns(econ)
    with pytest.raises(CostDiagnosisError, match="Missing required outcomes"):
        require_all_executed_outcomes(econ)


def test_evaluation_excluded() -> None:
    rows = []
    for d in (date(2020, 1, 6), date(2024, 1, 8)):
        for i, t in enumerate(list("ABCDE"), start=1):
            rows.append(_row(trade_date=d, ticker=t, M=1.0, H=0.1 * i, X=1.0, M1=0.1 * i))
    panel = pd.DataFrame(rows)
    dev = filter_development_panel(panel)
    assert all(dev["trade_date"] < EVAL_START)


def test_bonferroni_family_four() -> None:
    assert FAMILY_SIZE == 4
    assert ADJUSTED_CI_LEVEL == pytest.approx(0.9875)
    assert bonferroni_adjust_p(0.01, family_size=4) == pytest.approx(0.04)
    assert bonferroni_adjust_p(0.3, family_size=4) == pytest.approx(1.0)


def test_drawdown_includes_initial_zero_on_losing_path() -> None:
    # cum = -10000, -20000. Peak must stay at 0, so max drawdown is -20000
    # (omitting the start would report only -10000).
    assert fixed_budget_max_drawdown(np.array([-10_000.0, -10_000.0])) == pytest.approx(
        -20_000.0
    )


def test_drawdown_after_recovery() -> None:
    # Initial loss from $0, then a new high, then a smaller pullback.
    # cum: -10000, +15000, +7000. Peak path: 0, 15000, 15000.
    # Drawdowns: -10000, 0, -8000. Maximum is the initial -10000.
    assert fixed_budget_max_drawdown(
        np.array([-10_000.0, 25_000.0, -8_000.0])
    ) == pytest.approx(-10_000.0)
    # Subsequent decline from a high exceeds the initial loss.
    # cum: +10000, -15000. Peak 10000; drawdown -25000.
    assert fixed_budget_max_drawdown(np.array([10_000.0, -25_000.0])) == pytest.approx(
        -25_000.0
    )


def test_default_path_excludes_evaluation_dates_and_keeps_half_periods() -> None:
    rows = []
    for d in (date(2020, 6, 1), date(2024, 1, 8)):
        for i, t in enumerate(list("ABCDE"), start=1):
            rows.append(_row(trade_date=d, ticker=t, M=1.0, H=0.1 * i, X=1.0, M1=0.1 * i))
    panel = pd.DataFrame(rows)
    econ = attach_decomposition_columns(attach_scenario_economics(panel, 1.0))
    econ["group_M1"] = pd.NA
    for d in (date(2020, 6, 1), date(2024, 1, 8)):
        day = econ.loc[econ["trade_date"] == d]
        sel = select_groups_with_middle(day, "M1")
        econ.loc[sel["high"].index, "group_M1"] = "U"
    paired = pd.DataFrame(
        [
            {"measurement": "M1", "trade_date": date(2020, 6, 1)},
            {"measurement": "M1", "trade_date": date(2024, 1, 8)},
        ]
    )
    port, summary = build_portfolio_comparison(econ, paired, "M1")
    assert list(port["trade_date"]) == [date(2020, 6, 1)]
    assert "half_periods" in summary
    assert summary["half_period_reconcile_ok"] is True
    assert "reporting_periods" not in summary


def test_half_period_dollar_totals_reconcile() -> None:
    rows = []
    for d, x_high in ((date(2020, 6, 1), 0.2), (date(2022, 6, 6), 1.8)):
        for i, t in enumerate(list("ABCDE"), start=1):
            x = 1.4 if t == "A" else (x_high if t == "E" else 1.0)
            rows.append(_row(trade_date=d, ticker=t, M=1.0, H=0.1 * i, X=x, M1=0.1 * i))
    panel = pd.DataFrame(rows)
    econ = attach_decomposition_columns(attach_scenario_economics(panel, 1.0))
    # Groups labeled on econ directly.
    for d in (date(2020, 6, 1), date(2022, 6, 6)):
        day = econ.loc[econ["trade_date"] == d].copy()
        sel = select_groups_with_middle(day, "M1")
        econ.loc[sel["low"].index, "group_M1"] = "L"
        econ.loc[sel["high"].index, "group_M1"] = "U"
        econ.loc[sel["middle"].index, "group_M1"] = "middle"
    paired = pd.DataFrame(
        [
            {"measurement": "M1", "trade_date": date(2020, 6, 1)},
            {"measurement": "M1", "trade_date": date(2022, 6, 6)},
        ]
    )
    _, summary = build_portfolio_comparison(econ, paired, "M1")
    assert summary["half_period_reconcile_ok"] is True
    h1 = summary["half_periods"]["2020-2021"]
    h2 = summary["half_periods"]["2022-2023"]
    assert h1["n_dates"] == 1 and h2["n_dates"] == 1
    assert h1["pnl_baseline"] + h2["pnl_baseline"] == pytest.approx(
        summary["total_pnl_baseline"]
    )
    assert h1["pnl_filtered"] + h2["pnl_filtered"] == pytest.approx(
        summary["total_pnl_filtered"]
    )
    assert h1["losses_avoided"] + h2["losses_avoided"] == pytest.approx(
        summary["losses_avoided"]
    )
    assert h1["winning_profits_sacrificed"] + h2["winning_profits_sacrificed"] == pytest.approx(
        summary["winning_profits_sacrificed"]
    )


def test_summarize_decomposition_identity() -> None:
    paired = pd.DataFrame(
        [
            {
                "measurement": "M1",
                "trade_date": date(2020, 1, 6),
                "d_net": 0.10,
                "d_gross": 0.04,
                "spread_saving": 0.06,
                "mean_a_L": 0.1,
                "mean_a_U": 0.16,
                "mean_g_L": 0.2,
                "mean_g_U": 0.16,
                "mean_r_L": 0.1,
                "mean_r_U": 0.0,
                "mean_r_middle": 0.05,
            }
        ]
    )
    s = summarize_decomposition(paired, "M1")
    assert s["identity_check_ok"] is True
