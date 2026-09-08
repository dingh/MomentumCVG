"""Sprint 008 D1 within-date follow-up tests — synthetic frames only."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.backtest.sprint008_d0_input_readiness import BUDGET_B
from src.backtest.sprint008_d1_measurement_validation import (
    EVAL_START,
    attach_scenario_economics,
    filter_development_panel,
)
from src.backtest.sprint008_d1_within_date_followup import (
    ADJUSTED_CI_LEVEL,
    ALPHA,
    FAMILY_SIZE,
    FOLLOWUP_MEASUREMENTS,
    HAC_MAXLAGS,
    HAC_USE_CORRECTION,
    MIN_SCORED,
    ORDINARY_CI_LEVEL,
    FollowupValidationError,
    bartlett_weights,
    bonferroni_adjust_p,
    build_paired_observations,
    newey_west_intercept_inference,
    paired_ttest_diagnostic,
    select_within_date_groups,
    scored_candidates_for_date,
)


def _row(
    *,
    trade_date: date,
    ticker: str,
    M: float,
    H: float,
    X: float,
    S0: float = 100.0,
    M1: float | None = None,
    M2: float | None = None,
    crossed: bool = False,
    in_N: bool = True,
) -> dict:
    m1 = (H / M) if M1 is None and M > 0 else (np.nan if M1 is None else M1)
    m2 = (H / S0) if M2 is None and S0 > 0 else (np.nan if M2 is None else M2)
    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": "long",
        "in_N": in_N,
        "structure_ok": True,
        "M": M,
        "H": H,
        "X": X,
        "S0": S0,
        "M1": m1,
        "M2": m2,
        "M3": 0.5,
        "outcome_finite": np.isfinite(X),
        "analysis_eligible": not crossed,
        "crossed_quote_excluded": crossed,
        "delta_M_vs_stored": 0.0,
        "payoff_reconcile_ok": True,
    }


def test_within_date_grouping_rounding_and_ties() -> None:
    # n=11 → k=floor(11/5)=2; ticker breaks score ties deterministically.
    day = pd.DataFrame(
        [
            _row(trade_date=date(2020, 1, 6), ticker="Z", M=1.0, H=0.10, X=1.2, M1=0.10),
            _row(trade_date=date(2020, 1, 6), ticker="A", M=1.0, H=0.10, X=1.1, M1=0.10),
            _row(trade_date=date(2020, 1, 6), ticker="B", M=1.0, H=0.20, X=1.0, M1=0.20),
            _row(trade_date=date(2020, 1, 6), ticker="C", M=1.0, H=0.30, X=0.9, M1=0.30),
            _row(trade_date=date(2020, 1, 6), ticker="D", M=1.0, H=0.40, X=0.8, M1=0.40),
            _row(trade_date=date(2020, 1, 6), ticker="E", M=1.0, H=0.50, X=0.7, M1=0.50),
            _row(trade_date=date(2020, 1, 6), ticker="F", M=1.0, H=0.60, X=0.6, M1=0.60),
            _row(trade_date=date(2020, 1, 6), ticker="G", M=1.0, H=0.70, X=0.5, M1=0.70),
            _row(trade_date=date(2020, 1, 6), ticker="H", M=1.0, H=0.80, X=0.4, M1=0.80),
            _row(trade_date=date(2020, 1, 6), ticker="I", M=1.0, H=0.90, X=0.3, M1=0.90),
            _row(trade_date=date(2020, 1, 6), ticker="J", M=1.0, H=1.00, X=0.2, M1=1.00),
        ]
    )
    sel = select_within_date_groups(day, "M1")
    assert sel["eligible"] is True
    assert sel["k"] == 2
    assert sel["n_scored"] == 11
    # Tied at 0.10: ticker A before Z
    assert list(sel["low"]["ticker"]) == ["A", "Z"]
    assert list(sel["high"]["ticker"]) == ["I", "J"]
    assert sel["low_cutoff_tie"] is False
    assert sel["high_cutoff_tie"] is False


def test_within_date_insufficient_and_no_variation() -> None:
    few = pd.DataFrame(
        [
            _row(trade_date=date(2020, 1, 6), ticker=t, M=1.0, H=0.1 * i, X=1.0, M1=0.1 * i)
            for i, t in enumerate(["A", "B", "C", "D"], start=1)
        ]
    )
    assert select_within_date_groups(few, "M1")["reason"] == "fewer_than_five_scored"

    flat = pd.DataFrame(
        [
            _row(trade_date=date(2020, 1, 6), ticker=t, M=1.0, H=0.2, X=1.0, M1=0.2)
            for t in list("ABCDE")
        ]
    )
    assert select_within_date_groups(flat, "M1")["reason"] == "no_score_variation"
    assert MIN_SCORED == 5


def test_cutoff_tie_disclosure() -> None:
    # Scores: 1,1,1,2,3 → n=5,k=1; low boundary ties with next (both score 1)
    day = pd.DataFrame(
        [
            _row(trade_date=date(2020, 1, 6), ticker="A", M=1.0, H=0.1, X=1.0, M1=1.0),
            _row(trade_date=date(2020, 1, 6), ticker="B", M=1.0, H=0.1, X=1.0, M1=1.0),
            _row(trade_date=date(2020, 1, 6), ticker="C", M=1.0, H=0.1, X=1.0, M1=1.0),
            _row(trade_date=date(2020, 1, 6), ticker="D", M=1.0, H=0.2, X=1.0, M1=2.0),
            _row(trade_date=date(2020, 1, 6), ticker="E", M=1.0, H=0.3, X=1.0, M1=3.0),
        ]
    )
    sel = select_within_date_groups(day, "M1")
    assert sel["k"] == 1
    assert sel["low_cutoff_tie"] is True
    assert list(sel["low"]["ticker"]) == ["A"]
    assert list(sel["high"]["ticker"]) == ["E"]


def test_group_membership_ignores_future_outcomes() -> None:
    # Huge X on high-score names must not change L/U membership.
    day = pd.DataFrame(
        [
            _row(trade_date=date(2020, 1, 6), ticker="A", M=1.0, H=0.1, X=0.0, M1=0.1),
            _row(trade_date=date(2020, 1, 6), ticker="B", M=1.0, H=0.2, X=0.0, M1=0.2),
            _row(trade_date=date(2020, 1, 6), ticker="C", M=1.0, H=0.3, X=0.0, M1=0.3),
            _row(trade_date=date(2020, 1, 6), ticker="D", M=1.0, H=0.4, X=0.0, M1=0.4),
            _row(trade_date=date(2020, 1, 6), ticker="E", M=1.0, H=0.5, X=99.0, M1=0.5),
        ]
    )
    sel1 = select_within_date_groups(day, "M1")
    day2 = day.copy()
    day2.loc[day2["ticker"] == "E", "X"] = 0.0
    day2.loc[day2["ticker"] == "A", "X"] = 99.0
    sel2 = select_within_date_groups(day2, "M1")
    assert list(sel1["low"]["ticker"]) == list(sel2["low"]["ticker"]) == ["A"]
    assert list(sel1["high"]["ticker"]) == list(sel2["high"]["ticker"]) == ["E"]


def test_paired_differences_equal_date_weighting() -> None:
    # Date1: 5 names, k=1; Date2: 10 names, k=2 — each date one d_t, equal weight in mean.
    rows = []
    d1 = date(2020, 1, 6)
    for i, t in enumerate(list("ABCDE"), start=1):
        # low score A has high return; high score E has low return → d>0
        x = 2.0 if t == "A" else (0.5 if t == "E" else 1.0)
        rows.append(_row(trade_date=d1, ticker=t, M=1.0, H=0.1 * i, X=x, M1=0.1 * i))
    d2 = date(2020, 1, 13)
    for i, t in enumerate(list("ABCDEFGHIJ"), start=1):
        x = 1.5 if i <= 2 else (0.5 if i >= 9 else 1.0)
        rows.append(_row(trade_date=d2, ticker=t, M=1.0, H=0.1 * i, X=x, M1=0.1 * i))

    panel = pd.DataFrame(rows)
    econ = attach_scenario_economics(panel, 1.0)
    paired, excluded = build_paired_observations(econ, "M1")
    assert excluded.empty
    assert len(paired) == 2
    # Equal weight: mean_d = average of two date-level d_t, not trade-weighted.
    mean_d = float(paired["d_t"].mean())
    assert mean_d == pytest.approx(0.5 * (paired.iloc[0]["d_t"] + paired.iloc[1]["d_t"]))
    assert paired.iloc[0]["k"] == 1
    assert paired.iloc[1]["k"] == 2


def test_preserved_original_n_and_sizing() -> None:
    # One crossed-quote name stays in N; stake uses N=6, selected groups not resized.
    d0 = date(2020, 2, 3)
    rows = [
        _row(trade_date=d0, ticker=t, M=1.0, H=0.1 * i, X=1.1, M1=0.1 * i)
        for i, t in enumerate(list("ABCDE"), start=1)
    ]
    rows.append(
        _row(trade_date=d0, ticker="X", M=1.0, H=0.05, X=1.0, M1=0.05, crossed=True)
    )
    panel = pd.DataFrame(rows)
    econ = attach_scenario_economics(panel, 1.0)
    n = int((econ["in_N"] == True).sum())  # noqa: E712
    assert n == 6
    stake = float(econ.loc[econ["analysis_eligible"] == True, "stake_dollars"].iloc[0])  # noqa: E712
    assert stake == pytest.approx(BUDGET_B / 6.0)
    crossed_q = float(econ.loc[econ["ticker"] == "X", "q_h"].iloc[0])
    assert crossed_q == pytest.approx(0.0)
    scored = scored_candidates_for_date(econ, "M1")
    assert "X" not in set(scored["ticker"])
    sel = select_within_date_groups(scored, "M1")
    assert sel["n_scored"] == 5
    assert sel["k"] == 1
    low = sel["low"]
    assert float(
        econ.loc[econ["ticker"] == low.iloc[0]["ticker"], "stake_dollars"].iloc[0]
    ) == pytest.approx(BUDGET_B / 6.0)


def test_missing_selected_outcome_raises() -> None:
    d0 = date(2020, 3, 2)
    rows = [
        _row(trade_date=d0, ticker=t, M=1.0, H=0.1 * i, X=1.0, M1=0.1 * i)
        for i, t in enumerate(list("ABCDE"), start=1)
    ]
    rows[0]["X"] = np.nan
    rows[0]["outcome_finite"] = False
    panel = pd.DataFrame(rows)
    econ = attach_scenario_economics(panel, 1.0)
    with pytest.raises(FollowupValidationError, match="missing required outcome"):
        build_paired_observations(econ, "M1")


def test_evaluation_period_excluded() -> None:
    rows = []
    for d in (date(2020, 1, 6), date(2024, 1, 8)):
        for i, t in enumerate(list("ABCDE"), start=1):
            rows.append(
                _row(trade_date=d, ticker=t, M=1.0, H=0.1 * i, X=1.0 + 0.1 * i, M1=0.1 * i)
            )
    panel = pd.DataFrame(rows)
    assert any(panel["trade_date"] >= EVAL_START)
    dev = filter_development_panel(panel)
    assert all(dev["trade_date"] < EVAL_START)
    econ = attach_scenario_economics(dev, 1.0)
    paired, _ = build_paired_observations(econ, "M1")
    assert not paired.empty
    assert all(paired["trade_date"] < EVAL_START)


def test_hac_and_multiplicity_settings_frozen() -> None:
    assert FAMILY_SIZE == 2
    assert list(FOLLOWUP_MEASUREMENTS) == ["M1", "M2"]
    assert HAC_MAXLAGS == 3
    assert HAC_USE_CORRECTION is True
    assert ORDINARY_CI_LEVEL == 0.95
    assert ADJUSTED_CI_LEVEL == 0.975
    w = bartlett_weights(3)
    assert list(w) == pytest.approx([1 - 1 / 4, 1 - 2 / 4, 1 - 3 / 4])

    rng = np.random.default_rng(0)
    d = rng.normal(0.02, 0.05, size=40)
    hac = newey_west_intercept_inference(d, maxlags=3, use_correction=True)
    assert hac["maxlags"] == 3
    assert hac["kernel"] == "bartlett"
    assert hac["use_correction"] is True
    assert hac["df"] == 39
    mean = float(d.mean())
    e = d - mean
    s = float(np.dot(e, e) / len(d))
    for j, wt in enumerate(w, start=1):
        s += 2.0 * float(wt) * float(np.dot(e[j:], e[:-j]) / len(d))
    var = (s / len(d)) * (len(d) / (len(d) - 1))
    assert hac["se_hac"] == pytest.approx(np.sqrt(var))
    p_adj = bonferroni_adjust_p(hac["p_raw"], family_size=2)
    assert p_adj == pytest.approx(min(1.0, 2 * hac["p_raw"]))
    crit = float(stats.t.ppf(1.0 - (1.0 - 0.975) / 2.0, df=39))
    half = crit * hac["se_hac"]
    assert hac["ci_adjusted"][0] == pytest.approx(mean - half)
    assert hac["ci_adjusted"][1] == pytest.approx(mean + half)


def test_paired_ttest_matches_one_sample_on_d() -> None:
    l = np.array([0.1, 0.2, 0.0, -0.05])
    u = np.array([0.0, 0.05, 0.1, 0.0])
    diag = paired_ttest_diagnostic(l, u)
    one = stats.ttest_1samp(l - u, 0.0)
    assert diag["statistic"] == pytest.approx(float(one.statistic))
    assert diag["pvalue"] == pytest.approx(float(one.pvalue))


def test_bonferroni_family_size_two() -> None:
    assert bonferroni_adjust_p(0.04, family_size=2) == pytest.approx(0.08)
    assert bonferroni_adjust_p(0.6, family_size=2) == pytest.approx(1.0)
    assert ALPHA == 0.05
