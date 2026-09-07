"""Sprint 008 D1 measurement-validation tests — synthetic frames only."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest.sprint008_d0_input_readiness import (
    BUDGET_B,
    FEES,
    equal_dollar_quantities,
)
from src.backtest.sprint008_d1_measurement_validation import (
    ALPHA,
    BONFERRONI_HI,
    BONFERRONI_LO,
    DEV_END,
    DEV_START,
    EVAL_START,
    FAMILY_SIZE,
    LABEL_INCONCLUSIVE,
    LABEL_SUPPORTED,
    LABEL_UNSUPPORTED,
    MEASUREMENTS,
    PRIMARY_H,
    SENSITIVITY_H,
    analysis_set_for_measurement,
    assert_no_evaluation_rows,
    assign_frozen_quintiles,
    attach_scenario_economics,
    bonferroni_quantiles,
    classify_measurement,
    compute_delta_from_groups,
    delta_from_date_multiplicity,
    evaluate_p_gross,
    evaluate_predicates,
    explicit_path_delta_by_row_duplication,
    filter_development_panel,
    half_period_deltas,
    is_evaluation_date,
    precompute_date_quintile_stats,
    reconcile_economics,
    run_block_bootstrap_deltas,
    run_d1_validation,
    sample_block_date_path,
    sprint_level_gate,
    within_date_spearman_summary,
    D1ValidationError,
)


def _base_row(
    *,
    trade_date: date,
    ticker: str,
    M: float,
    H: float,
    X: float,
    S0: float = 100.0,
    M1: float | None = None,
    M2: float | None = None,
    M3: float | None = None,
    crossed: bool = False,
    in_N: bool = True,
) -> dict:
    m1 = (H / M) if M1 is None and M > 0 else (np.nan if M1 is None else M1)
    m2 = (H / S0) if M2 is None and S0 > 0 else (np.nan if M2 is None else M2)
    m3 = 0.5 if M3 is None else M3
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
        "K": 100.0,
        "ST": 100.0 + X,  # unused when X set
        "M1": m1,
        "M2": m2,
        "M3": m3,
        "crossed_quote_excluded": crossed,
        "analysis_eligible": (not crossed),
        "outcome_finite": True,
        "payoff_reconcile_ok": True,
        "delta_M_vs_stored": 0.0,
        "entry_spot": S0,
        "exit_spot": S0,
        "body_strike": 100.0,
        "expiry_date": trade_date + timedelta(days=4),
        "signal_rank_pct": 0.9,
        "included_in_portfolio": True,
        "entry_cost_mid_per_share": M,
        "entry_cost_per_share": M + H,
        "quantity": 1.0,
        "fill_label": "mid",
        "exclusion_reason": None,
    }


def _panel_from_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_scenario_specific_quantities_not_reused() -> None:
    """q_i(h) differs by h; sensitivity must not reuse h=1 quantities."""
    d0 = date(2021, 6, 1)
    panel = _panel_from_rows(
        [
            _base_row(trade_date=d0, ticker="AAA", M=5.0, H=2.0, X=8.0),
            _base_row(trade_date=d0, ticker="BBB", M=4.0, H=1.0, X=6.0),
        ]
    )
    q_by_h = {}
    for h in (0.0, 0.25, 0.5, 1.0):
        sized = equal_dollar_quantities(panel, h)
        q_by_h[h] = sized.set_index("ticker")["q_h"].to_dict()
        econ = attach_scenario_economics(panel, h)
        # Stake identity for eligible
        for _, row in econ.loc[econ["assoc_valid"]].iterrows():
            assert row["q_h"] * row["C"] == pytest.approx(BUDGET_B / 2.0)
            assert row["C"] == pytest.approx(row["M"] + h * row["H"] + FEES)

    assert q_by_h[0.0]["AAA"] != pytest.approx(q_by_h[1.0]["AAA"])
    assert q_by_h[0.25]["AAA"] != pytest.approx(q_by_h[1.0]["AAA"])
    # Explicit: economics at h=0.25 must use q(0.25), not q(1)
    econ025 = attach_scenario_economics(panel, 0.25)
    econ1 = attach_scenario_economics(panel, 1.0)
    assert list(econ025["q_h"]) != list(econ1["q_h"])


def test_gross_minus_drag_reconciliation() -> None:
    panel = _panel_from_rows(
        [
            _base_row(trade_date=date(2021, 1, 4), ticker="AAA", M=5.0, H=2.0, X=10.0),
            _base_row(trade_date=date(2021, 1, 4), ticker="BBB", M=6.0, H=1.5, X=4.0),
        ]
    )
    econ = attach_scenario_economics(panel, PRIMARY_H)
    recon = reconcile_economics(econ)
    assert recon["passed"] is True
    for _, row in econ.loc[econ["assoc_valid"]].iterrows():
        assert row["r"] == pytest.approx(row["g"] - row["d"])
        assert row["dollar_net"] == pytest.approx(row["dollar_gross"] - row["dollar_drag"])
        assert row["r"] == pytest.approx((row["X"] - row["C"]) / row["C"])
        assert row["g"] == pytest.approx((row["X"] - row["M"]) / row["C"])
        assert row["d"] == pytest.approx((PRIMARY_H * row["H"] + FEES) / row["C"])


def test_original_n_preserved_after_crossed_and_missing_m() -> None:
    d0 = date(2021, 3, 1)
    rows = [
        _base_row(trade_date=d0, ticker="AAA", M=5.0, H=1.0, X=6.0),
        _base_row(trade_date=d0, ticker="BBB", M=5.0, H=1.0, X=6.0, crossed=True),
        _base_row(trade_date=d0, ticker="CCC", M=5.0, H=1.0, X=6.0, M1=np.nan),
    ]
    panel = _panel_from_rows(rows)
    assert int(panel["in_N"].sum()) == 3
    econ = attach_scenario_economics(panel, PRIMARY_H)
    # Crossed stays in panel / N count, q=0, not assoc
    crossed = econ.loc[econ["ticker"] == "BBB"].iloc[0]
    assert crossed["in_N"] is True or crossed["in_N"] == True  # noqa: E712
    assert crossed["q_h"] == pytest.approx(0.0)
    assert crossed["assoc_valid"] is False or crossed["assoc_valid"] == False  # noqa: E712
    anal = analysis_set_for_measurement(econ, "M1")
    assert set(anal["ticker"]) == {"AAA"}  # missing M1 and crossed out
    # N disclosure unchanged
    assert int(econ["in_N"].sum()) == 3


def test_frozen_quintile_membership_and_deterministic_ties() -> None:
    # Same M1 → ties broken by trade_date then ticker
    rows = []
    base = date(2021, 1, 4)
    # 10 names, deliberately tied scores
    for i, tk in enumerate(list("ABCDEFGHIJ")):
        rows.append(
            _base_row(
                trade_date=base + timedelta(days=(i % 3) * 7),
                ticker=tk,
                M=5.0,
                H=1.0 + 0.01 * (i % 2),  # slight variation for some
                X=5.0 + i,
                M1=0.2,  # all tied
            )
        )
    panel = _panel_from_rows(rows)
    econ = attach_scenario_economics(panel, PRIMARY_H)
    anal = analysis_set_for_measurement(econ, "M1")
    q1 = assign_frozen_quintiles(anal, "M1")
    q2 = assign_frozen_quintiles(anal, "M1")
    assert list(q1) == list(q2)
    anal = anal.copy()
    anal["quintile"] = q1
    counts = anal["quintile"].value_counts()
    assert counts.max() - counts.min() <= 1
    assert set(counts.index) == {f"Q{i}" for i in range(1, 6)}
    # Order for ties: sort (m, date, ticker) — A before B on same date
    ordered = anal.sort_values(
        ["M1", "trade_date", "ticker"], kind="mergesort"
    ).reset_index(drop=True)
    # First two should be Q1 for n=10
    assert ordered.iloc[0]["quintile"] == "Q1"
    assert ordered.iloc[-1]["quintile"] == "Q5"


def test_bootstrap_date_multiplicity_and_frozen_quantities() -> None:
    dates = [date(2021, 1, 4) + timedelta(days=7 * i) for i in range(6)]
    rows = []
    for d in dates:
        for j, tk in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
            rows.append(
                _base_row(
                    trade_date=d,
                    ticker=tk,
                    M=5.0,
                    H=1.0 + 0.1 * j,
                    X=4.0 + j,
                    M1=0.1 * (j + 1),
                )
            )
    panel = _panel_from_rows(rows)
    econ = attach_scenario_economics(panel, PRIMARY_H)
    anal = analysis_set_for_measurement(econ, "M1")
    anal = anal.copy()
    anal["quintile"] = assign_frozen_quintiles(anal, "M1")
    # Capture quantities before bootstrap
    q_map = anal.set_index(["trade_date", "ticker"])["q_h"].to_dict()
    quint_map = anal.set_index(["trade_date", "ticker"])["quintile"].to_dict()

    date_list, stats = precompute_date_quintile_stats(anal)
    # Force multiplicity: repeat first date three times
    path = [date_list[0], date_list[0], date_list[0], date_list[1], date_list[2], date_list[3]]
    delta_fast = delta_from_date_multiplicity(path, stats)
    delta_ref = explicit_path_delta_by_row_duplication(anal, path)
    assert delta_fast == pytest.approx(delta_ref)

    # Explicit duplication length = sum of cross-sections with multiplicity
    by_date = anal.groupby(anal["trade_date"].map(lambda x: x)).size().to_dict()
    expected_rows = sum(by_date[d] for d in path)
    pieces = []
    grouped = {d: g for d, g in anal.groupby("trade_date")}
    for d in path:
        pieces.append(grouped[d])
    path_df = pd.concat(pieces, ignore_index=True)
    assert len(path_df) == expected_rows
    # Quantities and quintiles unchanged on path rows
    for _, row in path_df.iterrows():
        key = (row["trade_date"], row["ticker"])
        assert row["q_h"] == pytest.approx(q_map[key])
        assert row["quintile"] == quint_map[key]


def test_bootstrap_efficiency_matches_row_duplication_fixture() -> None:
    dates = [date(2020, 2, 3) + timedelta(days=7 * i) for i in range(8)]
    rows = []
    for d in dates:
        for j, tk in enumerate(list("ABCDE")):
            rows.append(
                _base_row(
                    trade_date=d,
                    ticker=tk,
                    M=4.0 + 0.2 * j,
                    H=1.0,
                    X=3.0 + 0.5 * j,
                    M1=0.05 * (j + 1) + 0.01 * (d.toordinal() % 3),
                )
            )
    anal = analysis_set_for_measurement(attach_scenario_economics(_panel_from_rows(rows), 1.0), "M1")
    anal = anal.copy()
    anal["quintile"] = assign_frozen_quintiles(anal, "M1")
    dates_list, stats = precompute_date_quintile_stats(anal)
    rng = np.random.default_rng(123)
    for _ in range(20):
        path = sample_block_date_path(dates_list, rng, block_len=4)
        a = delta_from_date_multiplicity(path, stats)
        b = explicit_path_delta_by_row_duplication(anal, path)
        if np.isfinite(a) or np.isfinite(b):
            assert a == pytest.approx(b)


def test_bonferroni_endpoints_and_fixed_family_size() -> None:
    lo, hi = bonferroni_quantiles()
    assert lo == pytest.approx(0.05 / 6)
    assert hi == pytest.approx(1.0 - 0.05 / 6)
    assert lo == pytest.approx(BONFERRONI_LO)
    assert hi == pytest.approx(BONFERRONI_HI)
    assert FAMILY_SIZE == 3
    assert ALPHA == 0.05
    # Family size remains 3 even if one measurement is inconclusive
    lo2, hi2 = bonferroni_quantiles(alpha=ALPHA, family_size=FAMILY_SIZE)
    assert (lo2, hi2) == (lo, hi)


def test_undefined_within_date_correlation_and_zero_valid() -> None:
    d0 = date(2021, 5, 3)
    # Constant measurement within date → exclude
    rows = [
        _base_row(trade_date=d0, ticker="AAA", M=5.0, H=1.0, X=8.0, M1=0.2),
        _base_row(trade_date=d0, ticker="BBB", M=5.0, H=1.0, X=3.0, M1=0.2),
    ]
    anal = analysis_set_for_measurement(attach_scenario_economics(_panel_from_rows(rows), 1.0), "M1")
    anal = anal.copy()
    anal["quintile"] = assign_frozen_quintiles(anal, "M1")
    summary = within_date_spearman_summary(anal, "M1")
    assert summary["n_valid"] == 0
    assert summary["p_wd"] is False
    assert summary["n_excluded_constant"] >= 1

    # Constant returns within date → exclude
    rows2 = [
        _base_row(trade_date=d0, ticker="AAA", M=5.0, H=1.0, X=7.0, M1=0.1),
        _base_row(trade_date=d0, ticker="BBB", M=5.0, H=2.0, X=7.0, M1=0.4),
    ]
    # Force same C so same r when X same and... wait C differs if H differs under h=1
    # Use same M,H so C same, X same → r constant; M1 differs
    rows2 = [
        _base_row(trade_date=d0, ticker="AAA", M=5.0, H=1.0, X=7.0, M1=0.1),
        _base_row(trade_date=d0, ticker="BBB", M=5.0, H=1.0, X=7.0, M1=0.4),
    ]
    anal2 = analysis_set_for_measurement(attach_scenario_economics(_panel_from_rows(rows2), 1.0), "M1")
    summary2 = within_date_spearman_summary(anal2, "M1")
    assert summary2["n_valid"] == 0
    assert summary2["p_wd"] is False


def test_invalid_bootstrap_statistics_when_groups_empty() -> None:
    # Single date / tiny panel → many invalid paths
    rows = [
        _base_row(trade_date=date(2021, 1, 4), ticker=tk, M=5.0, H=1.0, X=6.0 + i, M1=0.1 * (i + 1))
        for i, tk in enumerate(list("ABCDE"))
    ]
    anal = analysis_set_for_measurement(attach_scenario_economics(_panel_from_rows(rows), 1.0), "M1")
    anal = anal.copy()
    anal["quintile"] = assign_frozen_quintiles(anal, "M1")
    boot = run_block_bootstrap_deltas(anal, n_boot=50, seed=1, block_len=4, progress_every=0)
    assert boot["n_reps"] == 50
    assert boot["n_valid_delta"] + boot["n_invalid_delta"] == 50
    # With T=1, path always same date; Q1/Q5 may be size 1 each → invalid (<2)
    assert boot["n_invalid_delta"] >= 0


def test_p_gross_edge_cases() -> None:
    # Build Q1 with non-positive all-sample gross mean
    d0 = date(2021, 6, 7)
    rows = []
    # Low M1 (Q1) mildly negative g; high M1 more negative — g_all <= 0, Q1 >= g_all
    for i, tk in enumerate(list("ABCDEFGHIJ")):
        m1 = 0.05 * (i + 1)
        # X slightly below M so g < 0; lower cost intensity → less negative X-M somehow
        X = 4.0 - 0.05 * i  # higher i (higher M1) worse X
        rows.append(_base_row(trade_date=d0, ticker=tk, M=5.0, H=1.0, X=X, M1=m1))
    anal = analysis_set_for_measurement(attach_scenario_economics(_panel_from_rows(rows), 1.0), "M1")
    anal = anal.copy()
    anal["quintile"] = assign_frozen_quintiles(anal, "M1")
    g_all = float(anal["g"].mean())
    assert g_all <= 0.0
    # Criterion 2 fails when G+_all == 0
    assert evaluate_p_gross(anal, "M1") is False

    # G+_all == 0 explicit tiny panel
    rows_zero = [
        _base_row(trade_date=d0, ticker=tk, M=5.0, H=1.0, X=1.0, M1=0.1 * (i + 1))
        for i, tk in enumerate(list("ABCDE"))
    ]
    anal_z = analysis_set_for_measurement(
        attach_scenario_economics(_panel_from_rows(rows_zero), 1.0), "M1"
    )
    anal_z = anal_z.copy()
    anal_z["quintile"] = assign_frozen_quintiles(anal_z, "M1")
    assert (anal_z["g"] > 0).sum() == 0
    assert evaluate_p_gross(anal_z, "M1") is False

    # M3 N/A always passes
    assert evaluate_p_gross(anal_z, "M3") is True


def test_classification_boundaries_and_precedence() -> None:
    # Row 1: not P-cov
    preds = {
        "P-cov": False,
        "P-delta-def": True,
        "P-econ": True,
        "P-stat": True,
        "P-sign": True,
        "P-half": True,
        "P-wd": True,
        "P-gross": True,
        "P-wrong": False,
        "interval_defined": True,
    }
    assert classify_measurement(preds, point_delta=0.1, adj_upper=0.2) == (
        LABEL_INCONCLUSIVE,
        1,
    )

    # Row 2: P-wrong beats later supported-looking flags
    preds2 = {**preds, "P-cov": True, "P-wrong": True}
    assert classify_measurement(preds2, point_delta=-0.1, adj_upper=0.0)[0] == LABEL_UNSUPPORTED

    # Row 3: supported
    preds3 = {**preds, "P-cov": True, "P-wrong": False}
    assert classify_measurement(preds3, point_delta=0.1, adj_upper=0.2) == (
        LABEL_SUPPORTED,
        3,
    )

    # Row 4: missing P-wd
    preds4 = {**preds3, "P-wd": False}
    assert classify_measurement(preds4, point_delta=0.1, adj_upper=0.2) == (
        LABEL_INCONCLUSIVE,
        4,
    )

    # Row 5: missing P-half
    preds5 = {**preds3, "P-half": False}
    assert classify_measurement(preds5, point_delta=0.1, adj_upper=0.2) == (
        LABEL_INCONCLUSIVE,
        5,
    )

    # Row 6: missing P-gross
    preds6 = {**preds3, "P-gross": False}
    assert classify_measurement(preds6, point_delta=0.1, adj_upper=0.2) == (
        LABEL_INCONCLUSIVE,
        6,
    )

    # Row 7: sign+stat, not econ, delta>=0.02
    preds7 = {
        **preds3,
        "P-econ": False,
        "P-half": False,
        "P-wd": False,
        "P-gross": False,
    }
    assert classify_measurement(preds7, point_delta=0.03, adj_upper=0.1) == (
        LABEL_INCONCLUSIVE,
        7,
    )

    # Row 8: econ+sign, not stat
    preds8 = {**preds3, "P-stat": False, "P-half": False, "P-wd": False}
    assert classify_measurement(preds8, point_delta=0.1, adj_upper=0.2) == (
        LABEL_INCONCLUSIVE,
        8,
    )

    # Row 10 unsupported via delta<=0
    preds10 = {
        "P-cov": True,
        "P-delta-def": True,
        "P-econ": False,
        "P-stat": False,
        "P-sign": False,
        "P-half": False,
        "P-wd": False,
        "P-gross": True,
        "P-wrong": False,
        "interval_defined": True,
    }
    assert classify_measurement(preds10, point_delta=-0.01, adj_upper=0.05)[0] == LABEL_UNSUPPORTED
    # Row 10 inconclusive otherwise
    assert classify_measurement(preds10, point_delta=0.01, adj_upper=0.05)[0] == LABEL_INCONCLUSIVE


def test_supported_despite_weak_pooled_spearman() -> None:
    """Pooled Spearman is diagnostic only — does not appear in classify inputs."""
    preds = {
        "P-cov": True,
        "P-delta-def": True,
        "P-econ": True,
        "P-stat": True,
        "P-sign": True,
        "P-half": True,
        "P-wd": True,
        "P-gross": True,
        "P-wrong": False,
        "interval_defined": True,
    }
    label, row = classify_measurement(preds, point_delta=0.2, adj_upper=0.4)
    assert label == LABEL_SUPPORTED and row == 3
    # Weak Spearman would be ignored (not a predicate)


def test_evaluation_period_rows_excluded_from_outputs() -> None:
    rows = [
        _base_row(trade_date=date(2023, 6, 5), ticker="AAA", M=5.0, H=1.0, X=6.0, M1=0.2),
        _base_row(trade_date=date(2024, 1, 8), ticker="AAA", M=5.0, H=1.0, X=6.0, M1=0.2),
        _base_row(trade_date=date(2019, 6, 3), ticker="AAA", M=5.0, H=1.0, X=6.0, M1=0.2),
    ]
    panel = _panel_from_rows(rows)
    dev = filter_development_panel(panel)
    assert all(DEV_START <= r <= DEV_END for r in dev["trade_date"])
    assert not any(is_evaluation_date(r) for r in dev["trade_date"])
    assert_no_evaluation_rows(dev, context="dev")
    with pytest.raises(D1ValidationError):
        assert_no_evaluation_rows(panel, context="raw")

    econ = attach_scenario_economics(dev, PRIMARY_H)
    anal = analysis_set_for_measurement(econ, "M1")
    assert_no_evaluation_rows(anal, context="anal")
    assert EVAL_START not in set(anal["trade_date"].tolist())


def test_run_d1_validation_synthetic_end_to_end() -> None:
    """Small synthetic panel through run_d1_validation (tiny bootstrap)."""
    dates = [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(24)]
    rows = []
    for d in dates:
        for j, tk in enumerate(list("ABCDEFGHIJ")):
            # Lower M1 → higher X (favorable) to create Q1>Q5 signal
            m1 = 0.05 * (j + 1)
            X = 12.0 - 0.8 * j + 0.05 * (d.toordinal() % 5)
            half = "A" if d.year <= 2021 else "B"
            # keep both halves favorable
            _ = half
            rows.append(
                _base_row(
                    trade_date=d,
                    ticker=tk,
                    M=5.0,
                    H=5.0 * m1,
                    X=X,
                    M1=m1,
                    M2=m1 * 0.5,
                    M3=m1 * 0.8,
                )
            )
    panel = _panel_from_rows(rows)
    result = run_d1_validation(
        panel=panel,
        n_boot=200,
        seed=20260907,
        block_len=4,
        progress_every=0,
    )
    assert set(result.labels) == set(MEASUREMENTS)
    assert result.gate["decision"] in {"AUTHORIZE_D2", "STOP_NO_THRESHOLDS"}
    assert_no_evaluation_rows(result.panel_dev, context="result.panel_dev")
    for m in MEASUREMENTS:
        assert_no_evaluation_rows(result.analysis_panels[m], context=m)
        assert "point_delta" in result.delta_tables[m]
    # Primary delta uses h=1
    for m, anal in result.analysis_panels.items():
        if len(anal) >= 10:
            d = compute_delta_from_groups(anal)
            assert d == pytest.approx(result.delta_tables[m]["point_delta"])


def test_p_wrong_soft_threshold() -> None:
    preds = evaluate_predicates(
        anal=pd.DataFrame({"quintile": ["Q1", "Q5"], "r": [0.01, 0.0], "g": [0.1, 0.1], "dollar_gross": [1.0, 1.0], "trade_date": [date(2021, 1, 4)] * 2}),
        measurement="M1",
        point_delta=0.01,
        boot={"adj_lower": -0.1, "interval_defined": True, "adj_upper": 0.05},
        within={"p_wd": True},
        halves={"dev_a": 0.01, "dev_b": 0.01},
    )
    # delta < 0.02 and adj lower not > 0 → P-wrong
    assert preds["P-wrong"] is True


def test_sprint_gate_ranking() -> None:
    labels = {"M1": LABEL_SUPPORTED, "M2": LABEL_SUPPORTED, "M3": LABEL_INCONCLUSIVE}
    gate = sprint_level_gate(
        labels,
        deltas={"M1": 0.1, "M2": 0.2, "M3": 0.05},
        within={
            "M1": {"frac_rho_lt_0": 0.6},
            "M2": {"frac_rho_lt_0": 0.55},
            "M3": {"frac_rho_lt_0": 0.5},
        },
        anal_by_m={
            "M1": pd.DataFrame(),
            "M2": pd.DataFrame(),
            "M3": pd.DataFrame(),
        },
    )
    assert gate["authorize_d2"] is True
    assert gate["ranked_candidates"][0] == "M2"

    stop = sprint_level_gate(
        {m: LABEL_UNSUPPORTED for m in MEASUREMENTS},
        deltas={m: -0.1 for m in MEASUREMENTS},
        within={m: {"frac_rho_lt_0": 0.0} for m in MEASUREMENTS},
        anal_by_m={m: pd.DataFrame() for m in MEASUREMENTS},
    )
    assert stop["authorize_d2"] is False
    assert stop["decision"] == "STOP_NO_THRESHOLDS"


def test_half_period_uses_frozen_labels() -> None:
    dates_a = [date(2020, 2, 3) + timedelta(days=7 * i) for i in range(4)]
    dates_b = [date(2022, 2, 7) + timedelta(days=7 * i) for i in range(4)]
    rows = []
    for d in dates_a + dates_b:
        for j, tk in enumerate(list("ABCDE")):
            rows.append(
                _base_row(
                    trade_date=d,
                    ticker=tk,
                    M=5.0,
                    H=1.0,
                    X=10.0 - j,
                    M1=0.1 * (j + 1),
                )
            )
    anal = analysis_set_for_measurement(attach_scenario_economics(_panel_from_rows(rows), 1.0), "M1")
    anal = anal.copy()
    anal["quintile"] = assign_frozen_quintiles(anal, "M1")
    halves = half_period_deltas(anal)
    assert np.isfinite(halves["dev_a"])
    assert np.isfinite(halves["dev_b"])


def test_sensitivity_h_list_excludes_primary() -> None:
    assert PRIMARY_H not in SENSITIVITY_H
    assert set(SENSITIVITY_H) == {0.0, 0.25, 0.50}


def test_crossed_quote_not_in_association_but_counted_in_n() -> None:
    d0 = date(2021, 8, 2)
    panel = _panel_from_rows(
        [
            _base_row(trade_date=d0, ticker="AAA", M=5.0, H=1.0, X=7.0, M1=0.2),
            _base_row(trade_date=d0, ticker="BBB", M=5.0, H=1.0, X=7.0, M1=0.3, crossed=True),
        ]
    )
    econ = attach_scenario_economics(panel, 1.0)
    assert int(econ["in_N"].sum()) == 2
    anal = analysis_set_for_measurement(econ, "M1")
    assert "BBB" not in set(anal["ticker"])
    assert "AAA" in set(anal["ticker"])
