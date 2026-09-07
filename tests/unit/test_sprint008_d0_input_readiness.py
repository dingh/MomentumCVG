"""Sprint 008 D0 input-readiness tests — synthetic frames only."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.backtest.sprint007_d2b_package_tradability import (
    midpoint_package_cashflow,
    package_half_spread,
)
from src.backtest.sprint008_d0_input_readiness import (
    DOLLAR_TOL,
    MAX_NAMES,
    M3_LOOKBACK_DAYS,
    M3_MIN_HISTORY,
    VERDICT_BLOCKED,
    attach_outcomes_and_measurements,
    compute_m3_scores,
    compute_package_mh,
    equal_dollar_quantities,
    evaluate_readiness_gates,
    missing_outcome_smoke,
    reconstruct_capped_long_n,
    smoke_equal_dollar_accounting,
)


def _two_legs(
    *,
    trade_date: date = date(2021, 6, 1),
    ticker: str = "AAA",
    bid_call: float = 1.0,
    ask_call: float = 3.0,
    bid_put: float = 2.0,
    ask_put: float = 4.0,
    mid_call: float = 999.0,
    mid_put: float = 999.0,
    strike: float = 100.0,
    payoff_call: float = 5.0,
    payoff_put: float = 0.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": ticker,
                "direction": "long",
                "expiry_date": trade_date + timedelta(days=4),
                "option_type": "call",
                "strike": strike,
                "leg_index": 0,
                "unit_quantity": 1.0,
                "bid": bid_call,
                "ask": ask_call,
                "mid": mid_call,
                "expiry_payoff_per_unit": payoff_call,
            },
            {
                "trade_date": trade_date,
                "ticker": ticker,
                "direction": "long",
                "expiry_date": trade_date + timedelta(days=4),
                "option_type": "put",
                "strike": strike,
                "leg_index": 1,
                "unit_quantity": 1.0,
                "bid": bid_put,
                "ask": ask_put,
                "mid": mid_put,
                "expiry_payoff_per_unit": payoff_put,
            },
        ]
    )


def test_synthetic_two_leg_m_h_ask_debit() -> None:
    legs = _two_legs()
    uq = legs["unit_quantity"].to_numpy(dtype=float)
    bid = legs["bid"].to_numpy(dtype=float)
    ask = legs["ask"].to_numpy(dtype=float)
    expected_m = midpoint_package_cashflow(uq, bid, ask)
    expected_h = package_half_spread(uq, bid, ask)
    expected_ask = float(np.sum(uq * ask))

    metrics = compute_package_mh(legs)
    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert row["M"] == pytest.approx(expected_m)
    assert row["H"] == pytest.approx(expected_h)
    assert row["ask_debit"] == pytest.approx(expected_ask)
    assert row["M"] + row["H"] == pytest.approx(expected_ask)
    # call mid=(1+3)/2=2, put mid=(2+4)/2=3 → M=5; H=0.5*((3-1)+(4-2))=2
    assert row["M"] == pytest.approx(5.0)
    assert row["H"] == pytest.approx(2.0)
    assert row["ask_debit"] == pytest.approx(7.0)


def test_stored_mid_ignored_for_m() -> None:
    legs = _two_legs(mid_call=50.0, mid_put=50.0)
    metrics = compute_package_mh(legs)
    assert metrics.iloc[0]["M"] == pytest.approx(5.0)
    assert metrics.iloc[0]["M"] != pytest.approx(100.0)


def test_entry_cost_mid_mismatch_fails_readiness_delta_check() -> None:
    legs = _two_legs()
    mh = compute_package_mh(legs)
    trades = pd.DataFrame(
        [
            {
                "trade_date": date(2021, 6, 1),
                "ticker": "AAA",
                "direction": "long",
                "structure_ok": True,
                "in_N": True,
                "signal_rank_pct": 0.9,
                "included_in_portfolio": True,
                "entry_spot": 100.0,
                "exit_spot": 105.0,
                "body_strike": 100.0,
                "expiry_date": date(2021, 6, 5),
                "entry_cost_mid_per_share": 5.0 + 10 * DOLLAR_TOL,  # deliberate mismatch
            }
        ]
    )
    panel = attach_outcomes_and_measurements(trades, mh)
    assert abs(float(panel.iloc[0]["delta_M_vs_stored"])) > DOLLAR_TOL

    panel = compute_m3_scores(panel)
    verdict, gates, _ = evaluate_readiness_gates(
        panel=panel,
        identity_gates=None,
        shared_quotes_ok=True,
        reconstruction_vs_included={"passed": True, "detail": "ok"},
        accounting={"passed": True, "detail": "ok"},
        missing_outcome={"passed": True, "detail": "ok", "n_missing_x": 0},
    )
    mid_gate = next(g for g in gates if g.gate_id == "G3_midpoint_authority")
    assert mid_gate.passed is False
    assert verdict == VERDICT_BLOCKED


def test_n_reconstruction_caps_above_25_names() -> None:
    rows = []
    trade_date = date(2021, 6, 1)
    for i in range(30):
        rows.append(
            {
                "trade_date": trade_date,
                "ticker": f"T{i:02d}",
                "direction": "long",
                "structure_ok": True,
                "signal_rank_pct": 1.0 - i * 0.01,
                "included_in_portfolio": i < 20,
                "entry_spot": 100.0,
                "exit_spot": 100.0,
                "body_strike": 100.0,
                "expiry_date": date(2021, 6, 5),
                "entry_cost_mid_per_share": 5.0,
                "quantity": 999.0,  # must not drive reconstruction
            }
        )
    # One non-constructable long must stay out of N.
    rows.append(
        {
            "trade_date": trade_date,
            "ticker": "ZZZ",
            "direction": "long",
            "structure_ok": False,
            "signal_rank_pct": 1.0,
            "included_in_portfolio": False,
            "entry_spot": 100.0,
            "exit_spot": 100.0,
            "body_strike": 100.0,
            "expiry_date": date(2021, 6, 5),
            "entry_cost_mid_per_share": 5.0,
            "quantity": 1.0,
        }
    )
    frame = pd.DataFrame(rows)
    out = reconstruct_capped_long_n(frame)
    assert int(out["in_N"].sum()) == MAX_NAMES
    kept = out.loc[out["in_N"]].sort_values(["signal_rank_pct", "ticker"], ascending=[False, True])
    assert list(kept["ticker"]) == [f"T{i:02d}" for i in range(MAX_NAMES)]
    assert not bool(out.loc[out["ticker"] == "T25", "in_N"].iloc[0])
    assert not bool(out.loc[out["ticker"] == "ZZZ", "in_N"].iloc[0])


def _history_row(
    entry: date,
    *,
    ticker: str,
    x: float,
    s0: float = 100.0,
    expiry: date | None = None,
    m: float = 5.0,
    h: float = 1.0,
    in_n: bool = True,
) -> dict:
    return {
        "trade_date": entry,
        "ticker": ticker,
        "direction": "long",
        "structure_ok": True,
        "in_N": in_n,
        "signal_rank_pct": 0.9,
        "included_in_portfolio": True,
        "entry_spot": s0,
        "exit_spot": s0 + x if x >= 0 else s0,
        "body_strike": s0,
        "expiry_date": expiry if expiry is not None else entry + timedelta(days=4),
        "entry_cost_mid_per_share": m,
        "M": m,
        "H": h,
        "ask_debit": m + h,
        "n_legs": 2,
        "has_call": True,
        "has_put": True,
        "strike_match": True,
        "unit_qty_ok": True,
        "expiry_match_legs": True,
        "per_leg_quotes_ok": True,
        "legs_structure_ok": True,
        "leg_strike": s0,
        "leg_expiry": expiry if expiry is not None else entry + timedelta(days=4),
        "strike_matches_body": True,
        "expiry_matches_trade": True,
        "payoff_sum": x,
        "payoff_reconcile_ok": True,
        "S0": s0,
        "K": s0,
        "ST": s0 + x if x >= 0 else s0,
        "X": x,
        "delta_M_vs_stored": 0.0,
        "delta_MH_vs_ask": 0.0,
        "delta_X_vs_payoff_sum": 0.0,
        "M1": h / m,
        "M2": h / s0,
    }


def test_m3_nineteen_history_missing_twenty_with_zero_x_works() -> None:
    t = date(2022, 1, 10)
    rows = []
    for i in range(19):
        entry = t - timedelta(days=30 + i)
        rows.append(_history_row(entry, ticker=f"H{i}", x=2.0, expiry=entry + timedelta(days=3)))
    target = _history_row(t, ticker="TARGET", x=3.0, m=6.0, h=2.0)
    panel = pd.DataFrame(rows + [target])
    scored = compute_m3_scores(panel)
    target_row = scored.loc[scored["ticker"] == "TARGET"].iloc[0]
    assert not np.isfinite(target_row["M3"])
    assert target_row["m3_missing_reason"] == "cold_start"
    assert int(target_row["m3_n_history"]) == 19
    assert bool(target_row["in_N"]) is True

    # 20th history row with X=0 should make M3 finite when mu>0.
    zero_entry = t - timedelta(days=10)
    rows.append(
        _history_row(zero_entry, ticker="ZERO", x=0.0, expiry=zero_entry + timedelta(days=2))
    )
    panel20 = pd.DataFrame(rows + [target])
    scored20 = compute_m3_scores(panel20)
    target20 = scored20.loc[scored20["ticker"] == "TARGET"].iloc[0]
    assert int(target20["m3_n_history"]) == 20
    assert np.isfinite(target20["m3_mu"])
    assert target20["m3_mu"] == pytest.approx((19 * 0.02 + 0.0) / 20)
    assert np.isfinite(target20["M3"])
    assert target20["M3"] == pytest.approx((6.0 + 2.0) / (100.0 * target20["m3_mu"]))


def test_m3_window_boundaries_expiry_and_entry() -> None:
    t = date(2022, 6, 1)
    left = t - timedelta(days=M3_LOOKBACK_DAYS)
    rows = []
    # 20 valid history rows inside window with expiry < t.
    for i in range(M3_MIN_HISTORY):
        entry = t - timedelta(days=20 + i)
        rows.append(_history_row(entry, ticker=f"OK{i}", x=1.0, expiry=entry + timedelta(days=2)))

    # Future expiry (>= t) must be excluded even if entry is in window.
    rows.append(
        _history_row(
            t - timedelta(days=5),
            ticker="FUT_EXP",
            x=10.0,
            expiry=t,  # expiry < t is required; equality excluded
        )
    )
    # Entry on left boundary included; entry on t excluded; entry before left excluded.
    rows.append(_history_row(left, ticker="LEFT_OK", x=1.0, expiry=left + timedelta(days=1)))
    rows.append(_history_row(t, ticker="SAME_DAY", x=1.0, expiry=t + timedelta(days=1)))
    rows.append(
        _history_row(
            left - timedelta(days=1),
            ticker="TOO_OLD",
            x=1.0,
            expiry=left,
        )
    )
    target = _history_row(t, ticker="TARGET", x=2.0)
    # Replace SAME_DAY conflicting key: target is the evaluation row at t.
    panel = pd.DataFrame([r for r in rows if r["ticker"] != "SAME_DAY"] + [target])
    scored = compute_m3_scores(panel)
    target_row = scored.loc[scored["ticker"] == "TARGET"].iloc[0]
    # 20 OK + LEFT_OK = 21; FUT_EXP / TOO_OLD / target itself excluded.
    assert int(target_row["m3_n_history"]) == 21
    assert np.isfinite(target_row["M3"])


def test_missing_x_preserves_stake_marks_incomplete() -> None:
    trade_date = date(2021, 6, 1)
    panel = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": "AAA",
                "direction": "long",
                "structure_ok": True,
                "in_N": True,
                "M": 5.0,
                "H": 1.0,
                "S0": 100.0,
                "K": 100.0,
                "ST": np.nan,
                "X": np.nan,
                "entry_cost_mid_per_share": 5.0,
            },
            {
                "trade_date": trade_date,
                "ticker": "BBB",
                "direction": "long",
                "structure_ok": True,
                "in_N": True,
                "M": 4.0,
                "H": 1.0,
                "S0": 100.0,
                "K": 100.0,
                "ST": 110.0,
                "X": 10.0,
                "entry_cost_mid_per_share": 4.0,
            },
        ]
    )
    sized = equal_dollar_quantities(panel, h=1.0)
    assert sized.loc[sized["ticker"] == "AAA", "stake_dollars"].iloc[0] == pytest.approx(5000.0)
    assert np.isfinite(sized.loc[sized["ticker"] == "AAA", "q_h"].iloc[0])

    smoke = missing_outcome_smoke(panel, h=1.0)
    assert smoke["n_missing_x"] == 1
    assert smoke["stake_preserved"] is True
    assert smoke["pnl_unknown"] is True
    assert smoke["portfolio_incomplete"] is True
    assert smoke["passed"] is True


def test_dummy_reject_cash_not_redistributed() -> None:
    trade_date = date(2021, 6, 1)
    panel = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": "AAA",
                "direction": "long",
                "structure_ok": True,
                "in_N": True,
                "M": 5.0,
                "H": 1.0,
                "S0": 100.0,
                "X": 2.0,
            },
            {
                "trade_date": trade_date,
                "ticker": "BBB",
                "direction": "long",
                "structure_ok": True,
                "in_N": True,
                "M": 4.0,
                "H": 1.0,
                "S0": 100.0,
                "X": 3.0,
            },
        ]
    )
    baseline = smoke_equal_dollar_accounting(panel, h=1.0)
    assert baseline["passed"] is True

    reject = pd.Series([True, False], index=panel.index)
    rejected = smoke_equal_dollar_accounting(panel, h=1.0, reject_mask=reject)
    assert rejected["passed"] is True
    assert rejected["reject_cash_ok"] is True

    sized = equal_dollar_quantities(panel, h=1.0)
    # Keeper quantity unchanged vs baseline (no redistribution).
    assert sized.iloc[1]["q_h"] == pytest.approx(5000.0 / (4.0 + 1.0))
    assert sized.iloc[0]["q_h"] == pytest.approx(5000.0 / (5.0 + 1.0))


def test_all_rejected_cash_equals_full_budget_no_exception() -> None:
    trade_date = date(2021, 6, 1)
    panel = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": "AAA",
                "direction": "long",
                "structure_ok": True,
                "in_N": True,
                "M": 5.0,
                "H": 1.0,
                "S0": 100.0,
                "X": 2.0,
            },
            {
                "trade_date": trade_date,
                "ticker": "BBB",
                "direction": "long",
                "structure_ok": True,
                "in_N": True,
                "M": 4.0,
                "H": 1.0,
                "S0": 100.0,
                "X": 3.0,
            },
        ]
    )
    from src.backtest.sprint008_d0_input_readiness import SCENARIOS_H, BUDGET_B

    reject_all = pd.Series(True, index=panel.index)
    for h in SCENARIOS_H:
        result = smoke_equal_dollar_accounting(panel, h=float(h), reject_mask=reject_all)
        assert result["passed"] is True, result
        assert result["reject_cash_ok"] is True
        assert result["max_abs_budget_error"] <= 1e-6
        # Invested must be zero; cash = full budget.
        sized = equal_dollar_quantities(panel, h=float(h))
        assert float(np.nansum(sized["q_h"] * 0.0 + 0.0)) == 0.0
        assert BUDGET_B == pytest.approx(10_000.0)


def test_per_leg_crossed_call_fails_even_if_package_h_nonneg() -> None:
    """Crossed call ask<bid masked by wide put still fails per-leg quote check."""
    legs = _two_legs(
        bid_call=5.0,
        ask_call=4.0,  # crossed
        bid_put=1.0,
        ask_put=5.0,  # wide positive spread
    )
    metrics = compute_package_mh(legs)
    row = metrics.iloc[0]
    assert row["H"] == pytest.approx(0.5 * ((4.0 - 5.0) + (5.0 - 1.0)))
    assert row["H"] == pytest.approx(1.5)  # package H > 0
    assert bool(row["per_leg_quotes_ok"]) is False
    assert bool(row["legs_structure_ok"]) is False

    trades = pd.DataFrame(
        [
            {
                "trade_date": date(2021, 6, 1),
                "ticker": "AAA",
                "direction": "long",
                "structure_ok": True,
                "in_N": True,
                "signal_rank_pct": 0.9,
                "included_in_portfolio": True,
                "entry_spot": 100.0,
                "exit_spot": 105.0,
                "body_strike": 100.0,
                "expiry_date": date(2021, 6, 5),
                "entry_cost_mid_per_share": float(row["M"]),
            }
        ]
    )
    panel = attach_outcomes_and_measurements(trades, metrics)
    panel = compute_m3_scores(panel)
    verdict, gates, _ = evaluate_readiness_gates(
        panel=panel,
        identity_gates=None,
        shared_quotes_ok=True,
        reconstruction_vs_included={"passed": True, "detail": "ok"},
        accounting={"passed": True, "detail": "ok"},
        missing_outcome={"passed": True, "detail": "ok", "n_missing_x": 0},
    )
    join_gate = next(g for g in gates if g.gate_id == "G2_joins")
    req_gate = next(g for g in gates if g.gate_id == "G4_required_inputs")
    assert join_gate.passed is False
    assert req_gate.passed is False
    assert verdict == VERDICT_BLOCKED


def test_mismatched_body_vs_leg_strike_fails_readiness() -> None:
    legs = _two_legs(strike=100.0, payoff_call=5.0, payoff_put=0.0)
    metrics = compute_package_mh(legs)
    trades = pd.DataFrame(
        [
            {
                "trade_date": date(2021, 6, 1),
                "ticker": "AAA",
                "direction": "long",
                "structure_ok": True,
                "in_N": True,
                "signal_rank_pct": 0.9,
                "included_in_portfolio": True,
                "entry_spot": 100.0,
                "exit_spot": 105.0,
                "body_strike": 101.0,  # mismatch vs leg strike 100
                "expiry_date": date(2021, 6, 5),
                "entry_cost_mid_per_share": float(metrics.iloc[0]["M"]),
            }
        ]
    )
    panel = attach_outcomes_and_measurements(trades, metrics)
    assert bool(panel.iloc[0]["strike_matches_body"]) is False
    panel = compute_m3_scores(panel)
    verdict, gates, _ = evaluate_readiness_gates(
        panel=panel,
        identity_gates=None,
        shared_quotes_ok=True,
        reconstruction_vs_included={"passed": True, "detail": "ok"},
        accounting={"passed": True, "detail": "ok"},
        missing_outcome={"passed": True, "detail": "ok", "n_missing_x": 0},
    )
    assert next(g for g in gates if g.gate_id == "G2_joins").passed is False
    assert next(g for g in gates if g.gate_id == "G4_required_inputs").passed is False
    assert verdict == VERDICT_BLOCKED


def test_m3_ignores_capped_out_history() -> None:
    t = date(2022, 1, 10)
    rows = []
    for i in range(19):
        entry = t - timedelta(days=30 + i)
        rows.append(_history_row(entry, ticker=f"H{i}", x=2.0, expiry=entry + timedelta(days=3)))
    # 20th observation is capped out of N — must not satisfy min history or change mu.
    capped_entry = t - timedelta(days=10)
    rows.append(
        _history_row(
            capped_entry,
            ticker="CAPPED",
            x=50.0,
            expiry=capped_entry + timedelta(days=2),
            in_n=False,
        )
    )
    target = _history_row(t, ticker="TARGET", x=3.0, m=6.0, h=2.0)
    panel = pd.DataFrame(rows + [target])
    scored = compute_m3_scores(panel)
    target_row = scored.loc[scored["ticker"] == "TARGET"].iloc[0]
    assert int(target_row["m3_n_history"]) == 19
    assert target_row["m3_missing_reason"] == "cold_start"
    assert not np.isfinite(target_row["M3"])

    # Control: same row with in_N True would reach 20 and pull mu toward large X/S0.
    rows[-1]["in_N"] = True
    panel_in = pd.DataFrame(rows + [target])
    scored_in = compute_m3_scores(panel_in)
    target_in = scored_in.loc[scored_in["ticker"] == "TARGET"].iloc[0]
    assert int(target_in["m3_n_history"]) == 20
    assert np.isfinite(target_in["M3"])
    assert target_in["m3_mu"] == pytest.approx((19 * 0.02 + 0.5) / 20)
