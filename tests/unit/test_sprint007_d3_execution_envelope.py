"""Sprint 007 D3 envelope tests — synthetic frames only; no official economics."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.pipeline import _apply_tier_a_sizing
from src.backtest.sprint007_d1_gross_margin import VERDICT_CONTINUE, VERDICT_STOP
from src.backtest.sprint007_d2_shortfall_bridge import CLASS_EXECUTION, CLASS_MIXED
from src.backtest.sprint007_d3_execution_envelope import (
    H_TOL,
    NO_CROSSING,
    TIER_A_CONFIG,
    VERDICT_BLOCKED,
    Crossing,
    D3AnalysisError,
    D3Book,
    assemble_envelope,
    check_prerequisites,
    evaluate_path_at_h,
    fill_price_at_h,
    first_adverse_crossing,
    package_entry_cost_at_h,
    path_f_pnl_crossing,
    run_d3_from_book,
    size_book_at_h,
)


def test_fill_price_at_h_endpoints() -> None:
    assert fill_price_at_h(1.0, 3.0, 1, 0.0) == pytest.approx(2.0)
    assert fill_price_at_h(1.0, 3.0, 1, 1.0) == pytest.approx(3.0)
    assert fill_price_at_h(1.0, 3.0, -1, 1.0) == pytest.approx(1.0)
    assert fill_price_at_h(1.0, 3.0, 1, 0.5) == pytest.approx(2.5)


def test_package_entry_and_linear_pnl() -> None:
    legs = pd.DataFrame(
        [
            {"unit_quantity": 1, "bid": 1.0, "ask": 3.0},
            {"unit_quantity": 1, "bid": 1.0, "ask": 3.0},
        ]
    )
    c0 = package_entry_cost_at_h(legs, 0.0)
    c1 = package_entry_cost_at_h(legs, 1.0)
    c05 = package_entry_cost_at_h(legs, 0.5)
    assert c0 == pytest.approx(4.0)
    assert c1 == pytest.approx(6.0)
    assert c05 == pytest.approx(5.0)
    payoff = 10.0
    assert (payoff - c05) == pytest.approx(0.5 * (payoff - c0) + 0.5 * (payoff - c1))


def test_path_f_identity_and_closed_form() -> None:
    book = _long_book(p_mid=20.0, p_cross=-10.0, qty=2.0)
    p_mid = 40.0
    delta = 2.0 * (-10.0) - 40.0
    for h in (0.0, 0.25, 0.5, 1.0):
        got = evaluate_path_at_h(book, h, path="F")["pnl"]
        assert got == pytest.approx(p_mid + h * delta)
    assert path_f_pnl_crossing(0.50, p_mid, delta) == pytest.approx(0.5 * p_mid / (-delta))
    assert path_f_pnl_crossing(0.25, p_mid, delta) == pytest.approx(0.75 * p_mid / (-delta))
    assert path_f_pnl_crossing(0.00, p_mid, delta) == pytest.approx(p_mid / (-delta))
    with pytest.raises(ValueError):
        path_f_pnl_crossing(1.00, p_mid, delta)


def test_first_adverse_monotonic_order() -> None:
    def m(h: float) -> float:
        return 100.0 - 200.0 * h

    h50 = first_adverse_crossing(m, 50.0)
    h25 = first_adverse_crossing(m, 25.0)
    h0 = first_adverse_crossing(m, 0.0)
    assert h50 is not None and h25 is not None and h0 is not None
    assert h50 < h25 < h0
    assert h50 == pytest.approx(0.25, abs=H_TOL)
    assert h25 == pytest.approx(0.375, abs=H_TOL)
    assert h0 == pytest.approx(0.50, abs=H_TOL)


def test_first_adverse_non_monotonic_uses_first_root() -> None:
    def m(h: float) -> float:
        if h < 0.2:
            return 10.0 - 80.0 * h
        return 5.0

    h_star = first_adverse_crossing(m, 0.0)
    assert h_star == pytest.approx(0.125, abs=H_TOL)


def test_missed_dip_guard() -> None:
    def m(h: float) -> float:
        return -1.0 if abs(h - 0.015) < 1e-9 else 10.0

    h_star = first_adverse_crossing(m, 0.0)
    assert h_star is not None
    assert 0.01 < h_star <= 0.015 + H_TOL


def test_bisection_tolerance() -> None:
    def m(h: float) -> float:
        return 1.0 - 20.0 * h

    h_star = first_adverse_crossing(m, 0.0)
    assert h_star is not None
    assert abs(h_star - 0.05) <= H_TOL
    assert m(h_star) <= 0.0


def test_h_vis_is_not_the_root() -> None:
    def m(h: float) -> float:
        return 5.0 if h < 0.033 else -1.0

    h_star = first_adverse_crossing(m, 0.0)
    vis_interp = 0.0 + 0.05 * 5.0 / 6.0
    assert h_star is not None
    assert abs(h_star - 0.033) <= H_TOL
    assert abs(h_star - vis_interp) > H_TOL
    assert 0.05 not in {h_star}


def test_no_crossing_returns_none() -> None:
    assert first_adverse_crossing(lambda h: 10.0 - h, 0.0) is None


def test_path_r_sizing_matches_tier_a() -> None:
    trades = _sized_day(max_loss=10.0, short_credit=2.0, long_premium=4.0)
    sized = size_book_at_h(trades, 0.0)
    expected = trades.copy()
    expected["included_in_portfolio"] = True
    expected["quantity"] = float("nan")
    _apply_tier_a_sizing(expected, TIER_A_CONFIG)
    pd.testing.assert_series_equal(
        sized["quantity"].reset_index(drop=True),
        expected["quantity"].reset_index(drop=True),
    )
    assert abs(float(sized.loc[sized["direction"] == "short", "quantity"].iloc[0])) == pytest.approx(1000.0)
    assert abs(float(sized.loc[sized["direction"] == "long", "quantity"].iloc[0])) == pytest.approx(500.0)


def test_endpoint_q_recovers_sized_books() -> None:
    mid = _sized_day(max_loss=10.0, short_credit=2.0, long_premium=4.0)
    mid_sized = size_book_at_h(mid, 0.0)
    cross = _sized_day(max_loss=20.0, short_credit=1.0, long_premium=5.0)
    cross_sized = size_book_at_h(cross, 1.0)
    assert mid_sized["quantity"].notna().all()
    assert cross_sized["quantity"].notna().all()
    assert list(mid_sized["quantity"]) != list(cross_sized["quantity"])


def test_exclusion_guard() -> None:
    trades = _sized_day(max_loss=0.0, short_credit=2.0, long_premium=4.0)
    with pytest.raises(D3AnalysisError, match="drop a frozen key"):
        size_book_at_h(trades, 0.0)


def test_envelope_schema_has_no_h_req() -> None:
    result = assemble_envelope(
        h_R_50=0.4,
        h_R_25=0.5,
        h_R_P0=0.6,
        h_R_CAR0=0.55,
        h_F_50=0.2,
        h_F_25=0.3,
        h_F_P0=0.35,
        h_F_CAR0=0.33,
        crossings=[
            Crossing("R", "primary", "pnl_50", 50.0, 0.4, "bracket_bisection"),
            Crossing("F", "diagnostic", "pnl_50", 50.0, 0.2, "closed_form"),
        ],
        curves=pd.DataFrame(),
        reconciliation=[],
        monotonicity={"P_R": True},
        manifest={},
    )
    payload = result.to_dict()
    assert "h_req" not in payload
    assert "h_req" not in payload["path_r_envelope"]
    assert result.h_R_50 == 0.4
    assert result.h_F_50 == 0.2
    assert result.headroom_50_to_25 == pytest.approx(0.1)
    assert result.headroom_25_to_P0 == pytest.approx(0.1)
    assert result.distance_P0_to_cross == pytest.approx(0.4)
    assert payload["path_f_diagnostic"]["h_F_50"] == 0.2


def test_path_f_does_not_bind_path_r() -> None:
    result = assemble_envelope(
        h_R_50=0.40,
        h_R_25=0.55,
        h_R_P0=0.70,
        h_R_CAR0=0.65,
        h_F_50=0.10,
        h_F_25=0.15,
        h_F_P0=0.20,
        h_F_CAR0=0.18,
        crossings=[],
        curves=pd.DataFrame(),
        reconciliation=[],
        monotonicity={},
        manifest={},
    )
    assert result.h_R_50 == 0.40
    assert result.h_R_50 != result.h_F_50
    assert min(result.h_F_50, result.h_R_50) == result.h_F_50
    assert result.to_dict()["path_r_envelope"]["h_R_50"] == 0.40


def test_prerequisite_wrong_class_blocks() -> None:
    assert check_prerequisites(
        d0_passed=True, d1_verdict=VERDICT_CONTINUE, d2_final_class=CLASS_MIXED
    )
    book = _long_book(p_mid=20.0, p_cross=-10.0, qty=1.0)
    result = run_d3_from_book(
        book,
        d0_passed=True,
        d1_verdict=VERDICT_CONTINUE,
        d2_final_class=CLASS_MIXED,
    )
    assert result.verdict == VERDICT_BLOCKED
    assert result.h_R_50 is None
    assert "h_req" not in result.to_dict()


def test_prerequisite_d1_stop_blocks() -> None:
    book = _long_book(p_mid=20.0, p_cross=-10.0, qty=1.0)
    result = run_d3_from_book(
        book,
        d0_passed=True,
        d1_verdict=VERDICT_STOP,
        d2_final_class=CLASS_EXECUTION,
    )
    assert result.verdict == VERDICT_BLOCKED


def test_headroom_no_crossing_token() -> None:
    result = assemble_envelope(
        h_R_50=0.2,
        h_R_25=None,
        h_R_P0=None,
        h_R_CAR0=None,
        h_F_50=0.1,
        h_F_25=None,
        h_F_P0=None,
        h_F_CAR0=None,
        crossings=[],
        curves=pd.DataFrame(),
        reconciliation=[],
        monotonicity={},
        manifest={},
    )
    assert result.headroom_50_to_25 == NO_CROSSING
    assert result.distance_P0_to_cross == NO_CROSSING


def _long_book(*, p_mid: float, p_cross: float, qty: float) -> D3Book:
    trade_date = date(2021, 1, 4)
    trades = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": "AAA",
                "direction": "long",
                "quantity_mid": qty,
                "quantity_cross": qty * 0.5,
                "pnl_per_share_mid": p_mid,
                "pnl_per_share_cross": p_cross,
                "pnl_total_mid": qty * p_mid,
                "pnl_total_cross": (qty * 0.5) * p_cross,
                "wing_width": None,
                "instrument_type": "long_straddle",
            }
        ]
    )
    legs = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": "AAA",
                "direction": "long",
                "unit_quantity": 1,
                "bid": 1.0,
                "ask": 3.0,
            },
            {
                "trade_date": trade_date,
                "ticker": "AAA",
                "direction": "long",
                "unit_quantity": 1,
                "bid": 1.0,
                "ask": 3.0,
            },
        ]
    )
    status = pd.DataFrame(
        [{"trade_date": trade_date, "status": "traded", "reason": "ok"}]
    )
    return D3Book(trades=trades, legs=legs, date_status=status)


def _sized_day(*, max_loss: float, short_credit: float, long_premium: float) -> pd.DataFrame:
    trade_date = date(2021, 1, 4)
    return pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": "AAA",
                "direction": "short",
                "included_in_portfolio": True,
                "entry_cost_per_share": -short_credit,
                "net_credit_per_share": short_credit,
                "max_loss_per_share": max_loss,
            },
            {
                "trade_date": trade_date,
                "ticker": "BBB",
                "direction": "long",
                "included_in_portfolio": True,
                "entry_cost_per_share": long_premium,
                "net_credit_per_share": -long_premium,
                "max_loss_per_share": long_premium,
            },
        ]
    )
