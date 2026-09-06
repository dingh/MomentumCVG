"""Sprint 007 D3 envelope tests — synthetic frames only; no official economics."""
from __future__ import annotations

from datetime import date
import json

import pandas as pd
import pytest

from src.backtest.surface_decision_report import PRIMARY_END, PRIMARY_START

from src.backtest.pipeline import _apply_tier_a_sizing
from src.backtest.sprint007_d1_gross_margin import VERDICT_CONTINUE, VERDICT_STOP
from src.backtest.sprint007_d2_shortfall_bridge import CLASS_EXECUTION, CLASS_MIXED
from src.backtest.sprint007_d3_execution_envelope import (
    EVIDENCE_DIR_ENV,
    H_TOL,
    NO_CROSSING,
    PROGRESS_NAME,
    TIER_A_CONFIG,
    VERDICT_BLOCKED,
    WORKERS_ENV,
    Crossing,
    D3AnalysisError,
    D3Book,
    assemble_envelope,
    check_prerequisites,
    CAR_TOLERANCE,
    H_DET,
    H_VIS,
    book_entry_costs_at_h,
    evaluate_path_at_h,
    export_d3_evidence,
    fill_price_at_h,
    filter_primary_date_status,
    first_adverse_crossing,
    join_paired_trades,
    load_eval_checkpoint,
    package_entry_cost_at_h,
    path_f_pnl_crossing,
    reconcile_d3_endpoints,
    run_d3_from_book,
    side_snapshots_at_roots,
    size_book_at_h,
    _precompute_path_grid,
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


def test_vectorized_entry_cost_matches_per_trade_loop() -> None:
    mid, cross = _paired_trade_frames()
    joined = join_paired_trades(mid, cross)
    legs = pd.DataFrame(
        [
            {
                "trade_date": date(2021, 1, 4),
                "ticker": "AAA",
                "direction": "long",
                "unit_quantity": 1,
                "bid": 1.0,
                "ask": 3.0,
            },
            {
                "trade_date": date(2021, 1, 4),
                "ticker": "AAA",
                "direction": "long",
                "unit_quantity": 1,
                "bid": 1.0,
                "ask": 3.0,
            },
            {
                "trade_date": date(2021, 1, 4),
                "ticker": "BBB",
                "direction": "short",
                "unit_quantity": -1,
                "bid": 2.0,
                "ask": 4.0,
            },
            {
                "trade_date": date(2021, 1, 4),
                "ticker": "BBB",
                "direction": "short",
                "unit_quantity": 1,
                "bid": 0.5,
                "ask": 1.5,
            },
        ]
    )
    status = pd.DataFrame([{"trade_date": date(2021, 1, 4), "status": "traded", "reason": "ok"}])
    book = D3Book(trades=joined, legs=legs, date_status=status)
    for h in (0.0, 0.33, 1.0):
        vector = book_entry_costs_at_h(book, h)
        for trade in book.trades.itertuples(index=False):
            key = (trade.trade_date, trade.ticker, trade.direction)
            looped = package_entry_cost_at_h(book.legs_by_key[key], h)
            assert float(vector.loc[key]) == pytest.approx(looped)


def test_progress_file_records_elapsed_and_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(EVIDENCE_DIR_ENV, str(tmp_path))
    book = _long_book(p_mid=20.0, p_cross=-10.0, qty=2.0)
    evaluate_path_at_h(book, 0.1, path="R")
    lines = (tmp_path / PROGRESS_NAME).read_text(encoding="utf-8").strip().splitlines()
    assert lines
    rec = json.loads(lines[-1])
    assert "elapsed_s" in rec
    assert rec["eval_calls"] == 1
    assert rec["cache_size"] == 1


def test_checkpoint_resume_skips_recompute(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(EVIDENCE_DIR_ENV, str(tmp_path))
    book = _long_book(p_mid=20.0, p_cross=-10.0, qty=2.0)
    first = evaluate_path_at_h(book, 0.4, path="R")
    resumed = _long_book(p_mid=20.0, p_cross=-10.0, qty=2.0)
    assert load_eval_checkpoint(resumed, tmp_path) == 1
    calls = resumed.eval_calls
    second = evaluate_path_at_h(resumed, 0.4, path="R")
    assert resumed.eval_calls == calls
    assert second["pnl"] == pytest.approx(first["pnl"])
    assert second["car"] == pytest.approx(first["car"])


def test_precompute_parallel_matches_sequential(monkeypatch) -> None:
    grid = (0.0, 0.1, 0.2)
    sequential = _long_book(p_mid=20.0, p_cross=-10.0, qty=2.0)
    monkeypatch.setenv(WORKERS_ENV, "1")
    _precompute_path_grid(sequential, "R", grid, label="seq")
    parallel = _long_book(p_mid=20.0, p_cross=-10.0, qty=2.0)
    monkeypatch.setenv(WORKERS_ENV, "2")
    _precompute_path_grid(parallel, "R", grid, label="par")
    for h in grid:
        left = sequential.eval_cache[("R", float(f"{h:.12f}"))]
        right = parallel.eval_cache[("R", float(f"{h:.12f}"))]
        assert right["pnl"] == pytest.approx(left["pnl"])
        assert right["car"] == pytest.approx(left["car"])


def test_evaluate_path_at_h_memoizes_same_h() -> None:
    book = _long_book(p_mid=20.0, p_cross=-10.0, qty=2.0)
    first = evaluate_path_at_h(book, 0.2, path="R")
    calls = book.eval_calls
    second = evaluate_path_at_h(book, 0.2, path="R")
    assert book.eval_calls == calls
    assert second["pnl"] == pytest.approx(first["pnl"])
    assert second["car"] == pytest.approx(first["car"])
    evaluate_path_at_h(book, 0.2, path="F")
    assert book.eval_calls == calls + 1


def test_path_r_pnl_and_car_share_one_evaluation() -> None:
    book = _long_book(p_mid=20.0, p_cross=-10.0, qty=2.0)
    metrics = evaluate_path_at_h(book, 0.15, path="R")
    assert book.eval_calls == 1
    assert float(evaluate_path_at_h(book, 0.15, path="R")["pnl"]) == pytest.approx(metrics["pnl"])
    assert float(evaluate_path_at_h(book, 0.15, path="R")["car"]) == pytest.approx(metrics["car"])
    assert book.eval_calls == 1


def test_warm_h_det_does_not_repeat_grid_evals() -> None:
    book = _long_book(p_mid=20.0, p_cross=-10.0, qty=2.0)
    for h in H_DET:
        evaluate_path_at_h(book, h, path="R")
    calls_after_grid = book.eval_calls
    assert calls_after_grid == len(H_DET)

    def pnl_r(h: float) -> float:
        return float(evaluate_path_at_h(book, h, path="R")["pnl"])

    h_star = first_adverse_crossing(pnl_r, 0.0)
    assert h_star is not None
    # Midpoints and bisection may add points; the 101 H_det values must not recompute.
    assert book.eval_calls < calls_after_grid + len(H_DET)
    fresh = _long_book(p_mid=20.0, p_cross=-10.0, qty=2.0)
    h_uncached = first_adverse_crossing(
        lambda h: float(evaluate_path_at_h(fresh, h, path="R")["pnl"]),
        0.0,
    )
    assert h_star == pytest.approx(h_uncached)


def test_join_recovers_direction_from_trade_key() -> None:
    mid, cross = _paired_trade_frames()
    indexed = mid.set_index(["trade_date", "ticker", "direction"])
    assert "direction" not in indexed.columns
    joined = join_paired_trades(mid, cross)
    assert "direction" in joined.columns
    assert set(joined["direction"]) == {"long", "short"}
    assert len(joined) == 2
    assert float(joined.loc[joined["direction"] == "long", "quantity_mid"].iloc[0]) == 2.0


def test_primary_calendar_filters_and_requires_traded_dates() -> None:
    status = pd.DataFrame(
        [
            {"trade_date": date(2019, 12, 30), "status": "traded", "reason": "pre"},
            {"trade_date": date(2020, 1, 6), "status": "traded", "reason": "ok"},
            {"trade_date": date(2021, 2, 1), "status": "traded", "reason": "ok"},
            {"trade_date": date(2021, 2, 8), "status": "valid_no_trade", "reason": "skip"},
            {"trade_date": date(2026, 7, 17), "status": "traded", "reason": "post"},
        ]
    )
    filtered = filter_primary_date_status(status)
    dates = set(filtered["trade_date"])
    assert date(2019, 12, 30) not in dates
    assert date(2026, 7, 17) not in dates
    assert dates <= {d for d in dates if PRIMARY_START <= d <= PRIMARY_END}
    ok = filter_primary_date_status(status, require_traded_dates=2)
    assert int(ok.loc[ok["status"] == "traded", "trade_date"].nunique()) == 2
    with pytest.raises(D3AnalysisError, match="primary traded dates"):
        filter_primary_date_status(status, require_traded_dates=341)


def test_car_endpoint_reconciliation() -> None:
    book = _tier_a_matched_long_book()
    f0 = evaluate_path_at_h(book, 0.0, path="F")
    r0 = evaluate_path_at_h(book, 0.0, path="R")
    r1 = evaluate_path_at_h(book, 1.0, path="R")
    assert f0["car"] == pytest.approx(r0["car"], abs=CAR_TOLERANCE)
    book.car_mid_ref = f0["car"]
    book.car_cross_ref = r1["car"]
    rows = {row["metric"]: row for row in reconcile_d3_endpoints(book, official=False)}
    assert rows["F_h0_car"]["passed"]
    assert rows["R_h0_car"]["passed"]
    assert rows["R_h1_car"]["passed"]
    assert "F_h1_car" not in rows
    assert abs(rows["F_h0_car"]["delta"]) <= CAR_TOLERANCE
    missing = _tier_a_matched_long_book()
    with pytest.raises(D3AnalysisError, match="CAR references missing"):
        reconcile_d3_endpoints(missing, official=True)
    book.car_mid_ref = float(f0["car"]) + 0.01
    failed = {row["metric"]: row for row in reconcile_d3_endpoints(book, official=False)}
    assert failed["F_h0_car"]["passed"] is False
    assert failed["R_h0_car"]["passed"] is False


def test_side_snapshot_uses_exact_off_grid_root() -> None:
    h_star = 0.033
    assert h_star not in H_VIS
    book = _long_book(p_mid=20.0, p_cross=-10.0, qty=2.0)
    expected = evaluate_path_at_h(book, h_star, path="R")
    snap = side_snapshots_at_roots(book, {"h=0": 0.0, "h_R_50": h_star, "h=1": 1.0})
    row = snap.loc[snap["mark"] == "h_R_50"].iloc[0]
    assert row["h"] == pytest.approx(h_star)
    assert row["pnl"] == pytest.approx(expected["pnl"])
    assert row["pnl_long"] == pytest.approx(expected["pnl_long"])
    assert row["pnl_short"] == pytest.approx(expected["pnl_short"])
    vis_hit = snap.loc[snap["h"].isin(H_VIS) & (snap["mark"] == "h_R_50")]
    assert vis_hit.empty


def test_evidence_exporter_writes_design_files(tmp_path) -> None:
    result = assemble_envelope(
        h_R_50=0.4,
        h_R_25=0.5,
        h_R_P0=0.6,
        h_R_CAR0=0.55,
        h_F_50=0.2,
        h_F_25=0.3,
        h_F_P0=0.35,
        h_F_CAR0=0.33,
        crossings=[Crossing("R", "primary", "pnl_50", 50.0, 0.4, "bracket_bisection")],
        curves=pd.DataFrame([{"h": 0.0, "path": "R", "pnl": 1.0}]),
        reconciliation=[],
        monotonicity={},
        manifest={"official": False},
        side_snapshot=pd.DataFrame([{"mark": "h_R_50", "h": 0.4, "pnl_long": 1.0, "pnl_short": 0.0}]),
    )
    paths = export_d3_evidence(result, tmp_path)
    payload = (tmp_path / "d3_envelope.json").read_text(encoding="utf-8")
    assert "h_req" not in payload
    assert paths["envelope"].exists()
    assert paths["curves"].exists()
    assert paths["crossings"].exists()
    assert paths["side_snapshot"].exists()
    assert not (tmp_path / "d3_execution_envelope.executed.ipynb").exists()
    with pytest.raises(D3AnalysisError, match="result is required"):
        export_d3_evidence(evidence_dir=tmp_path)


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


def _paired_trade_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    mid = pd.DataFrame(
        [
            {
                "trade_date": date(2021, 1, 4),
                "ticker": "AAA",
                "direction": "long",
                "quantity": 2.0,
                "pnl_per_share": 10.0,
                "pnl_total": 20.0,
                "entry_cost_per_share": 4.0,
                "net_credit_per_share": -4.0,
                "max_loss_per_share": 4.0,
                "instrument_type": "long_straddle",
            },
            {
                "trade_date": date(2021, 1, 4),
                "ticker": "BBB",
                "direction": "short",
                "quantity": -3.0,
                "pnl_per_share": 5.0,
                "pnl_total": 15.0,
                "entry_cost_per_share": -2.0,
                "net_credit_per_share": 2.0,
                "max_loss_per_share": 8.0,
                "instrument_type": "iron_fly",
            },
        ]
    )
    cross = mid.copy()
    cross["quantity"] = [-1.0, -6.0]
    cross["pnl_per_share"] = [-4.0, -1.0]
    cross["pnl_total"] = [4.0, 6.0]
    cross["entry_cost_per_share"] = [6.0, -1.0]
    cross["net_credit_per_share"] = [-6.0, 1.0]
    cross["max_loss_per_share"] = [6.0, 9.0]
    return mid, cross


def _tier_a_matched_long_book() -> D3Book:
    """Long-only book whose Path R size at h=0/1 matches quantity_mid/cross."""
    trade_date = date(2021, 1, 4)
    qty_mid = 10000.0 / 4.0
    qty_cross = 10000.0 / 6.0
    trades = pd.DataFrame(
        [
            {
                "trade_date": trade_date,
                "ticker": "AAA",
                "direction": "long",
                "quantity_mid": qty_mid,
                "quantity_cross": qty_cross,
                "pnl_per_share_mid": 2.0,
                "pnl_per_share_cross": -1.0,
                "pnl_total_mid": qty_mid * 2.0,
                "pnl_total_cross": qty_cross * -1.0,
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
    status = pd.DataFrame([{"trade_date": trade_date, "status": "traded", "reason": "ok"}])
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
