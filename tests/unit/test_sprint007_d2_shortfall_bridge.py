"""Sprint 007 D2A bridge tests — synthetic frames only; no official economics."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest.sprint007_artifact_validation import D0ValidationResult, GateResult
from src.backtest.sprint007_d1_gross_margin import (
    VERDICT_CONTINUE,
    VERDICT_STOP,
    D1Result,
    D1Scorecard,
    ReconciliationResult,
)
from src.backtest.sprint007_d2_shortfall_bridge import (
    BRANCH_BODY_WING,
    BRANCH_NONE,
    BRANCH_SIZING,
    BRANCH_TRADABILITY,
    CLASS_EXECUTION,
    CLASS_MIXED,
    CLASS_SIZING,
    CLASS_STRUCTURE,
    VERDICT_BLOCKED,
    BridgeTerms,
    check_leg_to_trade,
    classify_d2a,
    compute_bridge_terms,
    run_d2a_analysis,
    side_bridge_rows,
)


def _trade(
    *,
    trade_date: date,
    ticker: str,
    direction: str,
    quantity: float,
    pnl_per_share: float,
    instrument_type: str | None = None,
) -> dict:
    qty = abs(float(quantity))
    inst = instrument_type or ("iron_fly" if direction == "short" else "straddle")
    return {
        "trade_date": trade_date,
        "ticker": ticker,
        "direction": direction,
        "included_in_portfolio": True,
        "quantity": qty if direction == "long" else -qty,
        "pnl_per_share": pnl_per_share,
        "pnl_total": qty * pnl_per_share,
        "instrument_type": inst,
        "fill_label": "mid",
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _leg(*, trade: dict, pnl_per_unit: float, fill: str = "mid") -> dict:
    return {
        "trade_date": trade["trade_date"],
        "ticker": trade["ticker"],
        "direction": trade["direction"],
        "expiry_date": trade["trade_date"],
        "option_type": "call",
        "strike": 100.0,
        "leg_index": 0,
        "unit_quantity": 1,
        "pnl_per_unit": pnl_per_unit,
        "pnl_total_leg": abs(trade["quantity"]) * pnl_per_unit,
        "included_in_portfolio": True,
        "fill_label": fill,
    }


def test_price_only_bridge_puts_the_gap_in_delta_price() -> None:
    mid = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=10.0, pnl_per_share=2.0)]
    )
    cross = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=10.0, pnl_per_share=1.0)]
    )
    terms = compute_bridge_terms(mid, cross)
    assert terms.gap == pytest.approx(-10.0)
    assert terms.delta_price == pytest.approx(-10.0)
    assert terms.delta_size == pytest.approx(0.0)
    assert terms.delta_set == pytest.approx(0.0)
    assert terms.residual == pytest.approx(0.0)
    assert terms.interaction == pytest.approx(0.0)
    assert terms.unmatched_keys == 0


def test_size_only_bridge_puts_the_gap_in_delta_size() -> None:
    mid = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="short", quantity=10.0, pnl_per_share=2.0)]
    )
    cross = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="short", quantity=4.0, pnl_per_share=2.0)]
    )
    terms = compute_bridge_terms(mid, cross)
    assert terms.gap == pytest.approx(-12.0)
    assert terms.delta_price == pytest.approx(0.0)
    assert terms.delta_size == pytest.approx(-12.0)
    assert terms.residual == pytest.approx(0.0)
    assert terms.interaction == pytest.approx(0.0)


def test_interaction_equals_laspeyres_paasche_price_difference() -> None:
    mid = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=10.0, pnl_per_share=2.0)]
    )
    cross = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=11.0, pnl_per_share=1.0)]
    )
    terms = compute_bridge_terms(mid, cross)
    assert terms.delta_price == pytest.approx(-10.0)
    assert terms.delta_size == pytest.approx(1.0)
    assert terms.gap == pytest.approx(-9.0)
    assert terms.residual == pytest.approx(0.0)
    assert abs(terms.interaction) == pytest.approx(abs(terms.delta_price - terms.delta_price_paasche))
    assert terms.s_order == pytest.approx(1.0 / 9.0)


def test_unmatched_keys_are_isolated_in_delta_set() -> None:
    mid = _frame(
        [
            _trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=1.0, pnl_per_share=10.0),
            _trade(trade_date=date(2021, 1, 4), ticker="BBB", direction="long", quantity=1.0, pnl_per_share=5.0),
        ]
    )
    cross = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=1.0, pnl_per_share=8.0)]
    )
    terms = compute_bridge_terms(mid, cross)
    assert terms.n_mid_only == 1
    assert terms.n_cross_only == 0
    assert terms.delta_set == pytest.approx(-5.0)
    assert terms.delta_price == pytest.approx(-2.0)
    assert terms.gap == pytest.approx(-7.0)
    assert terms.residual == pytest.approx(0.0)


def test_inconsistent_pnl_total_fails_residual_identity() -> None:
    mid = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=1.0, pnl_per_share=50.0)]
    )
    cross = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=1.0, pnl_per_share=40.0)]
    )
    mid = mid.copy()
    mid.loc[0, "pnl_total"] = 100.0
    terms = compute_bridge_terms(mid, cross)
    assert abs(terms.residual) > 0.01


def test_leg_to_trade_detects_pnl_mismatch() -> None:
    trade = _trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=2.0, pnl_per_share=3.0)
    trades = _frame([trade])
    good = check_leg_to_trade(trades, pd.DataFrame([_leg(trade=trade, pnl_per_unit=3.0)]))
    assert all(item.passed for item in good)
    bad = check_leg_to_trade(trades, pd.DataFrame([_leg(trade=trade, pnl_per_unit=1.0)]))
    assert any(not item.passed for item in bad)


def _classify(terms: BridgeTerms, side_price: dict[str, float], *, short_iron: bool = True) -> object:
    return classify_d2a(
        terms,
        blocked=False,
        side_delta_price=side_price,
        short_is_iron_fly=short_iron,
    )


def test_size_material_selects_sizing_branch_and_mixed_when_price_also_material() -> None:
    mid = _frame(
        [
            _trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="short", quantity=10.0, pnl_per_share=4.0),
            _trade(trade_date=date(2021, 1, 4), ticker="BBB", direction="long", quantity=10.0, pnl_per_share=4.0),
        ]
    )
    cross = _frame(
        [
            _trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="short", quantity=4.0, pnl_per_share=2.0),
            _trade(trade_date=date(2021, 1, 4), ticker="BBB", direction="long", quantity=4.0, pnl_per_share=2.0),
        ]
    )
    terms = compute_bridge_terms(mid, cross)
    sides = {row["slice"]: row["delta_price"] for row in side_bridge_rows(mid, cross)}
    cls = _classify(terms, sides)
    assert cls.price_material
    assert cls.size_material
    assert cls.d2b_branch == BRANCH_SIZING
    assert cls.provisional_d3_class == CLASS_MIXED


def test_price_dominant_short_concentration_recommends_body_wing() -> None:
    mid = _frame(
        [
            _trade(trade_date=date(2021, 1, 4), ticker="FLY", direction="short", quantity=10.0, pnl_per_share=5.0),
            _trade(trade_date=date(2021, 1, 4), ticker="STR", direction="long", quantity=10.0, pnl_per_share=1.0),
        ]
    )
    cross = _frame(
        [
            _trade(trade_date=date(2021, 1, 4), ticker="FLY", direction="short", quantity=10.0, pnl_per_share=1.0),
            _trade(trade_date=date(2021, 1, 4), ticker="STR", direction="long", quantity=10.0, pnl_per_share=0.8),
        ]
    )
    terms = compute_bridge_terms(mid, cross)
    sides = {row["slice"]: row["delta_price"] for row in side_bridge_rows(mid, cross)}
    cls = _classify(terms, sides)
    assert cls.price_material
    assert not cls.size_material
    assert cls.dominant == "price"
    assert cls.concentrating_side == "short"
    assert cls.structure
    assert cls.d2b_branch == BRANCH_BODY_WING
    assert cls.provisional_d3_class == CLASS_STRUCTURE


def test_price_dominant_long_concentration_recommends_tradability() -> None:
    mid = _frame(
        [
            _trade(trade_date=date(2021, 1, 4), ticker="STR", direction="long", quantity=10.0, pnl_per_share=5.0),
            _trade(trade_date=date(2021, 1, 4), ticker="FLY", direction="short", quantity=10.0, pnl_per_share=1.0),
        ]
    )
    cross = _frame(
        [
            _trade(trade_date=date(2021, 1, 4), ticker="STR", direction="long", quantity=10.0, pnl_per_share=1.0),
            _trade(trade_date=date(2021, 1, 4), ticker="FLY", direction="short", quantity=10.0, pnl_per_share=0.8),
        ]
    )
    terms = compute_bridge_terms(mid, cross)
    sides = {row["slice"]: row["delta_price"] for row in side_bridge_rows(mid, cross)}
    cls = _classify(terms, sides)
    assert cls.concentrating_side == "long"
    assert cls.d2b_branch == BRANCH_TRADABILITY
    assert cls.provisional_d3_class == CLASS_STRUCTURE


def test_diffuse_price_gap_is_execution_focused() -> None:
    mid = _frame(
        [
            _trade(trade_date=date(2021, 1, 4), ticker="STR", direction="long", quantity=10.0, pnl_per_share=3.0),
            _trade(trade_date=date(2021, 1, 4), ticker="FLY", direction="short", quantity=10.0, pnl_per_share=3.0),
        ]
    )
    cross = _frame(
        [
            _trade(trade_date=date(2021, 1, 4), ticker="STR", direction="long", quantity=10.0, pnl_per_share=1.0),
            _trade(trade_date=date(2021, 1, 4), ticker="FLY", direction="short", quantity=10.0, pnl_per_share=1.0),
        ]
    )
    terms = compute_bridge_terms(mid, cross)
    sides = {row["slice"]: row["delta_price"] for row in side_bridge_rows(mid, cross)}
    cls = _classify(terms, sides)
    assert cls.concentrating_side is None
    assert not cls.structure
    assert cls.d2b_branch == BRANCH_TRADABILITY
    assert cls.provisional_d3_class == CLASS_EXECUTION


def test_s_order_above_ten_percent_is_not_mixed_when_dual_agrees() -> None:
    mid = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=10.0, pnl_per_share=2.0)]
    )
    cross = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=11.0, pnl_per_share=1.0)]
    )
    terms = compute_bridge_terms(mid, cross)
    assert terms.s_order > 0.10
    cls = _classify(terms, {"long": terms.delta_price, "short": 0.0})
    assert not cls.order_sensitive
    assert cls.provisional_d3_class == CLASS_STRUCTURE
    assert cls.d2b_branch == BRANCH_TRADABILITY


def test_order_sensitive_flag_when_dual_makes_size_material() -> None:
    mid = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=10.0, pnl_per_share=5.0)]
    )
    cross = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=13.0, pnl_per_share=1.0)]
    )
    terms = compute_bridge_terms(mid, cross)
    # Δp=-4, ΔQ=3; Δ_price=-40; Δ_size=3; G=13-50=-37
    # |Δ_size|/37=3/37<0.25; dual Δ_size=5*3=15; 15/37=0.405 >= 0.25
    # dual Δ_price=13*(-4)=-52; dominant still price both ways; size materiality changes.
    cls = classify_d2a(
        terms,
        blocked=False,
        side_delta_price={"long": terms.delta_price, "short": 0.0},
        short_is_iron_fly=False,
    )
    assert not cls.size_material
    assert cls.order_sensitive
    assert cls.provisional_d3_class == CLASS_MIXED
    assert cls.d2b_branch == BRANCH_TRADABILITY


def test_blocked_classification_has_no_d2b_branch() -> None:
    terms = compute_bridge_terms(
        _frame([_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=1.0, pnl_per_share=1.0)]),
        _frame([_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="long", quantity=1.0, pnl_per_share=0.0)]),
    )
    cls = classify_d2a(
        terms,
        blocked=True,
        side_delta_price={"long": terms.delta_price, "short": 0.0},
        short_is_iron_fly=False,
    )
    assert cls.d2b_branch == BRANCH_NONE
    assert cls.provisional_d3_class == VERDICT_BLOCKED


def test_size_only_class_is_sizing_aware() -> None:
    mid = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="short", quantity=10.0, pnl_per_share=2.0)]
    )
    cross = _frame(
        [_trade(trade_date=date(2021, 1, 4), ticker="AAA", direction="short", quantity=4.0, pnl_per_share=2.0)]
    )
    terms = compute_bridge_terms(mid, cross)
    cls = classify_d2a(
        terms,
        blocked=False,
        side_delta_price={"long": 0.0, "short": 0.0},
        short_is_iron_fly=True,
    )
    assert cls.size_material
    assert not cls.price_material
    assert cls.d2b_branch == BRANCH_SIZING
    assert cls.provisional_d3_class == CLASS_SIZING


def _failed_d0() -> D0ValidationResult:
    return D0ValidationResult(
        verdict="BLOCKED_BY_SPECIFIC_EVIDENCE_GAP",
        gates=[GateResult("G1", False, "receipt hash mismatch")],
    )


def test_failed_d0_blocks_before_loading_economics(monkeypatch: pytest.MonkeyPatch) -> None:
    def _must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("D2A must not load economics after a failed D0")

    monkeypatch.setattr(
        "src.backtest.sprint007_d2_shortfall_bridge.run_d1_analysis", _must_not_run
    )
    monkeypatch.setattr(
        "src.backtest.sprint007_d2_shortfall_bridge.load_fill_primary_tables", _must_not_run
    )
    result = run_d2a_analysis(d0_result=_failed_d0())
    assert result.blocked
    assert result.verdict == VERDICT_BLOCKED
    assert "D0 prerequisite failed" in str(result.blocker)


def test_failed_d1_blocks_before_loading_paired_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    passed_d0 = D0ValidationResult(
        verdict="READY_WITH_NARROW_ENABLING_CHANGE",
        gates=[GateResult("G1", True, "ok")],
    )
    d1 = D1Result(
        verdict=VERDICT_STOP,
        scorecard=D1Scorecard(verdict=VERDICT_STOP),
        reconciliation=ReconciliationResult(),
    )

    def _must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("D2A must not open paired artifacts after a D1 stop")

    monkeypatch.setattr(
        "src.backtest.sprint007_d2_shortfall_bridge.load_fill_primary_tables", _must_not_run
    )
    result = run_d2a_analysis(d0_result=passed_d0, d1_result=d1)
    assert result.blocked
    assert result.verdict == VERDICT_BLOCKED
    assert VERDICT_STOP in str(result.blocker)
    assert result.verdict != VERDICT_CONTINUE
