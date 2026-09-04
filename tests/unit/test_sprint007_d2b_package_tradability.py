"""Sprint 007 D2B package-tradability tests — synthetic frames only."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.backtest.sprint007_d2_shortfall_bridge import (
    BRANCH_SIZING,
    BRANCH_TRADABILITY,
    CLASS_EXECUTION,
    CLASS_MIXED,
    CLASS_SIZING,
    CLASS_STRUCTURE,
    VERDICT_BLOCKED,
    BridgeTerms,
    D2AClassification,
    D2AResult,
    assign_final_d3_class,
    classify_d2a,
    compute_bridge_terms,
)
from src.backtest.sprint007_d2b_package_tradability import (
    D2BAnalysisError,
    assign_within_group_terciles,
    attach_terciles,
    check_shared_quotes,
    midpoint_package_cashflow,
    package_half_spread,
    package_metrics_by_trade,
    package_width_to_cashflow,
    per_trade_delta_price,
    run_d2b_analysis,
    run_d2b_tradability,
)


def test_package_formulas_on_two_legs() -> None:
    uq = np.array([1.0, -1.0])
    bid = np.array([1.0, 2.0])
    ask = np.array([3.0, 4.0])
    half = package_half_spread(uq, bid, ask)
    cash = midpoint_package_cashflow(uq, bid, ask)
    assert half == pytest.approx(2.0)
    assert cash == pytest.approx(-1.0)
    assert package_width_to_cashflow(half, cash) == pytest.approx(2.0)


def test_zero_cashflow_width_is_none() -> None:
    uq = np.array([1.0, -1.0])
    bid = np.array([1.0, 1.0])
    ask = np.array([3.0, 3.0])
    cash = midpoint_package_cashflow(uq, bid, ask)
    assert cash == pytest.approx(0.0)
    assert package_width_to_cashflow(1.0, cash) is None


def test_package_metrics_group_by_trade() -> None:
    legs = pd.DataFrame(
        [
            {
                "trade_date": date(2021, 1, 4),
                "ticker": "AAA",
                "direction": "long",
                "expiry_date": date(2021, 1, 8),
                "option_type": "call",
                "strike": 10.0,
                "leg_index": 0,
                "unit_quantity": 1,
                "bid": 1.0,
                "ask": 3.0,
            },
            {
                "trade_date": date(2021, 1, 4),
                "ticker": "AAA",
                "direction": "long",
                "expiry_date": date(2021, 1, 8),
                "option_type": "put",
                "strike": 10.0,
                "leg_index": 1,
                "unit_quantity": 1,
                "bid": 1.0,
                "ask": 3.0,
            },
        ]
    )
    metrics = package_metrics_by_trade(legs)
    assert len(metrics) == 1
    assert metrics.iloc[0]["package_half_spread"] == pytest.approx(2.0)
    assert metrics.iloc[0]["midpoint_package_cashflow"] == pytest.approx(4.0)
    assert metrics.iloc[0]["package_width_to_cashflow"] == pytest.approx(0.5)


def test_terciles_are_deterministic_within_a_group() -> None:
    widths = [float(i) for i in range(1, 10)]
    frame = pd.DataFrame(
        {
            "package_width_to_cashflow": widths,
            "trade_date": [date(2021, 1, d) for d in range(1, 10)],
            "ticker": [f"T{i}" for i in range(9)],
        }
    )
    assigned = assign_within_group_terciles(frame)
    assert assigned["tercile"].tolist() == [1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_terciles_are_not_pooled_across_direction() -> None:
    rows = []
    for direction, offset in (("long", 0.0), ("short", 100.0)):
        for i in range(3):
            rows.append(
                {
                    "direction": direction,
                    "instrument_type": "straddle" if direction == "long" else "iron_fly",
                    "package_width_to_cashflow": offset + i + 1,
                    "trade_date": date(2021, 1, i + 1),
                    "ticker": f"{direction}{i}",
                    "pnl_total_mid": 1.0,
                    "delta_price": -1.0,
                }
            )
    attached = attach_terciles(pd.DataFrame(rows))
    longs = attached.loc[attached["direction"] == "long"].sort_values("package_width_to_cashflow")
    shorts = attached.loc[attached["direction"] == "short"].sort_values("package_width_to_cashflow")
    assert longs["tercile"].tolist() == [1, 2, 3]
    assert shorts["tercile"].tolist() == [1, 2, 3]


def test_zero_cashflow_rows_are_skipped_not_ranked() -> None:
    trades = pd.DataFrame(
        [
            {
                "direction": "long",
                "instrument_type": "straddle",
                "package_width_to_cashflow": 0.2,
                "trade_date": date(2021, 1, 4),
                "ticker": "AAA",
                "pnl_total_mid": 5.0,
                "delta_price": -2.0,
            },
            {
                "direction": "long",
                "instrument_type": "straddle",
                "package_width_to_cashflow": None,
                "trade_date": date(2021, 1, 5),
                "ticker": "BBB",
                "pnl_total_mid": 1.0,
                "delta_price": -3.0,
            },
            {
                "direction": "long",
                "instrument_type": "straddle",
                "package_width_to_cashflow": 0.8,
                "trade_date": date(2021, 1, 6),
                "ticker": "CCC",
                "pnl_total_mid": 2.0,
                "delta_price": -4.0,
            },
        ]
    )
    attached = attach_terciles(trades)
    skipped = attached.loc[attached["tercile"].isna()]
    ranked = attached.loc[attached["tercile"].notna()]
    assert len(skipped) == 1
    assert skipped.iloc[0]["ticker"] == "BBB"
    assert ranked.sort_values("package_width_to_cashflow")["tercile"].tolist() == [1, 2]


def _trade(ticker: str, direction: str, qty: float, pnl_per_share: float, inst: str) -> dict:
    return {
        "trade_date": date(2021, 1, 4),
        "ticker": ticker,
        "direction": direction,
        "quantity": qty if direction == "long" else -qty,
        "pnl_per_share": pnl_per_share,
        "pnl_total": abs(qty) * pnl_per_share,
        "instrument_type": inst,
    }


def _legs(ticker: str, direction: str, bid: float, ask: float, uq: tuple[int, ...] = (1, 1)) -> list[dict]:
    return [
        {
            "trade_date": date(2021, 1, 4),
            "ticker": ticker,
            "direction": direction,
            "expiry_date": date(2021, 1, 8),
            "option_type": "call" if i == 0 else "put",
            "strike": 10.0 + i,
            "leg_index": i,
            "unit_quantity": uq[i],
            "bid": bid,
            "ask": ask,
        }
        for i in range(len(uq))
    ]


def test_per_trade_delta_price_reconciles_to_d2a_bridge() -> None:
    mid = pd.DataFrame(
        [
            _trade("AAA", "long", 10.0, 2.0, "straddle"),
            _trade("BBB", "short", 5.0, 3.0, "iron_fly"),
        ]
    )
    cross = pd.DataFrame(
        [
            _trade("AAA", "long", 10.0, 1.0, "straddle"),
            _trade("BBB", "short", 5.0, 1.0, "iron_fly"),
        ]
    )
    per_trade = per_trade_delta_price(mid, cross)
    terms = compute_bridge_terms(mid, cross)
    assert float(per_trade["delta_price"].sum()) == pytest.approx(terms.delta_price)


def test_tradability_reconciliation_and_expensive_share() -> None:
    names = [f"T{i}" for i in range(6)]
    mid_rows = [_trade(name, "long", 1.0, 3.0, "straddle") for name in names]
    cross_rows = [_trade(name, "long", 1.0, 1.0, "straddle") for name in names]
    mid = pd.DataFrame(mid_rows)
    cross = pd.DataFrame(cross_rows)
    mid_legs = pd.DataFrame(
        [
            row
            for i, name in enumerate(names)
            for row in _legs(name, "long", bid=1.0, ask=1.0 + 0.2 * (i + 1))
        ]
    )
    cross_legs = mid_legs.copy()
    terms = compute_bridge_terms(mid, cross)
    trades, groups, book, skipped, recon = run_d2b_tradability(
        mid_trades=mid,
        cross_trades=cross,
        mid_legs=mid_legs,
        cross_legs=cross_legs,
        d2a_delta_price=terms.delta_price,
        d2a_p_mid=terms.p_mid,
    )
    assert all(row["passed"] for row in recon)
    assert skipped["n_trades"] == 0
    assert {row["tercile"] for row in book} == {1, 2, 3}
    expensive = next(row for row in book if row["tercile"] == 3)
    assert expensive["n_trades"] == 2
    assert expensive["delta_price"] == pytest.approx(-4.0)
    assert terms.delta_price == pytest.approx(-12.0)
    assert abs(expensive["delta_price"]) / abs(terms.delta_price) == pytest.approx(1.0 / 3.0)
    outside = trades.loc[trades["tercile"].isin([1, 2]), "pnl_total_mid"].sum()
    assert outside == pytest.approx(12.0)
    assert groups  # within-group table exists


def test_quote_mismatch_is_a_d2b_failure() -> None:
    mid_legs = pd.DataFrame(_legs("AAA", "long", 1.0, 2.0))
    cross_legs = pd.DataFrame(_legs("AAA", "long", 1.0, 3.0))
    with pytest.raises(D2BAnalysisError, match="shared quote mismatch"):
        check_shared_quotes(mid_legs, cross_legs)


def _terms(price: float, size: float, gap: float | None = None) -> BridgeTerms:
    gap = price + size if gap is None else gap
    return BridgeTerms(
        p_mid=100.0,
        p_cross=100.0 + gap,
        p_cross_at_q_mid=100.0 + price,
        gap=gap,
        delta_price=price,
        delta_size=size,
        delta_set=0.0,
        residual=0.0,
        n_intersection=10,
        n_mid_only=0,
        n_cross_only=0,
        interaction=0.0,
        delta_price_paasche=price,
        delta_size_dual=size,
        s_order=0.0,
    )


def test_final_class_follows_precedence() -> None:
    price_only = classify_d2a(
        _terms(-80.0, 5.0, -75.0),
        blocked=False,
        side_delta_price={"long": -40.0, "short": -40.0},
        short_is_iron_fly=True,
    )
    assert price_only.d2b_branch == BRANCH_TRADABILITY
    assert assign_final_d3_class(price_only, blocked=False) == CLASS_EXECUTION

    structured = classify_d2a(
        _terms(-80.0, 5.0, -75.0),
        blocked=False,
        side_delta_price={"long": -10.0, "short": -70.0},
        short_is_iron_fly=True,
    )
    assert structured.structure
    assert assign_final_d3_class(structured, blocked=False) == CLASS_STRUCTURE

    mixed = classify_d2a(
        _terms(-40.0, -40.0, -80.0),
        blocked=False,
        side_delta_price={"long": -20.0, "short": -20.0},
        short_is_iron_fly=True,
    )
    assert mixed.d2b_branch == BRANCH_SIZING
    assert assign_final_d3_class(mixed, blocked=False) == CLASS_MIXED

    sizing = classify_d2a(
        _terms(0.0, -80.0, -80.0),
        blocked=False,
        side_delta_price={"long": 0.0, "short": 0.0},
        short_is_iron_fly=True,
    )
    assert assign_final_d3_class(sizing, blocked=False) == CLASS_SIZING
    assert assign_final_d3_class(price_only, blocked=True) == VERDICT_BLOCKED


def test_run_d2b_does_not_run_tradability_on_sizing_branch() -> None:
    cls = D2AClassification(
        price_material=False,
        size_material=True,
        dominant="size",
        order_sensitive=False,
        concentrating_side=None,
        structure=False,
        d2b_branch=BRANCH_SIZING,
        provisional_d3_class=CLASS_SIZING,
    )
    d2a = D2AResult(
        verdict=CLASS_SIZING,
        blocked=False,
        blocker=None,
        bridge=_terms(0.0, -80.0, -80.0),
        classification=cls,
    )
    result = run_d2b_analysis(d2a_result=d2a)
    assert result.final_d3_class == CLASS_SIZING
    assert result.book_terciles == []
    assert "tradability not selected" in str(result.blocker)


def test_blocked_d2a_yields_blocked_final_class() -> None:
    d2a = D2AResult(
        verdict=VERDICT_BLOCKED,
        blocked=True,
        blocker="residual failed",
        bridge=None,
        classification=None,
    )
    result = run_d2b_analysis(d2a_result=d2a)
    assert result.blocked
    assert result.final_d3_class == VERDICT_BLOCKED
